/*
 * llama-engine.js -- a small Llama inference engine in plain JavaScript.
 *
 * Runs a llama2.c-format model (stories15M) entirely inside the browser or Node:
 * no server, no WebAssembly, no network. Weights are int8 with one float scale
 * per row, which keeps a 15M-parameter model at ~15 MB with output that matches
 * a full-precision run.
 *
 * The maths mirrors local_llama/np_llama.py line for line (interleaved RoPE,
 * RMSNorm with eps 1e-5, SwiGLU, grouped-query attention, tied embeddings), so
 * the two can be diffed token for token.
 *
 * Weight container (.llm file, all little-endian):
 *
 *     magic  "LLM1"
 *     int32  version = 1
 *     int32  quant            0 = fp32, 1 = int8 rows
 *     int32  dim, hidden, layers, heads, kv_heads, vocab, seq
 *     then tensors in llama2.c order, each:
 *         int32 kind          0 = fp32, 1 = int8+row scales, 2 = alias token_embd
 *         int32 rows
 *         int32 cols
 *         float32[rows] scales      (kind 1 only)
 *         int8[rows*cols]           (kind 1)   |   float32[rows*cols]  (kind 0)
 */

(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.LlamaEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var ROPE_THETA = 10000.0;
  var RMS_EPS = 1e-5;
  var BOS_ID = 1;
  var EOS_ID = 2;

  var TENSOR_ORDER = [
    "token_embd", "rms_att", "wq", "wk", "wv", "wo", "rms_ffn",
    "w1", "w2", "w3", "rms_final", "wcls",
  ];

  /* Per-layer tensors: `rows` counts one layer only, the buffer holds all of
     them, so a layer's slice starts at layer*rows*cols. */
  var STACKED = { wq: 1, wk: 1, wv: 1, wo: 1, w1: 1, w2: 1, w3: 1 };

  // ---------------------------------------------------------------- utils --- //

  function bytesFromBase64(b64) {
    var bin = atob(b64);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function latin1(bytes, start, end) {
    var s = "";
    for (var i = start; i < end; i++) s += String.fromCharCode(bytes[i]);
    return s;
  }

  // ------------------------------------------------------------ tokenizer --- //

  /*
   * Reads llama2.c's tokenizer.bin: an int32 max token length, then a flat list
   * of (float score, int32 length, bytes) pieces. Encoding is greedy longest
   * match over raw UTF-8 bytes with <0xNN> byte fallback, which is what
   * karpathy's run.c does -- so it agrees with the Python reference.
   */
  function Tokenizer(buffer) {
    var view = new DataView(buffer);
    var bytes = new Uint8Array(buffer);
    var offset = 0;
    this.maxTokenLength = view.getInt32(0, true);
    offset = 4;

    this.pieces = [];       // Uint8Array per token, for decoding
    this.lookup = new Map(); // latin1 string -> id, for encoding
    this.scores = [];

    while (offset + 8 <= bytes.length) {
      var score = view.getFloat32(offset, true);
      var len = view.getInt32(offset + 4, true);
      offset += 8;
      if (len < 0 || offset + len > bytes.length) break;
      var piece = bytes.subarray(offset, offset + len);
      offset += len;
      var id = this.pieces.length;
      this.pieces.push(piece);
      this.scores.push(score);
      this.lookup.set(latin1(piece, 0, piece.length), id);
    }
    this.vocabSize = this.pieces.length;
    this.bosId = BOS_ID;
    this.eosId = EOS_ID;
  }

  Tokenizer.prototype.encode = function (text, addBos) {
    var raw = new TextEncoder().encode(text);
    var ids = [];
    if (addBos === undefined || addBos) ids.push(this.bosId);
    var i = 0;
    while (i < raw.length) {
      var best = -1, bestLen = 0;
      var limit = Math.min(this.maxTokenLength, raw.length - i);
      for (var len = limit; len > 0; len--) {
        var candidate = latin1(raw, i, i + len);
        var id = this.lookup.get(candidate);
        if (id !== undefined) { best = id; bestLen = len; break; }
      }
      if (best < 0) {
        var hex = "<0x" + ("0" + raw[i].toString(16).toUpperCase()).slice(-2) + ">";
        var fallback = this.lookup.get(hex);
        ids.push(fallback === undefined ? 0 : fallback);
        i += 1;
      } else {
        ids.push(best);
        i += bestLen;
      }
    }
    return ids;
  };

  Tokenizer.prototype.decodeBytes = function (ids) {
    var total = 0, k;
    for (k = 0; k < ids.length; k++) {
      var id = ids[k];
      if (id === this.bosId || id === this.eosId) continue;
      total += this.pieces[id].length;
    }
    var out = new Uint8Array(total);
    var at = 0;
    for (k = 0; k < ids.length; k++) {
      var token = ids[k];
      if (token === this.bosId || token === this.eosId) continue;
      var piece = this.pieces[token];
      // <0xNN> pieces stand for a single raw byte
      if (piece.length === 6 && piece[0] === 60 && piece[1] === 48 && piece[2] === 120
          && piece[5] === 62) {
        var code = parseInt(String.fromCharCode(piece[3], piece[4]), 16);
        if (!isNaN(code)) { out[at++] = code; continue; }
      }
      out.set(piece, at);
      at += piece.length;
    }
    return out.subarray(0, at);
  };

  Tokenizer.prototype.decode = function (ids) {
    return new TextDecoder("utf-8").decode(this.decodeBytes(ids));
  };

  // ---------------------------------------------------------------- model --- //

  function Llama(buffer) {
    var view = new DataView(buffer);
    var magic = String.fromCharCode(view.getUint8(0), view.getUint8(1),
                                    view.getUint8(2), view.getUint8(3));
    if (magic !== "LLM1") throw new Error("not a .llm model file (bad magic)");
    var version = view.getInt32(4, true);
    if (version !== 1) throw new Error("unsupported model version " + version);

    this.quant = view.getInt32(8, true);
    this.dim = view.getInt32(12, true);
    this.hidden = view.getInt32(16, true);
    this.nLayers = view.getInt32(20, true);
    this.nHeads = view.getInt32(24, true);
    this.nKvHeads = view.getInt32(28, true);
    this.vocabSize = view.getInt32(32, true);
    this.seqLen = view.getInt32(36, true);
    this.headSize = this.dim / this.nHeads;
    this.kvDim = this.nKvHeads * this.headSize;
    this.group = this.nHeads / this.nKvHeads;

    var offset = 40;
    this.tensors = {};
    var order = TENSOR_ORDER;
    for (var t = 0; t < order.length; t++) {
      var kind = view.getInt32(offset, true);
      var rows = view.getInt32(offset + 4, true);
      var cols = view.getInt32(offset + 8, true);
      offset += 12;
      var entry = { kind: kind, rows: rows, cols: cols };
      var blocks = STACKED[order[t]] ? this.nLayers : 1;   // norm tensors carry
                                                          // layers in `rows`
      if (kind === 1) {
        entry.scales = new Float32Array(buffer, offset, rows * blocks);
        offset += rows * blocks * 4;
        entry.data = new Int8Array(buffer, offset, rows * cols * blocks);
        offset += rows * cols * blocks;
      } else if (kind === 3) {
        entry.scales = new Float32Array(buffer, offset, rows * blocks);
        offset += rows * blocks * 4;
        entry.data = new Uint8Array(buffer, offset, rows * cols * blocks / 2);
        offset += rows * cols * blocks / 2;
      } else if (kind === 0) {
        entry.data = new Float32Array(buffer, offset, rows * cols * blocks);
        offset += rows * cols * blocks * 4;
      } else if (kind === 2) {
        entry.data = null;   // alias of token_embd (tied embeddings)
      }
      this.tensors[order[t]] = entry;
    }
    if (this.tensors.wcls.kind === 2) {
      // tied embeddings: the output projection *is* the token embedding, so it
      // inherits the embedding's shape as well as its data
      var emb = this.tensors.token_embd;
      this.tensors.wcls.data = emb.data;
      this.tensors.wcls.scales = emb.scales;
      this.tensors.wcls.rows = emb.rows;
      this.tensors.wcls.cols = emb.cols;
      this.tensors.wcls.kind = emb.kind;   // read it the same way as token_embd
    }

    // scratch buffers, allocated once
    var D = this.dim, H = this.hidden;
    this.x = new Float32Array(D);
    this.xb = new Float32Array(D);
    this.q = new Float32Array(D);
    this.k = new Float32Array(this.kvDim);
    this.v = new Float32Array(this.kvDim);
    this.att = new Float32Array(D);
    this.gate = new Float32Array(H);
    this.up = new Float32Array(H);
    this.scoresBuf = new Float32Array(4096);
    this.logits = new Float32Array(this.vocabSize);
    this.ropeCos = null;
    this.ropeSin = null;
  }

  Llama.prototype.buildRope = function (maxSeq) {
    var half = this.headSize / 2;
    this.ropeCos = new Float32Array(maxSeq * half);
    this.ropeSin = new Float32Array(maxSeq * half);
    var inv = new Float32Array(half);
    for (var i = 0; i < half; i++) inv[i] = 1.0 / Math.pow(ROPE_THETA, (2 * i) / this.headSize);
    for (var p = 0; p < maxSeq; p++) {
      for (var j = 0; j < half; j++) {
        var angle = p * inv[j];
        this.ropeCos[p * half + j] = Math.cos(angle);
        this.ropeSin[p * half + j] = Math.sin(angle);
      }
    }
  };

  Llama.prototype.newState = function (maxSeq) {
    this.buildRope(maxSeq);
    return {
      maxSeq: maxSeq,
      pos: 0,
      key: new Float32Array(this.nLayers * maxSeq * this.kvDim),
      value: new Float32Array(this.nLayers * maxSeq * this.kvDim),
    };
  };

  function matvecI8(t, x, out) {
    var w = t.data, scales = t.scales, rows = t.rows, cols = t.cols;
    for (var i = 0; i < rows; i++) {
      var off = i * cols, s = scales[i], sum = 0;
      for (var j = 0; j < cols; j++) sum += w[off + j] * x[j];
      out[i] = sum * s;
    }
  }

  function matvecF32(t, x, out) {
    var w = t.data, rows = t.rows, cols = t.cols;
    for (var i = 0; i < rows; i++) {
      var off = i * cols, sum = 0;
      for (var j = 0; j < cols; j++) sum += w[off + j] * x[j];
      out[i] = sum;
    }
  }

  /* One row of packed 4-bit values: two per byte, low nibble first, signed. */
  function matvecQ4(t, offset, x, out) {
    var packed = t.data, scales = t.scales, rows = t.rows, cols = t.cols;
    var half = cols >> 1, row0 = offset / cols;
    for (var i = 0; i < rows; i++) {
      var off = (row0 + i) * half, s = scales[row0 + i], sum = 0;
      for (var j = 0; j < half; j++) {
        var byte = packed[off + j];
        var lo = byte & 0x0F, hi = (byte >> 4) & 0x0F;
        if (lo >= 8) lo -= 16;
        if (hi >= 8) hi -= 16;
        sum += lo * x[2 * j] + hi * x[2 * j + 1];
      }
      out[i] = sum * s;
    }
  }

  Llama.prototype.matvec = function (name, x, out) {
    var t = this.tensors[name];
    if (t.kind === 1) matvecI8(t, x, out);
    else if (t.kind === 3) matvecQ4(t, 0, x, out);
    else matvecF32(t, x, out);
  };

  Llama.prototype.rmsnorm = function (src, weight, dst, n) {
    var sum = 0;
    for (var i = 0; i < n; i++) sum += src[i] * src[i];
    var scale = 1.0 / Math.sqrt(sum / n + RMS_EPS);
    for (var j = 0; j < n; j++) dst[j] = src[j] * scale * weight[j];
  };

  /* Rotate pairs of consecutive values -- the llama2.c / Meta convention. */
  Llama.prototype.applyRope = function (vec, base, headSize, pos) {
    var half = headSize / 2, rope = pos * half;
    for (var i = 0; i < half; i++) {
      var c = this.ropeCos[rope + i], s = this.ropeSin[rope + i];
      var a = vec[base + 2 * i], b = vec[base + 2 * i + 1];
      vec[base + 2 * i] = a * c - b * s;
      vec[base + 2 * i + 1] = a * s + b * c;
    }
  };

  Llama.prototype.forward = function (token, state) {
    var D = this.dim, HS = this.headSize, KV = this.kvDim;
    var pos = state.pos, maxSeq = state.maxSeq;
    var x = this.x, xb = this.xb, q = this.q, k = this.k, v = this.v, att = this.att;

    var emb = this.tensors.token_embd;
    if (emb.kind === 3) {
      // NOTE: `var` is function-scoped, so this loop variable must not be named
      // k/q/v/att -- doing so would overwrite those scratch buffers further down.
      var halfD = D >> 1, sNib = emb.scales[token], pOff = token * halfD;
      for (var kk = 0; kk < halfD; kk++) {
        var byte = emb.data[pOff + kk];
        var lo = byte & 0x0F, hi = (byte >> 4) & 0x0F;
        if (lo >= 8) lo -= 16;
        if (hi >= 8) hi -= 16;
        x[2 * kk] = lo * sNib;
        x[2 * kk + 1] = hi * sNib;
      }
    } else if (emb.scales) {
      var s = emb.scales[token], off = token * D;
      for (var i = 0; i < D; i++) x[i] = emb.data[off + i] * s;
    } else {
      var base = token * D;
      for (var i2 = 0; i2 < D; i2++) x[i2] = emb.data[base + i2];
    }

    for (var layer = 0; layer < this.nLayers; layer++) {
      var attWeight = this.tensors.rms_att.data.subarray(layer * D, (layer + 1) * D);
      var ffnWeight = this.tensors.rms_ffn.data.subarray(layer * D, (layer + 1) * D);

      this.rmsnorm(x, attWeight, xb, D);

      var wq = this.tensors.wq, wk = this.tensors.wk, wv = this.tensors.wv,
          wo = this.tensors.wo;
      var qOff = layer * wq.rows * wq.cols;
      var kOff = layer * wk.rows * wk.cols;
      var vOff = layer * wv.rows * wv.cols;
      var oOff = layer * wo.rows * wo.cols;

      matvecSlice(wq, qOff, xb, q);
      matvecSlice(wk, kOff, xb, k);
      matvecSlice(wv, vOff, xb, v);

      for (var h = 0; h < this.nHeads; h++) this.applyRope(q, h * HS, HS, pos);
      for (var hk = 0; hk < this.nKvHeads; hk++) this.applyRope(k, hk * HS, HS, pos);

      var kBase = layer * maxSeq * KV + pos * KV;
      var vBase = layer * maxSeq * KV + pos * KV;
      state.key.set(k, kBase);
      state.value.set(v, vBase);

      var scale = 1.0 / Math.sqrt(HS);
      for (var head = 0; head < this.nHeads; head++) {
        var kvHead = (head / this.group) | 0;
        var qBase = head * HS;
        var kvBase = kvHead * HS;
        var best = -Infinity;
        for (var t = 0; t <= pos; t++) {
          var kPos = layer * maxSeq * KV + t * KV + kvBase;
          var dot = 0;
          for (var d = 0; d < HS; d++) dot += state.key[kPos + d] * q[qBase + d];
          dot *= scale;
          this.scoresBuf[t] = dot;
          if (dot > best) best = dot;
        }
        var total = 0;
        for (var t2 = 0; t2 <= pos; t2++) {
          var e = Math.exp(this.scoresBuf[t2] - best);
          this.scoresBuf[t2] = e;
          total += e;
        }
        var inv = 1.0 / total;
        for (var d2 = 0; d2 < HS; d2++) {
          var acc = 0;
          for (var t3 = 0; t3 <= pos; t3++) {
            acc += this.scoresBuf[t3] * inv
                 * state.value[layer * maxSeq * KV + t3 * KV + kvBase + d2];
          }
          att[qBase + d2] = acc;
        }
      }

      matvecSlice(wo, oOff, att, xb);
      for (var i3 = 0; i3 < D; i3++) x[i3] += xb[i3];

      this.rmsnorm(x, ffnWeight, xb, D);
      var w1 = this.tensors.w1, w2 = this.tensors.w2, w3 = this.tensors.w3;
      var H = this.hidden;
      matvecSlice(w1, layer * w1.rows * w1.cols, xb, this.gate);
      matvecSlice(w3, layer * w3.rows * w3.cols, xb, this.up);
      for (var g = 0; g < H; g++) {
        var val = this.gate[g];
        this.gate[g] = (val / (1 + Math.exp(-val))) * this.up[g];
      }
      matvecSlice(w2, layer * w2.rows * w2.cols, this.gate, xb);
      for (var i4 = 0; i4 < D; i4++) x[i4] += xb[i4];
    }

    this.rmsnorm(x, this.tensors.rms_final.data, xb, D);
    this.matvec("wcls", xb, this.logits);
    state.pos = pos + 1;
    return this.logits;
  };

  /* matvec over one slice (one layer's worth) of a stacked weight tensor. */
  function matvecSlice(t, offset, x, out) {
    var w = t.data, scales = t.scales, rows = t.rows, cols = t.cols;
    if (t.kind === 3) { matvecQ4(t, offset, x, out); return; }
    if (scales) {
      for (var i = 0; i < rows; i++) {
        var off = offset + i * cols, s = scales[(offset / cols) + i], sum = 0;
        for (var j = 0; j < cols; j++) sum += w[off + j] * x[j];
        out[i] = sum * s;
      }
    } else {
      for (var i2 = 0; i2 < rows; i2++) {
        var off2 = offset + i2 * cols, sum2 = 0;
        for (var j2 = 0; j2 < cols; j2++) sum2 += w[off2 + j2] * x[j2];
        out[i2] = sum2;
      }
    }
  }

  // ---------------------------------------------------------------- random --- //

  /* Small deterministic PRNG so a seed reproduces a story exactly. */
  function Rng(seed) {
    this.state = (seed >>> 0) || 1;
  }
  Rng.prototype.next = function () {
    var x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >> 17;
    x ^= x << 5; x >>>= 0;
    this.state = x;
    return x / 4294967296;
  };

  Llama.prototype.sample = function (logits, opts, rng) {
    var temperature = opts.temperature;
    if (temperature <= 0) {
      var best = 0;
      for (var i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    var i2, n = logits.length;
    var probs = new Float32Array(n);
    for (i2 = 0; i2 < n; i2++) probs[i2] = logits[i2] / temperature;

    var topK = Math.min(opts.topK || 0, n);
    if (topK > 0) {
      var idx = Array.from(probs.keys()).sort(function (a, b) { return probs[b] - probs[a]; });
      var keep = idx.slice(0, topK);
      var masked = new Float32Array(n).fill(-Infinity);
      for (i2 = 0; i2 < keep.length; i2++) masked[keep[i2]] = probs[keep[i2]];
      probs = masked;
    }

    var max = -Infinity;
    for (i2 = 0; i2 < n; i2++) if (probs[i2] > max) max = probs[i2];
    var total = 0;
    for (i2 = 0; i2 < n; i2++) { var e = Math.exp(probs[i2] - max); probs[i2] = e; total += e; }
    for (i2 = 0; i2 < n; i2++) probs[i2] /= total;

    var topP = opts.topP;
    if (topP > 0 && topP < 1) {
      var order = Array.from(probs.keys()).sort(function (a, b) { return probs[b] - probs[a]; });
      var cumulative = 0, cutoff = order.length;
      for (i2 = 0; i2 < order.length; i2++) {
        cumulative += probs[order[i2]];
        if (cumulative >= topP) { cutoff = i2 + 1; break; }
      }
      var renorm = new Float32Array(n);
      var sum = 0;
      for (i2 = 0; i2 < cutoff; i2++) { renorm[order[i2]] = probs[order[i2]]; sum += probs[order[i2]]; }
      for (i2 = 0; i2 < n; i2++) renorm[i2] /= sum;
      probs = renorm;
    }

    var r = rng.next(), acc = 0;
    for (i2 = 0; i2 < n; i2++) {
      acc += probs[i2];
      if (r <= acc) return i2;
    }
    return n - 1;
  };

  return {
    Llama: Llama,
    Tokenizer: Tokenizer,
    Rng: Rng,
    bytesFromBase64: bytesFromBase64,
    EOS_ID: EOS_ID,
  };
});
