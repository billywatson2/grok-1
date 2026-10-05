#!/usr/bin/env python3
"""
Pure-NumPy inference for llama2.c checkpoints (the same format stories15M ships in).

Why this exists
---------------
llama.cpp (built by `build.sh`) is the fast path. This file is:

  1. a *reference oracle* -- it implements llama2.c's math literally, so you can
     prove a GGUF conversion didn't mangle the weights (see `--verify-rope` and
     `compare_with_llamacpp.py`),
  2. a fallback that needs nothing but numpy, and
  3. a readable implementation of a Llama forward pass in ~100 lines.

Model format (little-endian):
    header: 7 x int32 = dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len
    then fp32 tensors: token_embd, attn norms, wq/wk/wv/wo per layer, ffn norms,
    w1/w2/w3 per layer, final norm, rope cos/sin tables, [wcls if untied]

Usage:
    python np_llama.py --prompt "Once upon a time" --max-tokens 200
    python np_llama.py --prompt "Once upon a time" --max-tokens 50 --temperature 0  # greedy
    python np_llama.py --verify-rope
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "stories15M.bin"
DEFAULT_TOKENIZER = HERE / "models" / "tokenizer.model"
DEFAULT_TOKENIZER_BIN = HERE / "models" / "tokenizer.bin"
ROPE_THETA = 10000.0
RMS_EPS = 1e-5


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
class Llama2cTokenizer:
    """The self-contained llama2.c `tokenizer.bin` (no third-party deps).

    Encoding is greedy-longest-match over the vocab with `<0xNN>` byte fallback,
    which is what karpathy's run.c/runq.c do. It is close to, but not always
    identical to, the real SentencePiece segmentation -- use the .model file
    (below) when you want byte-for-byte agreement with llama.cpp.
    """

    def __init__(self, path: Path):
        with path.open("rb") as f:
            (self.max_token_length,) = struct.unpack("<i", f.read(4))
            self.tokens: list[bytes] = []
            self.scores: list[float] = []
            while True:
                head = f.read(8)
                if len(head) < 8:
                    break
                score, length = struct.unpack("<fi", head)
                self.tokens.append(f.read(length))
                self.scores.append(score)
        self.vocab_size = len(self.tokens)
        self.bos_id, self.eos_id = 1, 2
        self._lookup = {t: i for i, t in enumerate(self.tokens)}

    def _byte_token(self, b: int) -> int:
        return self._lookup[f"<0x{b:02X}>".encode()]

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = [self.bos_id] if bos else []
        raw = text.encode("utf-8")
        i = 0
        while i < len(raw):
            best_id, best_len = None, 0
            for length in range(min(self.max_token_length, len(raw) - i), 0, -1):
                candidate = self._lookup.get(raw[i:i + length])
                if candidate is not None:
                    best_id, best_len = candidate, length
                    break
            if best_id is None:
                ids.append(self._byte_token(raw[i]))
                i += 1
            else:
                ids.append(best_id)
                i += best_len
        return ids

    def decode_one(self, token_id: int) -> bytes:
        if token_id == self.bos_id:
            return b""
        piece = self.tokens[token_id]
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            return bytes([int(piece[3:5], 16)])
        return piece

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.decode_one(i) for i in ids).decode("utf-8", errors="replace")


class SentencePieceTokenizer:
    """Wraps the real SentencePiece model -- matches llama.cpp's tokenizer."""

    def __init__(self, path: Path):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.vocab_size = self.sp.vocab_size()
        self.bos_id, self.eos_id = self.sp.bos_id(), self.sp.eos_id()

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        return ([self.bos_id] + ids) if bos else list(ids)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


def load_tokenizer(tokenizer_model: Path | None, tokenizer_bin: Path | None):
    if tokenizer_model is not None and tokenizer_model.exists():
        try:
            return SentencePieceTokenizer(tokenizer_model)
        except ImportError:
            print("[tokenizer] sentencepiece not installed, using tokenizer.bin", file=sys.stderr)
    if tokenizer_bin is None or not tokenizer_bin.exists():
        raise SystemExit("no tokenizer found: need tokenizer.model (sentencepiece) "
                         "or tokenizer.bin")
    return Llama2cTokenizer(tokenizer_bin)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Llama2cModel:
    def __init__(self, path: Path):
        with path.open("rb") as f:
            self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads, \
                self.vocab_size, self.seq_len = struct.unpack("<7i", f.read(28))
            self.head_size = self.dim // self.n_heads
            self.kv_dim = self.n_kv_heads * self.head_size

            def read(shape):
                n = int(np.prod(shape))
                return np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape).astype(np.float32)

            L, H, D, KV = self.n_layers, self.hidden_dim, self.dim, self.kv_dim
            self.token_embedding = read((self.vocab_size, D))
            self.rms_att = read((L, D))
            self.wq = read((L, self.n_heads * self.head_size, D))
            self.wk = read((L, KV, D))
            self.wv = read((L, KV, D))
            self.wo = read((L, D, self.n_heads * self.head_size))
            self.rms_ffn = read((L, D))
            self.w1 = read((L, H, D))
            self.w2 = read((L, D, H))
            self.w3 = read((L, H, D))
            self.rms_final = read((D,))
            self.freq_cis_real = read((self.seq_len, self.head_size // 2))
            self.freq_cis_imag = read((self.seq_len, self.head_size // 2))
            rest = f.read(self.vocab_size * D * 4)
            self.tied = len(rest) < self.vocab_size * D * 4
            self.wcls = self.token_embedding if self.tied else \
                np.frombuffer(rest, dtype="<f4").reshape(self.vocab_size, D).astype(np.float32)

    # ---- rope ------------------------------------------------------------- #
    def rope_tables(self, length: int):
        """cos/sin tables, computed the same way llama2.c precomputes them."""
        inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, self.head_size, 2) / self.head_size))
        pos = np.arange(length, dtype=np.float32)
        angles = np.outer(pos, inv_freq)          # [length, head_size/2]
        return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)

    def verify_rope(self) -> float:
        """Compare computed tables against the ones stored in the checkpoint."""
        cos, sin = self.rope_tables(self.seq_len)
        dcos = float(np.abs(cos - self.freq_cis_real).max())
        dsin = float(np.abs(sin - self.freq_cis_imag).max())
        print(f"rope table max |delta| vs checkpoint: cos={dcos:.2e} sin={dsin:.2e}")
        return max(dcos, dsin)

    # ---- forward ---------------------------------------------------------- #
    @staticmethod
    def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return (x / np.sqrt(np.mean(x * x) + RMS_EPS)) * weight

    @staticmethod
    def apply_rope(vec: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray) -> np.ndarray:
        """Rotate pairs of *consecutive* head values (llama2.c / Meta convention)."""
        even = vec[0::2]
        odd = vec[1::2]
        out = np.empty_like(vec)
        out[0::2] = even * cos_row - odd * sin_row
        out[1::2] = even * sin_row + odd * cos_row
        return out

    def new_state(self, max_seq: int):
        return {
            "pos": 0,
            "k": np.zeros((self.n_layers, max_seq, self.kv_dim), dtype=np.float32),
            "v": np.zeros((self.n_layers, max_seq, self.kv_dim), dtype=np.float32),
        }

    def forward(self, token: int, state: dict, cos_all, sin_all) -> np.ndarray:
        """One token through the network. Returns logits. Updates the KV cache."""
        pos = state["pos"]
        x = self.token_embedding[token].copy()
        cos_row, sin_row = cos_all[pos], sin_all[pos]

        for layer in range(self.n_layers):
            xb = self.rmsnorm(x, self.rms_att[layer])

            q = self.wq[layer] @ xb
            k = self.wk[layer] @ xb
            v = self.wv[layer] @ xb

            q = q.reshape(self.n_heads, self.head_size)
            k = k.reshape(self.n_kv_heads, self.head_size)
            q = np.stack([self.apply_rope(q[h], cos_row, sin_row) for h in range(self.n_heads)])
            k = np.stack([self.apply_rope(k[h], cos_row, sin_row) for h in range(self.n_kv_heads)])

            state["k"][layer, pos] = k.reshape(-1)
            state["v"][layer, pos] = v

            keys = state["k"][layer, : pos + 1]      # [pos+1, kv_dim]
            vals = state["v"][layer, : pos + 1]
            keys = keys.reshape(pos + 1, self.n_kv_heads, self.head_size)
            vals = vals.reshape(pos + 1, self.n_kv_heads, self.head_size)

            attn_out = np.empty((self.n_heads, self.head_size), dtype=np.float32)
            scale = 1.0 / np.sqrt(self.head_size)
            group = self.n_heads // self.n_kv_heads
            for h in range(self.n_heads):
                kv_head = h // group
                scores = keys[:, kv_head, :] @ q[h] * scale          # [pos+1]
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                attn_out[h] = probs @ vals[:, kv_head, :]

            x = x + self.wo[layer] @ attn_out.reshape(-1)

            xb = self.rmsnorm(x, self.rms_ffn[layer])
            gate = self.w1[layer] @ xb
            silu = gate / (1.0 + np.exp(-gate))
            x = x + self.w2[layer] @ (silu * (self.w3[layer] @ xb))

        x = self.rmsnorm(x, self.rms_final)
        state["pos"] = pos + 1
        return self.wcls @ x


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int,
           rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits.astype(np.float64) / temperature
    if top_k > 0:
        keep = np.argpartition(-logits, min(top_k, len(logits) - 1))[:top_k]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumulative, top_p) + 1)
        keep = order[:cutoff]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(model: Llama2cModel, tokenizer, prompt: str, max_tokens: int,
             temperature: float, top_p: float, top_k: int, seed: int,
             stream: bool = True):
    tokens = tokenizer.encode(prompt, bos=True)
    max_seq = min(4096, len(tokens) + max_tokens + 8)
    cos_all, sin_all = model.rope_tables(max_seq)
    state = model.new_state(max_seq)
    rng = np.random.default_rng(seed)

    out_ids: list[int] = []
    logits = None
    for i, tok in enumerate(tokens):
        logits = model.forward(tok, state, cos_all, sin_all)
        if state["pos"] >= max_seq - 1:
            break
    # SentencePiece only restores inter-word spaces when it sees the full
    # sequence, so stream by re-decoding and emitting the newly added suffix.
    emitted = ""
    for _ in range(max_tokens):
        nxt = sample(logits, temperature, top_p, top_k, rng)
        out_ids.append(nxt)
        if nxt == tokenizer.eos_id:
            break
        if stream:
            text = tokenizer.decode(out_ids)
            sys.stdout.write(text[len(emitted):])
            sys.stdout.flush()
            emitted = text
        logits = model.forward(nxt, state, cos_all, sin_all)
        if state["pos"] >= max_seq - 1:
            break
    if stream:
        sys.stdout.write("\n")
    return tokenizer.decode(out_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER_BIN)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-rope", action="store_true",
                    help="check computed rope tables against the checkpoint's own")
    args = ap.parse_args()

    model = Llama2cModel(args.model)
    print(f"# model: dim={model.dim} layers={model.n_layers} heads={model.n_heads} "
          f"kv_heads={model.n_kv_heads} vocab={model.vocab_size} "
          f"tied_embeddings={model.tied}", file=sys.stderr)

    if args.verify_rope:
        worst = model.verify_rope()
        raise SystemExit(0 if worst < 1e-4 else 1)

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_bin)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature,
             args.top_p, args.top_k, args.seed)


if __name__ == "__main__":
    main()
