/*
 * test_engine.js -- check the browser engine against the Python reference.
 *
 *   node phone/test_engine.js <model.llm> [prompt] [tokens]
 *
 * Greedy decoding is deterministic, so the token ids printed here must equal the
 * ones from local_llama/np_llama.py (see phone/README.md for the exact command).
 * That is the check that catches a transposed matrix, a RoPE convention slip, a
 * bad quantiser or a tokenizer mismatch.
 */

const fs = require("fs");
const path = require("path");
const { Llama, Tokenizer, Rng } = require("./web/llama-engine.js");

const modelPath = process.argv[2];
const prompt = process.argv[3] || "Once upon a time";
const maxTokens = parseInt(process.argv[4] || "32", 10);

if (!modelPath) {
  console.error("usage: node test_engine.js <model.llm> [prompt] [tokens]");
  process.exit(2);
}

const ROOT = path.resolve(__dirname, "..");
const tokenizerPath = path.join(ROOT, "local_llama", "models", "tokenizer.bin");

function load(file) {
  const buf = fs.readFileSync(file);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

const t0 = Date.now();
const model = new Llama(load(modelPath));
const tokenizer = new Tokenizer(load(tokenizerPath));
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(2)}s  ` +
            `quant=${model.quant} dim=${model.dim} layers=${model.nLayers} vocab=${model.vocabSize}`);

const ids = tokenizer.encode(prompt, true);
const maxSeq = Math.min(768, ids.length + maxTokens + 8);
const state = model.newState(maxSeq);

let logits = null;
for (const id of ids) logits = model.forward(id, state);

const rng = new Rng(0);
const out = [];
const genStart = Date.now();
for (let i = 0; i < maxTokens; i++) {
  const next = model.sample(logits, { temperature: 0, topP: 1, topK: 0 }, rng);
  if (next === 2) break;
  out.push(next);
  logits = model.forward(next, state);
  if (state.pos >= maxSeq - 1) break;
}
const seconds = (Date.now() - genStart) / 1000;

console.log(`prompt_ids: ${ids.join(",")}`);
console.log(`gen_ids   : ${out.join(",")}`);
console.log(`text      : ${JSON.stringify(tokenizer.decode(out))}`);
console.log(`speed     : ${(out.length / seconds).toFixed(1)} tok/s ` +
            `(${(seconds * 1000 / Math.max(out.length, 1)).toFixed(0)} ms/token)`);
