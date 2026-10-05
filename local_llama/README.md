# Local Llama

Run a **Llama model locally, offline, with no API keys** — three ways, from a
dependency-free reference implementation up to llama.cpp with an
OpenAI-compatible HTTP server and a browser playground.

**See also:** [`../lm_arena`](../lm_arena) builds a blind A/B battle arena and Elo
leaderboard on top of these models, and [`../phone`](../phone) runs either one
entirely on an Android phone — no computer, no WiFi.

This directory is self-contained and independent of the Grok-1 code at the repo
root. (Grok-1 itself is a 314B-parameter model that needs a multi-GPU cluster —
see [the repo README](../README.md) — so it is *not* what runs here.)

```
┌──────────────────────────────────────────────────────────────────────┐
│  stories15M.bin  (60.8 MB, fp32, Llama-2 architecture, TinyStories)  │
└───────┬──────────────────────────────────┬───────────────────────────┘
        │ pure NumPy                       │ convert_llama2c_to_gguf.py
        ▼                                  ▼
   np_llama.py                        stories15M-f16.gguf (31 MB)
   ~60 tok/s on 2 CPU cores                 │ llama.cpp
        │                                   ▼
        └──────────► serve.py ◄──── llama-cli / llama-server
                  (OpenAI API + web UI on :8000)
```

## Quickstart

```bash
cd local_llama
./download_model.sh                 # ~61 MB, verified checksums
python3 np_llama.py --prompt "Once upon a time" --max-tokens 200
```

On a phone, this whole directory works under Termux — see
[`../phone/README.md`](../phone/README.md) for the scripted setup.

That is the whole minimum: **numpy is the only dependency**. You get streaming
text at roughly 60 tokens/sec on two CPU cores, ~250 MB RAM.

Want it faster and with an API?

```bash
./build.sh                          # builds llama.cpp (10–25 min on 2 cores)
python3 convert_llama2c_to_gguf.py --bin models/stories15M.bin \
    --tokenizer models/tokenizer.model --out models/stories15M-f16.gguf
python3 serve.py                    # http://localhost:8000  (web UI + OpenAI API)
```

`build.sh` and the conversion are optional — `serve.py` falls back to the NumPy
backend automatically if llama.cpp is not built yet.

## The model

| | |
|---|---|
| Name | `stories15M` (karpathy/llama2.c "tinyllamas") |
| Architecture | Llama-2: 6 layers, 288 dim, 6 heads, 6 KV heads, RoPE, RMSNorm, SwiGLU |
| Parameters | 15M (tied embeddings) |
| Vocab | 32,000 (Llama-2 SentencePiece) |
| Trained on | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Licence | MIT (llama2.c) |

It writes simple, grammatical children's stories. It is a **base** model: it
continues text, it does not answer questions and it does not know facts. That is
the honest trade-off for something that runs on a CPU in 250 MB of RAM — and it
is a real, fully-verifiable Llama forward pass, which is the point.

The canonical download lives on Hugging Face (`karpathy/tinyllamas`). Because
many sandboxes (including the one this was built in) block Hugging Face,
`download_model.sh` fetches the same bytes from public GitHub mirrors and
verifies them:

| File | Size | MD5 |
|---|---|---|
| `stories15M.bin` | 60,816,028 | `644db0bc012b405d6baf99559272ab11` |
| `tokenizer.model` | 499,723 | `eeec4125e9c7560836b4873b6f8e3025` |
| `tokenizer.bin` | 433,869 | `c5a4f2f24b728689a3c4f9e4f79d5112` |

## Files

| File | What it does |
|---|---|
| `download_model.sh` | Fetches the checkpoint + tokenizers, verifies checksums |
| `np_llama.py` | Pure-NumPy Llama forward pass + sampler + CLI (the reference) |
| `build.sh` | Builds `llama-cli` and `llama-server` from llama.cpp source |
| `convert_llama2c_to_gguf.py` | Converts the llama2.c `.bin` checkpoint to GGUF |
| `serve.py` | OpenAI-compatible API + web playground, llama.cpp or NumPy backend |
| `compare_with_llamacpp.py` | Proves the GGUF conversion is faithful (greedy match) |
| `web/index.html` | The playground page (no CDN, works offline) |
| `models/` | Checkpoints and tokenizers (git-ignored, ~150 MB) |
| `llama.cpp/` | Third-party llama.cpp checkout (git-ignored) |

## Using it from other tools

`serve.py` speaks the OpenAI HTTP API, so any client that accepts a base URL works.

```bash
# text completion
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":80,"temperature":0.8}'

# chat-completions shape (messages are flattened into a prompt)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Once upon a time"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.completions.create(model="stories15M",
      prompt="Once upon a time", max_tokens=120).choices[0].text)
```

Point Open WebUI, Continue, LangChain, or any OpenAI-compatible tool at
`http://localhost:8000/v1`.

## Raw llama.cpp, if you prefer

```bash
./llama.cpp/build/bin/llama-cli -m models/stories15M-f16.gguf \
    -p "Once upon a time" -n 200 --temp 0.8

# its own server instead of ours
./llama.cpp/build/bin/llama-server -m models/stories15M-f16.gguf --port 8081
```

## What the conversion actually does

A few details that make the GGUF faithful — these are the usual places a
llama2.c → GGUF conversion goes wrong:

* **RoPE convention.** llama2.c rotates *pairs of consecutive head values*
  (`x[2i], x[2i+1]`). llama.cpp's `LLM_ARCH_LLAMA` uses exactly that
  (`LLAMA_ROPE_TYPE_NORM`), while Hugging Face checkpoints store q/k rows
  permuted for the split-half convention — which is why llama.cpp's HF converter
  *un-permutes* them (`undo_permute = True` in `conversion/llama.py`). Starting
  from llama2.c weights, no permutation is needed, and doing one would break it.
* **Rope tables are dropped.** The checkpoint stores precomputed cos/sin tables
  for its 256-token training window; llama.cpp computes RoPE itself. Passing them
  through as `rope_freqs.weight` would be wrong. You can confirm the tables are
  reproducible from `theta=10000` with `python3 np_llama.py --verify-rope`
  (matches to ~1e-5, i.e. float32 noise).
* **Tied embeddings.** `stories15M.bin` has no `wcls` tensor (embedding sharing),
  so `output.weight` is omitted and llama.cpp reuses `token_embd.weight`.
* **RMSNorm epsilon** is `1e-5`, matching llama2.c, not the HF default.
* **No transposes.** The llama2.c layout is already `[out, in]` per weight, which
  is what GGUF stores.

`compare_with_llamacpp.py` checks all of this end to end: it greedy-decodes the
same prompt with both engines and diffs the output.

## Measured performance

Measured in this sandbox (2 vCPU, no GPU at all, 3.8 GB RAM):

| Engine | Speed | Memory |
|---|---|---|
| `np_llama.py` (NumPy, fp32 checkpoint) | ~60–68 tok/s | ~250 MB |
| `llama.cpp` (`llama-server`, f16 GGUF, via `serve.py`) | ~294–340 tok/s | ~50 MB |

So llama.cpp is ~5x faster here; both are fast enough to feel interactive, and
`compare_with_llamacpp.py` confirms they compute the same thing.

## Scaling up to bigger Llama models

Nothing here is stories15M-specific — `serve.py --gguf <any.gguf>` runs any
llama.cpp-compatible model, and `np_llama.py` reads any llama2.c checkpoint.
With network access and more RAM/GPU, drop in a real instruct model:

```bash
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "*Q4_K_M.gguf" --local-dir models
python3 serve.py --gguf models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Rough guidance for CPU-only: 1B Q4 ≈ 0.8 GB RAM and a few tokens/sec; 3B Q4 ≈
2 GB; 8B Q4 ≈ 5 GB. For 70B+ you want a GPU (rebuild with `-DGGML_CUDA=ON`).

## Troubleshooting

* **`ModuleNotFoundError: numpy`** → `pip install numpy`
  (add `--break-system-packages` on Debian/Ubuntu system Python).
* **Spaces missing between words** → you are on an old `np_llama.py` that decoded
  token-by-token; SentencePiece needs the full sequence to place spaces.
* **GGUF loads but output is gibberish** → the conversion mangled the weights;
  run `compare_with_llamacpp.py`, and check you did not permute q/k or keep the
  stored rope tables.
* **`cmake` not found** → `build.sh` installs it from PyPI automatically; if you
  use a venv, activate it first.
* **llama.cpp build is slow** → it is single-machine C++ work; more cores or a
  prebuilt release binary from the llama.cpp GitHub releases page is far quicker.
* **Hugging Face is blocked** → that is why `download_model.sh` uses GitHub
  mirrors. Behind a proxy, set `HTTPS_PROXY` or `GITHUB_TOKEN` for higher rate
  limits.

## Licence and provenance

* `stories15M` and the tokenizer are from [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
  (MIT), trained on the TinyStories dataset.
* `llama.cpp` is MIT, fetched by `build.sh` into `llama.cpp/` (not vendored here).
* The mirror used to fetch the checkpoint in this sandbox is a third-party
  GitHub repo; prefer the canonical `karpathy/tinyllamas` on Hugging Face when
  you have access, which is why checksums are pinned above.
