#!/usr/bin/env python3
"""
Verify that the GGUF conversion is faithful.

Runs the same prompt through two independent engines with greedy decoding:

  * np_llama.py    -- the pure-NumPy reference (implements llama2.c's math)
  * llama-server   -- llama.cpp reading the converted .gguf

Greedy decoding is deterministic, so if the conversion is correct the two texts
match character for character. A mismatch means a transposed weight matrix, a
RoPE convention mix-up, a wrong RMSNorm epsilon or a broken tokenizer -- and the
diff below shows where they first diverge.

Usage:
    python3 compare_with_llamacpp.py --prompt "Once upon a time" --max-tokens 48
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from np_llama import Llama2cModel, load_tokenizer, sample  # noqa: E402

LLAMA_SERVER = HERE / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_GGUF = HERE / "models" / "stories15M-f16.gguf"


def run_numpy(prompt: str, max_tokens: int) -> str:
    model = Llama2cModel(HERE / "models" / "stories15M.bin")
    tokenizer = load_tokenizer(HERE / "models" / "tokenizer.model",
                               HERE / "models" / "tokenizer.bin")
    tokens = tokenizer.encode(prompt, bos=True)
    max_seq = min(4096, len(tokens) + max_tokens + 8)
    cos_all, sin_all = model.rope_tables(max_seq)
    state = model.new_state(max_seq)

    logits = None
    for tok in tokens:
        logits = model.forward(tok, state, cos_all, sin_all)

    rng = np.random.default_rng(0)
    out_ids: list[int] = []
    for _ in range(max_tokens):
        nxt = sample(logits, 0.0, 1.0, 0, rng)  # temperature 0 == greedy
        if nxt == tokenizer.eos_id:
            break
        out_ids.append(nxt)
        logits = model.forward(nxt, state, cos_all, sin_all)
        if state["pos"] >= max_seq - 1:
            break
    return tokenizer.decode(out_ids)


def run_llamacpp(prompt: str, max_tokens: int, gguf: Path, port: int,
                 ctx: int) -> str:
    if not LLAMA_SERVER.exists():
        raise SystemExit(f"{LLAMA_SERVER} not found -- run ./build.sh first")

    proc = subprocess.Popen(
        [str(LLAMA_SERVER), "-m", str(gguf), "--port", str(port), "--host",
         "127.0.0.1", "-c", str(ctx), "-t", str(2)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError("llama-server did not become ready")

        body = json.dumps({
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
            "top_k": 1, "seed": 0, "stream": False, "cache_prompt": False,
            "n_keep": 0,
        }).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.load(resp)
        return data.get("content", "")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def first_divergence(a: str, b: str) -> tuple[int, str]:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i, f"first difference at char {i}: llama.cpp={ca!r} numpy={cb!r}"
    if len(a) != len(b):
        longer = "llama.cpp" if len(a) > len(b) else "numpy"
        return min(len(a), len(b)), f"{longer} produced extra text"
    return -1, "identical"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()

    print(f"prompt: {args.prompt!r}   greedy, {args.max_tokens} tokens\n")

    print("--- llama.cpp / GGUF ---")
    ref = run_llamacpp(args.prompt, args.max_tokens, args.gguf, args.port, args.ctx)
    print(ref or "<empty>")

    print("\n--- numpy / raw checkpoint ---")
    mine = run_numpy(args.prompt, args.max_tokens)
    print(mine or "<empty>")

    idx, why = first_divergence(ref, mine)
    print()
    if idx < 0:
        print("MATCH: both engines produced identical greedy output "
              "-> the GGUF conversion is faithful.")
    else:
        print(f"DIVERGED: {why}")
        lo = max(0, idx - 30)
        print(f"  llama.cpp: ...{ref[lo:idx + 30]!r}")
        print(f"  numpy    : ...{mine[lo:idx + 30]!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
