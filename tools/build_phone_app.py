#!/usr/bin/env python3
"""
Build the no-server phone app: one self-contained HTML file.

Takes the stories15M checkpoint and emits

    phone/LlamaPhone.html          <- copy this to the phone, open it, done

The HTML has the model, the tokenizer and the JavaScript engine baked in as
base64, so there is nothing to install, no server to run, and no network access
of any kind. A 15M model at int8 is ~15 MB, which becomes ~21 MB of HTML.

Also writes the raw container next to it (phone/LlamaPhone.llm) so the engine can
be tested from Node against the Python reference.

    python3 tools/build_phone_app.py                 # int8 (default)
    python3 tools/build_phone_app.py --quant fp32    # full precision, ~80 MB HTML
    python3 tools/build_phone_app.py --quant q4      # int4 nibbles, ~11 MB HTML
    python3 tools/build_phone_app.py --no-html       # just the .llm container

Container format is documented at the top of phone/web/llama-engine.js.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA = ROOT / "local_llama"
DEFAULT_BIN = LLAMA / "models" / "stories15M.bin"
DEFAULT_TOKENIZER = LLAMA / "models" / "tokenizer.bin"
ENGINE = ROOT / "phone" / "web" / "llama-engine.js"
TEMPLATE = ROOT / "phone" / "web" / "app-template.html"
OUT_DIR = ROOT / "phone"

TENSOR_ORDER = [
    "token_embd", "rms_att", "wq", "wk", "wv", "wo", "rms_ffn",
    "w1", "w2", "w3", "rms_final", "wcls",
]
# (name, rows-per-layer) -- the stacked per-layer tensors, in llama2.c order
STACKED = {"wq", "wk", "wv", "wo", "w1", "w2", "w3"}
NORM_TENSORS = {"rms_att", "rms_ffn", "rms_final"}   # always fp32: tiny and sensitive


def read_checkpoint(path: Path):
    with path.open("rb") as f:
        dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack("<7i", f.read(28))
        head_size = dim // heads
        kv_dim = kv_heads * head_size

        def read(shape):
            n = int(np.prod(shape))
            return np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape).astype(np.float32)

        weights = {
            "token_embd": read((vocab, dim)),
            "rms_att": read((layers, dim)),
            "wq": read((layers, heads * head_size, dim)),
            "wk": read((layers, kv_dim, dim)),
            "wv": read((layers, kv_dim, dim)),
            "wo": read((layers, dim, heads * head_size)),
            "rms_ffn": read((layers, dim)),
            "w1": read((layers, hidden, dim)),
            "w2": read((layers, dim, hidden)),
            "w3": read((layers, hidden, dim)),
            "rms_final": read((dim,)),
        }
        # the rope tables and wcls: wcls exists only when embeddings are untied
        read((seq, head_size // 2))     # freq_cis_real   (unused, llama.cpp does this too)
        read((seq, head_size // 2))     # freq_cis_imag
        rest = f.read(vocab * dim * 4)
        tied = len(rest) < vocab * dim * 4
        weights["wcls"] = weights["token_embd"] if tied else \
            np.frombuffer(rest, dtype="<f4").reshape(vocab, dim).astype(np.float32)

    return (dim, hidden, layers, heads, kv_heads, vocab, seq), weights, tied


def quantize_int8(rows: int, cols: int, w: np.ndarray):
    """Per-row symmetric int8. Returns (kind, scales, bytes)."""
    flat = w.reshape(rows, cols)
    scales = np.abs(flat).max(axis=1) / 127.0
    scales[scales == 0] = 1e-8
    q = np.clip(np.rint(flat / scales[:, None]), -127, 127).astype(np.int8)
    return 1, scales.astype("<f4").tobytes(), q.tobytes()


def quantize_q4(rows: int, cols: int, w: np.ndarray):
    """Per-row 4-bit, two values per byte (low nibble first), stored as int8 scale."""
    flat = w.reshape(rows, cols)
    scales = np.abs(flat).max(axis=1) / 7.0
    scales[scales == 0] = 1e-8
    q = np.clip(np.rint(flat / scales[:, None]), -7, 7).astype(np.int8)
    q = q.reshape(rows, cols // 2, 2)
    packed = (q[:, :, 0] & 0x0F) | ((q[:, :, 1] & 0x0F) << 4)
    return 3, scales.astype("<f4").tobytes(), packed.astype(np.uint8).tobytes()


def encode_tensor(name: str, w: np.ndarray, tied: bool, quant: str):
    """Serialise one tensor: (kind, rows, cols, payload_bytes).

    The reader's rule is: a tensor holds `rows * cols` values, times n_layers for
    the per-layer weight matrices. So a 3-D block keeps rows = rows-per-layer,
    while the 2-D norm stacks use rows = n_layers and fit in a single block.
    """
    if name == "wcls" and tied:
        return 2, 1, 1, b""

    if name in NORM_TENSORS:
        if w.ndim == 1:                       # rms_final: a single vector
            return 0, 1, w.shape[0], np.ascontiguousarray(w, dtype="<f4").tobytes()
        return 0, w.shape[0], w.shape[1], np.ascontiguousarray(w, dtype="<f4").tobytes()

    quantize = quantize_int8 if quant == "int8" else quantize_q4

    if w.ndim == 3:                           # wq/wk/wv/wo/w1/w2/w3
        layers, rows, cols = w.shape
        flat = np.ascontiguousarray(w).reshape(layers * rows, cols)
        if quant == "fp32":
            return 0, rows, cols, flat.astype("<f4").tobytes()
        kind, scales, payload = quantize(layers * rows, cols, flat)
        return kind, rows, cols, scales + payload

    rows, cols = w.shape
    if quant == "fp32":
        return 0, rows, cols, np.ascontiguousarray(w, dtype="<f4").tobytes()
    kind, scales, payload = quantize(rows, cols, w)
    return kind, rows, cols, scales + payload


def build_container(bin_path: Path, quant: str) -> tuple[bytes, dict]:
    dims, weights, tied = read_checkpoint(bin_path)
    dim, hidden, layers, heads, kv_heads, vocab, seq = dims
    quant_id = {"fp32": 0, "int8": 1, "q4": 3}[quant]

    out = bytearray()
    out += b"LLM1"
    out += struct.pack("<iii", 1, quant_id, dim)
    out += struct.pack("<iiiii", hidden, layers, heads, kv_heads, vocab)
    out += struct.pack("<i", seq)

    for name in TENSOR_ORDER:
        kind, rows, cols, payload = encode_tensor(name, weights[name], tied, quant)
        out += struct.pack("<iii", kind, rows, cols)
        out += payload

    stats = {
        "dim": dim, "hidden": hidden, "layers": layers, "heads": heads,
        "kv_heads": kv_heads, "vocab": vocab, "seq": seq, "quant": quant,
        "tied_embeddings": tied,
        "bytes": len(out),
        # tied embeddings share one matrix, so count it once
        "params": sum(int(np.prod(w.shape)) for name, w in weights.items()
                      if not (name == "wcls" and tied)),
    }
    return bytes(out), stats


def build_html(container: bytes, tokenizer: bytes, stats: dict, out_path: Path,
               engine_js: str, template: str) -> None:
    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    html = template
    html = html.replace("/*__ENGINE__*/", engine_js)
    html = html.replace("__MODEL_B64__", b64(container))
    html = html.replace("__TOKENIZER_B64__", b64(tokenizer))
    html = html.replace("__STATS__", json.dumps({
        "dim": stats["dim"], "layers": stats["layers"], "heads": stats["heads"],
        "vocab": stats["vocab"], "quant": stats["quant"],
        "params": stats["params"], "mb": round(len(container) / 1e6, 1),
    }))
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "LlamaPhone.html")
    ap.add_argument("--quant", choices=("int8", "q4", "fp32"), default="int8")
    ap.add_argument("--no-html", action="store_true", help="only write the .llm container")
    args = ap.parse_args()

    for required in (args.bin, args.tokenizer, ENGINE, TEMPLATE):
        if not required.exists():
            sys.exit(f"missing {required} -- run local_llama/download_model.sh first")

    print(f"checkpoint : {args.bin}")
    container, stats = build_container(args.bin, args.quant)
    print(f"  {stats['params'] / 1e6:.1f}M params, {stats['layers']} layers, "
          f"dim {stats['dim']}, vocab {stats['vocab']}, "
          f"tied embeddings: {stats['tied_embeddings']}")
    print(f"  quantised  : {args.quant} -> {len(container) / 1e6:.1f} MB container")

    llm_path = args.out.with_suffix(".llm")
    llm_path.write_bytes(container)
    print(f"  wrote      : {llm_path}")

    if not args.no_html:
        html = build_html(container, args.tokenizer.read_bytes(), stats, args.out,
                          ENGINE.read_text(), TEMPLATE.read_text())
        size = args.out.stat().st_size
        print(f"  wrote      : {args.out} ({size / 1e6:.1f} MB, self-contained)")
        print()
        print("  Copy that one file to the phone (USB, email, cloud, anything) and")
        print("  open it in Chrome. It runs in airplane mode.")


if __name__ == "__main__":
    main()
