#!/usr/bin/env python3
"""
Convert a llama2.c checkpoint (karpathy/llama2.c `.bin` format) into a GGUF
file that llama.cpp can load.

The llama2.c format is dead simple:

    int32  dim              (embedding size)
    int32  hidden_dim       (feed-forward size)
    int32  n_layers
    int32  n_heads
    int32  n_kv_heads
    int32  vocab_size
    int32  seq_len          (context the model was trained with)
    ...then raw little-endian float32 tensors, in this exact order...

      1. token_embedding_table   [vocab, dim]
      2. rms_att_weight          [n_layers, dim]
      3. wq                      [n_layers, n_heads*head_size, dim]
      4. wk                      [n_layers, n_kv_heads*head_size, dim]
      5. wv                      [n_layers, n_kv_heads*head_size, dim]
      6. wo                      [n_layers, dim, n_heads*head_size]
      7. rms_ffn_weight          [n_layers, dim]
      8. w1                      [n_layers, hidden_dim, dim]
      9. w2                      [n_layers, dim, hidden_dim]
     10. w3                      [n_layers, hidden_dim, dim]
     11. rms_final_weight        [dim]
     12. freq_cis_real           [seq_len, head_size/2]   (ignored, see note)
     13. freq_cis_imag           [seq_len, head_size/2]   (ignored, see note)
     14. wcls                    [vocab, dim]  (only present when embeddings are untied)

Note on rope frequencies: llama2.c precomputes cos/sin tables for the position
range it was trained on. llama.cpp computes RoPE on the fly, so those two
tensors are skipped on purpose and `llama.rope.freq_base` is written instead
(stories15M was trained with the default 10000.0).

Note on tied embeddings: stories15M shares the output matrix with the token
embedding, so its checkpoint stops after the rope tables. When that happens we
omit `output.weight` from the GGUF and let llama.cpp reuse `token_embd.weight`.

Usage:
    python convert_llama2c_to_gguf.py --bin models/stories15M.bin \
        --tokenizer models/tokenizer.model --out models/stories15M-f16.gguf
    python convert_llama2c_to_gguf.py ... --out models/stories15M-f32.gguf --dtype f32
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# llama.cpp ships its GGUF python package in-tree; allow a pip-installed `gguf`
# as well by only prepending the local copy when it exists.
_HERE = Path(__file__).resolve().parent
_LOCAL_GGUF = _HERE / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF.is_dir():
    sys.path.insert(1, str(_LOCAL_GGUF))

import gguf  # noqa: E402


def read_header(path: Path) -> tuple[int, int, int, int, int, int, int]:
    with path.open("rb") as f:
        raw = f.read(7 * 4)
    if len(raw) != 7 * 4:
        raise ValueError(f"{path} is too small to be a llama2.c checkpoint")
    return struct.unpack("<7i", raw)  # dim, hidden, layers, heads, kv_heads, vocab, seq_len


def tensor_plan(dim: int, hidden_dim: int, n_layers: int, n_heads: int,
                n_kv_heads: int, vocab_size: int, seq_len: int):
    head_size = dim // n_heads
    return [
        ("token_embedding_table", (vocab_size, dim)),
        ("rms_att_weight", (n_layers, dim)),
        ("wq", (n_layers, n_heads * head_size, dim)),
        ("wk", (n_layers, n_kv_heads * head_size, dim)),
        ("wv", (n_layers, n_kv_heads * head_size, dim)),
        ("wo", (n_layers, dim, n_heads * head_size)),
        ("rms_ffn_weight", (n_layers, dim)),
        ("w1", (n_layers, hidden_dim, dim)),
        ("w2", (n_layers, dim, hidden_dim)),
        ("w3", (n_layers, hidden_dim, dim)),
        ("rms_final_weight", (dim,)),
        ("freq_cis_real", (seq_len, head_size // 2)),
        ("freq_cis_imag", (seq_len, head_size // 2)),
        ("wcls", (vocab_size, dim)),
    ]


def load_weights(path: Path, shapes) -> tuple[dict[str, np.ndarray], bool]:
    """Read the raw fp32 tensors. Returns (weights, tied_embeddings)."""
    weights: dict[str, np.ndarray] = {}
    tied = False
    with path.open("rb") as f:
        f.seek(7 * 4)
        for name, shape in shapes:
            count = int(np.prod(shape))
            buf = f.read(count * 4)
            if len(buf) != count * 4:
                if name == "wcls":
                    tied = True  # export.py skips wcls when weights are shared
                    break
                raise ValueError(f"unexpected EOF while reading tensor {name!r}")
            weights[name] = np.frombuffer(buf, dtype="<f4").reshape(shape)
    if tied:
        weights["wcls"] = weights["token_embedding_table"]
    return weights, tied


def load_tokenizer(tokenizer_path: Path, vocab_size: int):
    """Pull token text, scores and token types out of the SentencePiece model."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if sp.vocab_size() != vocab_size:
        raise ValueError(f"tokenizer vocab ({sp.vocab_size()}) != model vocab ({vocab_size})")

    tokens, scores, types = [], [], []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        tokens.append(piece)
        scores.append(sp.get_score(i))
        if i == sp.unk_id():
            types.append(gguf.TokenType.UNKNOWN)
        elif i in (sp.bos_id(), sp.eos_id()):
            types.append(gguf.TokenType.CONTROL)
        elif len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            types.append(gguf.TokenType.BYTE)  # the 256 byte-fallback pieces
        else:
            types.append(gguf.TokenType.NORMAL)
    return tokens, scores, types, sp


def convert(bin_path: Path, tokenizer_path: Path, out_path: Path, dtype: str,
            context_length: int) -> None:
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = read_header(bin_path)
    head_size = dim // n_heads
    print(f"checkpoint : {bin_path}")
    print(f"  dim={dim} hidden={hidden_dim} layers={n_layers} heads={n_heads} "
          f"kv_heads={n_kv_heads} head_size={head_size} vocab={vocab_size} "
          f"trained_ctx={seq_len}")

    weights, tied = load_weights(bin_path, tensor_plan(dim, hidden_dim, n_layers,
                                                       n_heads, n_kv_heads, vocab_size, seq_len))
    if tied:
        print("  embeddings are tied -> output.weight omitted (llama.cpp reuses token_embd)")
    tokens, scores, types, sp = load_tokenizer(tokenizer_path, vocab_size)

    np_dtype = np.float32 if dtype == "f32" else np.float16
    file_type = (gguf.LlamaFileType.ALL_F32 if dtype == "f32"
                 else gguf.LlamaFileType.MOSTLY_F16)

    writer = gguf.GGUFWriter(str(out_path), "llama")
    writer.add_name(f"stories15M-{dtype}")
    writer.add_file_type(file_type)
    writer.add_context_length(context_length)
    writer.add_embedding_length(dim)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(hidden_dim)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(head_size)
    writer.add_layer_norm_rms_eps(1e-5)  # the value llama2.c uses

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    if sp.bos_id() >= 0:
        writer.add_bos_token_id(sp.bos_id())
    if sp.eos_id() >= 0:
        writer.add_eos_token_id(sp.eos_id())
    if sp.unk_id() >= 0:
        writer.add_unk_token_id(sp.unk_id())

    def add(name: str, array: np.ndarray) -> None:
        # Norm weights stay f32 even in f16 models: ggml applies them to f32
        # activations with a typed binary op, which rejects mixed dtypes.
        dtype = np.float32 if name.endswith("norm.weight") else np_dtype
        writer.add_tensor(name, np.ascontiguousarray(array, dtype=dtype))

    add("token_embd.weight", weights["token_embedding_table"])
    add("output_norm.weight", weights["rms_final_weight"])
    if not tied:
        add("output.weight", weights["wcls"])

    for i in range(n_layers):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", weights["rms_att_weight"][i])
        add(p + "attn_q.weight", weights["wq"][i])
        add(p + "attn_k.weight", weights["wk"][i])
        add(p + "attn_v.weight", weights["wv"][i])
        add(p + "attn_output.weight", weights["wo"][i])
        add(p + "ffn_norm.weight", weights["rms_ffn_weight"][i])
        add(p + "ffn_gate.weight", weights["w1"][i])
        add(p + "ffn_down.weight", weights["w2"][i])
        add(p + "ffn_up.weight", weights["w3"][i])

    # freq_cis_real/imag are intentionally dropped: llama.cpp recomputes RoPE.
    print(f"writing    : {out_path} ({dtype})")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done       : {out_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, type=Path, help="llama2.c checkpoint (.bin)")
    ap.add_argument("--tokenizer", required=True, type=Path, help="SentencePiece tokenizer.model")
    ap.add_argument("--out", required=True, type=Path, help="output .gguf path")
    ap.add_argument("--dtype", default="f16", choices=("f16", "f32"))
    ap.add_argument("--context", type=int, default=2048,
                    help="context length to advertise in the GGUF (default: 2048)")
    args = ap.parse_args()
    convert(args.bin, args.tokenizer, args.out, args.dtype, args.context)


if __name__ == "__main__":
    main()
