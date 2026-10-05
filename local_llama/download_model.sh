#!/usr/bin/env bash
# Download the model checkpoint and tokenizer used by this playground.
#
#   ./download_model.sh            # into ./models
#   ./download_model.sh /some/dir
#
# This fetches ~61 MB. The files come from public GitHub repos because that is
# what is reachable from locked-down sandboxes (Hugging Face is often blocked).
# Checksums are verified after download, so a corrupted/mirrored file is caught.
#
# Canonical alternative (if you have Hugging Face access):
#   huggingface-cli download karpathy/tinyllamas stories15M.bin --local-dir models
#   huggingface-cli download karpathy/llama2.c tokenizer.model tokenizer.bin --local-dir models

set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")" && pwd)/models}"
mkdir -p "$DEST"
cd "$DEST"

# repo, path, local-name, expected-bytes, expected-md5
MODEL_REPO="Chi-Isaac/LLM-Inference-Engine"
MODEL_MIRROR="Srajay-2005/tiny-llm-safety"
TOKENIZER_REPO="karpathy/llama2.c"

fetch() { # repo path outfile
  local repo="$1" path="$2" out="$3"
  echo "  fetching $repo/$path -> $out"
  curl -sfL --retry 3 --retry-delay 2 \
    -H "Accept: application/vnd.github.raw" \
    ${GITHUB_TOKEN:+-H "Authorization: Bearer $GITHUB_TOKEN"} \
    "https://api.github.com/repos/$repo/contents/$path" -o "$out"
}

verify() { # file expected_md5
  local file="$1" want="$2"
  local got
  got=$(md5sum "$file" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "  !! checksum mismatch for $file"
    echo "     expected $want"
    echo "     got      $got"
    return 1
  fi
  echo "  ok $file ($(du -h "$file" | cut -f1))"
}

TOKENIZER_MODEL_MD5="eeec4125e9c7560836b4873b6f8e3025"
TOKENIZER_BIN_MD5="c5a4f2f24b728689a3c4f9e4f79d5112"
MODEL_MD5="644db0bc012b405d6baf99559272ab11"

echo "==> stories15M.bin (60.8 MB)"
if [ -f stories15M.bin ] && [ "$(stat -c%s stories15M.bin)" = "60816028" ]; then
  echo "  already present"
else
  fetch "$MODEL_REPO" "data/stories15M.bin" stories15M.bin || \
    fetch "$MODEL_MIRROR" "engine/stories15M.bin" stories15M.bin
fi
size=$(stat -c%s stories15M.bin)
[ "$size" = "60816028" ] || { echo "  !! unexpected size $size"; exit 1; }
verify stories15M.bin "$MODEL_MD5"

echo "==> tokenizer.model + tokenizer.bin (SentencePiece, from karpathy/llama2.c)"
[ -f tokenizer.model ] || fetch "$TOKENIZER_REPO" "tokenizer.model" tokenizer.model
[ -f tokenizer.bin ]   || fetch "$TOKENIZER_REPO" "tokenizer.bin"   tokenizer.bin
verify tokenizer.model "$TOKENIZER_MODEL_MD5"
verify tokenizer.bin "$TOKENIZER_BIN_MD5"

echo
echo "Done. Files in $DEST:"
ls -la
echo
echo "Next:  python np_llama.py --prompt 'Once upon a time'   # instant, numpy only"
echo "       ./build.sh && python convert_llama2c_to_gguf.py ...   # faster llama.cpp path"
