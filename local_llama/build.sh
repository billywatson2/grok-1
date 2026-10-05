#!/usr/bin/env bash
# Build llama.cpp (CPU build) and produce the binaries this playground uses:
#   llama.cpp/build/bin/llama-cli     -- interactive / one-shot generation
#   llama.cpp/build/bin/llama-server  -- OpenAI-compatible HTTP server
#
#   ./build.sh              # ~10-25 min on 2 cores, much faster on more
#
# No GPU is required. If you have one, see the notes at the bottom.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/llama.cpp"
JOBS="$(nproc 2>/dev/null || echo 2)"

# --- 1. get cmake ----------------------------------------------------------- #
if ! command -v cmake >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/cmake" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "==> cmake missing; installing from PyPI (needs no root)"
    pip3 install --user --break-system-packages cmake 2>/dev/null \
      || pip3 install --user cmake
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
echo "==> cmake: $(cmake --version | head -1)"

# --- 2. get llama.cpp source ------------------------------------------------ #
if [ ! -d "$SRC" ]; then
  if command -v git >/dev/null 2>&1; then
    echo "==> cloning llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
  else
    echo "==> downloading llama.cpp source tarball"
    tmp="$(mktemp -d)"
    curl -sfL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master" \
      -o "$tmp/llama.cpp.tar.gz"
    tar xzf "$tmp/llama.cpp.tar.gz" -C "$tmp"
    mv "$tmp"/llama.cpp-master "$SRC"
    rm -rf "$tmp"
  fi
else
  echo "==> reusing existing $SRC"
fi

# --- 3. configure + build --------------------------------------------------- #
cd "$SRC"
echo "==> configuring (-j$JOBS)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF

echo "==> building llama-cli + llama-server"
cmake --build build --target llama-cli llama-server -j "$JOBS"

echo
echo "==> done"
ls -la "$SRC/build/bin/llama-cli" "$SRC/build/bin/llama-server"
echo
echo "Next:"
echo "  python3 convert_llama2c_to_gguf.py --bin models/stories15M.bin \\"
echo "      --tokenizer models/tokenizer.model --out models/stories15M-f16.gguf"
echo "  ./llama.cpp/build/bin/llama-cli -m models/stories15M-f16.gguf -p 'Once upon a time' -n 120"
echo "  python3 serve.py            # web UI + OpenAI API on :8000"
echo
echo "# GPU shortcut: instead of building, grab a prebuilt release binary from"
echo "#   https://github.com/ggml-org/llama.cpp/releases  (e.g. *-bin-ubuntu-*-cuda-*.zip)"
echo "# or just use the PyPI wrapper:  pip install llama-cpp-python"
echo "# Hardware accel flags you can add to the cmake line above:"
echo "#   -DGGML_CUDA=ON      (NVIDIA)   -DGGML_METAL=ON   (Apple, usually on by default)"
echo "#   -DGGML_VULKAN=ON    (AMD/Intel) -DGGML_HIP=ON    (AMD ROCm)"
