#!/data/data/com.termux/files/usr/bin/bash
# One-shot setup for running Local Llama + LM Arena on an Android phone, with no
# computer involved and no network needed after setup.
#
#   bash phone/termux_setup.sh              # python only (fast, no compiling)
#   bash phone/termux_setup.sh --build      # also build llama.cpp (5-20 min, ~5x faster)
#   bash phone/termux_setup.sh --build --model-1b   # + a 1B model (~800 MB download)
#
# Run it from the repository root *inside Termux*. See phone/README.md.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=0
MODEL_1B=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    --model-1b) MODEL_1B=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

if [ ! -d "$PREFIX" ] || [ "${PREFIX#/data/data/com.termux}" = "$PREFIX" ]; then
  echo "!! This script is for Termux on Android (no \$PREFIX found)."
  echo "   On a computer, use local_llama/download_model.sh and local_llama/build.sh instead."
  exit 1
fi

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

step "Device check"
echo "    prefix   : $PREFIX"
echo "    cores    : $(nproc 2>/dev/null || echo '?')"
if command -v termux-info >/dev/null 2>&1; then
  free -h 2>/dev/null | awk 'NR<=2 {print "    memory   : " $0}' || true
fi

step "Installing packages (python, numpy, git, curl)"
pkg update -y >/dev/null 2>&1 || true
# python-numpy comes from Termux itself: the manylinux wheels on PyPI do not
# apply to Android's bionic libc, so `pip install numpy` would compile for ages.
pkg install -y python python-numpy git curl || {
  echo "!! pkg install failed -- run 'pkg update' manually and retry"; exit 1; }

step "Checking Python + numpy"
python - <<'EOF'
import sys
print(f"    python {sys.version.split()[0]}")
try:
    import numpy
    print(f"    numpy  {numpy.__version__}")
except ImportError:
    sys.exit("!! numpy missing: try 'pkg install python-numpy'")
EOF

step "Downloading models (~95 MB, WiFi is fine here -- it is a one-time step)"
cd "$REPO/local_llama"
bash download_model.sh

if [ "$MODEL_1B" = "1" ]; then
  step "Fetching a larger model for the arena (optional)"
  if python -c "import huggingface_hub" 2>/dev/null; then
    python - <<'EOF' || echo "    (skipped: 1B download failed -- probably no Hugging Face access here)"
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    local_dir="models")
print("    saved:", p)
EOF
  else
    echo "    pip install huggingface_hub, then re-run with --model-1b"
  fi
fi

if [ "$BUILD" = "1" ]; then
  step "Building llama.cpp (this is the slow part, expect 5-20 minutes)"
  pkg install -y cmake clang make pkg-config
  cd "$REPO/local_llama"
  # build.sh detects Termux and skips the PyPI cmake fallback automatically
  bash build.sh
  step "Converting the checkpoint to GGUF"
  python convert_llama2c_to_gguf.py --bin models/stories15M.bin \
    --tokenizer models/tokenizer.model --out models/stories15M-f16.gguf
else
  echo
  echo "Skipping the llama.cpp build. The NumPy backend works right now --"
  echo "it needs no compiler at all. Re-run with --build later for ~5x speed."
fi

step "Done. To run the arena:"
cat <<'EOF'
    cd ~/grok-1/lm_arena
    termux-wake-lock                       # stop Android from freezing the server
    python3 arena.py --port 8100 --host 127.0.0.1

    then open http://localhost:8100 in Chrome and use
    "Add to Home Screen" to get it as an app.

    --host 127.0.0.1 means the server is reachable *only from this phone*:
    no WiFi, no data, nothing on the network at all.
EOF
