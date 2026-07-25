#!/usr/bin/env bash
# Build the Tier-2 attention-capture host against a local llama.cpp build.
# Usage: LLAMA_DIR=~/llama.cpp bash build.sh (LLAMA_DIR defaults to ~/llama.cpp)
# Produces ./rex_attn_capture next to this script; agent.attn_introspect finds it there.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BIN="$LLAMA_DIR/build/bin"

[ -f "$LLAMA_DIR/include/llama.h" ] || { echo "llama.h not found under $LLAMA_DIR/include - set LLAMA_DIR" >&2; exit 1; }
[ -d "$BIN" ] || { echo "llama.cpp build/bin not found ($BIN) - build llama.cpp first" >&2; exit 1; }

c++ -O2 -std=c++17 "$HERE/rex_attn_capture.cpp" \
    -I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include" \
    -L"$BIN" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BIN" \
    -o "$HERE/rex_attn_capture"
echo "built: $HERE/rex_attn_capture   (links llama.cpp in $BIN)"
