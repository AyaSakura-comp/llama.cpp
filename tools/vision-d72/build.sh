#!/usr/bin/env bash
# Experimental standalone builds only. Does not alter normal llama.cpp targets.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${D72_BUILD_DIR:-"$ROOT/build-vision-d72"}
ROCM=${ROCM_PATH:-/opt/rocm-7.2.2}
mkdir -p "$OUT"
FLAGS=(-O3 -std=c++17 --offload-arch=gfx1151 -D__AMDGCN_WAVEFRONT_SIZE=32
       -DHIP_ENABLE_WARP_SYNC_BUILTINS -I"$ROCM/include" -I"$ROOT/ggml/include"
       -fPIC -gline-tables-only)
"$ROCM/bin/hipcc" "${FLAGS[@]}" -shared \
  "$ROOT/ggml/src/ggml-cuda/vision-d72-intrinsic.hip" -o "$OUT/libvision_d72.so"
"$ROCM/bin/hipcc" "${FLAGS[@]}" -shared "$ROOT/tools/vision-d72/adapter.hip" \
  -ldl -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lrocblas -o "$OUT/libd72_adapter.so"
"$ROCM/lib/llvm/bin/clang++" -x hip --cuda-device-only "${FLAGS[@]}" -S \
  "$ROOT/ggml/src/ggml-cuda/vision-d72-intrinsic.hip" -o "$OUT/vision-d72.s"
if [[ -n "${D72_LLAMA_LIBDIR:-}" ]]; then
  "${CXX:-g++}" -O2 -std=c++17 "$ROOT/tools/vision-d72/encoder.cpp" \
    -I"$ROOT/tools/mtmd" -I"$ROOT/ggml/include" -I"$ROOT/include" -I"$ROCM/include" \
    -L"$D72_LLAMA_LIBDIR" -L"$ROCM/lib" -Wl,-rpath,"$D72_LLAMA_LIBDIR:$ROCM/lib" \
    -lmtmd -lggml -lggml-base -lrocprofiler-sdk-roctx -o "$OUT/encoder"
fi
printf 'Experimental artifacts: %s\n' "$OUT"
