#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_BUILD=${FLASHHEAD_TEST_BUILD:-$ROOT/build-gpu-tests}

cmake --build "$TEST_BUILD" --target test-backend-ops -j "${JOBS:-16}"
"$TEST_BUILD/bin/test-backend-ops" test -o TOP_K -b ROCm0 \
    -p 'ne=\[7760,1,1,1\],k=256' --output console
"$TEST_BUILD/bin/test-backend-ops" test -o CONCAT -b ROCm0 \
    -p 'type=i32' --output console
cmake --build "$ROOT/build" --target llama-server -j "${JOBS:-16}"
git -C "$ROOT" diff --check

echo "PASS: FlashHead GPU Top-K, I32 concat, and llama-server build"
