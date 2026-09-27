# Experimental gfx1151 D=72 intrinsic vision Flash Attention

**Status: research branch; NOT enabled in llama-server, NOT deployed.**

This is an AI-assisted experimental implementation, test harness and measured
performance record. It is deliberately separate from the regular CMake targets.
Do not add the diagnostic adapter to a production service's `LD_PRELOAD`.

## Scope and implementation

The measured Qwen3.6-35B-A3B vision projector uses 27 attention layers, 16 heads,
head dimension 72 and 16-pixel patches. On gfx1151, the deployed generic
`flash_attn_tile<72,72,...>` consumed 5.031 seconds (81.1% of GPU kernel time)
for a 4032x3024 image, internally resized to 2336x1760. Attention runs over
16,060 patches **before** the merger emits 4,015 image embeddings.

The new operator uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for QK and
PV, FP32 accumulators, zero-filled D=80 tails and online softmax. It retains
`1/sqrt(72)` scaling; Q is scaled before FP16 conversion to match the generic
kernel convention. Four wave32 groups reuse each 16-key tile through LDS.
The optional software pipeline prefetches the next K/V tile into registers,
with a scheduling barrier separating prefetch from current-tile computation.
No quadratic-size attention matrix is allocated by this candidate.

It does **not** implement arbitrary masks, GQA, attention sinks, softcap or
batch > 1. Only the experiment's unmasked, 16-head, self-attention shape is
eligible in the adapter. Regular llama.cpp dispatch has not been modified.

## Build and numerical tests

Requirements: ROCm 7.2.2, gfx1151, a C++ compiler, Python with NumPy. The wave32
and HIP warp-sync defines in the build script are required on the tested host.

```sh
bash tools/vision-d72/build.sh
python3 tests/test-vision-d72.py
```

Overrides: `ROCM_PATH`, `D72_BUILD_DIR` and `D72_LIB` (for the Python test).

The test covers ten deterministic nonuniform shapes/scales, both pipeline
variants, sequence tails and multiple heads. Its CPU FP32 reference is
softmax(QK^T/sqrt(72))V. Acceptance is cosine > .9999 and max absolute error
< .01. Both variants passed all twenty cases; neither is claimed bit-exact to
the CPU reference.

For the actual encoder harness, additionally set `D72_LLAMA_LIBDIR` to a
**compatible** directory of llama.cpp shared libraries when building. The
measurement used deployed libraries based on `503855413` plus pre-existing
local performance changes, not a clean upstream release. The clip header API
is internal; do not assume compatibility with arbitrary library versions.

```sh
D72_LLAMA_LIBDIR=/path/to/compatible/bin bash tools/vision-d72/build.sh
export D72_MMPROJ=/path/to/mmproj.gguf
# Input is RGBRGB... raw uint8, width*height*3 bytes, converted locally.
GGML_CUDA_DISABLE_GRAPHS=1 build-vision-d72/encoder image.rgb 4032 3024 3 original
# Research-only interposition, same graphs setting on both sides:
D72_EXPERIMENTAL=1 GGML_CUDA_DISABLE_GRAPHS=1 \
  LD_PRELOAD="$PWD/build-vision-d72/libd72_adapter.so" \
  build-vision-d72/encoder image.rgb 4032 3024 3 original
```

`ENCODER_DUMP` writes final embeddings; `ENCODER_NO_FA` selects the existing
non-FA path for cross-checks. `D72_REFERENCE=1` selects a **slow validation-only**
bounded-memory FP32 SGEMM + softmax reference in the adapter. It uses blocks of
128 queries rather than allocating the full N*N matrix. `D72_CAPTURE` optionally
captures first-layer inputs/outputs to an existing local directory. Captures
contain image-derived private data: do not upload them without permission.

## Measured results — 2026-09-27

Three unprofiled measurements after warmup, median, shared GPU; not an isolated
hardware performance claim. Encoder A/B disables HIP graphs on **both** sides.

| Workload | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Entire original-photo encoder | 6200.860 ms | 4649.410 ms | latency -25.02% |
| Standalone N=16060, H=16; plain vs prefetch | 167.036 ms | 118.106 ms | latency -29.29% |
| Standalone N=3072, H=16; plain vs prefetch | 6.867 ms | 4.545 ms | latency -33.82% |

The first row compares generic vs intrinsic. The other rows compare two new
intrinsic variants on synthetic QKV; these are not old-vs-new encoder results.
Full latency samples and comparison summaries are in `results/`.

### Correctness: why the old output is not the oracle

Initial equivalence to the deployed generic encoder **failed**: cosine was
about .985. Rather than loosen that old-output test and silently accept it,
we compared both outputs to the independent FP32 attention reference:

| Full-image embedding vs FP32 attention reference | Cosine | Relative L2 |
|---|---:|---:|
| Deployed generic attention | .9852756751 | .17799656 |
| Intrinsic candidate | .9997875187 | .02061363 |

The FP32 reference was separately checked against NumPy on sampled queries
from actual first-layer data: maximum absolute error 9.66e-6. The candidate
passes the full-embedding gate of cosine > .999 and relative L2 < .05, but is
**not bit-exact**, and one photo is not a complete quality evaluation.

### End-to-end functional checks

Two private Chinese screenshots were tested with identical OCR prompts,
temperature 0, seed 42, no prompt reuse and thinking disabled. The new path
was verified to dispatch exactly 27 times per image. Critical numeric tables,
merchant fields and principal paragraphs agreed; both versions retained a
shared Chinese-character typo and unsupported UI identification. No new major
regression was observed in these two samples, not a general accuracy guarantee.

Due to memory constraints the functional A/B used **CPU text-model decoding,
GPU encoder, F16 KV and MTP disabled** on both sides. Therefore it is not final
acceptance of the production GPU + MTP stack. Responses, screenshots, paths and
raw embeddings are intentionally not included in this repository.

## Instruction-level evidence and limitations

PC sampling attempts with stochastic/instructions and host_trap/time both failed:
`Given PC sampling configuration is not supported on any of the agents`.
No per-PC runtime stall attribution is claimed.

Hardware instruction-class counters did work. For N=3072, H=16, median of three
non-warmup dispatches, the raw reported aggregate counter values were:

| Counter | Plain | Prefetch |
|---|---:|---:|
| SQ_INSTS_VALU | 183530496 | 215973888 |
| SQ_INSTS_SALU | 39247872 | 33503232 |
| SQ_INSTS_LDS | 77186594 | 74331914 |

Profiler/counter timings are **not** used for speedup claims. Static ISA shows
10 `v_wmma` instructions in the K-loop body (5 QK + 5 PV), `v_exp_f32` for
softmax, 256 VGPR, 7744-byte LDS and 24-byte scratch metadata for the pipelined
variant (36-byte scratch without prefetch). Trace LDS allocation rounds to
8192 bytes. Static counts/occupancy estimates are not dynamic stall measurements.
**Latency is not fully hidden; register pressure and scratch remain.**

## Deployment gate — currently blocked

The adapter exists solely to test real encoder outputs without rebuilding or
replacing the live service. It allocates workspace per layer and uses
`hipDeviceSynchronize`; it is not safe for HIP graph capture. Disabling graphs
service-wide may regress text-only/MTP workloads and was not accepted.

Before production activation:

1. Integrate narrowly in native HIP attention dispatch using the caller's stream
   and graph-safe workspace, retaining explicit architecture/shape/type guards.
2. Test fallback cases, arbitrary strides, tails, concurrent slots and replay;
   reduce live-register ranges/scratch and compare pipeline depths.
3. Re-run full GPU + MTP output checks over multiple images, including dense OCR,
   and verify no text-only decode regression with interleaved measurements.
4. Obtain independent review; no independent review has been performed yet.
5. Deploy an immutable artifact with an exact previous binary/config rollback,
   restart through the service-management procedure, and verify health plus
   visual/text smoke requests. Never deploy the diagnostic preload as a shortcut.

No production systemd changes, restart scripts or enable-by-default settings
are part of this commit.
