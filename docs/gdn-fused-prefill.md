# Fused Gated DeltaNet prefill kernel (gfx1151)

`LLAMA_GDN_FUSED=1` replaces six ops of every Qwen3.6 Gated DeltaNet (GDN) layer
with one HIP kernel during prefill:

```
before:  CONCAT(conv_state, qkvᵀ) → SSM_CONV → SILU → L2_NORM(q) → L2_NORM(k) → GATED_DELTA_NET
after :  GATED_DELTA_NET (fused variant: op_params[0] = 1)
```

Result on Strix Halo (Radeon 8060S, gfx1151), Qwen3.6-35B-A3B selective-Q4_0 FlashHead,
20 000-token prompt, 3 cold samples each:

| build | prefill | 20k prompt | decode |
|---|---|---|---|
| previous deploy (`gfx1151-fattn-unroll4-q4-503855413`) | 1132 tok/s | 17.66 s | 76.9 tok/s |
| this tree, `LLAMA_GDN_FUSED=0` | 1382 tok/s | 14.47 s | 72.5 tok/s |
| this tree, `LLAMA_GDN_FUSED=1` | **1586 tok/s** | **12.61 s** | 73.8 tok/s |

One GDN layer, one 2048-token ubatch: **1.47 ms** fused vs 6–17 ms for the op chain.

## Motivation

A rocprofv3 trace of a 20k-token prefill (deployed config) put Gated DeltaNet first:

| kernel | share of prefill |
|---|---|
| `gated_delta_net_cuda` | 30.9 % (5.2 s) |
| Flash Attention (10 full-attention layers) | 20.4 % |
| MoE / dense quantised matmuls | ~30 % |

The GDN kernel reached only ~0.5 TFLOPS and ~8 GB/s, i.e. it was bound by neither compute
nor DRAM bandwidth. ATT (thread trace) showed where the cycles went:

- 49 % waiting for global loads (`s_waitcnt vmcnt`)
- 33 % stalled *issuing* loads (memory pipeline full)
- 8 % storing one output float per token
- 8 % warp reductions (`ds_bpermute`)
- **0.8 % actual math**

Its structure explains it. One warp owns one state column, and the 2048-token loop runs inside
the kernel. Every token, every warp re-reads q/k/v/g/β from global memory. The same q/k values
are read by 8 warps × 16 blocks × 2 v-heads, which is 43 % of *all* L2 requests of the prefill.
L2 hit rate falls from 84 % (T=512) to 39 % (T=2048). Each miss stalls a serial per-token
dependency chain.

So prefill is **latency-bound on a sequential scan**, not compute- or bandwidth-bound. Two
things follow:

- Quantising/bf16-ing the GDN inputs does not help. It was measured end to end: −1.3 %, because
  the extra cast kernels cost more than the halved reads save.
- The fix is to restructure data movement: share q/k across a block, keep the state and all
  operands on chip, and pipeline loads against compute.

The small neighbours (conv1d, SiLU, two l2norms: ~3–5 % of prefill) depend on GDN only per
token, not per tensor. Folding them into the same kernel hides their math in GDN's idle VALU
and removes two round trips of the 8192-channel activation through memory.

## Math (identical to the ggml ops)

For each v-head `h` (32 v-heads; q/k head = `h % 16`, ggml's tiled broadcast) and token `t`:

```
xc   = SiLU( Σ_k in[t+k]·w[k] ),  in = [conv_state(3) | x(T)]      per channel, width 4
q,k  = xc / sqrt(max(Σ xc², eps²))                                  l2norm over 128
S    = e^g · S
δ    = β · (v − Sᵀk)
S   += k δᵀ
o    = Sᵀq / sqrt(128)
```

## Kernel design — `ggml/src/ggml-cuda/gdn-fused.cuh`, `gdn_fused_prefill_ws<8>`

One block per v-head (`grid = 32 × n_seqs`), 320 threads = 8 consumer + 2 producer waves.

**Consumers (recurrence)**

- **Column ownership.** Each lane holds MC = 4 state columns × 16 rows in registers. A group of
  G = 8 lanes covers the 128 rows of 4 columns, so `Sᵀk` and `Sᵀq` need only 3 DPP steps
  (`row_xmask:1/2/4`, `__builtin_amdgcn_mov_dpp`) and no LDS traffic.
- **Row interleave.** Rows are interleaved across the group in float4 units:
  `row(r) = (r/4)·32 + 4·lane + r%4`. One `ds_load_b128` by a group therefore reads 128
  contiguous bytes. The naive block layout put all 8 lanes on 2 LDS banks, a 4-way conflict.
- **Lazy decay.** `S_true = c·S'` with a block-uniform scalar `c ← c·e^g`, so the update is
  one FMA per element: `S' += k·(δ/c)` and `o = c·S'ᵀq`. `S'` is renormalised when
  `c < 2⁻⁴⁰`, and tokens with `e^g < 2⁻⁴⁰` decay explicitly (real models reach g ≈ −92). The
  branches are uniform per block.
- **Token-level software pipeline.** Everything token t+1 needs from LDS (k rows, v for its
  4 columns, packed `(e^g, β)`) is issued while token t computes. q(t) is issued at the top of
  step(t) and consumed after the `Sᵀk` reduction. `__builtin_amdgcn_sched_barrier(0)` pins
  the issue points; without it the scheduler sank the loads next to their use and 35 % of time
  was `lgkmcnt` waits.

**Producers (conv + norm)**

Two waves build chunk c+1 (C = 8 tokens) in the other LDS buffer: global loads of the raw qkv
row (12 channels per lane), sliding conv window, SiLU, in-wave l2norm (`wave_sum` via DPP +
`permlanex16`) and `(e^g, β)`. They also store chunk c−1's outputs to global memory from the
output staging buffer, so consumers never touch global memory inside the loop.

**Chunk hand-off**

LDS is double-buffered for qk / v / out / egb (≈ 33 KB). There is **one barrier per chunk**,
and it is `s_waitcnt lgkmcnt(0); s_barrier` — **not** `__syncthreads()`. `__syncthreads`
also drains `vmcnt`, which forces every in-flight global prefetch to complete before the
barrier and destroys the pipeline.

## Optimisation log (one layer, T = 2048)

| step | time | what the profile showed |
|---|---|---|
| v1: quad per column (G=4, MC=1) | 6.3 ms | LDS instructions 59 %: arrays not 16-B aligned → `ds_load_2addr_b32` |
| MC=4 columns per lane, `alignas(16)` | 1.85 ms | lanes of a wave still re-read identical LDS data |
| row interleave | 1.55 ms | 4-way LDS bank conflicts removed (MC=2 went 3.1 → 1.55 ms) |
| lazy decay | 1.56 ms | fewer VALU ops but no speedup: not VALU-bound |
| producer/consumer + LDS-only barrier | **1.47 ms** | per-chunk conv/norm/store off the critical path |

Two dead ends are worth recording:

- **A value-select on loads** (`x = ok ? load(p) : 0`) compiles to load + `v_cndmask`, which
  forces an immediate `s_waitcnt` per load and serialises a prefetch. Select the *address*
  instead.
- **A `break` inside a `#pragma unroll` loop** can turn register arrays into dynamically
  indexed ones. That did not happen here (ScratchSize 0), but the whole-chunk producer preload
  variant was slower anyway (2.0 ms, higher VGPR, occupancy 7) and was dropped.

## Correctness

**Op level.** The kernel was compared with the ggml op chain (`npu-fa/gdn_fused/bench.cpp`)
on random inputs (T = 3, 33, 37, 2048, 1568 × 2 seqs; g down to −100) and on real inputs dumped
from six different layers of the model (`GDN_FUSED_DUMP=<dir>`). Max relative error is about
1e-6 and cosine is 1.000000000 for both outputs and the new state.

**Model level.** This model amplifies fp32 rounding differences a lot (MoE top-k routing and
Q4 KV quantisation boundaries), so model-level metrics have to be read against the
build-to-build baseline:

| logits over 256 positions (2 ubatches) | cosine mean / min | KL mean | top-1 same |
|---|---|---|---|
| unfused vs unfused (rerun) | 1.000000000 / 1.000000000 | 0 | 100 % |
| **fused vs unfused (same build)** | 0.9913 / 0.853 | 0.024 | 95.7 % |
| old deploy vs this build unfused (no GDN change at all) | 0.9908 / 0.854 | 0.032 | 93.4 % |

`llama-perplexity --kl-divergence` over 8 × 2048 tokens gives the same picture:

| comparison | KL | top-1 same |
|---|---|---|
| fused vs unfused | 0.044 | 92.7 % |
| old deploy vs unfused | 0.047 | 92.8 % |

**Production smoke test.** A 9010-token prompt (5 ubatches, conv/SSM state carried across
them) prefills at 1787 tok/s. A fact placed at token 0 is recalled, the cached follow-up turn
works, and fused and unfused produce identical text on the same conversation.

## Usage and scope

- Set `LLAMA_GDN_FUSED=1`. The fused path is used for ubatches with ≥ 16 tokens per sequence
  (prefill). Decode and MTP verification keep the existing path.
- It is only built for HIP (`GGML_USE_HIP`) and only engaged for the Qwen3.6 GDN shape:
  head dim 128, 16 q/k heads, 32 v heads, conv width 4. Other models fall back to the
  unfused graph. The CPU backend asserts if it ever receives the fused op.
- The API is `ggml_gated_delta_net_conv(x, conv_state, conv_w, g, beta, state, eps)`.
  It produces the same `[attn | new_state]` layout as `ggml_gated_delta_net` and is marked
  by `op_params[0] = 1` on `GGML_OP_GATED_DELTA_NET`.
- **Ordering.** The fused op reads `conv_state` before `conv_state_update` overwrites it. The
  graph builder expands the fused node first; `build_rs` also hands it a `get_rows` copy.
- **Debug hook.** `GDN_FUSED_DUMP=<dir>` together with `GDN_FUSED_DUMP_CALL=<n>` dumps the
  n-th call's inputs, for replay in the standalone bench.

## Possible next steps

- **Use all 40 CUs.** One block per v-head only occupies 32 CUs.
- **Chunked (WY/UT) delta rule on WMMA.** This turns per-token rank-1 updates into per-chunk
  rank-C matmuls and would remove the sequential chain altogether.
- **Revisit the other prefill hot spots** with the same method: Q4_K MMQ is VALU-bound on
  per-sub-block scale handling, with WMMA only 5–14 % of its cycles; the Flash Attention
  prefill tile kernel does not use WMMA.
