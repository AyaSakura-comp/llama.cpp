# Qwen 3.6 / gfx1151 optimization working memory

> **Mandatory:** Read [`AGENTS.md`](AGENTS.md) completely before doing any work.
> This private fork is heavily hardware-specialized. Do not remove an unfamiliar
> guard, fallback, test, or deployment constraint until its history below has been
> checked.

## Contributor and agent rules

- The upstream llama.cpp AI policy in `AGENTS.md` applies. The human must understand,
  own, and be able to defend every change.
- Never write a commit message, PR description, reviewer reply, commit, push, or open
  a PR for the user. A commit message must be supplied explicitly by the human.
- Use test-driven development for fixes and new optimization paths. Establish RED,
  implement the smallest GREEN, then refactor.
- A performance claim requires an interleaved cold A/B, medians, matching workloads,
  numeric validation, and restoration of production afterward.
- Do not trade target-model correctness for throughput without explicit approval.
- Never deploy directly from mutable `build/bin`. Production must use an immutable,
  self-contained deployment with `$ORIGIN:/opt/rocm-7.2.2/lib` RUNPATH.
- Never kill or manually replace the systemd-owned llama-server process. Use the
  approved Qwen restart helper documented below.

## Scope and current state

This work optimizes Qwen 3.6 35B-A3B on an AMD Strix Halo APU:

| Item | Value |
| --- | --- |
| Hardware | AMD RDNA 3.5, `gfx1151`, UMA |
| ROCm | 7.2.2 |
| Source | `/home/chihmin/llama-mtp-opt` |
| Active branch at this snapshot | `flashhead-lm-head` |
| Active committed HEAD | `15bd9b002877` |
| Branch baseline | `a421d661a` (`mtp-clean`) |
| Production service | `qwen-mtp.service`, port 8001 |
| Model alias | `qwen3.6-35b-q4` |
| Context / slots | `-c 260000 -np 1` (API reports 260096) |
| Batch | `-b 4096 -ub 2048` |
| Speculation | `--spec-type mtp --spec-draft-n-max 3` |
| Production policy | FlashHead draft, **dense target verifier**, F16 target/draft KV |

The worktree is intentionally dirty at this snapshot. The uncommitted Dynamic-PP
change is described in its own section. Do not lose it during rebases or branch
switches.

### Current production artifact

```text
/home/chihmin/llama-mtp-deploy/
  gfx1151-dynamic-mtp-pp-15bd9b0028-120407c7b2/bin/llama-server
```

This artifact was built from base commit `15bd9b0028` plus Dynamic-PP patch hash
`120407c7b2`. It is immutable but should eventually be replaced by a deployment
built from the formal Dynamic-PP commit.

Production currently uses:

```text
GGUF=/home/chihmin/models/Qwen3.6-35B-A3B-selective-Q4_0-proof/
     Qwen3.6-35B-A3B-UD-Q4_K_M-selective-Q4_0-flashhead.gguf
LLAMA_FLASHHEAD_PROBES=256
LLAMA_FLASHHEAD_TARGET is absent
KV cache: F16 target + F16 draft
```

`LLAMA_FLASHHEAD_TARGET=1` enables the approximate All-FlashHead experiment. It
must remain absent in default production.

## Correctness invariants

1. **The dense target distribution is the authority.** Draft FlashHead may change
   proposal efficiency, but the target LM head must remain dense by default.
2. Dense numeric checks compare the complete 248,320-entry token-ID-aligned vector,
   not only generated text or top-k values.
3. Required dense result: sampled token exact, every logprob bit-exact, maximum
   finite delta `0`, cosine similarity `1`.
4. F16 target and draft KV are the correctness baseline. Do not silently re-enable
   the rejected F16 attention-mask experiment; it reached only cosine `0.9963`.
5. Target and MTP draft contexts share the same `model_tgt`; do not duplicate model
   weights. Their KV caches are intentionally separate because the target ten-layer
   state and one-layer MTP draft state are different.
6. MTP prompt optimizations must preserve all `h_pre_norm` rows, recurrent state,
   checkpoint replay, public final logits, token IDs, and decode behavior.
7. Multimodal requests must not feed media placeholder positions through unsupported
   MTP draft replay. MTP resumes only after the media checkpoint is safely restored.
8. Hardware-specific kernels need exact device/type/shape guards and an unchanged
   portable fallback for other GPUs, quant types, split buffers, and unsupported
   shapes.
9. Health alone is insufficient. Every deployment needs a real inference request,
   live executable/library inspection, model-table activation check, KV-type check,
   and log scan.

## Architecture notes

### MTP target and draft

- The server loads one target model and creates target and draft contexts from it.
- The target has ten recurrent layers; the MTP draft uses the model's one `nextn`
  layer. Their compute schedulers and KV/state buffers are separate.
- Prompt propagation needs target pre-normalized hidden rows, not a complete
  248,320-logit vector for every prompt position.
- Speculative draft quality affects acceptance and speed. With a dense target, it
  must not change the target probability distribution.
- **Dual-Track Slot Save/Restore (`ctx_tgt` & `ctx_dft`):** Slot snapshots (`/slots/:id?action=save|restore`)
  must persist both `ctx_tgt` (`.bin`) and `ctx_dft` (`.bin.dft`). Without draft state synchronization,
  subsequent multi-turn continuations or restored sessions cause M-RoPE position divergence and `GGML_ABORT`
  in `ctx_dft` decode. On restore, tokens beyond `pos_max` are safely pruned via `llama_memory_seq_rm`.
- **Hybrid / Recurrent Memory Checkpoint Retention:** In models with recurrent state (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`),
  checkpoints restored with valid prompt tokens are protected against premature memory purge on non-zero `p0`.

### FlashHead

The FlashHead GGUF contains:

```text
flashhead.centroids: [2048, 7760]
flashhead.c2t:       [32, 7760]
flashhead.static:    [4096]
```

With 256 probes, retrieval computes 7,760 centroid scores, selects 256 clusters,
adds `256 * 32` cluster token IDs plus 4,096 static IDs, and computes exact selected
output rows. Duplicate IDs leave about 11,700-11,900 unique candidates, roughly
4.7-4.8% of the 248,320-token vocabulary. Unselected logits are filled with
`-10000`.

- **Draft-only FlashHead:** approximation is confined to proposals; dense target
  verification remains authoritative.
- **All FlashHead:** the target output projection is also shortlisted. This changes
  the verifier and final sampling distribution and is not distribution-preserving.
- Current retrieval is heuristic centroid top-k. It has no residual-norm or omitted
  softmax-mass upper bound.

## Changes on the active branch

There are 34 commits from `a421d661a` through `15bd9b002`. The sections below group
what they do; the complete chronological ledger follows.

### 1. RDNA 3.5 prefill MMQ dispatch

- Limited oversized Q4_K/Q5_K MMQ tiles on RDNA 3.5 to reduce VGPR pressure and
  eliminate Q4_K spilling.
- Follow-up lowered the Q4_K cap again after profiling. Kernel time improved, while
  end-to-end 64-vs-32 tile differences were mostly noise.
- Changes are dispatch-only and preserve quantized arithmetic.

### 2. MTP prompt logits and LM-head suppression

- Added controls for MTP-only prompt evaluation so the server does not materialize
  and copy a full FP32 vocabulary vector for every prompt token.
- First narrowed prompt output to the required row; then suppressed non-final
  chunk/ubatch LM heads globally while retaining all hidden rows.
- This eliminated 55 observed copies totaling 26.05 GiB in the profiled long prompt
  and later reduced Q6_K full-output dequant from 39 launches to zero.
- Decode, multi-slot/coupled sequences, media fallbacks, and public final logits keep
  the full path where required.

### 3. Chunked Qwen SSM convolution

- Fused `CONCAT -> SSM_CONV -> SILU` for split recurrent-state and QKV inputs.
- Long chunks read split inputs directly and update recurrent state from the QKV
  tail. Short autoregressive chunks retain the original path.
- Materialized non-contiguous concat dispatches fell from 1,290 to 60 in the recorded
  trace. Numeric tests cover contiguous and strided state and multiple token/state
  shapes.

### 4. gfx1151 kernel and MoE prefill work

- Widened the gfx1151 GDN workgroup; this was correct but only a small end-to-end
  win.
- Reordered conventional MMQ grid traversal to improve weight reuse on gfx1151 while
  retaining portable grid-limit and non-gfx1151 fallbacks.
- Shared MoE Gate/Up routing compaction and Q8_1 activation preprocessing.
- Fused paired Q4_K Gate/Up MMQ, SwiGLU, and Q8_1 requantization so intermediate
  F32 Gate/Up tensors are not written to global memory.
- Fused selected-expert weighting into the Q5_K Down MMQ writeback.
- These paths are narrowly guarded for gfx1151, expected quant types, contiguous
  local buffers, and non-stream-K execution.

### 5. Q4_0 tiled Flash Attention

- Added an opt-in gfx1151 Q4_0 tiled FA path.
- Added a fused ncols=3/4 MTP draft path with aligned 32-token loaders.
- The historical Q4_0-KV experiment reduced KV memory from 5,080 MiB to 1,428 MiB,
  but production later standardized on F16 KV for numeric reliability.

### 6. Decode microkernels and graph fusion

- Replaced Q4_0 nibble zero-point subtraction with RDNA 3.5 `__vsubss4`.
- Added a prefill-derived 8,192-token active-vocabulary shortlist for the older
  Qwen35 path.
- Cached Q8_1 activation quantization across repeated Q/K/V projections.
- Added native Q4_0 fused QKV MMVQ and graph lookahead fusion.
- Simplified the MTP draft loop to use the sampler bound to the draft context rather
  than CPU sorting over the entire vocabulary.
- Added native fused MoE Gate/Up MMVQ for Q4_0, Q4_K, and Q5_K.

### 7. FlashHead retrieval

- Added FlashHead tensors to the model architecture and a sparse MTP draft LM head.
- Added selected-row matrix multiplication, full-vocabulary fill/scatter, and
  fallbacks for unsupported tensors, LoRA, wide batches, or insufficient sparsity.
- Replaced CPU Top-K fallback for 7,760 columns with GPU radix-sort/merge Top-K.
- Extended CUDA/HIP concat to contiguous and strided I32 tensors.
- Kept centroid scoring, Top-K, candidate expansion, static-ID append, selected
  projection, fill, and scatter GPU-resident.
- Recorded result: GPU-resident retrieval improved median decode from 76.09 to
  78.41 tok/s (`+3.05%`) in the controlled five-pair test.

### 8. Optional target FlashHead

- Added `LLAMA_FLASHHEAD_TARGET=1` as an explicit, approximate target-side mode.
- The dense target remains the code and production default.
- Added target-path and full-grid trace regression tests.
- Numeric and agentic findings show that this mode should remain experimental; see
  the measurements section.

### 9. Multimodal MTP recovery

- Disabled unsupported MTP draft processing while a request contains media.
- Added checkpoint/replay regression coverage.
- Added bounded replay after the media region, then enabled full configured MTP depth
  after safe checkpoint restoration.

### 10. Qwen chat template preserve_thinking and slot checkpoint alignment

- **Qwen Jinja template fix**: When `preserve_thinking: true`, empty or absent `reasoning_content` in assistant turns previously caused spurious `<think>\n\n</think>\n\n` tag emission (generating rogue token 271), diverging from the KV cache and causing full prompt re-evaluations. Added a `has_thinking` guard in `common/chat.cpp` to emit thinking tags only when actual reasoning tokens or tags are present.
- **Multimodal metadata lifecycle**: In `tools/server/server-context.cpp`, saving text-only slots now removes stale `.media.json` sidecars, preventing obsolete image metadata from corrupting subsequent restores.
- **Safe position bounds and checkpoint restoration**: In `tools/server/server-common.cpp` and `server-context.cpp`, out-of-bounds media entries (`idx >= tokens.size()`) are ignored, `pos_next()` is clamped to non-negative, and restored checkpoint `pos_max` is bounded by `llama_memory_seq_pos_max()` to prevent M-RoPE position divergence (`X < Y`).

## Active-branch commit ledger

| Commit | Change |
| --- | --- |
| `4eb5733f8` | `fix(server): compute checkpoint pos_max after loading multimodal media chunks` |
| `8a10ea94d` | `[verified] hip: tune K-quant MMQ tiles for RDNA 3.5` |
| `5c39e48f2` | `[verified] hip: lower RDNA 3.5 Q4_K MMQ tile cap` |
| `2d3f15e5b` | `[verified] mtp: skip unused Qwen prompt logits` |
| `d0d55d13d` | Documentation: add gfx1151 Qwen investigation record |
| `ec167ee26` | Documentation: add gfx1151 roofline and cache profile |
| `675d2d69d` | `[verified] mtp: skip non-final Qwen prompt heads` |
| `4d17b443c` | Documentation: record global-final MTP head optimization |
| `258a21646` | Documentation: update production deployment reference |
| `30b8617c8` | `[verified] cuda: fuse Qwen chunked SSM convolution concat` |
| `5add639aa` | Documentation: record fused Qwen SSM convolution results |
| `f68293633` | `[verified] hip: widen gfx1151 GDN workgroup` |
| `89b6e594f` | `[verified] hip: reorder gfx1151 MMQ traversal` |
| `ebcf34c41` | `[verified] hip: share gfx1151 MoE preprocessing` |
| `d77c844c1` | `[verified] hip: fuse gfx1151 MoE MMQ epilogue` |
| `8c1a71abb` | Documentation: record gfx1151 MoE MMQ deployment |
| `07689bc03` | `[verified] hip: fuse gfx1151 MoE Down weighting` |
| `0985f20ae` | Documentation: record gfx1151 Down weighting deployment |
| `ed3b6d863` | `[verified] hip: opt-in gfx1151 Q4_0 tiled flash attention` |
| `44acd0de0` | `[verified] hip: fuse gfx1151 Q4_0 tiled FA for ncols=3/4 MTP draft` |
| `6b50c61bf` | Optimize Q4_0/Q8_1 dot product with `__vsubss4` |
| `1a5982a19` | Implement prefill-driven active-vocabulary shortlist |
| `36729303d` | Cache Q8_1 activation quantization in MMVQ |
| `db1d087bb` | Add native HIP fused QKV kernel and graph auto-fusion |
| `54372e4ba` | Documentation: handoff milestone 5 |
| `8f0987060` | Optimize MTP draft generation loop and candidate sampling |
| `3e02d6104` | Documentation: handoff milestone 6 |
| `6489d11f9` | Add native HIP fused MoE Gate/Up kernel |
| `76a8742a2` | Documentation: handoff milestone 7 |
| `ada1c7983` | Add FlashHead retrieval head to Qwen35MoE MTP draft path |
| `47f056019` | `[verified] keep FlashHead retrieval operations on GPU` |
| `9a9ea7b51` | Add opt-in approximate target FlashHead |
| `b46e3e8d4` | `[verified] disable MTP for multimodal requests` |
| `63584e4de` | `[verified] resume bounded MTP after media` |
| `15bd9b002` | `[verified] enable full MTP depth after media` |

## Uncommitted change: Dynamic MTP prompt-processing reserve

### Problem

At `-c 260000 -ub 2048`, target and draft schedulers each pre-reserved a worst-case
prompt-processing graph. The dominant attention mask scaled with context and ubatch,
leaving approximately 10,496 MiB of combined target/draft compute reserve while the
server was idle.

### Design

A single internal policy, `mtp_dynamic_pp`, was added:

- `llama_cparams::mtp_dynamic_pp`
- `llama_context::set_mtp_dynamic_pp(bool)`
- public extension `llama_set_mtp_dynamic_pp(ctx, bool)`
- server activation for both target and draft MTP contexts

When enabled, `sched_reserve()` keeps the token-generation graph pre-reserved but
skips the worst-case prompt-processing reserve. The scheduler grows PP buffers to
the actual prompt shape on first use. Non-MTP contexts retain the original reserve
policy.

### Dirty files

```text
M  src/llama-context.cpp
M  src/llama-context.h
M  src/llama-cparams.h
M  src/llama-ext.h
M  tools/server/server-context.cpp
?? tests/test-mtp-reserve-memory.py
```

`tests/test-mtp-reserve-memory.py` parses startup logs and rejects a return to the
oversized target/draft reserve.

### Verified effects

- Idle RAM: `43.98 -> 34.10 GiB` (`-9.88 GiB`).
- Idle GTT: `33.12 -> 26.91 GiB` (`-6.21 GiB`).
- Long 54,460-token prompt grew cgroup memory by 5.823 GiB and GTT by 1.678 GiB as
  intended; prefill was about 967.5 tok/s.
- Dense complete-vocabulary numeric output remained bit-exact with cosine `1`.
- Draft-only FlashHead plus dense target also remained complete-vocabulary bit-exact.
- Sustained decode median changed from 73.30 to 71.79 tok/s (`-2.06%`) in the
  recorded A/B, accepted in exchange for roughly 9.88 GiB less idle memory.
- The target-context null check was strengthened before invoking the setter.

Do not replace this with a shared scratch arena without proving target/draft lifetime,
concurrency, and backend ownership. Dynamic growth was selected as the lower-risk
solution.

## Other branches and experiments not in the active branch

Do not assume these commits are present in `flashhead-lm-head`.

### `perf/gfx1151-mmq`

This branch continues after milestone 7 with decode micro-optimizations:

| Commit | Milestone |
| --- | --- |
| `3389d3189` | Hoist activation-block VGPR preloading in fused QKV and Gate/Up |
| `238a872e5` | Replace Q8_0/Q8_1 16-bit loads with aligned 32-bit loads |
| `826b71f15` | Bypass strided 64-bit indexing for contiguous Q8_1 quantization |
| `8e9a85f96` | Cache RMSNorm input values in VGPRs between passes |
| `3c98a99b1` | Hoist fused-QKV Q4_0 weights across MTP columns |
| `e282ce5d1` | Hoist fused Gate/Up expert weights across MTP columns |
| `0fc8d1566` | Use aligned 32-bit loads in Q5_0/Q8_1 dot product |
| `1384e859e` | Fuse Q4_K VMMQ scale/min accumulation |
| `e102e7bab` | Fuse Q5_K VMMQ scale/min accumulation |
| `56728285d` | Prefetch fused-QKV activation scales into VGPR arrays |
| `019fef4a3` | Prefetch fused Gate/Up activation scales into VGPR arrays |
| `f09bb3a36` | Precompute RMSNorm pass-2 scale multiplication |
| `8507dbdd4` | Fuse Q5_K MMQ scale/min accumulation |

Documentation commits between these code commits update milestones 8-19 in that
branch's `handoff-gemini.md`. The final reported peak was 68.05 tok/s, but individual
milestone numbers were not a monotonic controlled cumulative improvement; re-run a
clean A/B before cherry-picking any item.

### `memory-optimization`

Commit `3a80fae08` adds a gfx1151 GPU `KQ_MASK` operation and associated context/KV
plumbing to avoid CPU mask construction/copy. It is not on the active branch and is
not production. A later F16 attention-mask direction was rejected because full-vocab
cosine was only about `0.9963`.

### Checkpoints and WIP

- `checkpoint/mtp-multimodal-before-fix-20260806` points at target FlashHead before
  multimodal MTP fixes.
- `wip/gemini-broken-20260802` is explicitly broken; do not deploy or casually
  cherry-pick it.
- Q4-KV/turboquant worktrees are historical experiments, not the current F16-KV
  production policy.

## Measurements and decisions

### Dense, draft-only, and All FlashHead throughput

Controlled 20K-prompt / 1,024-output sustained result:

| Variant | Decode TPS | Relative to Dense |
| --- | ---: | ---: |
| Dense target + dense draft head | 62.71 | baseline |
| Dense target + FlashHead draft | 71.24 | `+13.60%` |
| FlashHead target + FlashHead draft | 75.42 | `+20.27%` |

All FlashHead adds only about `+5.9%` over draft-only in this sustained workload.
Short-output gains over draft-only were about `+4.4%` at 128 output tokens and
`+3.7%` at 256.

### Target FlashHead numeric tail

A 32-token, 248,320-logprob comparison between draft-only and All FlashHead produced:

- generated text and sampled tokens exact in that run
- concatenated cosine `0.9993788996`
- mean cosine `0.9993199705`
- minimum event cosine `0.9960851466`
- maximum common finite logprob delta `0.75894165`

Cosine masked important support error. For the eight comparable sparse events:

| Event | Unique candidates | Dense mass omitted | Actual TV distance | Dense top-20 recall |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 11,720 | 0.000082% | 0.000082% | 20/20 |
| 1 | 11,901 | 0.0132% | 0.0132% | 20/20 |
| 14 | 11,915 | 0.634% | 0.634% | 20/20 |
| 20 | 11,896 | 0.390% | 2.636% | 20/20 |
| 24 | 11,932 | 12.374% | 12.470% | 16/20 |
| 25 | 11,884 | 0.162% | 2.389% | 20/20 |
| 28 | 11,930 | 4.500% | 5.638% | 19/20 |
| 31 | 11,875 | 0.000783% | 0.00260% | 19/20 |

If selected-row logits exactly matched dense logits, target FlashHead would be the
dense distribution conditioned on the shortlist. With omitted dense mass `delta`,
ideal total variation is exactly `delta` and `KL(Q || P) = -log(1-delta)`. Events
whose actual TV exceeds omitted mass also contain retained-logit numeric differences.

The production conclusion is that target FlashHead at 256 probes has a heavy error
tail and is not worth a further 4-6% throughput for a general agent.

### Agentic test

In the controlled live "Din Tai Fung Taipei 101 queue" task:

- Dense completed in 47 seconds with four tool calls.
- Draft-only completed in 39 seconds with three tool calls.
- All FlashHead exceeded 200 repeated calls and timed out at 300 seconds without a
  final answer.

Dense and draft-only both found the correct dashboard but consumed a stale Firecrawl
snapshot; this single task is a warning, not causal proof. Use many prompts and seeds
before making broad quality claims.

### UMA allocation and copy findings

- No persistent second full GGUF mapping was found. The stable CPU-mapped model range
  was about 515 MiB.
- Cold loading temporarily had file-backed PSS near 21.19 GiB while GTT was already
  near 28.91 GiB; this is transient page-cache residency, not two permanent weights.
- A HIP interposer observed 537 Host-to-Device `hipMemcpyAsync` operations totaling
  20.922 GiB during startup and zero `hipMallocManaged` calls.
- Target and draft reuse the same loaded model. The large extra idle memory was
  compute reserve, not duplicate model weights.
- `mmap` and direct I/O do not provide GPU zero-copy. Direct I/O can reduce page-cache
  duplication but still stages and copies into HIP allocations.
- Device-resident target-to-draft hidden-state handoff remains future work.

## Rejected or deferred approaches

- **F16 attention mask:** rejected; full-vocabulary cosine only about `0.9963`.
- **All FlashHead as default:** rejected; target approximation has a heavy numeric
  tail and weak risk/reward versus draft-only.
- **KV sharing:** invalid; target and draft recurrent states are structurally
  different.
- **Assuming UMA means automatic zero-copy:** false for current HIP allocations and
  loader behavior.
- **Inferring memory from `smaps` alone:** insufficient; pair process maps with GTT,
  cgroup data, HIP allocation/copy traces, and stop/start deltas.
- **rocprofv3 as the sole memory tracer:** launch shutdown hit
  `ring_buffer mmap errno 22`; non-root attach was blocked by Yama; root detach did
  not flush CSV. The HIP interposer is the known-working fallback.
- **GGUF file-backed zero-copy:** deferred. A future prototype may test
  `hipHostRegister(...Mapped | ReadOnly)` plus `hipHostGetDevicePointer`, but it must
  compare startup peak, RSS/GTT, coherency behavior, prefill, and decode TPS.

## Build and validation

### Build

```bash
cd /home/chihmin/llama-mtp-opt
cmake --build build -j 16 --target llama-server
```

Use a Release-configured build. Never infer production behavior from a stale build
folder; inspect the live ELF and mapped libraries.

### Fast source gates

```bash
git diff --check
python3 -m py_compile \
  tests/test-mtp-prefill-copy.py \
  tests/test-mtp-prefill-lm-head.py \
  tests/test-mtp-reserve-memory.py \
  tests/test-mtp-multimodal-checkpoint-replay.py \
  tests/test-qwen35moe-flashhead-target.py
```

Run the focused structural tests relevant to touched code. Important tests include:

```text
tests/test-mtp-prefill-copy.py
tests/test-mtp-prefill-lm-head.py
tests/test-mtp-reserve-memory.py
tests/test-mtp-multimodal-checkpoint-replay.py
tests/test-qwen35moe-flashhead-target.py
tests/test-flashhead-full-grid-trace.py
tests/test-qwen35-fused-ssm-conv.py
tests/test-rdna35-gdn-launch.py
tests/test-gfx1151-mmq-weight-stationary-guards.py
tests/test-gfx1151-moe-shared-preprocess.py
tests/test-gfx1151-moe-mmq-q8-epilogue.py
tests/test-gfx1151-moe-down-weight-epilogue.py
```

For GPU retrieval changes, also run backend Top-K and concat cases and
`scripts/verify-flashhead-gpu.sh`.

### Numeric gate

For target-path changes:

1. Capture baseline and candidate on the same prompt and same prefix.
2. Request all 248,320 logprobs indexed by token ID.
3. Compare sampled IDs, finite support, bit patterns, maximum finite delta, cosine,
   omitted probability mass, total variation, JS divergence, and top-k recall.
4. Compare only vectors with identical preceding prefixes. Once a sampled token
   diverges, later contexts are not directly comparable.
5. Report median, p95, p99, and maximum. A mean cosine is not an adequate quality
   gate.

### Performance gate

Use the local skills/scripts for cold, interleaved A/B and kernel attribution:

```text
/home/chihmin/.pi/agent/skills/ab-qwen-profiling/
/home/chihmin/.pi/agent/skills/benchmark-qwen/
/home/chihmin/.pi/agent/skills/rocm-kernel-trace/
```

Distinguish server prefill TPS, server decode TPS, Pi wall time, MTP acceptance, and
agent task success. Restore production even when a benchmark is interrupted.

## Production operations

Use only:

```bash
HELPER=/home/chihmin/.pi/agent/skills/restart-qwen-mtp/scripts/restart-qwen-mtp.sh

$HELPER draft-flashhead  # production: FlashHead draft, dense target
$HELPER flashhead        # experimental: FlashHead draft + approximate target
$HELPER f16-baseline     # dense draft and target heads, F16 KV
```

The helper manages:

```text
/etc/systemd/system/qwen-mtp.service.d/zz-flashhead.conf
```

It backs up the prior drop-in, restarts systemd, waits for health, and verifies the
live binary, matching HIP library, GGUF, model alias, KV type, and denied environment
variables. `powerprofilesctl set performance` is unavailable on this platform; the
helper warns and continues.

Mandatory post-deployment checks:

```bash
systemctl show qwen-mtp.service -p MainPID -p ExecStart -p Environment -p DropInPaths
pid=$(systemctl show -p MainPID --value qwen-mtp.service)
readlink -f /proc/$pid/exe
grep -m1 'libggml-hip' /proc/$pid/maps
grep 'FlashHead tables found' /tmp/qwen35-server.log | tail -1
grep 'llama_kv_cache: size' /tmp/qwen35-server.log | tail -2
tr '\0' '\n' </proc/$pid/environ | grep -E \
  '^(LLAMA_FLASHHEAD_PROBES|LLAMA_FLASHHEAD_TARGET|LD_LIBRARY_PATH|GPU_MAX_HW_QUEUES|ROCP_TOOL_ATTACH)=' || true
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | jq '.data[0] | {id, context: .meta.n_ctx}'
```

Expected draft-only environment:

```text
LLAMA_FLASHHEAD_PROBES=256
LLAMA_FLASHHEAD_TARGET absent
GPU_MAX_HW_QUEUES absent
ROCP_TOOL_ATTACH absent
```

Run a real non-streaming inference and scan for `unknown op`, fallback, error,
assertion, and server failures. Confirm Gemma remains inactive unless explicitly
requested.

### Host protection outside the repository

`/etc/default/earlyoom` contains:

```text
--ignore '^llama-server$'
```

This prevents earlyOOM from selecting production llama-server. Do not remove it as a
memory workaround. Fix allocation policy or stop the service through systemd when an
approved benchmark requires it.

## Reference evidence

- Main investigation and UMA record: [`README.md`](README.md), section
  "Qwen 3.6 35B-A3B / gfx1151 optimization and profiling record".
- FlashHead GPU Top-K validation:
  [`docs/flashhead-gpu-topk-results.md`](docs/flashhead-gpu-topk-results.md).
- Decode milestones 1-7: [`handoff-gemini.md`](handoff-gemini.md).
- Later milestones 8-19: `git show perf/gfx1151-mmq:handoff-gemini.md`.
- Dense Dynamic-PP numeric evidence:
  `/tmp/qwen-dynamic-pp-prod-numeric-20260808-022034/`.
- All-FlashHead numeric evidence:
  `/tmp/qwen-all-flashhead-target-numeric-mtp32-20260808-130308/`.
- Temperature-0.8 comparison:
  `/tmp/qwen-abc-temp08-cosine-20260808-143736/`.
- Sustained ABC throughput:
  `/tmp/abc-qwen-flashhead-sustained-20260808-140641/`.
- Dynamic-PP deployment evidence:
  `/tmp/qwen-dynamic-mtp-deploy-20260808-021955/`.
- HIP memory trace:
  `/tmp/qwen-hipmem-profile-20260808-015301/hip-memory.csv`.

`/tmp` evidence may expire. Preserve any result needed for a formal commit or future
reproduction under durable benchmark storage before cleanup.
