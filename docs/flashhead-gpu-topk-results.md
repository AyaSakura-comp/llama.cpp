# FlashHead GPU Top-K validation

## Change

- HIP `TOP_K` above 1024 columns uses hipCUB/rocPRIM radix sort on the GPU.
- Existing bitonic Top-K remains unchanged for 1024 columns or fewer.
- CUDA/HIP concat now supports contiguous and strided I32 tensors.
- The FlashHead path is GPU-resident through centroid scoring, Top-K, cluster expansion, static-ID append, selected projection, fill, and scatter.

## Correctness

`test-backend-ops` results on gfx1151 / ROCm 7.2.2:

- FlashHead `TOP_K(f32, 7760 -> 256)`: 1/1 passed.
- All GPU Top-K cases: 446/446 passed, including ties and multi-row inputs.
- All concat cases: 32/32 passed; the 16 I32 contiguous/strided cases passed.

## Top-K overhead

rocprof trace: `/home/chihmin/benchmark-data/trace-flashhead-gpu-20260805-023249`

There were 239 FlashHead selections for 256 committed output tokens.

| GPU operation | Mean per selection |
| --- | ---: |
| Initialize indices | 1.31 us |
| rocPRIM block radix sort | 6.20 us |
| Three merge passes | 7.04 us |
| GPU Top-K kernels total | 14.55 us |
| I32 concat | 1.76 us |

The first Top-K of a cold server had a one-time 15.8 ms graph/library warm-up gap. Every later stage-1-to-Top-K gap had p90 below 3.3 us.

| FlashHead critical path | CPU fallback | GPU operations |
| --- | ---: | ---: |
| Median stage 1 through scatter | 411.8 us | 212.9 us |
| Mean GPU kernels in path | 160.0 us | 173.6 us |
| Decode CPU utilization | about 4.44 cores | about 1.78 cores |

GPU Top-K adds about 13.6 us of kernel work per selection, but removes about 199 us from the steady-state critical path by eliminating backend synchronization.

## End-to-end A/B

Evidence: `/home/chihmin/benchmark-data/ab-flashhead-gpu-5pair-20260805-023515`

Five cold, interleaved 20,000-input / 256-output samples used identical model data and settings. The baseline was the committed CPU-fallback FlashHead build; the candidate was the GPU-operations build.

| Variant | Server decode TPS runs | Median | Median ms/token | MTP acceptance |
| --- | --- | ---: | ---: | ---: |
| CPU fallback | 76.24, 75.83, 76.09, 76.00, 76.45 | 76.09 | 13.142 | 98.676% |
| GPU operations | 78.41, 78.28, 77.87, 79.26, 79.01 | 78.41 | 12.753 | 98.677% |

Result: **+3.05% median decode TPS** over CPU-fallback FlashHead. Prefill was unchanged within run noise. Kernel dispatches increased from about 809 to 814 per token, but unrelated kernel categories did not regress consistently; total traced GPU kernel time decreased slightly within run-to-run variation.

No production deployment or service was changed. Qwen, Gemma, and the display manager remained stopped after validation.
