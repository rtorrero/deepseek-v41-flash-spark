
## 2026-09-15 02:25Z — prefill attention GEMM math type, frontend at keep 0.36

`DSV41_PREFILL_ATTN_GEMM` over `fp32 tf32 bf16`, one identical prompt (`results/prefill/prompt.txt`,
28000 chars), `EXPERT_PROFILE=frontend PRUNE_KEEP=0.36 ARENA_GB=81`,
`EXPERT_FORMAT=cb3 DSV41_PREFILL_CHUNK=2048 MAX_SEQ=262144`,
thinking off for the probes and on for the gate, effort 45. One prompt per mode, no sweep.

| | fp32 (shipped) | tf32 | bf16 |
|---|---|---|---|
| prompt tokens | 6678 | 6678 | 6678 |
| **prefill tok/s** | 253.73 | 266.47 | 357.93 |
| prefill s | 26.32 | 25.061 | 18.657 |
| **TTFT s** | 26.341 | 25.079 | 18.673 |
| wall s | 31.96 | 30.39 | 23.31 |
| decode tok/s | 11.03 | 11.87 | 12.95 |
| DSpark acceptance | 3.1 | 2.62 | 2.86 |
| **prefill tok/s against fp32** | 1.000x | 1.050x | 1.411x |

Kernel audits: results/prefill/gemm_dispatch-fp32.json results/prefill/gemm_dispatch-tf32.json — the fp32 table must show the two 672-call
`cutlass_*_simt_sgemm_*_{tn,nn}_align1` rows; the tf32 table must not.

Generation gate on `tf32`: `results/prefill/GATE-frontend-tf32.md` — compare against
`results/keepsets/frontend/GATE.md` for the shipped engine on the same prompts. A
non-default mode is accepted by that comparison, not by the tok/s above.

Transcript: `/tmp/prefill_hc.log`.
