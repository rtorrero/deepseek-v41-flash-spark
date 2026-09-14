# Prefill fixes — the record

What `tools/verify_prefill_fixes.sh` writes on the box. The two fixes are the ones
`docs/gemm-dispatch.md` names at the end of its inventory:

* **Fix A**, `DSV41_PREFILL_FP8_DEQUANT` — the dense fp8 weights that are still dequantised to
  bf16 per call, per layer, per chunk above `M = 16`;
* **Fix B**, `DSV41_PREFILL_FUSED_SINKHORN` — prefill's 20-iteration Hyper-Connection Sinkhorn,
  run through `tiled_rows` in 16-row tiles when a fused kernel for it already exists.

Both default to off, and off is the engine every number in `RESULTS.md` was measured on. This
directory is where the numbers that would justify turning them on live.

## What lands here

| file | what it is |
|---|---|
| `prompt.txt` | the one ~7,000-token prompt every run uses. Written once, on the first run, and reused verbatim afterwards — "identical prompt" is then a fact about a file rather than a claim about a generator. Delete it to re-derive. |
| `run<N>-probe.json` | run `N`'s requests: a discarded warm-up (the first request of a load compiles the Triton kernels), then `REPEATS` timed streamed requests with client-side TTFT and the server's own `x_engine_stats.prefill_s` / `prefill_tok_s`, plus the median of the three. |
| `run<N>-audit.json` | `tools/audit_gemm_dispatch.py --phase prefill` for runs 1 and 4 only, taken with the server stopped. The launch count and the GPU-busy fraction are here; they are the only numbers that can show Fix B did anything, because each of the ~700,000 Sinkhorn launches it removes is individually too small to appear in a top-20-by-GPU-time table. |

The four runs are: 1 both off (the baseline), 2 Fix A, 3 Fix B, 4 both. The generation gate that
follows run 4 appends to `results/keepsets/frontend/GATE.md`, where every other Frontend gate run
already is, rather than to a second record here.

## Reading it

The probe table the script prints at the end is the answer. Two things to check before believing
it:

* **the prompt token count is the same in all four files.** It is the same file, so it should be;
  if it is not, something re-derived it between runs and the comparison is void.
* **run 1's audit reproduces the shape the audit page predicts** — a GPU-busy fraction well under
  1 and a launch count in the hundreds of thousands. If it does not, the runs are measuring
  something other than the served configuration and run 4 means nothing either.

Nothing here is a release number until it has been taken twice, on two days, on the same `.env`.
