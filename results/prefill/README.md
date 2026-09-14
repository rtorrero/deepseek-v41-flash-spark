# `results/prefill/` — prefill memory and speed runs

One file per experiment, appended to, never rewritten. These are throughput and memory results:
how long a prompt takes to become a first token, and what a configuration had to give up to get
there.

They are deliberately **not** in a profile's `GATE.md`. That file is the record of what a keep-set
can write — a quality record — and mixing a memory experiment into it would make both harder to
read. When an experiment runs the generation gate as a control (which it should, whenever it
touches the path the model computes on), the record here says PASS or FAIL and the gate's own
report stays where it belongs.

## What is here

| file | what it measures | written by |
|---|---|---|
| `<stamp>-a.json` / `-b.json` / `-c.json`, `-summary.json`, `-GATE.md` | `DSV41_PREFILL_KV_FP8` and a 4,096-token chunk against the baseline | `tools/verify_prefill_fp8.sh` |
| `unpack-cache-<date>.md` | `DSV41_PREFILL_UNPACK_CACHE_GB` on against off, same box, same prompt | `tools/verify_prefill_cache.sh` |
| `prompt.txt`, `run<N>-probe.json`, `run<N>-audit.json` | `DSV41_PREFILL_FP8_DEQUANT` and `DSV41_PREFILL_FUSED_SINKHORN`, each alone and together | `tools/verify_prefill_fixes.sh` |
| `RECORDS.md`, `gemm_dispatch-<mode>.json`, `GATE-frontend-<mode>.md` | `DSV41_PREFILL_ATTN_GEMM=fp32\|tf32\|bf16`, the attention softmax's math type | `tools/verify_prefill_hc.sh` |

## `verify_prefill_fp8.sh`

One JSON file per run, plus a summary and, when the last run completed, the profile gate it was
judged by.

```
<stamp>-a.json         baseline:  DSV41_PREFILL_KV_FP8=0  DSV41_PREFILL_CHUNK=2048
<stamp>-b.json         fp8:       DSV41_PREFILL_KV_FP8=1  DSV41_PREFILL_CHUNK=2048
<stamp>-c.json         fp8+chunk: DSV41_PREFILL_KV_FP8=1  DSV41_PREFILL_CHUNK=4096
<stamp>-summary.json   the three rows together
<stamp>-GATE.md        gate_profile.py --profile Frontend --thinking on, run on (c)
```

Every run is the same prompt — a fixed slice of `corpus/sources/dsv41_techreport.txt`, about 7,000
tokens — so the three prefill rates are comparable.

| field | what it is |
|---|---|
| `prefill_s`, `prefill_tok_s` | the engine's own, out of `x_engine_stats` |
| `ttft_s` | wall clock at the client to the first streamed content byte |
| `mem_available_min_gb` | the low-water mark of `/proc/meminfo` `MemAvailable`, sampled 5×/s for the whole request |
| `predicted_reserve_gb` | what `tools/budget.py` said this configuration would need |
| `engine_prefill_chunk`, `engine_prefill_kv_fp8`, `engine_ring` | read back from the server, not trusted from the environment — a variable that never reached the engine is exactly the failure this comparison would otherwise report as "fp8 changes nothing" |

`mem_available_min_gb` is the one that matters. The prefill reserve in `tools/budget.py` is a linear
fit to this measurement taken on 2026-09-12, and a run here is how it gets a better one. The engine's
watchdog kills the process below 2.5 GB, so a run whose floor is near that number is a configuration
that does not serve, whatever its tok/s says.

The gate is appended here rather than to `results/keepsets/frontend/GATE.md`: that file is the record
of the shipped configuration, and a run with `DSV41_PREFILL_KV_FP8=1` is not it.

## Reading an unpack-cache record

Every record names the profile, the keep fraction, the cache budget and the checksum of the prompt,
because a prefill number means nothing without them:

* **prefill work scales with the number of chunks, not with the prompt length.** Every chunk unpacks
  nearly every expert of every layer it visits again, so 291 / 326 / 369 tok/s at chunk 512 / 1024 /
  2048 is the same engine (`NOTES.md` 2026-09-12). A record taken at a different
  `DSV41_PREFILL_CHUNK` is a different experiment.
* **the prompt has to be the same one.** `tools/longprefill.py` builds it deterministically from
  the checkout's own corpus in a fixed file order, so two loads on two days see identical bytes.
  The checksum in the record is how you know.
* **one engine load per configuration.** The arena is sized from what is free at start-up, so two
  configurations compared inside one load are not two configurations.

## `verify_prefill_fixes.sh` — the prefill fixes

What `tools/verify_prefill_fixes.sh` writes on the box. The two fixes are the ones
`docs/gemm-dispatch.md` names at the end of its inventory:

* **Fix A**, `DSV41_PREFILL_FP8_DEQUANT` — the dense fp8 weights that are still dequantised to
  bf16 per call, per layer, per chunk above `M = 16`;
* **Fix B**, `DSV41_PREFILL_FUSED_SINKHORN` — prefill's 20-iteration Hyper-Connection Sinkhorn,
  run through `tiled_rows` in 16-row tiles when a fused kernel for it already exists.

Both default to off, and off is the engine every number in `RESULTS.md` was measured on. This
directory is where the numbers that would justify turning them on live.

### What lands here

| file | what it is |
|---|---|
| `prompt.txt` | the one ~7,000-token prompt every run uses. Written once, on the first run, and reused verbatim afterwards — "identical prompt" is then a fact about a file rather than a claim about a generator. Delete it to re-derive. |
| `run<N>-probe.json` | run `N`'s requests: a discarded warm-up (the first request of a load compiles the Triton kernels), then `REPEATS` timed streamed requests with client-side TTFT and the server's own `x_engine_stats.prefill_s` / `prefill_tok_s`, plus the median of the three. |
| `run<N>-audit.json` | `tools/audit_gemm_dispatch.py --phase prefill` for runs 1 and 4 only, taken with the server stopped. The launch count and the GPU-busy fraction are here; they are the only numbers that can show Fix B did anything, because each of the ~700,000 Sinkhorn launches it removes is individually too small to appear in a top-20-by-GPU-time table. |

The four runs are: 1 both off (the baseline), 2 Fix A, 3 Fix B, 4 both. The generation gate that
follows run 4 appends to `results/keepsets/frontend/GATE.md`, where every other Frontend gate run
already is, rather than to a second record here.

### Reading it

The probe table the script prints at the end is the answer. Two things to check before believing
it:

* **the prompt token count is the same in all four files.** It is the same file, so it should be;
  if it is not, something re-derived it between runs and the comparison is void.
* **run 1's audit reproduces the shape the audit page predicts** — a GPU-busy fraction well under
  1 and a launch count in the hundreds of thousands. If it does not, the runs are measuring
  something other than the served configuration and run 4 means nothing either.

Nothing here is a release number until it has been taken twice, on two days, on the same `.env`.

## `verify_prefill_hc.sh` — the attention GEMM math type

What `tools/verify_prefill_hc.sh` leaves behind. Empty until the script has been run on the box —
this directory ships with the question, not with an answer.

| file | what it is |
|---|---|
| `RECORDS.md` | one section per run: the three modes' prefill tok/s and TTFT on one identical prompt, what the kernel audits said, and which gate (if any) was taken |
| `prompt.txt` | the prompt the run used, written once and reused by all three modes, so "identical prompt" is a property of the bytes |
| `gemm_dispatch-fp32.json`, `gemm_dispatch-tf32.json` | `tools/audit_gemm_dispatch.py --phase prefill --json`, server stopped, one mode each |
| `GATE-frontend-<mode>.md` | `tools/gate_profile.py --profile Frontend --thinking on` under a non-default math type |

### The question these records answer

A 2,048-token prefill chunk's largest single kernel cost is not a quantised kernel. It is the pair
of fp32 GEMMs inside the attention softmax — 672 calls each, 855 ms together, against 893 ms for
both FP4 MoE kernels — and fp32 on this box means the SIMT pipeline rather than the tensor cores.
`DSV41_PREFILL_ATTN_GEMM=fp32|tf32|bf16` is the switch; the arithmetic, the shapes and the error
bounds are in [`../../docs/gemm-dispatch.md`](../../docs/gemm-dispatch.md).

### Reading a record

The speed rows are the cheap half and they settle nothing on their own. Two things decide whether
a non-default mode is worth anything:

* **Did the flag take?** The `fp32` audit must show the two 672-call
  `cutlass_*_simt_sgemm_*_{tn,nn}_align1` rows and the `tf32` audit must not. If both tables look
  the same, cuBLAS never saw the math type and the run measured fp32 three times.
* **What does the model write?** `tf32` is ~1e-3 on the product and `bf16` ~4e-3, against a design
  whose chunk-invariance argument is built on 1e-7. The failure mode is a flipped router top-k
  deep in a long prompt, not a wrong number, so the gate output compared against
  `../keepsets/frontend/GATE.md` is the only evidence that accepts either mode. A record whose
  gate section says none was run accepts nothing.

The gate output lives here and not in `../keepsets/frontend/GATE.md`: those files are the record of
the shipped configuration, and an experimental math type has no business in them.
