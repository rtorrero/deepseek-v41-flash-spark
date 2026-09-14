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
| `<stamp>-topk-a.json` / `-c.json` / `-b.json`, `-topk-summary.json`, `-topk-GATE.md` | fewer routed experts per prefill token, folded (`DSV41_PREFILL_TOPK` / `DSV41_PREFILL_FOLD`) against the shipped six | `tools/verify_prefill_topk.sh` |

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

## `verify_prefill_topk.sh`

One JSON file per run, a summary, and the gate the k=4 configuration was judged by.

```
<stamp>-topk-a.json       baseline:  DSV41_PREFILL_TOPK unset        (six routed experts)
<stamp>-topk-c.json       three:     DSV41_PREFILL_TOPK=3  FOLD=exfold
<stamp>-topk-b.json       four:      DSV41_PREFILL_TOPK=4  FOLD=exfold
<stamp>-topk-summary.json the three rows together, plus what the bound test predicted
<stamp>-topk-GATE.md      gate_profile.py --profile Frontend --thinking on, run on (b)
```

The runs happen in the order a, c, b so that b is the server still standing when the gate runs.

Two things to read the numbers with. The speed ceiling is small and known in advance: the FP4 MoE
kernels are ~893 ms of a ~3.7 s chunk, so scaling only that share gives ~1.075x prefill at k=4 and
~1.115x at k=3 — the summary prints those next to what was measured, and a measurement far above
them means something other than the routed k moved. And the row that decides anything is the gate,
not the tok/s: a prompt's routing approximation is still being read hundreds of tokens into the
answer, through the window KV the decoder layers inherit from prefill.
