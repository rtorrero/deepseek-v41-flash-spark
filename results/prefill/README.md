# Prefill memory and speed runs

What `tools/verify_prefill_fp8.sh` writes. One JSON file per run, plus a summary and, when the last
run completed, the profile gate it was judged by.

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
