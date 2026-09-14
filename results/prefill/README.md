# `results/prefill/` — prefill experiments

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
| `unpack-cache-<date>.md` | `DSV41_PREFILL_UNPACK_CACHE_GB` on against off, same box, same prompt | `tools/verify_prefill_cache.sh` |

## Reading one

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
