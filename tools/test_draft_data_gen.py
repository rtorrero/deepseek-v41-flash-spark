"""What the drafter's training data is made of, checked without a GPU, a model or a network.

Run: python3 tools/test_draft_data_gen.py

Three things are worth pinning here, and none of them needs torch:

  * **The mix.** The fine-tune exists to move prose acceptance without losing markup acceptance, so
    a sampler that quietly draws 95 % prose is the failure that would make the whole exercise look
    like a bad idea. The share is checked, the determinism is checked, and so is the one exclusion
    that matters: `corpus/heldout_sources/` is the corpus every teacher-forced number in RESULTS.md
    is measured on and must never be trained on.
  * **The record format.** 30,988 bytes per position is not a detail: it is what makes 300,000
    positions 9.3 GB instead of 500. The size and the field layout are asserted, because a reader
    and a writer that disagree about them produce plausible garbage rather than an error.
  * **The sample index.** `usable_starts` decides which recorded positions can be trained on, and
    the two rules it enforces -- a full window inside ONE run, a recorded distribution at every
    supervised step -- are exactly the rules that keep a training sample from straddling two
    different generations.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "engine"))

import draft_data_gen as G  # noqa: E402
import draft_record as DR  # noqa: E402
import train_mtp as T  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


# =============================================================================
# the corpus and the mix
# =============================================================================

sources = G.discover_sources()
check("the corpus has sources", len(sources) > 20, f"{len(sources)}")
check("both registers are present",
      bool(G.register_pool(sources, G.PROSE)) and bool(G.register_pool(sources, G.CODEY)),
      f"{len(G.register_pool(sources, G.PROSE))} prose/reasoning, "
      f"{len(G.register_pool(sources, G.CODEY))} code/markup")
check("the held-out corpus is never a source",
      not any("heldout_sources" in s["path"] for s in sources))
check("every source carries a register the mix knows",
      all(s["register"] in G.PROSE + G.CODEY for s in sources))
check("a .txt under prose/ is prose",
      any(s["register"] == "prose" and s["path"].endswith("prose/fiction1.txt") for s in sources))
check("an .html file is markup",
      any(s["register"] == "markup" and s["path"].endswith(".html") for s in sources))
check("a reasoning_*.txt is its own register",
      any(s["register"] == "reasoning" for s in sources))

picks = G.plan_mix(sources, 400, 0.75, seed=1)
share = sum(1 for s in picks if s["register"] in G.PROSE) / len(picks)
check("the prose share is the one that was asked for", abs(share - 0.75) < 0.02, f"{share:.3f}")
check("the mix is deterministic in the seed",
      [s["path"] for s in G.plan_mix(sources, 50, 0.75, 7)] ==
      [s["path"] for s in G.plan_mix(sources, 50, 0.75, 7)])
check("a different seed is a different draw",
      [s["path"] for s in G.plan_mix(sources, 50, 0.75, 7)] !=
      [s["path"] for s in G.plan_mix(sources, 50, 0.75, 8)])

# Round-robin within a register, not a uniform draw: over a few hundred requests every code/markup
# source has to appear, or the fine-tune has a register it never saw.
codey = {s["path"] for s in sources if s["register"] in G.CODEY}
drawn = {s["path"] for s in G.plan_mix(sources, 2000, 0.75, 3) if s["register"] in G.CODEY}
check("every code/markup source is drawn at least once over a long plan",
      drawn == codey, f"{len(drawn)}/{len(codey)}")

check("the next request comes from the group behind its TOKEN share, not its request share",
      G.prose_behind({}, 0.75) and G.prose_behind({"prose": 100, "code": 300}, 0.75)
      and not G.prose_behind({"prose": 400, "reasoning": 50, "code": 100}, 0.75)
      and not G.prose_behind({"code": 10}, 0.0))
check("prose_share 1.0 needs no code sources drawn",
      all(s["register"] in G.PROSE for s in G.plan_mix(sources, 100, 1.0, 5)))

# =============================================================================
# passages and prompts
# =============================================================================

import random  # noqa: E402

rng = random.Random(0)
text = "\n".join(f"line {i} with a few words on it" for i in range(200))
p = G.passage(text, rng, 20, 40)
check("a passage starts at a line boundary", p.startswith("line "))
check("a passage is about the length it was asked for", 20 <= len(p.split()) <= 60, f"{len(p.split())} words")
check("a passage out of blank text is empty", G.passage("\n\n\n", rng) == "")

prose_src = {"register": "prose", "topic": "t", "fence": "", "path": "x"}
code_src = {"register": "code", "topic": "t", "fence": "python", "path": "y"}
check("a prose prompt asks for prose only", "Write prose only" in G.build_prompt(prose_src, "abc"))
check("a code prompt fences the passage with its language",
      "```python" in G.build_prompt(code_src, "abc"))
check("the sample key is stable", G.sample_key(prose_src, "abc") == G.sample_key(prose_src, "abc"))
check("the sample key follows the text", G.sample_key(prose_src, "abc") != G.sample_key(prose_src, "abd"))

# =============================================================================
# the estimate and the resume state
# =============================================================================

est = G.estimate(300000)
check("300k tokens at 17 tok/s is about 5 hours", 4.5 < est["hours"] < 5.7, f"{est['hours']:.2f} h")
check("the estimate counts the requests it will make", 1500 < est["requests"] < 2000,
      f"{est['requests']}")

with tempfile.TemporaryDirectory() as d:
    st = G.load_state(d)
    check("a missing resume file reads as an empty run", st["tokens"] == 0 and st["done"] == [])
    st["tokens"] = 1234
    st["done"].append("prose/x/abc")
    G.save_state(d, st)
    back = G.load_state(d)
    check("the resume state round-trips", back["tokens"] == 1234 and back["done"] == ["prose/x/abc"])
    check("the resume file is where the shards are",
          os.path.exists(os.path.join(d, "draft_data_gen.state.json")))

# =============================================================================
# the record format and its manifest
# =============================================================================

dt = DR.record_dtype()
check("a record is 30,988 bytes", DR.record_bytes() == 30988, f"{DR.record_bytes()}")
check("the record fields are the five the trainer reads",
      list(dt.names) == ["position", "token", "flags", "top_ids", "top_vals", "hidden"])
check("the hidden state is 15,360 raw bfloat16 halves",
      dt["hidden"].shape == (15360,) and dt["hidden"].base == np.dtype("<u2"))
check("the top-32 ids and values are 32 wide",
      dt["top_ids"].shape == (32,) and dt["top_vals"].shape == (32,))
check("300,000 positions are 9.3 GB", abs(DR.estimate_gb(300000) - 9.30) < 0.02,
      f"{DR.estimate_gb(300000):.2f} GB")

m = DR.manifest("/x/shard-1.bin", 15360, 32, {"tag": "p42"},
                [{"index": 0, "position": 100, "n": 128}], 128)
check("the manifest names the format and its version",
      m["format"] == "dsv41-draft-data" and m["version"] == DR.FORMAT_VERSION)
check("the manifest carries what a reader needs to open the shard",
      m["hidden_dim"] == 15360 and m["topk"] == 32 and m["record_bytes"] == 30988)
check("the manifest keeps the runs", m["runs"][0]["position"] == 100)
check("the manifest is JSON", json.loads(json.dumps(m))["shard"] == "shard-1.bin")

# The writer's own bookkeeping, without torch: `add` is numpy in, bytes out.
with tempfile.TemporaryDirectory() as d:
    r = DR.DraftRecorder(d, hidden_dim=8, topk=2)
    r.open_run("test", 3)
    hid = np.arange(16, dtype=np.uint16).reshape(2, 8)
    r.add(np.array([10, 11]), np.array([5, 6]), hid, np.zeros((2, 2), np.float32),
          np.zeros((2, 2), np.int32), has_dist=False)
    r.add(np.array([12]), np.array([7]), hid[:1], np.array([[1.0, 0.5]], np.float32),
          np.array([[9, 8]], np.int32))
    path = r.path
    r.close_run("done")
    rec = DR.read_shard(path, 8, 2)
    man = DR.load_manifest(path)
    check("the shard holds one record per position", len(rec) == 3)
    check("positions and tokens survive the round trip",
          list(rec["position"]) == [10, 11, 12] and list(rec["token"]) == [5, 6, 7])
    check("context rows carry no distribution and settled rows do",
          list(rec["flags"]) == [0, 0, DR.FLAG_HAS_DIST])
    check("the top-k of a settled row survives", list(rec["top_ids"][2]) == [9, 8])
    check("the hidden bits survive verbatim", list(rec["hidden"][1]) == list(range(8, 16)))
    check("contiguous appends are ONE run", len(man["runs"]) == 1 and man["runs"][0]["n"] == 3)
    check("the manifest counts what the file holds", man["records"] == 3)

    r2 = DR.DraftRecorder(d, hidden_dim=8, topk=2)
    r2.open_run("gap", 0)
    r2.add(np.array([0]), np.array([1]), hid[:1], None, None, has_dist=False)
    r2.add(np.array([40]), np.array([2]), hid[:1], None, None, has_dist=False)  # a jump
    p2 = r2.path
    r2.close_run("done")
    check("a gap in positions starts a new run", len(DR.load_manifest(p2)["runs"]) == 2)

# =============================================================================
# which recorded positions can be trained on
# =============================================================================

W, D = 8, 5
pos = np.arange(40)
flags = np.full(40, DR.FLAG_HAS_DIST, dtype=np.int32)
flags[:W] = 0                                  # the prompt tail: window only
starts = T.usable_starts(pos, flags, W, D)
check("a sample needs a full window behind it", starts.min() == W, f"min {starts.min()}")
check("a sample needs `draft` records ahead of it", starts.max() == 40 - D - 1, f"max {starts.max()}")

pos_gap = np.concatenate([np.arange(20), np.arange(100, 120)])
starts_gap = T.usable_starts(pos_gap, np.full(40, DR.FLAG_HAS_DIST, np.int32), W, D)
check("no sample straddles two runs",
      all(not (s < 20 <= s + D) for s in starts_gap) and (starts_gap >= W).all())
check("a run shorter than window + draft yields nothing",
      len(T.usable_starts(np.arange(10), np.full(10, 1, np.int32), W, D)) == 0)

flags_hole = np.full(40, DR.FLAG_HAS_DIST, dtype=np.int32)
flags_hole[20] = 0
n_all = len(T.usable_starts(pos, np.full(40, DR.FLAG_HAS_DIST, np.int32), W, D))
n_hole = len(T.usable_starts(pos, flags_hole, W, D))
check("a position with no recorded distribution cannot be supervised",
      n_hole == n_all - D, f"{n_all} -> {n_hole}")

# =============================================================================
# the trainable groups
# =============================================================================

check("every default group is a real group", all(g in T.GROUPS for g in T.DEFAULT_GROUPS))
check("the routed experts (`ffn.experts.N`) are in no trainable group -- only the shared one is",
      not any(s.startswith("ffn.experts") for g in T.GROUPS.values() for s in g))
check("main_proj is trainable by default",
      "main_proj" in {s for g in T.DEFAULT_GROUPS for s in T.GROUPS[g]})

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
