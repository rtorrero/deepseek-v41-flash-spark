"""test_trace_extras.py -- top-1 and co-routing, from the trace arrays to the atlas file.

Run: python3 tools/test_trace_extras.py

Two summaries were missing from the expert atlas because the reduction never wrote them, not
because the trace never saw them: which of a token's six experts carried the largest gate
weight (`top1_<topic>`, drawn as `top1_share`), and which experts a token tends to take
together (`pairs_<topic>`, drawn as the co-routing ring). Both are derived from arrays
`tools/expert_trace.py` has stored since the first run -- `indices` and `weights` -- so nothing
was added to the tracer.

That derivation is where the arithmetic can go wrong, and all of it is checked here against a
brute-force count on a trace small enough to count by hand:

  1. `top1_hist` picks the largest WEIGHT, not the first column. The tracer stores `indices` in
     the router's selection order, which includes the aux-loss-free selection bias, and
     `weights` are the renormalised gate weights gathered in that same order -- so the six are
     not sorted by weight and column 0 is not the first pick. The synthetic trace below is
     built so that the two answers differ, and the test fails if the column is taken.
  2. the histograms are the definition: every `top1_<topic>` row sums to that topic's tokens
     (where `counts_<topic>` sums to tokens x top-k), and equals a token-by-token argmax.
  3. `pair_table` counts the C(top-k, 2) pairs of each token: every count it reports is the
     brute-force count, and every lift is `count / (count_a * count_b / tokens)`.
  4. the three `--pairs` modes put the tables where they say they do.
  5. `tools/atlas_export.py` turns them into `top1_share` (rows sum to 1) and `coroute` in the
     shape the page reads -- and, on a stats file that carries neither, still writes exactly
     what it wrote before, with both fields absent.

No torch, no GPU, no checkpoint.
"""
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import atlas_export as AX   # noqa: E402
import budget as B          # noqa: E402
import expert_stats as S    # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


# --- a trace small enough to count by hand ----------------------------------
# 3 layers, 8 experts, 50 tokens, 2 categories. Eight experts and top-6 means 28 possible
# pairs against 15 taken per token, so every pair is observed often enough for a lift to mean
# something -- the real trace has 384 experts and the same 15 pairs a token.

N_LAYERS, N_USED, N_TOK, TOPK = 3, 8, 50, 6
CATS = ("alpha", "beta")


def synth_layer(L: int, rng):
    """One layer's arrays, in the tracer's own layout.

    `indices` is the router's selection order and `weights` are that selection's gate weights,
    NOT sorted: the largest weight lands on a column drawn at random, which is what the bias
    in `(scores + gate_bias).topk(k)` does to a real trace (on the reference run only 44 % of
    tokens have their six weights in descending order)."""
    idx = np.stack([rng.permutation(N_USED)[:TOPK] for _ in range(N_TOK)]).astype(np.int16)
    w = rng.uniform(0.05, 0.2, size=(N_TOK, TOPK))
    lead = rng.integers(0, TOPK, size=N_TOK)          # which column carries the largest weight
    w[np.arange(N_TOK), lead] = rng.uniform(0.4, 0.6, size=N_TOK)
    nrm = rng.uniform(0.5, 4.0, size=(N_TOK, TOPK))
    cat = np.array([CATS[t % len(CATS)] for t in range(N_TOK)])
    return (idx, w.astype(np.float16), (w * nrm).astype(np.float32), cat)


def write_trace(root: str) -> dict:
    os.makedirs(os.path.join(root, "trace"), exist_ok=True)
    rng = np.random.default_rng(20260914)
    arrays = {}
    for L in range(N_LAYERS):
        idx, w, contrib, cat = synth_layer(L, rng)
        arrays[L] = (idx, w, contrib, cat)
        np.savez_compressed(os.path.join(root, "trace", f"layer{L}.npz"),
                            indices=idx, weights=w, contrib_norms=contrib, category=cat,
                            token=np.arange(N_TOK, dtype=np.int32),
                            scores=np.zeros(0, np.float16))
    json.dump({"n_tokens": N_TOK * N_LAYERS, "n_seqs": len(CATS)},
              open(os.path.join(root, "meta.json"), "w"))
    return arrays


TMP = tempfile.mkdtemp(prefix="trace-extras-")
try:
    TRACE = os.path.join(TMP, "run")
    ARR = write_trace(TRACE)

    idx0, w0, _c0, cat0 = ARR[0]
    idx0i, w0f = idx0.astype(np.int64), w0.astype(np.float32)

    # --- 1. the first pick is the largest weight, not the first column -------
    lead_col = w0f.argmax(1)
    check("the synthetic trace does not sort the six by weight -- as the real one does not",
          (lead_col != 0).sum() > N_TOK // 4,
          f"{int((lead_col == 0).sum())} of {N_TOK} tokens lead on column 0")
    want = np.bincount(idx0i[np.arange(N_TOK), lead_col], minlength=S.N_EXP)
    by_column = np.bincount(idx0i[:, 0], minlength=S.N_EXP)
    got = S.top1_hist(idx0i, w0f)
    check("top1_hist is the token-by-token argmax of the gate weights",
          np.array_equal(got, want))
    check("  and is NOT the first column of `indices`", not np.array_equal(got, by_column),
          "the two agree, so this trace cannot tell the two definitions apart")
    check("  its row sums to the tokens, where a counts row sums to tokens x top-k",
          got.sum() == N_TOK and np.bincount(idx0i.reshape(-1), minlength=S.N_EXP).sum() == N_TOK * TOPK)
    check("  and an empty slice is an empty histogram, not a crash",
          S.top1_hist(idx0i[:0], w0f[:0]).sum() == 0)

    # --- 2. the pair table is the brute-force count --------------------------
    table = S.pair_table(idx0i, top=32, min_count=5)
    brute = Counter()
    for row in idx0i:
        for a, b in itertools.combinations(sorted(set(int(x) for x in row)), 2):
            brute[(a, b)] += 1
    check("every pair_table count is the brute-force count of that pair",
          all(brute[(a, b)] == n for a, b, n, _ in table),
          str([(a, b, n, brute[(a, b)]) for a, b, n, _ in table if brute[(a, b)] != n][:3]))
    check("  a < b in every row, and no pair appears twice",
          all(a < b for a, b, _, _ in table) and len({(a, b) for a, b, _, _ in table}) == len(table))
    check("  rows come out ordered by count, descending",
          [n for _, _, n, _ in table] == sorted((n for _, _, n, _ in table), reverse=True))
    marg = np.bincount(idx0i.reshape(-1), minlength=S.N_EXP).astype(float)
    bad_lift = [(a, b, lf, n * N_TOK / (marg[a] * marg[b]))
                for a, b, n, lf in table
                if abs(lf - n * N_TOK / (marg[a] * marg[b])) > 1e-3 * max(1.0, lf)]
    check("  lift is count / (count_a x count_b / tokens)", not bad_lift, str(bad_lift[:2]))
    check("  with all 28 pairs of 8 experts observed, the whole table is kept",
          len(table) == len(brute), f"{len(table)} of {len(brute)}")
    check("  a table with fewer than two picks a token is empty, not a crash",
          S.pair_table(idx0i[:, :1]) == [] and S.pair_table(idx0i[:0]) == [])

    # --- 3. expert_stats writes both, per topic ------------------------------
    def run_stats(out, *extra, trace=None):
        r = subprocess.run([sys.executable, os.path.join(HERE, "expert_stats.py"),
                            "--trace", trace or TRACE, "--out", os.path.join(TMP, out),
                            "--budgets", "8", *extra], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
        return os.path.join(TMP, out)

    STATS = run_stats("stats")
    cov = json.load(open(os.path.join(STATS, "coverage.json")))
    L0 = cov["per_layer"]["0"]
    check("expert_stats writes top1_<topic> beside counts_<topic>",
          all(f"top1_{c}" in L0 and f"counts_{c}" in L0 for c in CATS) and "top1" in L0,
          ", ".join(sorted(k for k in L0 if k.startswith("top1"))))
    for c in CATS:
        m = cat0 == c
        check(f"  top1_{c} is the argmax histogram of that topic's tokens",
              np.array_equal(np.asarray(L0[f"top1_{c}"]), S.top1_hist(idx0i[m], w0f[m])))
        check(f"  and its row sums to the {int(m.sum())} tokens tagged {c}",
              sum(L0[f"top1_{c}"]) == int(m.sum()), str(sum(L0[f"top1_{c}"])))
    check("  the mixed top1 row is the topics added up",
          np.array_equal(np.asarray(L0["top1"]),
                         sum(np.asarray(L0[f"top1_{c}"]) for c in CATS)))
    check("  and the counts histograms are untouched",
          np.array_equal(np.asarray(L0["counts"]),
                         np.bincount(idx0i.reshape(-1), minlength=S.N_EXP)))

    # --- 4. where the pair tables go -----------------------------------------
    side = json.load(open(os.path.join(STATS, "pairs.json")))
    check("--pairs sibling (the default) writes pairs.json beside coverage.json",
          set(side["per_layer"]) == {str(L) for L in range(N_LAYERS)}
          and all(f"pairs_{c}" in side["per_layer"]["0"] for c in CATS)
          and side["min_count"] == 5 and side["top"] == 32 and side["topk"] == TOPK,
          json.dumps({k: v for k, v in side.items() if k != "per_layer"}))
    check("  and keeps coverage.json free of them",
          not any(k.startswith("pairs") for k in L0) and "pairs_meta" not in cov)
    for c in CATS:
        m = cat0 == c
        check(f"  pairs_{c} is the pair table of that topic's tokens",
              side["per_layer"]["0"][f"pairs_{c}"] == S.pair_table(idx0i[m], 32, 5))

    INLINE = run_stats("stats-inline", "--pairs", "inline")
    cov_i = json.load(open(os.path.join(INLINE, "coverage.json")))
    check("--pairs inline puts them in coverage.json instead",
          all(f"pairs_{c}" in cov_i["per_layer"]["0"] for c in CATS)
          and cov_i["pairs_meta"]["min_count"] == 5
          and not os.path.exists(os.path.join(INLINE, "pairs.json")))
    check("  and they are the same tables either way",
          cov_i["per_layer"]["0"]["pairs_alpha"] == side["per_layer"]["0"]["pairs_alpha"])

    OFF = run_stats("stats-off", "--pairs", "off")
    cov_o = json.load(open(os.path.join(OFF, "coverage.json")))
    check("--pairs off computes none, and still writes every histogram",
          not os.path.exists(os.path.join(OFF, "pairs.json"))
          and not any(k.startswith("pairs") for k in cov_o["per_layer"]["0"])
          and all(f"top1_{c}" in cov_o["per_layer"]["0"] for c in CATS))

    # --- 5. the exporter ------------------------------------------------------
    # The export is 40 x 384 whatever the trace was, so the toy above cannot be handed to it
    # directly. This is the same arithmetic on a file of the right shape, built out of the
    # same functions: 40 layers, 2 topics, 200 tokens of top-6 routing over 24 experts.
    def toy_stats(root: str, extras: bool = True):
        os.makedirs(root, exist_ok=True)
        rng = np.random.default_rng(7)
        per_layer, pairs_layer = {}, {}
        for L in range(B.N_LAYERS):
            row, prow = {}, {}
            mixed = np.zeros(B.N_EXPERTS, dtype=np.int64)
            for t in CATS:
                idx = np.stack([rng.permutation(24)[:B.TOPK] for _ in range(200)]).astype(np.int64)
                w = rng.uniform(0.05, 0.5, size=(200, B.TOPK)).astype(np.float32)
                sal = w * rng.uniform(0.5, 3.0, size=(200, B.TOPK)).astype(np.float32)
                counts = np.bincount(idx.reshape(-1), minlength=B.N_EXPERTS)
                row[f"counts_{t}"] = counts.tolist()
                row[f"saliency_{t}"] = S.saliency_hist(idx, sal).tolist()
                mixed += counts
                if extras:
                    row[f"top1_{t}"] = S.top1_hist(idx, w).tolist()
                    prow[f"pairs_{t}"] = S.pair_table(idx, 32, 5)
            row["counts"] = mixed.tolist()
            per_layer[str(L)] = row
            pairs_layer[str(L)] = prow
        json.dump({"per_layer": per_layer, "global": []},
                  open(os.path.join(root, "coverage.json"), "w"))
        if extras:
            json.dump({"schema": 1, "min_count": 5, "top": 32, "topk": B.TOPK,
                       "per_layer": pairs_layer}, open(os.path.join(root, "pairs.json"), "w"))
        json.dump({"runs": []}, open(os.path.join(root, "gates.json"), "w"))
        return root

    TOY = toy_stats(os.path.join(TMP, "toy"))
    out = os.path.join(TMP, "atlas")
    r = AX.export(stats=os.path.join(TOY, "coverage.json"), gates=os.path.join(TOY, "gates.json"),
                  out=out, profiles=[])
    R = r["payload"]["routing"]
    N, E = B.N_LAYERS, B.N_EXPERTS

    check("the exporter fills top1_share from top1_<topic>",
          "top1_share" in R and len(R["top1_share"]) == N
          and all(len(row) == E for row in R["top1_share"]))
    off = [(L, s) for L, row in enumerate(R.get("top1_share", []))
           for s in [sum(row)] if abs(s - 1.0) > 2e-3]
    check("  and every row of it sums to 1", not off, str(off[:3]))
    check("  no negative share", not any(v < 0 for row in R["top1_share"] for v in row))
    check("  the layer dynamics gain top1_gini",
          all("top1_gini" in d for d in R["dynamics"]["all"])
          and all(0.0 <= d["top1_gini"] <= 1.0 for d in R["dynamics"]["all"]))

    co = R.get("coroute")
    check("the exporter fills coroute, one entry per layer", isinstance(co, list) and len(co) == N)
    shapes = all(len(c["count"]) <= 16 and len(c["lift"]) <= 16 and c["min_count"] == 5
                 and all(len(q) == 4 for q in c["count"] + c["lift"]) for c in co)
    check("  in the page's shape: 16 by count, 16 by lift, a and b first", shapes)
    # the page reads q[3] of a `count` row as a fraction of tokens and q[3] of a `lift` row as
    # a multiple, and recomputes neither
    toy_pl = json.load(open(os.path.join(TOY, "coverage.json")))["per_layer"]
    toks = [sum(sum(toy_pl[str(L)][f"counts_{t}"]) for t in CATS) / B.TOPK for L in range(N)]
    bad = []
    for L, c in enumerate(co):
        for a, b, n, frac in c["count"]:
            if abs(frac - round(n / toks[L], 5)) > 1e-5:
                bad.append((L, a, b, frac))
    check("  a count row's fourth number is n / tokens", not bad, str(bad[:2]))
    check("  count rows are ordered by n, descending",
          all([q[2] for q in c["count"]] == sorted((q[2] for q in c["count"]), reverse=True)
              for c in co))
    check("  the lift half never ranks a pair seen fewer than min_count times",
          all(q[2] >= c["min_count"] for c in co for q in c["lift"]))
    check("  and the export says where both came from",
          "top1" in r["payload"]["meta"]["source"] and "coroute" in r["payload"]["meta"]["source"])

    # --- 6. a trace that kept neither still exports exactly as before --------
    BARE = toy_stats(os.path.join(TMP, "toy-bare"), extras=False)
    out2 = os.path.join(TMP, "atlas-bare")
    r2 = AX.export(stats=os.path.join(BARE, "coverage.json"),
                   gates=os.path.join(BARE, "gates.json"), out=out2, profiles=[])
    R2 = r2["payload"]["routing"]
    check("a stats file with neither summary omits both fields rather than faking them",
          "top1_share" not in R2 and "coroute" not in R2,
          ", ".join(k for k in ("top1_share", "coroute") if k in R2))
    check("  and leaves top1_gini out of the dynamics too",
          not any("top1_gini" in d for d in R2["dynamics"]["all"]))
    check("  while everything that was there before still is",
          all(k in R2 for k in ("reap", "reap_domains", "route_share", "contribution",
                                "dynamics", "prune_sets", "layers", "experts", "domains")))
    # The sibling is not a freshness input of its own: expert_stats writes it in the same run
    # as coverage.json, so the stats file's mtime already moves with it.
    # --- 7. the merge: a topic can be the SUM of two traces ------------------
    # The corpus behind one shipped topic was collected in two passes, each traced and reduced
    # on its own, and the shipped histogram is the two added together. A merge that copies one
    # of them across instead of summing ships a topic measured on a fraction of its corpus and
    # looks perfectly well-formed while doing it -- every key present, every array the right
    # shape. This is that shape in miniature: two trace directories for one topic, one shipped
    # histogram that is their sum.
    def shard(root: str, topic: str, seed: int, n_tok: int):
        """A trace directory whose every token is tagged `topic`."""
        os.makedirs(os.path.join(root, "trace"), exist_ok=True)
        rng = np.random.default_rng(seed)
        for L in range(N_LAYERS):
            idx = np.stack([rng.permutation(N_USED)[:TOPK] for _ in range(n_tok)]).astype(np.int16)
            w = rng.uniform(0.05, 0.6, size=(n_tok, TOPK))
            np.savez_compressed(os.path.join(root, "trace", f"layer{L}.npz"),
                                indices=idx, weights=w.astype(np.float16),
                                contrib_norms=(w * rng.uniform(0.5, 3.0, size=w.shape)).astype(np.float32),
                                category=np.array([topic] * n_tok),
                                token=np.arange(n_tok, dtype=np.int32),
                                scores=np.zeros(0, np.float16))
        json.dump({"n_tokens": n_tok * N_LAYERS, "n_seqs": 1},
                  open(os.path.join(root, "meta.json"), "w"))
        return root

    MERGE = os.path.join(TMP, "merge")
    runs = []
    for name, seed, n in (("pass_one", 11, 14), ("pass_two", 12, 40)):
        shard(os.path.join(MERGE, "trace", name), "gamma", seed, n)
        runs.append(run_stats(os.path.join("merge", "red", name),
                              trace=os.path.join(MERGE, "trace", name)))
    red = [json.load(open(os.path.join(r, "coverage.json"))) for r in runs]
    ship_pl = {}
    for L in red[0]["per_layer"]:
        ship_pl[L] = dict(red[0]["per_layer"][L])
        for fam in ("counts_gamma", "saliency_gamma", "top1_gamma"):
            ship_pl[L][fam] = [a + b for a, b in zip(red[0]["per_layer"][L][fam],
                                                     red[1]["per_layer"][L][fam])]
    SHIPDIR = os.path.join(MERGE, "shipped")
    os.makedirs(SHIPDIR, exist_ok=True)
    SHIP = os.path.join(SHIPDIR, "coverage.json")
    json.dump({"per_layer": ship_pl, "global": []}, open(SHIP, "w"))
    one, two = (sum(red[i]["per_layer"]["0"]["counts_gamma"]) for i in (0, 1))
    check("the toy shipped topic really is the sum of two traces, not either of them",
          sum(ship_pl["0"]["counts_gamma"]) == one + two and one != two,
          f"{one} + {two}")

    # both shards on hand: the merge must find them and sum them
    ok_merge = S.merge_reductions(SHIP, runs, log=lambda *_: None)
    check("merge_reductions rebuilds the union from both traces",
          not ok_merge["mismatch"] and sorted(ok_merge["chosen"]["gamma"]) ==
          sorted(os.path.basename(r) for r in runs),
          f"mismatch={ok_merge['mismatch']} chosen={ok_merge['chosen']}")
    check("  and the rebuilt counts_gamma is the shipped array, element for element",
          ok_merge["cov"]["per_layer"]["0"]["counts_gamma"] == ship_pl["0"]["counts_gamma"])
    check("  saliency too",
          S.rows_equal({"0": ok_merge["cov"]["per_layer"]["0"]["saliency_gamma"]},
                       {"0": ship_pl["0"]["saliency_gamma"]}))
    check("  and top1_gamma, which is summed the same way",
          ok_merge["cov"]["per_layer"]["0"]["top1_gamma"] == ship_pl["0"]["top1_gamma"])
    check("  the merged pair table is one table, with the union's own lift",
          all(len(q) == 4 and q[0] < q[1]
              for q in ok_merge["pairs"]["per_layer"]["0"]["pairs_gamma"]))

    # one shard only: the guard must catch it, which key presence alone would not
    half = S.merge_reductions(SHIP, runs[:1], log=lambda *_: None)
    check("one trace of a two-trace topic is REFUSED, not shipped",
          "counts_gamma" in half["mismatch"] and "saliency_gamma" in half["mismatch"],
          str(half["mismatch"]))
    check("  even though every key is present and the right shape -- which is why the guard "
          "compares arrays and not key names",
          all(len(half["cov"]["per_layer"]["0"][f"{f}_gamma"]) == S.N_EXP
              for f in ("counts", "saliency", "top1")))
    check("  and the numbers are the giveaway",
          sum(half["cov"]["per_layer"]["0"]["counts_gamma"]) == one)

    # the whole tool, through its CLI: refusal writes coverage.new.json and touches nothing
    before = open(SHIP, "rb").read()
    rc = subprocess.run([sys.executable, os.path.join(HERE, "expert_stats.py"),
                         "--merge", SHIP, "--runs", runs[0]], capture_output=True, text=True)
    check("--merge exits 5 on a refusal", rc.returncode == 5, rc.stdout[-300:])
    check("  leaves the shipped file byte-for-byte alone", open(SHIP, "rb").read() == before)
    check("  and writes coverage.new.json beside it",
          os.path.exists(os.path.join(SHIPDIR, "coverage.new.json"))
          and not os.path.exists(SHIP + ".before-extras"))
    rc = subprocess.run([sys.executable, os.path.join(HERE, "expert_stats.py"),
                         "--merge", SHIP, "--runs", *runs], capture_output=True, text=True)
    check("--merge exits 0 when both traces are there", rc.returncode == 0, rc.stdout[-300:])
    check("  replaces the file and keeps the old one as .before-extras",
          os.path.exists(SHIP + ".before-extras")
          and open(SHIP + ".before-extras", "rb").read() == before)
    now = json.load(open(SHIP))["per_layer"]["0"]
    check("  the installed file carries the union and the new summaries",
          now["counts_gamma"] == ship_pl["0"]["counts_gamma"] and "top1_gamma" in now)
    check("  and a pairs.json beside it", os.path.exists(os.path.join(SHIPDIR, "pairs.json")))

    check("  and the freshness inputs are the three they always were",
          [os.path.basename(x) for x in AX.inputs(os.path.join(TOY, "coverage.json"),
                                                  os.path.join(TOY, "gates.json"))]
          == ["coverage.json", "gates.json", "tune.py"])
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
