"""Checks on the tail metric -- the point is that the number computed from a
histogram is the number a per-token trace would have given.

Run: python3 tools/test_tail_metric.py

Four things are held here.

  1. A toy `coverage.json` whose answer can be worked out by hand, so the
     definition is pinned rather than merely reproduced.
  2. The exact metric, computed from per-token `indices` arrays, against the
     histogram estimate of it -- on a toy trace where they must agree to the bit
     for `nonres`, and on the real 40-layer trace in this checkout, where the
     `all6` bias quoted in `budget.tail_curves` is measured rather than claimed.
  3. Shapes and edges: curve length, monotonicity, caching, a topic absent from
     a layer, an empty selection.
  4. The correlation runner over the shipped files: that it runs, that it picks
     the same gate record `tools/tune.py` would, and that its rows carry the
     record they came from.

No torch, no GPU, no checkpoint. numpy only for the parts that read a real
`.npz`, and those skip where it is absent.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402
import tail_metric as TM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want, tol=0.0):
    if isinstance(want, float) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(name)


def ok(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


TMP = tempfile.mkdtemp(prefix="tail-metric-")


# =============================================================================
# 1. a toy catalogue whose answer is known by hand
# =============================================================================
# One topic, `flat`, that in every layer spreads its picks evenly over the first
# twelve experts of the layer and nowhere else. Ranked by its own histogram, the
# admission order takes those twelve first (ties broken by argsort, which for an
# all-equal prefix is ascending id), so:
#
#     keep n experts a layer, n <= 12  ->  resident pick fraction n/12
#     nonres = 6 * (1 - n/12)          ->  at n = 6 that is exactly 3.0
#     all6   = (n/12)^6                ->  at n = 6 that is 1/64 = 0.015625
#     worst layer = n/12               (every layer is identical)
#
# and at n >= 12 everything is resident: nonres 0, all6 1.
FLAT = [100.0] * 12 + [0.0] * (B.N_EXPERTS - 12)
# A second topic that lives on twelve DIFFERENT experts, so a selection of both
# under `maxmin` hands out slots one at a time and each topic has half the
# budget spent on it: at n = 12 each holds 6 of its 12 -> nonres 3.0 again.
OTHER = [0.0] * 12 + [100.0] * 12 + [0.0] * (B.N_EXPERTS - 24)
# A topic that is silent in layer 0 and flat elsewhere -- the zero-mass layer.
toy = {"per_layer": {}}
for L in range(B.N_LAYERS):
    toy["per_layer"][str(L)] = {
        "counts_flat": FLAT,
        "counts_other": OTHER,
        "counts_quiet": ([0.0] * B.N_EXPERTS) if L == 0 else FLAT,
        # A saliency family that ranks the SAME experts, so a keep-set ranked on
        # it is the counts keep-set and the two measurements can be compared.
        "saliency_flat": FLAT,
        "saliency_other": OTHER,
        "saliency_quiet": ([0.0] * B.N_EXPERTS) if L == 0 else FLAT,
    }
TOY = os.path.join(TMP, "toy.json")
json.dump(toy, open(TOY, "w"))

ix = B.TopicIndex(TOY)
check("toy topics", sorted(ix.topics), ["flat", "other", "quiet"])
check("toy picks loaded", sorted(ix.picks), ["flat", "other", "quiet"])

t = ix.tail(("flat",), 6 / B.N_EXPERTS, rank="maxmin", only=("flat",))["flat"]
check("flat: cov_picks at 6 of 12", t["cov_picks"], 0.5, 1e-12)
check("flat: nonres at 6 of 12", t["nonres"], 3.0, 1e-12)
check("flat: all6 at 6 of 12", t["all6"], 1 / 64, 1e-12)
check("flat: worst layer at 6 of 12", t["worst_layer"], 0.5, 1e-12)

t = ix.tail(("flat",), 12 / B.N_EXPERTS, rank="maxmin", only=("flat",))["flat"]
check("flat: nonres once all twelve are kept", t["nonres"], 0.0, 1e-12)
check("flat: all6 once all twelve are kept", t["all6"], 1.0, 1e-12)

# Two disjoint topics, one budget: maxmin alternates, so twelve slots is six each.
t = ix.tail(("flat", "other"), 12 / B.N_EXPERTS, rank="maxmin",
            only=("flat", "other"))
check("maxmin splits: flat nonres", t["flat"]["nonres"], 3.0, 1e-12)
check("maxmin splits: other nonres", t["other"]["nonres"], 3.0, 1e-12)

# A layer the topic never routes in must not be scored -- neither as a miss
# (there is nothing to miss) nor as a free hit (that would flatter the rest).
t = ix.tail(("quiet",), 6 / B.N_EXPERTS, rank="maxmin", only=("quiet",))["quiet"]
check("a silent layer does not vote: nonres", t["nonres"], 3.0, 1e-12)
check("a silent layer does not vote: all6", t["all6"], 1 / 64, 1e-12)

# --- the identity the whole thing rests on ----------------------------------
# `counts_<topic>` IS the histogram of picks and every token contributes exactly
# TOPK of them per layer, so the mean non-resident count per token-layer is the
# missed mass times TOPK, with no assumption at all about the joint.
for n_keep in (2, 6, 11, 20):
    keep = n_keep / B.N_EXPERTS
    d = ix.tail(("flat", "other"), keep, rank="maxmin", only=("flat", "other"))
    for topic, row in d.items():
        ok(f"nonres is a real miss at n={n_keep} [{topic}]", row["nonres"] > 0.0)
        check(f"nonres = 6(1-cov_picks) [{topic} @ n={n_keep}]",
              row["nonres"], B.TOPK * (1 - row["cov_picks"]), 1e-12)


# =============================================================================
# 2. shapes, monotonicity, caching
# =============================================================================
curves = ix.tail_curves(("flat",), only=("flat",), rank="maxmin")["flat"]
check("curve keys", sorted(curves), ["all6", "cov_picks", "nonres", "worst_layer"])
for k, v in curves.items():
    check(f"curve length [{k}]", len(v), B.N_EXPERTS + 1)
check("curve starts at zero experts kept", curves["cov_picks"][0], 0.0, 1e-12)
check("curve reaches every pick", curves["cov_picks"][B.N_EXPERTS], 1.0, 1e-12)
ok("cov_picks is monotone in the keep fraction",
   all(curves["cov_picks"][n] <= curves["cov_picks"][n + 1] + 1e-12
       for n in range(B.N_EXPERTS)))
ok("nonres falls monotonically",
   all(curves["nonres"][n] >= curves["nonres"][n + 1] - 1e-12 for n in range(B.N_EXPERTS)))
ok("all6 is bounded by cov_picks",
   all(curves["all6"][n] <= curves["cov_picks"][n] + 1e-12 for n in range(B.N_EXPERTS + 1)))
ok("worst layer never exceeds the mean",
   all(curves["worst_layer"][n] <= curves["cov_picks"][n] + 1e-12
       for n in range(B.N_EXPERTS + 1)))

before = len(ix._tail)
ix.tail_curves(("flat",), only=("flat",), rank="maxmin")
check("cached on the same key as curves", len(ix._tail), before)
ok("a different rank is a different entry",
   ix.tail_curves(("flat",), only=("flat",), rank="sum") is not curves)
check("an empty selection yields nothing", ix.tail((), 0.36, rank="maxmin"), {})


# =============================================================================
# 3. the estimate against the exact metric, on real per-token arrays
# =============================================================================
TRACE = os.path.join(ROOT, "results", "trace-full-20260910")
try:
    import numpy as np
except Exception as e:  # noqa: BLE001
    np = None
    print(f"skip the per-token checks (no numpy: {type(e).__name__})")

if np is not None and os.path.isdir(os.path.join(TRACE, "trace")):
    import glob
    import re as _re

    # Build the histograms this trace would have produced, rank a keep-set from
    # them exactly as the engine would, then ask both instruments the same
    # question about the same keep-set.
    lay = {}
    for p in sorted(glob.glob(os.path.join(TRACE, "trace", "layer*.npz")),
                    key=lambda p: int(_re.search(r"layer(\d+)", p).group(1))):
        z = np.load(p)
        lay[int(_re.search(r"layer(\d+)", p).group(1))] = (z["indices"].astype(np.int64),
                                                           z["category"])
    cats = sorted(set(lay[0][1].tolist()))
    check("the checked-in trace is 40 layers", len(lay), 40)
    check("its picks are [tokens, 6]", lay[0][0].shape[1], B.TOPK)

    per_layer = {}
    for L in sorted(lay):
        idx, cat = lay[L]
        row = {}
        for c in cats:
            row["counts_" + c] = np.bincount(idx[cat == c].reshape(-1),
                                             minlength=B.N_EXPERTS).astype(float).tolist()
        per_layer[str(L)] = row
    RAW = os.path.join(TMP, "raw.json")
    json.dump({"per_layer": per_layer}, open(RAW, "w"))
    rix = B.TopicIndex(RAW)
    check("topics recovered from the trace", sorted(rix.topics), cats)
    check("token count recovered from the histogram totals",
          rix.tokens[cats[0]], int((lay[0][1] == cats[0]).sum()))

    worst_all6 = 0.0
    pairs = []
    for sel in ((cats[0],), tuple(cats)):
        for keep in (0.20, 0.36, 0.40):
            order = rix.curves(sel, only=tuple(sorted(sel)), rank="maxmin")[1]
            n = B.keep_n(keep)
            keep_sets = {L: order[L][:n] for L in order}
            # `only=None`: the topic that was NOT selected is measured too, and
            # it is the interesting one -- a keep-set spent on `coding` is where
            # `general` shows what a displaced token costs.
            est = rix.tail(sel, keep, rank="maxmin")
            for c in cats:
                exact = TM.exact_from_trace(TRACE, keep_sets, category=c)
                check(f"nonres is EXACT [{c} | sel={'+'.join(sel)} @ {keep}]",
                      est[c]["nonres"], exact["nonres"], 5e-4)
                # all6 assumes the six picks are independent draws. They are
                # not -- experts co-fire -- so the estimate is LOW, never high,
                # and that one-sidedness is the property worth pinning.
                ok(f"all6 estimate is not above the truth [{c} | {keep}]",
                   est[c]["all6"] <= exact["all6"] + 1e-9,
                   f"est {est[c]['all6']:.4f} vs exact {exact['all6']:.4f}")
                worst_all6 = max(worst_all6, exact["all6"] / max(est[c]["all6"], 1e-12))
                pairs.append((est[c]["all6"], exact["all6"]))
                # And the whole-token reading is useless at forty layers, which
                # is why the shipped metric counts token-LAYERS.
                ok(f"clean whole tokens are ~zero at forty layers [{c} | {keep}]",
                   exact["clean_token"] < 1e-3, f"{exact['clean_token']:.6f}")
    ok("the all6 bias stays inside the 75 % quoted in budget.tail_curves",
       worst_all6 < 1.8, f"worst exact/estimate ratio {worst_all6:.2f}")
    # Biased low is survivable for a ranking statistic; biased UNEVENLY is not,
    # because then a keep-set that reads better could be worse. Hold the rank
    # correlation with the truth, which is the property the docstring claims.
    rho = TM.spearman([p[0] for p in pairs], [p[1] for p in pairs])
    ok("all6 ranks keep-sets the way the exact metric does",
       rho is not None and rho >= 0.95, f"spearman {rho:.3f} over {len(pairs)} points")
else:
    print("skip the per-token checks (no results/trace-full-20260910/trace)")


# =============================================================================
# 4. the runner over the shipped files
# =============================================================================
STATS = os.path.join(ROOT, "results", "keepsets", "topics", "coverage.json")
GATES = os.path.join(ROOT, "results", "keepsets", "gates.json")
if os.path.exists(STATS) and os.path.exists(GATES):
    rows = TM.profile_rows(STATS, GATES)
    ok("every shipped profile with a gate record produced a row", len(rows) >= 10,
       f"{len(rows)} rows")
    for r in rows:
        ok(f"{r['record']}: cites the record it came from",
           r["gate_file"].endswith("GATE.md") and r["record"] in r["gate_file"])
        ok(f"{r['record']}: keep is the gated one", 0.0 < r["keep"] <= 1.0, str(r["keep"]))
        ok(f"{r['record']}: coverage in picks is below coverage in {r['source']}",
           r["cov_picks_worst"] < r["cov_sal_worst"],
           f"{r['cov_picks_worst']:.4f} vs {r['cov_sal_worst']:.4f}")
        ok(f"{r['record']}: nonres = 6(1 - cov_picks) holds on the shipped file",
           abs(r["nonres_worst"] - B.TOPK * (1 - r["cov_picks_worst"])) < 1e-9)

    # The record-selection rule is a second implementation of tune.py's. Hold
    # the two to the same record for every profile -- importing tune here is
    # fine, this is a test and it has a terminal's worth of patience for curses.
    try:
        import tune as T  # noqa: E402
        for name, _b, topics, record, _rank in TM.read_profiles():
            mine = TM.gate_for(record, topics)
            theirs = T.gate_for(record, topics)
            check(f"same gate record as tune.py [{record}]",
                  None if mine is None else mine["run"],
                  None if theirs is None else theirs["run"])
            if mine and theirs:
                check(f"same strict count [{record}]", mine["strict"], theirs["strict"])
    except Exception as e:  # noqa: BLE001
        print(f"skip the tune.py cross-check ({type(e).__name__}: {e})")

    cor = TM.correlations(rows)
    for key, _label, _red in TM.PREDICTORS:
        for axis, _lbl in TM.GATE_AXES:
            rho = cor[key][axis]["rho"]
            ok(f"rho is a correlation [{key} vs {axis}]",
               rho is None or -1.0 - 1e-9 <= rho <= 1.0 + 1e-9, str(rho))
    check("spearman of a series with itself",
          TM.spearman([3, 1, 2, 5], [3, 1, 2, 5]), 1.0, 1e-12)
    check("spearman of a series with its reverse",
          TM.spearman([3, 1, 2, 5], [-3, -1, -2, -5]), -1.0, 1e-12)
    check("spearman survives ties", TM.spearman([1, 1, 2, 2], [1, 1, 2, 2]), 1.0, 1e-12)
    check("spearman refuses two points", TM.spearman([1, 2], [1, 2]), None)
    p = TM.permutation_p([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], trials=2000)
    ok("a perfect rank correlation at n=8 is significant", p is not None and p < 0.05, str(p))

    swaps = TM.swap_rows(STATS)
    check("all four A/B swaps resolve", len(swaps), len(TM.SWAPS))
    for s in swaps:
        ok(f"{s['record']}: the swap cites its configuration", bool(s["cite"]))
        ok(f"{s['record']}: both sides measured", "a" in s and "b" in s)

    # And the CLI itself, end to end, the way a reader would run it.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tail_metric.py")],
                       capture_output=True, text=True)
    ok("the CLI exits 0", r.returncode == 0, r.stderr.strip()[-300:])
    ok("the CLI prints the per-profile table", "# Per profile" in r.stdout)
    ok("the CLI prints the correlations", "Spearman against the gate" in r.stdout)
    ok("the CLI prints the swaps", "a different keep-set" in r.stdout)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tail_metric.py"),
                        "--profile", "frontend"], capture_output=True, text=True)
    ok("--profile narrows to one", r.returncode == 0 and r.stdout.count("| Frontend") == 1,
       r.stderr.strip()[-300:])
    out = os.path.join(TMP, "out.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tail_metric.py"),
                        "--json", out], capture_output=True, text=True)
    ok("--json writes a readable file", r.returncode == 0 and os.path.exists(out))
    if os.path.exists(out):
        d = json.load(open(out))
        ok("the json carries the profiles and the swaps",
           len(d.get("profiles", [])) >= 10 and len(d.get("swaps", [])) == len(TM.SWAPS))
        ok("the shipped json is current",
           not os.path.exists(os.path.join(ROOT, "results", "keepsets", "tail_metric.json"))
           or json.load(open(os.path.join(ROOT, "results", "keepsets",
                                          "tail_metric.json")))["profiles"] == d["profiles"],
           "re-run: python3 tools/tail_metric.py --json results/keepsets/tail_metric.json")
    # A missing file is an error with a name in it, not a traceback.
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tail_metric.py"),
                        "--stats", os.path.join(TMP, "nope.json")],
                       capture_output=True, text=True)
    ok("a missing stats file exits non-zero and says which", r.returncode != 0
       and "nope.json" in r.stderr)
else:
    print("skip the shipped-file checks (no results/keepsets/topics/coverage.json)")


# =============================================================================
# 5. the record-selection rule, on a synthetic GATE.md
# =============================================================================
# tools/verify_think_controls.sh appends gate runs taken with DSV41_THINK_BUDGET
# (and DSV41_THINK_REPEAT_BREAK) to results/keepsets/{frontend,backend}/GATE.md.
# They measure a decode path the box does not serve with, so the newest of them
# is never the profile's record -- the newest run taken with the controls OFF is.
# The predicate is tools/gate_records.py, imported by this module and by
# tools/tune.py alike; the same case is in tools/test_tune_profiles.py.
def section(when, controls, strict=5, finished=None, runs=10):
    """One gate card in the shape tools/gate_profile.py writes it."""
    rows = [f"# Generation gate \u2014 {when}", "", "| | |", "|---|---|",
            "| profile | Synthetic |", "| topics | html, css |",
            f"| prompts | {runs} runs over {runs} prompts |", "| thinking | on |",
            "| keep-set | PRUNE_KEEP=0.36, DSV41_PRUNE_RANK=maxmin, "
            "DSV41_PRUNE_SOURCE=saliency |"]
    if controls is not None:
        rows.append(f"| reasoning-span controls | {controls} |")
    rows += ["", "| prompt | thinking | finish | why |", "|---|---|---|---|",
             "| `css-card` | on | stop | ok |", "",
             f"**Verdict: FAIL** \u2014 {runs - strict} of {runs} runs failed: `css-card` (on) no",
             "",
             f"{finished if finished is not None else strict} of {runs} finished a correct "
             "answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, "
             f"guard 0, corrupt 0, content {runs - strict}.", ""]
    return "\n".join(rows)


synth = os.path.join(TMP, "results", "keepsets", "synthetic")
os.makedirs(synth, exist_ok=True)
open(os.path.join(synth, "GATE.md"), "w").write("\n".join([
    section("2026-09-14 03:36",
            "off (neither DSV41_THINK_BUDGET nor DSV41_THINK_REPEAT_BREAK was set)",
            strict=3, finished=10),
    section("2026-09-14 20:04", "DSV41_THINK_BUDGET=2000", strict=9),
    section("2026-09-14 22:41", "DSV41_THINK_BUDGET=2000, DSV41_THINK_REPEAT_BREAK=12", strict=8),
]))
seen = TM.read_gate(os.path.join(synth, "GATE.md"))
check("the control runs are read, and named", [x["controls"] for x in seen],
      [[], ["DSV41_THINK_BUDGET=2000"],
       ["DSV41_THINK_BUDGET=2000", "DSV41_THINK_REPEAT_BREAK=12"]])
g = TM.gate_for("synthetic", ["html", "css"], root=TMP)
check("the record is the newest DEFAULT run, not the newest run", g["run"], "2026-09-14 03:36")
check("  with that run's counts", (g["strict"], g["runs"], g["finished"]), (3, 10, 10))
try:
    import tune as T2  # noqa: E402
    check("  and tune.py picks the same one",
          T2.gate_for("synthetic", ["html", "css"], {}, root=TMP)["run"], g["run"])
except Exception as e:  # noqa: BLE001
    print(f"skip the tune.py cross-check on the synthetic file ({type(e).__name__}: {e})")

print()
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails[:8])}{' ...' if len(fails) > 8 else ''}")
    sys.exit(1)
print("all checks passed")
