#!/usr/bin/env python3
"""The ExFold routing arithmetic, on a toy: tools/test_exfold.py

Torch-free by default, so it runs on the CPU runner with the rest of the suite. If torch happens
to be importable it also runs `engine/exfold.py`'s tensor implementation over the same toy and
holds the two to each other, elementwise; CUDA is never required and never touched.

What is under test is the part that decides output quality: which of the router's six routes
survive, which retained expert each excluded one is folded onto, how much of its weight goes
there, and where the rest lands. The kernel timing is a separate question and a separate script.

Equation numbers are the paper's (arXiv 2608.24938):
  (10) B = TopK_e (w_e * h_e, K')            (12) pi(s) = argmin_t l_{s->t}
  (13) w~_t = w_t + sum w_s S[s,t]           (26) c_s = clip(1 - min_t l, 0, 1), remainder pro rata
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.prefill_topk import (UNOBSERVED_LOSS, fold_reference,  # noqa: E402
                                 reduce_reference)

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def allclose(a, b, tol=1e-9):
    return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))


# ============================================================ the toy
# Six experts, ids chosen so that "expert id" and "column" can never be confused. `h` is
# deliberately NOT flat: expert 30's output norm is four times expert 11's, which is the regime
# the paper's Figure 3(d) describes (>3x spread within a layer) and the reason Eq. 10 ranks by
# w*h rather than by w.
E = [10, 11, 12, 30, 31, 32]
W = [0.50, 0.35, 0.25, 0.20, 0.12, 0.08]          # already normalised x route_scale
H = {10: 1.0, 11: 1.0, 12: 1.0, 30: 4.0, 31: 4.0, 32: 4.0}
FLAT = {e: 1.0 for e in E}

# A loss table with one very good pair, one mediocre pair and one unobserved pair, so each branch
# of the safeguard is reached by some token.
LOSS = {s: {t: 0.5 for t in E} for s in E}
SCALAR = {s: {t: 1.0 for t in E} for s in E}
for e in E:
    LOSS[e][e], SCALAR[e][e] = 0.0, 1.0
LOSS[31][10] = 0.10; SCALAR[31][10] = 0.90        # good target
LOSS[31][11] = 0.80; SCALAR[31][11] = 0.30
LOSS[32][10] = UNOBSERVED_LOSS                    # never co-routed with 10 in calibration
LOSS[32][11] = 0.40; SCALAR[32][11] = 1.70        # the retained expert is SMALLER; s > 1


# ============================================================ off is identity
for k in (6, 7):
    idx, w = fold_reference(W, E, k, H, SCALAR, LOSS)
    check(f"k'={k} >= k leaves the routing untouched", idx == E and w == W)
idx, w = reduce_reference(W, E, 6, 1.5)
check("the none arm at k'=k is identity too", idx == E and w == W)


# ============================================================ (10) the selection rule
idx, _ = fold_reference(W, E, 3, FLAT, SCALAR, LOSS)
check("with a flat h the ranking is the router's own order", idx == [10, 11, 12])
idx, _ = fold_reference(W, E, 3, H, SCALAR, LOSS)
# w*h = .50 .35 .25 | .80 .48 .32 -> 30 (.80), 31 (.48), 10 (.50)... ranked: .80 .50 .48
check("w * h reorders the routes against the router's weights alone", idx == [10, 30, 31])
check("the kept ids come back in the router's column order", idx == sorted(idx, key=E.index))


# ============================================================ (13)+(26) the fold itself
# Keep 3 by w*h: {10, 30, 31}. Excluded: 11 (w .35), 12 (w .25), 32 (w .08).
kept, folded = fold_reference(W, E, 3, H, SCALAR, LOSS)
base = [W[E.index(e)] for e in kept]
check("folding never drops a retained route", kept == [10, 30, 31])
check("folding moves every retained weight off its original value",
      all(not close(a, b) for a, b in zip(base, folded)))


def hand_fold(keep_set, excluded, k_prime):
    """The same arithmetic written out longhand, from the equations, with no shared code."""
    kept_w0 = [W[E.index(e)] for e in keep_set]
    out = list(kept_w0)
    mass = sum(kept_w0)
    for s in excluded:
        w_s = W[E.index(s)]
        t = min(keep_set, key=lambda t: (LOSS[s][t], keep_set.index(t)))
        c = min(1.0, max(0.0, 1.0 - LOSS[s][t]))
        out[keep_set.index(t)] += w_s * c * SCALAR[s][t]
        for j, kw in enumerate(kept_w0):
            out[j] += w_s * (1.0 - c) * kw / mass
    return out


check("the fold matches the equations written out by hand",
      allclose(folded, hand_fold([10, 30, 31], [11, 12, 32], 3)))

# (12): the minimum-loss target, not the nearest weight and not the first column
kept2, folded2 = fold_reference([0.50, 0.30, 0.20], [10, 11, 31], 2,
                                {10: 1.0, 11: 1.0, 31: 1.0}, SCALAR, LOSS)
check("an excluded route picks its MINIMUM-LOSS target", kept2 == [10, 11])
# 31 -> 10 has loss 0.10 (c = 0.90, scalar 0.90); 31 -> 11 has loss 0.80. All of the fold must
# land on column 0 even though column 1 is the one the router ranked next.
check("... so the folded mass lands on that target and not on the next-ranked route",
      close(folded2[0], 0.50 + 0.20 * 0.90 * 0.90 + 0.20 * 0.10 * (0.50 / 0.80))
      and close(folded2[1], 0.30 + 0.20 * 0.10 * (0.30 / 0.80)))


# ============================================================ (26) the safeguard
# A pair the calibration never observed: loss 1e30 -> c = 0 -> nothing is folded, and the whole
# weight falls back to the retained routes in proportion. This is the property that makes a stale
# or partial table safe.
kept3, folded3 = fold_reference([0.60, 0.30, 0.10], [10, 11, 32], 2, FLAT, SCALAR, LOSS)
# 32's targets are 10 (unobserved, 1e30) and 11 (loss 0.40, scalar 1.70). Eq. 12 must pick 11,
# Eq. 26 then sends 1-0.40 = 0.60 of 32's weight there through the scalar and leaves 0.40 of it
# to be split over {10, 11} in proportion to 0.60 : 0.30.
rest3 = 0.10 * 0.40
check("an unobserved target is never chosen while an observed one exists", kept3 == [10, 11])
check("... and the confidence splits the excluded weight between fold and fallback",
      close(folded3[0], 0.60 + rest3 * (0.60 / 0.90))
      and close(folded3[1], 0.30 + 0.10 * 0.60 * 1.70 + rest3 * (0.30 / 0.90)))

LOSS_ALL_BLIND = {s: {t: UNOBSERVED_LOSS for t in E} for s in E}
kept4, folded4 = fold_reference([0.60, 0.30, 0.10], [10, 11, 12], 2, FLAT, SCALAR,
                                LOSS_ALL_BLIND)
check("a table with NO observed pair degrades to a pro-rata reweighting, not to a drop",
      close(sum(folded4), 1.0) and close(folded4[0] / folded4[1], 0.60 / 0.30))

LOSS_PERFECT = {s: {t: 0.0 for t in E} for s in E}
ONE = {s: {t: 1.0 for t in E} for s in E}
kept5, folded5 = fold_reference([0.60, 0.30, 0.10], [10, 11, 12], 2, FLAT, ONE, LOSS_PERFECT)
check("a perfectly reconstructed pair with a scalar of 1 conserves the routed mass exactly",
      close(sum(folded5), 1.0))
check("... and puts all of the excluded weight on its target",
      close(folded5[0], 0.60 + 0.10) and close(folded5[1], 0.30))

# Mass is NOT conserved in general, and that is the point: S[s,t] is a magnitude correction, so a
# retained expert that is weaker than the one it replaces takes MORE weight than was removed.
kept6, folded6 = fold_reference([0.60, 0.30, 0.10], [10, 11, 32], 2, FLAT,
                                {s: {t: 2.0 for t in E} for s in E}, LOSS_PERFECT)
check("a scalar above 1 deliberately raises the routed mass above the router's",
      sum(folded6) > 1.0 and close(sum(folded6), 0.60 + 0.30 + 0.10 * 2.0))

# A negative scalar (the DeepSeek artifact does not clip) must subtract, not be clamped away.
kept7, folded7 = fold_reference([0.60, 0.40], [10, 11], 1, FLAT,
                                {s: {t: -0.5 for t in E} for s in E}, LOSS_PERFECT)
check("a negative scalar subtracts rather than being silently clamped",
      close(folded7[0], 0.60 - 0.40 * 0.5))


# ============================================================ the none arm
idx, w = reduce_reference(W, E, 3, 1.5)
check("the none arm keeps the router's first k'", idx == [10, 11, 12])
check("... and renormalises them back to route_scale", close(sum(w), 1.5))
check("... preserving their ratios exactly", close(w[0] / w[1], W[0] / W[1]))


# ============================================================ pins
src = open(os.path.join(ROOT, "engine/exfold.py")).read()
_reduce_body = src.split("def reduce(", 1)[1].split("def reduce_plain", 1)[0]
check("the tensor fold never renormalises and never rescales: the scalar IS the correction",
      "* route_scale" not in _reduce_body and "route_scale)" not in _reduce_body
      and "kept_w.sum" not in _reduce_body)
check("decode cannot reach the fold: fastdecode does not import it",
      "exfold" not in open(os.path.join(ROOT, "engine/fastdecode.py")).read())
check("the fold is applied only where the prefill reduction is",
      len(re.findall(r"self\.exfold\.reduce\(", open(os.path.join(ROOT, "engine/model.py")).read())) == 1)


# ============================================================ torch, if it is here
try:
    import torch
except ImportError:
    print("\nskip: torch not importable here -- the tensor implementation is checked on the box")
else:
    from engine.exfold import FoldTable, reduce_plain

    n_e = 64
    S = torch.ones(1, n_e, n_e)
    N = torch.full((1, n_e, n_e), 0.5)
    h = torch.ones(1, n_e)
    for s in E:
        for t in E:
            S[0, s, t] = SCALAR[s][t]
            N[0, s, t] = LOSS[s][t]
    for e, v in H.items():
        h[0, e] = v
    tbl = FoldTable(S, N, h, {}, "<toy>")

    ti = torch.tensor([E, E], dtype=torch.long)
    tw = torch.tensor([W, W], dtype=torch.float64)
    for k_prime in (1, 2, 3, 4, 5):
        gi, gw = tbl.reduce(ti, tw, 0, k_prime)
        ri, rw = fold_reference(W, E, k_prime, H, SCALAR, LOSS)
        check(f"torch and the reference agree at k'={k_prime}",
              gi[0].tolist() == ri and allclose(gw[0].tolist(), rw, 1e-6),
              f"{gi[0].tolist()} {gw[0].tolist()} vs {ri} {rw}")

    pi, pw = reduce_plain(ti, tw, 3, 1.5)
    ri, rw = reduce_reference(W, E, 3, 1.5)
    check("torch and the reference agree on the none arm",
          pi[0].tolist() == ri and allclose(pw[0].tolist(), rw, 1e-6))

    big_i = torch.stack([torch.tensor(E)] * 257)
    big_w = torch.stack([torch.tensor(W, dtype=torch.float64)] * 257)
    gi, gw = tbl.reduce(big_i, big_w, 0, 3)
    check("every row of a chunk gets the same answer as one row",
          gi.shape == (257, 3) and gw.shape == (257, 3)
          and bool((gi == gi[0]).all()) and allclose(gw[256].tolist(), gw[0].tolist(), 1e-9))

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
