#!/usr/bin/env python3
"""The reasoning budget, where the decode loop actually has to honour it.

    python3 engine/test_think_budget.py

Two parts, and the first one runs anywhere:

**A. Wiring (no torch, no CUDA, no checkpoint).** The controls are only worth
anything if the decode loop calls them at *every* site it calls the tool-call
gate at -- a burst emitted without an ``observe`` silently mis-counts the span,
and a path without a ``mask_rows`` silently ignores the budget. This part reads
``engine/v41_engine.py`` with ``ast`` and as text and checks: ``think`` is a
parameter of ``generate``, ``_generate`` and ``_decode_loop`` and is passed down
both hops; there are as many ``th.observe`` sites as ``grammar.observe`` sites
and as many ``th.mask_rows`` as ``grammar.mask_rows``; and on both decode paths
``th.mask_rows`` comes **after** ``pen.apply``, because a forced ``</think>``
leaves one legal token in the row and a penalty applied afterwards could ban it
and leave the row all ``-inf``.

**B. The forcing, against the real verification arithmetic (torch, CPU is
enough -- this skips if torch is missing, and needs no GPU and no weights).**
Masking row 0 is only equivalent to "emit ``</think>`` next" because of what the
three verify paths then compute, so this part builds a [6, V] logits block whose
drafts would all be accepted, applies ``ThinkControls.mask_rows`` to it, and runs
the accept/reject arithmetic of each path over the result:

* the lean greedy path (``argmax`` -> ``eq(drafts)`` -> ``cumprod`` -> ``sum``):
  accepted length must be 0 and the bonus token ``</think>``;
* the per-position greedy path (``sample_probs`` at temperature 0): the draft is
  rejected and the bonus is ``</think>``;
* the per-position sampled path (temperature 1): the draft is rejected, and the
  residual ``(p - q).clamp_min(0)`` of the one-hot p is a point mass on
  ``</think>``, so the bonus is that token whatever the RNG does;
* the plain autoregressive path on a [1, V] row.

Those expressions are copied from ``_decode_loop``, so each one is checked to be
present in the source verbatim first: if the loop is rewritten, this test fails
as a mirror that has drifted rather than passing on code nobody runs any more.
It also checks the control case -- budget not yet spent, block accepted whole --
so the mask is shown to be the cause.
"""

from __future__ import annotations

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

from think_controls import ThinkControls  # noqa: E402  (server/, stdlib only)

ENGINE = os.path.join(ROOT, "engine", "v41_engine.py")
SRC = open(ENGINE, encoding="utf-8").read()
FAILS: list = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


# =============================================================================
# A. wiring
# =============================================================================

def params_of(name: str) -> set:
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            a = node.args
            return {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    raise AssertionError(f"{name} not found in {ENGINE}")


def part_a() -> None:
    print("A. wiring (no torch needed)")
    for fn in ("generate", "_generate", "_decode_loop"):
        check("think" in params_of(fn), f"`think` is a parameter of {fn}()")
    check(bool(re.search(r"self\._generate\((?:.|\n)*?grammar, penalties, think\)", SRC)),
          "generate() passes think to _generate()")
    check(bool(re.search(r"self\._decode_loop\((?:.|\n)*?grammar, penalties, think\)", SRC)),
          "_generate() passes think to _decode_loop()")
    check("th = think if (think is not None and think.active) else None" in SRC,
          "an inactive control object is dropped once, not tested every step")

    n_obs_g, n_obs_t = SRC.count("grammar.observe("), SRC.count("th.observe(")
    check(n_obs_g == n_obs_t and n_obs_t == 4,
          f"every settled-token site feeds both ({n_obs_g} grammar.observe, {n_obs_t} th.observe)")
    n_mask_g, n_mask_t = SRC.count("grammar.mask_rows("), SRC.count("th.mask_rows(")
    check(n_mask_g == n_mask_t and n_mask_t == 2,
          f"both decode paths mask ({n_mask_g} grammar.mask_rows, {n_mask_t} th.mask_rows)")

    # ordering: on each path pen.apply must come before th.mask_rows
    pens = [m.start() for m in re.finditer(r"pen\.apply\(logits, out\)", SRC)]
    masks = [m.start() for m in re.finditer(r"th\.mask_rows\(logits, ", SRC)]
    check(len(pens) == 2 and len(masks) == 2
          and all(p < m for p, m in zip(sorted(pens), sorted(masks)))
          and sorted(masks)[0] > sorted(pens)[0] and sorted(masks)[1] > sorted(pens)[1],
          "th.mask_rows runs after pen.apply on both paths (the forced close must survive)")
    check("supports_think_controls = True" in SRC,
          "the engine advertises the control object to the server")


# =============================================================================
# B. the forcing, against the loop's own verification arithmetic
# =============================================================================

# Copied from _decode_loop; each is asserted to still be in the source.
MIRRORED = [
    "am = logits.argmax(-1)",
    "acc = am[:self._tv - 1].eq(drafts).to(torch.int32).cumprod(0)",
    "pt = sample_probs(logits[i], temperature, top_p)",
    "ok = int(pt.argmax()) == d",
    "resid = (pt - q[i]).clamp_min(0)",
    "bonus = int(torch.multinomial(resid / resid.sum(), 1))",
    "pt = sample_probs(logits[0], temperature, top_p)",
    "tok = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())",
]

CLOSE = 7           # stands in for </think>; the ids do not matter to the arithmetic
VOCAB = 32
T_VERIFY = 6        # one accepted token + five drafts


def part_b() -> None:
    print("\nB. the forcing, against the real verify arithmetic")
    for line in MIRRORED:
        check(line in SRC, f"mirrored line still in v41_engine.py: {line[:52]}")

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("SKIP torch is not installed; the arithmetic checks did not run "
              "(they need torch on the CPU only -- no GPU, no checkpoint)")
        return
    try:
        from engine.v41_engine import sample_probs  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 - the module pulls in the whole engine
        print(f"SKIP cannot import engine.v41_engine ({e}); the arithmetic checks did not run")
        return

    drafts = torch.tensor([1, 2, 3, 4, 5])

    def block_logits():
        """[6, V] whose argmax on every row is the draft that row verifies, so
        without a mask the whole block is accepted."""
        lg = torch.zeros(T_VERIFY, VOCAB)
        for i in range(T_VERIFY - 1):
            lg[i, int(drafts[i])] = 10.0
        lg[T_VERIFY - 1, 9] = 10.0
        return lg

    def spent(budget=1, ngram=0):
        tc = ThinkControls(CLOSE, budget=budget, repeat_ngram=ngram)
        tc.observe([11] * budget)
        return tc

    # -- control: budget not spent, nothing masked, the block is accepted whole
    lg = block_logits()
    tc = ThinkControls(CLOSE, budget=1000)
    tc.observe([11])
    tc.mask_rows(lg, None)
    am = lg.argmax(-1)
    acc = am[:T_VERIFY - 1].eq(drafts).to(torch.int32).cumprod(0)
    check(int(acc.sum()) == 5 and not tc.budget_hit,
          "control: under budget the drafts are accepted (5 of 5) and nothing fired")

    # -- lean greedy path
    lg = block_logits()
    tc = spent()
    check(tc.mask_rows(lg, None) == 1, "the budget masked exactly one row")
    am = lg.argmax(-1)
    acc = am[:T_VERIFY - 1].eq(drafts).to(torch.int32).cumprod(0)
    a, cand = int(acc.sum()), am.tolist()
    check(a == 0, "lean path: accepted length is 0, the block is rolled back to one token")
    check(cand[a] == CLOSE, "lean path: the bonus token is </think>")
    check(lg[1].argmax().item() == int(drafts[1]),
          "rows 1..5 are untouched (the answer continues from the model's own logits)")

    # -- per-position greedy path
    lg, tc = block_logits(), spent()
    tc.mask_rows(lg, None)
    pt = sample_probs(lg[0], 0.0, 0.95)
    d = int(drafts[0])
    check(not (int(pt.argmax()) == d), "greedy path: draft 0 is rejected")
    check(int(pt.argmax()) == CLOSE, "greedy path: the bonus token is </think>")

    # -- per-position sampled path (rejection sampling)
    lg, tc = block_logits(), spent()
    tc.mask_rows(lg, None)
    pt = sample_probs(lg[0], 1.0, 0.95)
    q = torch.full((T_VERIFY - 1, VOCAB), 1.0 / VOCAB)
    ok_any = False
    for _ in range(200):
        r = torch.rand(())
        ok_any |= bool(r < (pt[d] / q[0][d].clamp_min(1e-20)).clamp(max=1.0))
    check(not ok_any, "sampled path: draft 0 is rejected for every RNG draw")
    resid = (pt - q[0]).clamp_min(0)
    if float(resid.sum()) <= 0:
        resid = pt
    bonuses = {int(torch.multinomial(resid / resid.sum(), 1)) for _ in range(200)}
    check(bonuses == {CLOSE}, "sampled path: the residual is a point mass on </think>")

    # -- plain autoregressive path, [1, V]
    lg, tc = torch.zeros(1, VOCAB), spent()
    lg[0, 3] = 10.0
    tc.mask_rows(lg, None)
    pt = sample_probs(lg[0], 1.0, 0.95)
    check({int(torch.multinomial(pt, 1)) for _ in range(200)} == {CLOSE},
          "plain path: </think> is the only token that can be sampled")

    # -- the loop breaker on the same shapes
    lg = block_logits()
    tc = ThinkControls(CLOSE, budget=0, repeat_ngram=3)
    tc.observe([1, 2, 3] * 4)            # the token that extends the loop is `1`
    tc.mask_rows(lg, None)
    check(float(lg[0][1]) == float("-inf") and float(lg[0][2]) == 0.0,
          "loop breaker: only the continuation of the repeated window is banned")
    check(lg[1].argmax().item() == int(drafts[1]), "loop breaker: rows 1..5 untouched")


def main() -> int:
    part_a()
    part_b()
    print()
    print(f"{len(FAILS)} failed: " + "; ".join(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
