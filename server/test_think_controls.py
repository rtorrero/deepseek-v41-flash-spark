#!/usr/bin/env python3
"""The reasoning-span controls, and how a request asks for them.

Standard library only -- no torch, no CUDA, no checkpoint, no server -- because
the whole state machine of ``server/think_controls.py`` is plain Python and the
only thing it needs from a tensor is item assignment. ``FakeLogits`` below is
that much of a tensor and no more, so the masking these tests check is the same
code the decode loop runs on the GPU.

    python3 server/test_think_controls.py
"""

from __future__ import annotations

import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from think_controls import NEG_INF, ThinkControls, make_controls  # noqa: E402

CLOSE = 128822          # </think> on this checkpoint
V = 129280              # the checkpoint's vocabulary: the ids below are the real ones


class FakeRow:
    """One logits row: enough of a tensor for ``mask_rows`` to write into."""

    def __init__(self, n: int) -> None:
        self.v = [0.0] * n

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            for i in range(*key.indices(len(self.v))):
                self.v[i] = value
        elif isinstance(key, list):
            for i in key:
                self.v[i] = value
        else:
            self.v[key] = value

    def __getitem__(self, key):
        return self.v[key]

    def argmax(self) -> int:
        return max(range(len(self.v)), key=lambda i: self.v[i])

    def banned(self) -> set:
        return {i for i, x in enumerate(self.v) if x == NEG_INF}


class FakeLogits:
    """[R, V] of :class:`FakeRow`, with torch's ``dim()``."""

    def __init__(self, rows: int = 6, n: int = V) -> None:
        self.rows = [FakeRow(n) for _ in range(rows)]

    def dim(self) -> int:
        return 2

    def __getitem__(self, i):
        return self.rows[i]


def step(tc: ThinkControls, rows: int = 6) -> FakeLogits:
    """One decode step's mask, returned for inspection."""
    lg = FakeLogits(rows)
    tc.mask_rows(lg, None)
    return lg


# -- tests -------------------------------------------------------------------

def test_off_by_default():
    """No budget and no n: the object is inert and the server does not build one."""
    tc = ThinkControls(CLOSE)
    assert not tc.active
    tc.observe([5, 6, 7])
    lg = step(tc)
    assert lg[0].banned() == set() and tc.mask_rows(lg, None) == 0
    assert make_controls(CLOSE, budget=0, repeat_ngram=0) is None
    # ... and with no id for </think> neither control can be expressed
    assert make_controls(None, budget=100, repeat_ngram=12) is None
    assert make_controls(CLOSE, budget=100, repeat_ngram=0) is not None


def test_budget_forces_the_close_and_only_once():
    tc = make_controls(CLOSE, budget=4, repeat_ngram=0)
    tc.observe([10, 11, 12])
    assert step(tc)[0].banned() == set(), "under budget: nothing is masked"
    assert tc.force_token() is None and not tc.budget_hit

    tc.observe([13])                      # the 4th reasoning token
    lg = step(tc)
    row = lg[0]
    assert row.argmax() == CLOSE, "the close token is the only one left"
    assert len(row.banned()) == V - 1 and CLOSE not in row.banned()
    assert tc.budget_hit and tc.stats["budget_forced"] >= 1
    assert lg[1].banned() == set(), "rows 1..R-1 follow a different prefix and are left alone"

    tc.observe([CLOSE])                   # the loop emitted it
    assert not tc.in_think
    assert step(tc)[0].banned() == set(), "the answer is never masked"
    assert tc.reasoning_tokens == 4


def test_budget_reasserts_itself_until_the_close_lands():
    """A step that did not emit the close (a stop id ended the block first) is
    forced again on the next one rather than silently giving up."""
    tc = make_controls(CLOSE, budget=2, repeat_ngram=0)
    tc.observe([1, 2])
    assert step(tc)[0].argmax() == CLOSE
    tc.observe([3])                       # something else came out anyway
    assert step(tc)[0].argmax() == CLOSE


def test_budget_ignores_a_request_that_is_not_thinking():
    tc = ThinkControls(CLOSE, budget=2, repeat_ngram=0, in_think=False)
    tc.observe([1, 2, 3, 4])
    assert tc.reasoning_tokens == 0 and step(tc)[0].banned() == set()


def test_repeat_break_fires_on_the_third_copy_only():
    """`a b c` twice is a model drafting; three times is a loop."""
    n = 3
    tc = make_controls(CLOSE, budget=0, repeat_ngram=n)
    block = [7, 8, 9]
    tc.observe(block + [40])              # 1st copy, followed by 40
    assert step(tc)[0].banned() == set()
    tc.observe(block + [41])              # 2nd copy, followed by 41
    assert step(tc)[0].banned() == set(), "twice is not yet a loop"
    tc.observe(block)                     # 3rd copy: this is the one that gets cut
    assert step(tc)[0].banned() == {40, 41}, "both continuations of the earlier copies"
    assert tc.stats["repeat_breaks"] == 1 and tc.stats["banned_tokens"] == 2


def test_repeat_break_cuts_a_single_token_run():
    """The `e e e e` corruption: the window is one token repeated."""
    tc = make_controls(CLOSE, budget=0, repeat_ngram=4)
    tc.observe([5] * 8)
    assert step(tc)[0].banned() == {5}


def test_repeat_break_needs_the_window_to_match_exactly():
    tc = make_controls(CLOSE, budget=0, repeat_ngram=4)
    for i in range(6):
        tc.observe([1, 2, 3, i])          # the last token differs every time
    assert step(tc)[0].banned() == set()


def test_repeat_break_is_scoped_to_the_reasoning_span():
    """The refuted control: an n-gram ban over an answer destroys CSS, which
    repeats `px` and `0` legitimately. Nothing after `</think>` is touched."""
    tc = make_controls(CLOSE, budget=0, repeat_ngram=3)
    tc.observe([4, 5, 6] * 4)
    assert step(tc)[0].banned(), "inside the span it is on"
    tc.observe([CLOSE])
    tc.observe([4, 5, 6] * 8)             # the same n-gram, now in the answer
    assert step(tc)[0].banned() == set()
    assert tc.reasoning_tokens == 12, "answer tokens are not reasoning tokens"


def test_both_controls_together_and_the_budget_wins_the_row():
    tc = make_controls(CLOSE, budget=9, repeat_ngram=3)
    tc.observe([2, 3, 4] * 3)             # 9 tokens: a loop AND the budget, same step
    row = step(tc)[0]
    assert row.argmax() == CLOSE and len(row.banned()) == V - 1


def test_masking_failure_disables_the_control_instead_of_the_request():
    class Exploding(FakeLogits):
        def __getitem__(self, i):
            raise RuntimeError("no such row")

    tc = make_controls(CLOSE, budget=1, repeat_ngram=0)
    tc.observe([1])
    assert tc.mask_rows(Exploding(), None) == 0
    assert "error" in tc.stats
    assert tc.mask_rows(FakeLogits(), None) == 0, "and stays off for the rest of the request"


def test_one_dimensional_logits():
    tc = make_controls(CLOSE, budget=1, repeat_ngram=0)
    tc.observe([1])

    class Row1D(FakeRow):
        def dim(self):
            return 1

    row = Row1D(V)
    assert tc.mask_rows(row, None) == 1 and row.argmax() == CLOSE


def test_rolling_hash_tracks_a_long_span():
    """The window is a rolling hash, so a 20k-token span must still ban on the
    n-gram that actually repeats and on nothing else."""
    rng = random.Random(7)
    tc = make_controls(CLOSE, budget=0, repeat_ngram=12)
    for _ in range(200):
        tc.observe([rng.randrange(1000, 3000) for _ in range(100)])
    assert step(tc)[0].banned() == set(), "20,000 tokens of noise, no ban"
    filler = list(range(20, 32))
    for _ in range(3):
        tc.observe(filler + [999])
    tc.observe(filler)
    assert step(tc)[0].banned() == {999}, "the one window that does repeat, three times"
    assert tc.reasoning_tokens == 20000 + 3 * 13 + 12


# -- how a request asks for them ---------------------------------------------

def _app():
    import app  # noqa: PLC0415 - stdlib-only import, but only needed by these tests
    return app


def test_request_fields():
    app = _app()
    os.environ.pop("DSV41_THINK_BUDGET", None)
    os.environ.pop("DSV41_THINK_REPEAT_BREAK", None)
    r = app.resolve_reasoning_controls
    assert r({}) == (0, 0), "both off unless asked for"
    assert r({"reasoning_budget": 8000}) == (8000, 0)
    assert r({"reasoning": {"max_tokens": 8000}}) == (8000, 0)
    assert r({"chat_template_kwargs": {"reasoning_budget": 123}}) == (123, 0)
    assert r({"reasoning_budget": 1, "reasoning": {"max_tokens": 2}}) == (1, 0), "top level wins"
    assert r({"think_repeat_break": 12}) == (0, 12)
    assert r({"chat_template_kwargs": {"think_repeat_break": 12}}) == (0, 12)
    assert r({"reasoning_budget": 0, "think_repeat_break": 0}) == (0, 0), "0 is a valid off"

    for bad in ({"reasoning_budget": -1}, {"reasoning_budget": "8000"}, {"reasoning_budget": True},
                {"think_repeat_break": 1}, {"think_repeat_break": 129}, {"think_repeat_break": 1.5},
                {"reasoning": 7}):
        try:
            r(bad)
        except app.APIError as e:
            assert e.status == 400, (bad, e.status)
        else:
            raise AssertionError(f"{bad} should have been rejected")


def test_environment_defaults():
    app = _app()
    os.environ["DSV41_THINK_BUDGET"] = "8000"
    os.environ["DSV41_THINK_REPEAT_BREAK"] = "12"
    try:
        assert app.resolve_reasoning_controls({}) == (8000, 12)
        # a request still overrides the deployment default, including back to off
        assert app.resolve_reasoning_controls({"reasoning_budget": 0}) == (0, 12)
        os.environ["DSV41_THINK_BUDGET"] = "nonsense"
        assert app.resolve_reasoning_controls({}) == (0, 12), "a bad value is off, not a 500"
        os.environ["DSV41_THINK_REPEAT_BREAK"] = "1"
        assert app.resolve_reasoning_controls({}) == (0, 0)
    finally:
        os.environ.pop("DSV41_THINK_BUDGET", None)
        os.environ.pop("DSV41_THINK_REPEAT_BREAK", None)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
