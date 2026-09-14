"""Decode-side controls scoped to the reasoning span (`<think>` ... `</think>`).

Two controls, both OFF by default, built per request by ``server/app.py`` and
handed to the engine exactly like the tool-call gate in ``tool_grammar.py``:
the engine owes this object ``observe`` and ``mask_rows`` and nothing else.
Standard library only -- no torch -- so the whole state machine is testable
without a GPU (``server/test_think_controls.py``).

Why they exist, measured on this repo's own gate (RESULTS.md 5.4 and the
2026-09-14 addenda):

* A run that passes deliberates for about 4,000 reasoning CHARACTERS; a run
  that fails deliberates for about 19,000. The Backend keep-set at 0.36 produced
  a correct answer on all ten prompts, but seven of them restated themselves
  three to eight times first, at 18-38k reasoning characters against 4-9k where
  they do not loop. The failure is not "it cannot answer", it is "it will not
  stop thinking": a long deliberation ends in a "Maybe add X? Skip." loop, or
  it never closes the think block at all and the answer is empty.

  Those are the gate's own columns, and they are character counts -- until
  2026-09-14 they were quoted here as token counts, which they never were. A
  budget below is in TOKENS, and a token is about four characters in this
  register (2,001 forced reasoning tokens = 7,953 characters, measured
  2026-09-14), so the numbers above are roughly 1,000 tokens for a run that
  passes, 4,800 for one that fails and 4.5-9.5k for the ones that loop. A budget
  chosen from the character figures is about four times too large to ever fire
  (RESULTS.md, correction of 2026-09-14 21:10).
* So: (1) cap the reasoning span and force the close token when the cap is
  reached -- the model then writes its answer from the deliberation it already
  has; (2) inside the span only, refuse the token that would start a third
  verbatim copy of an n-gram.

Both are scoped to the reasoning span on purpose. A repeat ban over the *answer*
was refuted here: CSS repeats `px`, `0`, `;` and `}` legitimately and an n-gram
ban destroys it. Inside a think block there is no such register.

The whole object is a no-op unless the request asked for one of the two, and a
request with thinking off never builds one.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, Sequence, Tuple

log = logging.getLogger("dsv41.think")

NEG_INF = float("-inf")

# Rolling hash over the last n token ids of the span. A 61-bit modulus keeps the
# products in one machine word and makes a collision a ~1e-18 event per window;
# a collision would ban one token for one step, which is the same cost as the
# control firing correctly, so it is not worth a second exact check.
_HASH_MOD = (1 << 61) - 1
_HASH_BASE = 1_000_003

MIN_NGRAM = 2
MAX_NGRAM = 128


def _dim(logits) -> int:
    """Rows of a logits tensor, without importing torch."""
    d = getattr(logits, "dim", None)
    if callable(d):
        return int(d())
    return int(getattr(logits, "ndim", 1))


class ThinkControls:
    """Reasoning-span budget and loop breaker for one request.

    ``think_end_id`` is the id of ``</think>`` -- the single token the V4.1
    tokenizer has for it (128822 on this checkpoint). With thinking on, the
    prompt ends with ``<think>``, so generation *starts* inside the reasoning
    span and the span ends at the first ``</think>`` the model writes. That is
    the whole span detection: no marker scanning, no text.

    :param budget: force ``</think>`` once this many reasoning tokens have been
        generated without one. 0 = off.
    :param repeat_ngram: window length n for the loop breaker. When the last n
        tokens of the span repeat a window that has already occurred twice, the
        tokens that followed those earlier occurrences are masked, so the third
        copy cannot be extended. 0 = off.

    The budget is checked once per decode *step*, and a speculative step settles
    up to six tokens at once, so a span can overrun the budget by a few tokens
    before the close lands. That is measured honestly:
    ``usage.completion_tokens_details.reasoning_tokens`` is the real length.

    Memory is one dict entry per window of the span (a 19k-token deliberation
    costs a few MB) and it is dropped with the request.
    """

    def __init__(self, think_end_id: Optional[int], *, budget: int = 0,
                 repeat_ngram: int = 0, in_think: bool = True) -> None:
        self.think_end_id = think_end_id
        self.budget = max(0, int(budget or 0))
        self.repeat_ngram = max(0, int(repeat_ngram or 0))
        self.in_think = bool(in_think)
        self.reasoning_tokens = 0
        self.budget_hit = False
        self._failed = False
        n = self.repeat_ngram
        self._win: deque = deque(maxlen=n or 1)
        self._h = 0
        self._pn = pow(_HASH_BASE, n - 1, _HASH_MOD) if n else 0
        self._count: dict = {}     # window hash -> occurrences so far, including the latest
        self._follow: dict = {}    # window hash -> the token ids that followed earlier occurrences
        self._prev_h: Optional[int] = None   # hash of the window ending at the last observed token
        self.stats = {
            "budget": self.budget,
            "repeat_ngram": self.repeat_ngram,
            "budget_forced": 0,
            "repeat_breaks": 0,
            "banned_tokens": 0,
            "reasoning_tokens": 0,
            "mask_calls": 0,
            "mask_s": 0.0,
        }

    # -- state -----------------------------------------------------------
    @property
    def active(self) -> bool:
        """False means the engine can skip this object entirely."""
        return self.think_end_id is not None and (self.budget > 0 or self.repeat_ngram > 0)

    def observe(self, token_ids: Sequence[int]) -> None:
        """Every token the decode loop has settled on, in order, once.

        Speculated tokens that were rolled back must never be shown: the span is
        what the model actually wrote.
        """
        if not self.in_think or not self.active:
            return
        for t in token_ids:
            t = int(t)
            if t == self.think_end_id:
                self.in_think = False
                self.stats["reasoning_tokens"] = self.reasoning_tokens
                return
            if self.repeat_ngram:
                # the window ending at the previous token is now known to be followed by t
                if self._prev_h is not None:
                    self._follow.setdefault(self._prev_h, set()).add(t)
                if len(self._win) == self.repeat_ngram:
                    self._h = (self._h - self._win[0] * self._pn) % _HASH_MOD
                self._win.append(t)
                self._h = (self._h * _HASH_BASE + t) % _HASH_MOD
                if len(self._win) == self.repeat_ngram:
                    self._prev_h = self._h
                    self._count[self._h] = self._count.get(self._h, 0) + 1
                else:
                    self._prev_h = None
            self.reasoning_tokens += 1
        self.stats["reasoning_tokens"] = self.reasoning_tokens

    # -- decisions -------------------------------------------------------
    def force_token(self) -> Optional[int]:
        """``</think>`` when the budget is spent, else None.

        This *records* the fire (``budget_hit``), so ask once per decode step and
        only where the answer is acted on. It is the one decision both the real
        engine (which masks every other token out of the row) and the mock engine
        (which splices the token in) take, so they cannot drift apart.
        """
        if not (self.in_think and self.budget and self.think_end_id is not None):
            return None
        if self.reasoning_tokens < self.budget:
            return None
        self.budget_hit = True
        self.stats["budget_forced"] += 1
        return self.think_end_id

    def banned_tokens(self) -> Tuple[int, ...]:
        """Tokens that would extend a third verbatim copy of the current window."""
        if not (self.in_think and self.repeat_ngram) or self._prev_h is None:
            return ()
        if self._count.get(self._prev_h, 0) < 3:
            return ()
        return tuple(self._follow.get(self._prev_h, ()))

    # -- the engine-facing mask ------------------------------------------
    def mask_rows(self, logits, block_ids=None) -> int:
        """Mask ``logits`` in place; return the number of rows masked (0 or 1).

        ``logits`` is [R, V] (the DSpark verify block's rows) or [V]; only **row
        0** is ever touched, because both decisions are functions of the settled
        history and rows 1..R-1 follow a different prefix. That is enough on the
        speculative path: row 0 is where the first draft is verified, so masking
        it rejects that draft, the block is rolled back to one token, and the
        bonus token sampled from row 0 is the forced ``</think>`` (greedy takes
        the argmax of the masked row; the rejection sampler's residual
        ``(p - q).clamp_min(0)`` of a one-hot p is that same point mass).

        Failure here is recoverable and killing the request is not, so a broken
        mask disables the control for the rest of the generation and says so.
        """
        if self._failed or not self.in_think or not self.active:
            return 0
        close = self.force_token()
        ban = () if close is not None else self.banned_tokens()
        if close is None and not ban:
            return 0
        t0 = time.perf_counter()
        try:
            row = logits[0] if _dim(logits) > 1 else logits
            if close is not None:
                row[:] = NEG_INF
                row[close] = 0.0
            else:
                row[list(ban)] = NEG_INF
                self.stats["repeat_breaks"] += 1
                self.stats["banned_tokens"] += len(ban)
        except Exception as e:  # noqa: BLE001 - losing the control beats losing the request
            self._failed = True
            self.stats["error"] = f"{type(e).__name__}: {e}"
            log.warning("think controls failed to mask (%s); dropped for this request", e)
            return 0
        self.stats["mask_calls"] += 1
        self.stats["mask_s"] += time.perf_counter() - t0
        return 1


def make_controls(think_end_id: Optional[int], *, budget: int, repeat_ngram: int) -> Optional[ThinkControls]:
    """A :class:`ThinkControls` when one of the two is asked for, else None.

    Returns None (with a warning) when the tokenizer has no single id for
    ``</think>``: without it neither control can be expressed.
    """
    if budget <= 0 and repeat_ngram <= 0:
        return None
    if think_end_id is None:
        log.warning("reasoning_budget/think_repeat_break asked for but the tokenizer has no single "
                    "id for </think>; both controls are off for this request")
        return None
    tc = ThinkControls(think_end_id, budget=budget, repeat_ngram=repeat_ngram)
    return tc if tc.active else None
