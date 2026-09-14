"""escape.py -- the transient-expert escape hatch: when a masked-off pick is worth fetching.

A keep-set is a hard mask. The router's logits are `masked_fill(-inf)` outside the resident set
and the token is computed with six experts it did not ask for (`substitute`, the shipped default).
That is what makes the engine fast and it is also what corrupts rare tokens at subword boundaries:
the token loses its true pick, the substitute injects a signal the model was never trained to
receive, and the model then loops trying to repair what it wrote (RESULTS.md 5.3, docs/keep-sets.md).

The two ends of that trade are both measured. Streaming *every* expert -- no mask at all -- passes
the three prompts that fail on every keep-set, at 1,008-1,549 s per prompt
(`results/keepsets/null-unmasked/GATE.md`). Masking everything not resident is 15-25 tok/s and
fails them. Nothing in between has been tried, and "a little streaming" is the obvious thing to
look at: not every displaced pick matters, and the ones that do are identifiable from the router's
own scores before the mask is applied.

This module is the POLICY half of that experiment -- which picks are worth 18.80 MB of NVMe and a
stall in the decode step, how many are allowed, and what is counted. It holds no torch and touches
no device: the engine's `EscapeRuntime` (engine/v41_engine.py) does the fetching, the mask patch
and the slot bookkeeping, and calls in here to decide. Keeping the decision arithmetic torch-free
is what lets `tools/test_escape_rule.py` test the shipped code rather than a copy of it.

    DSV41_ESCAPE_K=0        how many non-resident experts a layer-step may fetch (0 = off)
    DSV41_ESCAPE_MARGIN     how much better than the pick it displaces the candidate must be

Off by default, and off means the engine never constructs this object at all.

The margin
----------
The gate is `scores = sqrt(softplus(y @ gate_w))`, `logits = scores + gate_bias`, and the routed
weights are the *scores* of the top-k by logit, renormalised over the k and multiplied by
`route_scale`. There is no softmax anywhere, so "probability" has to be defined rather than read
off. The quantity that decides what a substitution costs is how much of the token's routed weight
changes hands, so that is what the margin is:

    margin = (scores[candidate] - scores[displaced]) / sum(scores[resident top-k])

where `displaced` is the resident pick the candidate would push out of the top-k. It is
dimensionless, comparable across layers (the denominator is the same one the renormalisation
uses), and reads directly: at `DSV41_ESCAPE_MARGIN=0.10` a pick is fetched when it would carry a
tenth of this token's routed weight more than the expert standing in for it. For scale, a uniform
top-6 gives every pick 1/6 = 0.167 of the mass.

For the j-th candidate (1-based, candidates ordered by logit) the displaced pick is the j-th
weakest of the resident top-k: admitting one expert pushes out the last, admitting two pushes out
the last two. That bookkeeping is only right for a PREFIX of the candidates, so the rule admits a
prefix -- it stops at the first candidate that does not clear the margin rather than skipping it.
The margins are not monotone in j and nothing here assumes they are: the ranking is by logit
(score + gate_bias) and the margin is in scores, so a candidate can outrank the pick it would
displace and still be worth less routed weight than it. The hatch does not fetch that expert,
which is the whole reason the margin is defined in scores and not in logits.

A candidate also has to actually DISPLACE something. Being the best non-resident expert does not
mean the mask took anything away: if its logit is below the weakest resident pick's, the router
would not have chosen it with or without the keep-set and fetching it would change nothing. The
scan gives such a candidate a margin of -inf, so one rule -- "margin >= DSV41_ESCAPE_MARGIN" --
covers both conditions, and at K=1 the rule reads exactly as it sounds: the router's unmasked
top-1 is non-resident, and it is worth this much more than the expert standing in for it.

The budget
----------
A fetch is not free and it is not overlapped: 18.80 MB of O_DIRECT reads (the checkpoint stores
FP4; 14.45 MB is what the expert occupies in a CB3 arena, not what is read) plus, in CB3, a GPU
repack of the three matrices before the slot is usable. `K` caps a layer-step. The whole step is
capped too, at `STEP_BUDGET_PER_K * K`, because K alone bounds nothing useful: with 40 layers,
K=1 permits 40 fetches in one decode step, and at even 4 ms each that is a step of 295 ms instead
of 135. Eight fetches a step is the largest number that keeps the hatch inside the noise of a
~135 ms step if a fetch is cheap, and a visible but not ruinous cost if it is not.
"""

from __future__ import annotations

import os

#: `DSV41_ESCAPE_MARGIN` when the variable is unset but `DSV41_ESCAPE_K` is not.
#:
#: No gate has been run behind this number and no measurement of this model's gate distribution
#: exists to derive it from. What it is: the value at which the torch-free reference in
#: `tools/test_escape_rule.py` fires on about one layer-step in twenty at the shipped budget
#: (139 resident of 384) -- roughly two escapes in a 40-layer decode step, which fits inside the
#: per-step budget below with room to spare. That reference draws its gate scores from a synthetic
#: distribution, not from the checkpoint, so treat it as a starting point and not as a finding.
#: `tools/verify_escape.sh` prints `escapes_per_token` from the server's own `x_engine_stats`;
#: that is the number this should be re-derived from.
DEFAULT_MARGIN = 0.10

#: Per decode step, the hatch may fetch this many experts per unit of K, across all 40 layers.
#: See "The budget" above.
STEP_BUDGET_PER_K = 8


class EscapeHatch:
    """The decision rule and its accounting. No torch, no device, no I/O.

    One instance per engine. `select()` is the whole policy; everything else is counters the
    engine reports in `x_engine_stats` and a small map of what is currently live so the runtime
    knows which mask bits and LUT entries it owns.
    """

    __slots__ = ("k", "margin", "topk", "step_budget", "step_left", "live", "stats")

    def __init__(self, k: int, margin: float, topk: int = 6,
                 step_budget_per_k: int = STEP_BUDGET_PER_K):
        k = int(k)
        margin = float(margin)
        if k < 0:
            raise ValueError(f"DSV41_ESCAPE_K must be >= 0 (got {k})")
        if k > topk:
            # More than `topk` fetches cannot change more than `topk` picks: the candidates past
            # the k-th displace experts that are themselves already displaced. Refuse rather than
            # silently spend NVMe on experts the router cannot reach.
            raise ValueError(f"DSV41_ESCAPE_K={k} exceeds the router's top-k ({topk}); at most "
                             f"{topk} picks exist to change")
        if not (margin >= 0.0):  # NaN lands here too
            raise ValueError(f"DSV41_ESCAPE_MARGIN must be >= 0 (got {margin!r})")
        self.k = k
        self.margin = margin
        self.topk = int(topk)
        self.step_budget = int(step_budget_per_k) * k
        self.step_left = self.step_budget
        self.live: dict = {}          # (layer, expert) -> arena slot, currently routable
        self.stats = _zero_stats()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env=None, topk: int = 6):
        """The hatch `DSV41_ESCAPE_K` asks for, or None when it is off.

        `or`, not a default argument, for the same reason `DSV41_PRUNE_MODE` reads that way:
        start.sh sources .env with `set -a`, so a line left empty in env.example arrives here as
        '' and must still mean "off".
        """
        env = os.environ if env is None else env
        raw_k = (env.get("DSV41_ESCAPE_K") or "0").strip() or "0"
        try:
            k = int(raw_k)
        except ValueError:
            raise ValueError(f"DSV41_ESCAPE_K: {raw_k!r}; use an integer 0..{topk} "
                             f"(0 = off, the default)") from None
        if k == 0:
            return None
        raw_m = (env.get("DSV41_ESCAPE_MARGIN") or "").strip()
        try:
            margin = DEFAULT_MARGIN if not raw_m else float(raw_m)
        except ValueError:
            raise ValueError(f"DSV41_ESCAPE_MARGIN: {raw_m!r}; use a float >= 0 "
                             f"(default {DEFAULT_MARGIN})") from None
        return cls(k, margin, topk=topk)

    # ------------------------------------------------------------------ the policy
    @staticmethod
    def margins(cand_scores, displaced_scores, den: float):
        """The margin of each candidate: the share of the token's routed weight that changes hands.

        `cand_scores[j]` is the gate SCORE (not the logit) of the j-th non-resident candidate,
        `displaced_scores[j]` the score of the resident pick it would push out of the top-k, and
        `den` the sum of the resident top-k scores -- the denominator the engine renormalises the
        routed weights by. Returns a list of floats, one per candidate.
        """
        if not (den > 0.0):
            return [0.0] * len(cand_scores)
        return [(float(c) - float(d)) / float(den) for c, d in zip(cand_scores, displaced_scores)]

    def select(self, cands, margins):
        """Which of this layer-step's candidates to fetch. Reserves budget; counts what it refused.

        `cands` are expert ids ordered by router logit, descending, and `margins[j]` the margin of
        `cands[j]` computed as above. Returns the admitted prefix as a list of (expert, margin).

        A prefix, not a filter: candidate j's margin was computed against the j-th weakest
        resident pick, and that bookkeeping only holds if the j-1 candidates before it were taken
        too. So the rule stops at the first candidate that does not clear the margin instead of
        skipping it and looking at the next. A candidate the mask did not actually displace
        arrives here with a margin of -inf and stops the prefix there, which is what it should do:
        the candidates after it rank lower still.
        """
        out = []
        n = min(len(cands), len(margins))
        self.stats["considered"] += n
        for j in range(n):
            m = float(margins[j])
            if m < self.margin:
                self.stats["blocked_margin"] += n - j
                break
            if len(out) >= self.k:
                self.stats["blocked_k"] += n - j
                break
            if self.step_left <= 0:
                self.stats["blocked_budget"] += n - j
                break
            self.step_left -= 1
            out.append((int(cands[j]), m))
        return out

    def begin_step(self):
        """New decode step: the per-step fetch budget is back."""
        self.step_left = self.step_budget

    # ------------------------------------------------------------------ what is live
    def admitted(self, key, slot: int, margin: float = 0.0):
        """Record that (layer, expert) is now routable in `slot`."""
        self.live[key] = int(slot)
        self.stats["escapes"] += 1
        self.stats["margin_sum"] += float(margin)
        if len(self.live) > self.stats["peak_live"]:
            self.stats["peak_live"] = len(self.live)

    def evicted(self, key):
        """Its transient slot was recycled: it is not routable any more."""
        if self.live.pop(key, None) is not None:
            self.stats["evictions"] += 1
            return True
        return False

    def record_fetch(self, nbytes: int, seconds: float):
        self.stats["bytes"] += int(nbytes)
        self.stats["fetch_s"] += float(seconds)

    def reset(self):
        """Between requests. The engine clears the mask bits and the LUT entries that go with it.

        Escapes do not survive a request. They are decided from one prompt's routing and an
        escape set carried into the next request would make a gate's second prompt depend on its
        first -- every keep-set measurement in this repository assumes a request starts from the
        shipped keep-set, and that has to stay true with the hatch armed.
        """
        self.live.clear()
        self.stats = _zero_stats()
        self.step_left = self.step_budget

    # ------------------------------------------------------------------ reporting
    def report(self, completion_tokens: int = 0):
        """The accounting `x_engine_stats` carries: escapes per token, bytes read, ms spent."""
        s = self.stats
        n = max(int(completion_tokens), 1)
        esc = s["escapes"]
        return {
            "escape_k": self.k,
            "escape_margin": self.margin,
            "escapes": esc,
            "escapes_per_token": round(esc / n, 4),
            "escape_gb": round(s["bytes"] / 1e9, 3),
            "escape_ms": round(s["fetch_s"] * 1e3, 1),
            "escape_ms_each": round(s["fetch_s"] * 1e3 / esc, 2) if esc else None,
            "escape_margin_mean": round(s["margin_sum"] / esc, 4) if esc else None,
            "escape_live": len(self.live),
            "escape_peak_live": s["peak_live"],
            "escape_evictions": s["evictions"],
            "escape_considered": s["considered"],
            "escape_blocked_margin": s["blocked_margin"],
            "escape_blocked_k": s["blocked_k"],
            "escape_blocked_budget": s["blocked_budget"],
        }


def _zero_stats():
    return {"escapes": 0, "considered": 0, "blocked_margin": 0, "blocked_k": 0,
            "blocked_budget": 0, "bytes": 0, "fetch_s": 0.0, "evictions": 0,
            "peak_live": 0, "margin_sum": 0.0}
