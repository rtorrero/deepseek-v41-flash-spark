"""A kept set that does not fit the arena is a refusal, not a warning.

Twice on 2026-09-15 a server came up with more kept experts than LRU slots (a Writing keep-set at
0.40 on an arena sized for 0.36, and a calibration run on a 60 GB arena) and answered every
request with HTTP 500: a prefill chunk routes to the whole kept set of a layer, the tail that is
not resident has to come through the transient ring, and the ring has eight slots. The warning
that used to be printed here scrolled past in a start log nobody reads while a gate recorded
eight failures against the wrong cause. Torch-free so the rule is under test on a CPU runner.
"""
from __future__ import annotations

import os

ENV_ALLOW = "DSV41_ALLOW_STREAMING_TAIL"


def check_fit(n_kept: int, lru_slots: int, env=None) -> str | None:
    """None when the kept set fits (or the tail is explicitly allowed); otherwise the message to
    refuse with. Pure: the caller decides how to fail."""
    if n_kept <= lru_slots:
        return None
    env = os.environ if env is None else env
    if (env.get(ENV_ALLOW) or "").strip() in ("1", "true", "yes"):
        return None
    return (f"pruned set {n_kept} experts > {lru_slots} LRU slots. A prefill chunk needs the whole "
            f"kept set of a layer resident and the transient ring cannot carry the tail, so every "
            f"request would fail with 'transient ring exhausted'. Raise ARENA_GB (or leave it empty "
            f"so ./start.sh sizes it for the keep), lower PRUNE_KEEP, or set {ENV_ALLOW}=1 to serve "
            f"the streaming tail anyway.")
