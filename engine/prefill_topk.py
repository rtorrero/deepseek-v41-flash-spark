"""DSV41_PREFILL_TOPK_TEST -- a measurement-only override of the router's top-k, prefill only.

Why it exists. Prefill and decode are bound by different things on this box, and the question
"does a prefill chunk get cheaper if the router activates fewer experts?" decides whether an
ExFold-style top-k reduction in prefill is worth building at all. If the time scales with k the
prefill MoE is compute-bound and the reduction pays; if it is flat the chunk is bandwidth- or
unpack-bound and the levers are elsewhere. `tools/prefill_bound_test.py` asks exactly that, and
this module is the one line of engine code it needs.

Why it is not `DSV41_TOPK`. That switch (tools/v41_ref.py::routed_topk) rewrites
`Args.n_activated_experts` as the config is read, so it changes the checkpoint's routing
everywhere: the decode path's static buffers, the CUDA graphs, the prune fallback table and the
served output all follow it. This one changes nothing but the `k` a PREFILL call passes to
`logits.topk`, and only while a measurement is running.

Off is the default and off is byte-identical: with the variable unset `OVERRIDE` is None and
`prefill_topk` returns the k it was handed, so `engine/model.py::moe` issues the same topk it
always did. `tools/test_prefill_topk.py` pins both halves of that.

Values: an integer 1..k, or unset / "off" / "0" for the shipped behaviour. It cannot be combined
with DSV41_PRUNE_MODE=drop, whose per-column fallback table is sized for the checkpoint's k.
"""

from __future__ import annotations

import os

ENV_VAR = "DSV41_PREFILL_TOPK_TEST"


def parse(raw, default_k: int = 6):
    """The env value -> an override, or None for "leave the routing alone".

    Pure and torch-free so the unit test can exercise every branch on a CPU runner.
    """
    v = (raw or "").strip().lower()
    if v in ("", "off", "0", "none", "default"):
        return None
    try:
        k = int(v)
    except ValueError:
        raise ValueError(f"{ENV_VAR}: {v!r}; use an integer 1..{default_k}, "
                         f"or leave it unset") from None
    if not 1 <= k <= int(default_k):
        raise ValueError(f"{ENV_VAR}: {k}; must be 1..{default_k} "
                         f"(the checkpoint's n_activated_experts)")
    return k


#: None unless the environment asked for an override, read once at import. A harness that wants to
#: sweep k inside ONE process (which is the only way to A/B anything on this box -- see NOTES,
#: "any A/B that changes a resident allocation's size has to be run inside one process") assigns
#: this attribute directly between runs. Nothing in the engine ever writes it.
OVERRIDE = parse(os.environ.get(ENV_VAR), 6)


def prefill_topk(k: int, drop: bool = False) -> int:
    """The number of routed experts a PREFILL call should activate.

    Returns `k` unchanged whenever the override is off, which is every served configuration.
    """
    if OVERRIDE is None:
        return k
    if drop:
        raise ValueError(f"{ENV_VAR} cannot be combined with DSV41_PRUNE_MODE=drop: the "
                         f"per-column fallback table is sized for the checkpoint's k")
    return min(OVERRIDE, k)
