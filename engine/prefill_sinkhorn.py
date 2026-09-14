"""DSV41_PREFILL_FUSED_SINKHORN -- run prefill's Hyper-Connection Sinkhorn on the fused kernel.

The problem, from docs/gemm-dispatch.md (rows 8-9 of the inventory). `v41_ref.hc_mixes` finishes
with the 20-iteration Sinkhorn, and at prefill it runs it through `tiled_rows` in `MM_TILE = 16`
row tiles:

    return tiled_rows(lambda t: hc_split_sinkhorn(t, ...), mixes)

`hc_split_sinkhorn` is ~130 tiny elementwise/reduction kernels on a `[m, 4, 4]` tensor, so one
`hc_mixes` at T = 2048 is 128 tiles x ~130 kernels = ~16,640 launches; `Model.block` calls it
twice per layer over the 21 encoder layers, which is **698,880 launches per 2,048-token chunk**,
every one of them on a 16x4x4 tensor. A launch-bound phase like that is invisible in a
top-20-by-GPU-time list -- each kernel is microscopic -- which is why the audit script reports the
launch count and the GPU-busy fraction next to the kernel table.

The fix is not a new kernel. `engine/hc_sinkhorn.py` already replaces the whole thing with one
Triton launch, one program per token row, and `engine/fastdecode.py` has used it since the fast
decode path existed -- `engine/model.py` simply never picked it up. One program per row makes it
row-count- *and* row-offset-invariant by construction, which is the property `tiled_rows` exists to
manufacture, so the tiling is not merely unnecessary there, it is replaced by something stronger.
Nothing in the kernel assumes decode shapes: the grid is `(max(n, 1),)` with an `if row >= n_rows:
return` guard, `HC` is a `tl.constexpr` of 4 and there is no batch or row limit to extend. With the
flag on the count goes **698,880 -> 42 launches per chunk**, one per `hc_mixes` call.

Why it is off by default. The fused kernel and the torch path do the same arithmetic in a
different order -- the softmax, the 20 alternating normalisations and `tl.exp` against
`torch.exp` -- so the last bits of `pre`/`post`/`comb` are not guaranteed equal, and the shipped
prefill numbers were measured on the tiled path. Decode has run on the fused kernel all along, so
the agreement is not in doubt; it is measured rather than assumed by
`tools/test_hc_sinkhorn_prefill.py`, which compares the two paths at prefill row counts on a CUDA
box and self-skips anywhere else.

Values: `1` / `on` / `true` / `yes` to route prefill through the fused kernel, anything in the off
list (or unset) for the shipped behaviour. The switch is **prefill-only** -- `engine/model.py`
passes it as `fused=... and prefill`, so the un-graphed decode forward keeps the tiled path and the
DSpark verify block's numbers cannot move. Torch-free, like `engine/prefill_topk.py`.
"""

from __future__ import annotations

import os

ENV_VAR = "DSV41_PREFILL_FUSED_SINKHORN"

_OFF = ("", "0", "off", "no", "false", "none", "default")
_ON = ("1", "on", "yes", "true")


def parse(raw) -> bool:
    """The env value -> True (fused) or False (the shipped tiled path).

    Pure and torch-free so the unit test can exercise every branch on a CPU runner.
    """
    v = (raw or "").strip().lower()
    if v in _OFF:
        return False
    if v in _ON:
        return True
    raise ValueError(f"{ENV_VAR}: {v!r}; use 1 / 0 (on / off), or leave it unset")


#: False unless the environment asked for it, read once at import. A harness that wants to A/B
#: inside one process assigns this attribute directly; nothing in the engine ever writes it.
ENABLED = parse(os.environ.get(ENV_VAR))


def fused_prefill(prefill: bool) -> bool:
    """Should THIS `hc_mixes` call take the fused kernel? Only a prefill one, only when asked."""
    return bool(prefill) and ENABLED
