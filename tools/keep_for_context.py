#!/usr/bin/env python3
"""keep_for_context.py -- the keep fraction a context length can afford.

`PRUNE_KEEP` is the one number on this box that has to be chosen against
another number, and the other number is `MAX_SEQ`. The KV and indexer caches
are allocated for the whole context up front, and a prefill chunk's transient
cost grows with the context behind it, so the same arena that serves a 32k
context is over budget at 256k -- and the way it goes over is not a friendly
one. The record:

  * 0.36 / arena 81 GB served a FILLED 256k context (RESULTS.md, 2026-09-13
    22:50 and the 2026-09-14 addenda).
  * 0.40 / arena 89.2 GB serves every short prompt and was killed by the memory
    watchdog 582 s into a 195k-token prefill -- it passes the engine's own
    pre-flight, loads, reports ready, and dies on the request.

So `PRUNE_KEEP=auto` resolves here, before the engine is launched, and the
answer is the largest step on the keep ladder that still fits the context that
was asked for. The arithmetic is not this file's: `tools/budget.py` owns the
memory model (`plan()`, and `Plan.verdict` in particular), and this is the one
question asked of it, with the engine's own pre-flight margin included the way
`Plan.launch_need` mirrors it.

Two things are deliberately NOT done here.

It does not go below what has been gated. Nothing under keep 0.36 has ever been
put through a generation gate on this box (`results/keepsets/*/GATE.md`), so a
resolved value below that is reported with a warning and served anyway: a
smaller keep-set is a coverage question, not a memory one, and clamping the
memory answer upward would hand back a configuration the watchdog kills.

It does not second-guess `ARENA_GB`. A pinned arena is a pinned arena; when one
is set the answer is the largest step that fits INSIDE it (slots minus the
transient ring), and if the pinned arena itself will not fit the box, that is
said rather than quietly corrected.

  python3 tools/keep_for_context.py --max-seq 262144
  PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144 (fits with 3.3 GB spare)

  python3 tools/keep_for_context.py --max-seq 262144 --bare
  0.36

No torch, no CUDA, no model load -- `./start.sh` calls this on every start and
it has to cost nothing.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

# The ladder. Two points of keep is 307 experts per layer-set and about 4 GB of
# arena, which is fine enough to spend the box's spare memory on and coarse
# enough that a resolved value is a number somebody has typed before.
# `tools/tune.py` slides the keep along the same steps, and imports them from
# here so that the screen and the launcher can never land between each other's
# rungs.
KEEP_STEPS = [round(0.02 * i, 2) for i in range(3, 31)]          # 6 % .. 60 %

# The smallest keep fraction any generation gate has ever been run at on this
# box. Below it the memory model still answers, and nothing says what the
# output looks like. A warning, never a clamp -- see the module docstring.
GATED_FLOOR = 0.36

AUTO = "auto"


@dataclass
class Resolution:
    """What `auto` came to, and everything needed to explain it."""
    keep: float
    max_seq: int
    plan: B.Plan
    pinned: bool = False          # the user gave a number; nothing was resolved
    fits: bool = True
    spare: float = 0.0            # GB of margin on the tighter of the two gates
    cap: float | None = None      # the keep ARENA_GB alone allows, when pinned
    notes: list = field(default_factory=list)

    @property
    def why(self) -> str:
        head = "pinned; " if self.pinned else ""
        if self.fits:
            return f"{head}fits with {self.spare:.1f} GB spare"
        return f"{head}{-self.spare:.1f} GB short of what a filled context needs"

    @property
    def line(self) -> str:
        """The one line `./start.sh` prints, and the tool's own stdout."""
        given = f"{self.keep:.2f}" if self.pinned else AUTO
        return (f"PRUNE_KEEP={given} -> {self.keep:.2f} "
                f"for MAX_SEQ={self.max_seq} ({self.why})")


def _plan(host, keep, max_seq, fmt, arena_gb, transient_slots, keep_free_gb):
    return B.plan(host, None, (), keep, max_seq, fmt=fmt, arena_gb=arena_gb,
                  transient_slots=transient_slots, keep_free_gb=keep_free_gb)


def _spare(p: B.Plan) -> float:
    """GB of margin on whichever gate is tighter.

    Both have to hold and they fail differently: the launcher's pre-flight
    refuses to start, and the serving margin is the one the watchdog enforces
    on the first request. The smaller of the two is what "spare" can honestly
    mean.
    """
    return min(p.launch_slack, p.free_after_load - p.need_free)


def arena_cap(arena_gb: float, fmt: str, transient_slots: int) -> tuple:
    """(slots, the largest keep a pinned arena can hold). The arena has to hold
    the kept set PLUS the ring that prefill misses go through, which is why the
    transient slots come off the top -- a kept set larger than the LRU streams
    its tail from NVMe on every step, which is exactly what an all-resident
    keep-set exists to prevent (engine/experts.py)."""
    slots = int(arena_gb * B.GB / B.EXPERT_BYTES[fmt])
    return slots, max(0.0, (slots - transient_slots) / B.N_ROUTED)


def resolve(host: B.Host, max_seq: int, keep=AUTO, fmt: str = "cb3",
            arena_gb: float | None = None,
            transient_slots: int = B.TRANSIENT_SLOTS_DEFAULT,
            keep_free_gb: float = B.KEEP_FREE_GB_DEFAULT,
            steps=KEEP_STEPS) -> Resolution:
    """The keep fraction to serve `max_seq` with on this host.

    `keep` is either a number -- passed through untouched, because a pinned
    value is a decision somebody made and this tool's job is then only to say
    what it costs -- or "auto", in which case it is the largest step that fits.
    """
    notes = []
    if host.note:
        notes.append(host.note)

    cap = None
    if arena_gb:
        slots, cap = arena_cap(arena_gb, fmt, transient_slots)
        notes.append(f"ARENA_GB={arena_gb:g} is {slots:,} {fmt} slots, "
                     f"{slots - transient_slots:,} of them outside the transient ring, "
                     f"which caps the keep at {cap:.2f}")

    pinned = not (isinstance(keep, str) and keep.strip().lower() == AUTO)
    if pinned:
        value = float(keep)
    else:
        # The largest STEP that fits, not the continuous ceiling: the engine
        # rounds the per-layer count UP (`ceil(keep x 384)`, budget.keep_n), so
        # a plan at the continuous maximum is already over it. Walk the ladder
        # down until one actually fits. `tools/tune.py` picks its profile
        # ceiling the same way and must agree with this.
        def ok(k):
            return _plan(host, k, max_seq, fmt, arena_gb, transient_slots,
                         keep_free_gb).verdict != "over"

        # With ARENA_GB pinned the memory verdict does not move with the keep
        # fraction at all -- the arena is the arena -- so walking the ladder
        # down cannot rescue an arena that is too big for the context. Answer
        # with the most the arena holds and say that, rather than handing back
        # a 0.06 that is just as over budget and serves nothing.
        within = [k for k in steps if cap is None or k <= cap + 1e-9] or [steps[0]]
        value = next((k for k in reversed(within) if ok(k)), None)
        if value is None and cap is not None:
            value = within[-1]
            notes.append(f"ARENA_GB={arena_gb:g} does not fit MAX_SEQ={max_seq} on this box at "
                         f"any keep fraction -- the arena is pinned, so the keep cannot fix it; "
                         f"unset ARENA_GB, or lower it, or shorten MAX_SEQ")
        elif value is None:
            # Nothing on the ladder fits. Report the bottom rung and how far
            # short it is; the caller decides whether that is fatal.
            value = within[0]
            notes.append(f"nothing on the keep ladder fits MAX_SEQ={max_seq} on this box; "
                         f"{value:.2f} is the smallest step there is")

    p = _plan(host, value, max_seq, fmt, arena_gb, transient_slots, keep_free_gb)
    spare = _spare(p)
    if value < GATED_FLOOR - 1e-9:
        notes.append(f"keep {value:.2f} is below {GATED_FLOOR:.2f}, and nothing below "
                     f"{GATED_FLOOR:.2f} has been gated on this box "
                     f"(results/keepsets/*/GATE.md) -- the memory fits, the generation "
                     f"is unmeasured")
    if cap is not None and value > cap + 1e-9:
        notes.append(f"keep {value:.2f} needs {B.keep_n(value) * B.N_LAYERS:,} resident experts "
                     f"and the pinned arena holds {p.slots - transient_slots:,}; the engine will "
                     f"keep fewer than asked")
    return Resolution(keep=value, max_seq=max_seq, plan=p, pinned=pinned,
                      fits=p.verdict != "over", spare=spare, cap=cap, notes=notes)


# --- CLI --------------------------------------------------------------------

def _env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="the largest keep fraction a context length can afford on this box",
        epilog="defaults come from the environment ./start.sh has already sourced .env into")
    ap.add_argument("--max-seq", type=int, default=int(_env("MAX_SEQ", 32768)),
                    help="the context the caches are allocated for (default %(default)s)")
    ap.add_argument("--keep", default=_env("PRUNE_KEEP", AUTO), metavar="VALUE",
                    help=f"'{AUTO}' to resolve, or a number to pass through and cost "
                         f"(default %(default)s)")
    ap.add_argument("--format", default=_env("EXPERT_FORMAT", "cb3"), choices=("cb3", "fp4"),
                    help="the arena's expert layout (default %(default)s)")
    ap.add_argument("--arena-gb", type=float, default=_env("ARENA_GB"), metavar="GB",
                    help="a pinned arena; the answer then has to fit inside it")
    ap.add_argument("--transient-slots", type=int,
                    default=int(_env("TRANSIENT_SLOTS", B.TRANSIENT_SLOTS_DEFAULT)),
                    help="prefill slots outside the LRU (default %(default)s)")
    ap.add_argument("--keep-free-gb", type=float,
                    default=float(_env("KEEP_FREE_GB", B.KEEP_FREE_GB_DEFAULT)),
                    help="host memory the launcher leaves free (default %(default)s)")
    ap.add_argument("--bare", action="store_true",
                    help="print the number alone, for a shell to capture")
    ap.add_argument("--arena", action="store_true",
                    help="print the arena in GB that the resolved keep needs, and nothing else. "
                         "The engine's own automatic sizing takes 82 %% of what is free, which on "
                         "this box is more arena than a filled context can afford; sizing it to "
                         "the kept set instead is what makes an unpinned ARENA_GB safe")
    a = ap.parse_args(argv)

    keep = str(a.keep).strip()
    if keep.lower() != AUTO:
        try:
            v = float(keep)
        except ValueError:
            print(f"PRUNE_KEEP must be a number or '{AUTO}' (got {keep!r})", file=sys.stderr)
            return 2
        if not 0 < v <= 1:
            print(f"PRUNE_KEEP must be in (0, 1] (got {v})", file=sys.stderr)
            return 2
        keep = v
    else:
        keep = AUTO
    if a.max_seq <= 0:
        print(f"MAX_SEQ must be positive (got {a.max_seq})", file=sys.stderr)
        return 2

    r = resolve(B.read_host(), a.max_seq, keep, fmt=a.format,
                arena_gb=float(a.arena_gb) if a.arena_gb else None,
                transient_slots=a.transient_slots, keep_free_gb=a.keep_free_gb)
    for n in r.notes:
        print(f"# {n}", file=sys.stderr)
    if a.arena:
        # ceil, not round: the arena has to HOLD the kept set plus the ring.
        print(f"{math.ceil(r.plan.arena)}")
    else:
        print(f"{r.keep:.2f}" if a.bare else r.line)
    # Non-zero only when the answer will not serve: a pinned number that is over
    # budget, or a context this box cannot hold at any keep fraction. `auto` on
    # a box with room is the quiet path.
    return 0 if r.fits else 1


if __name__ == "__main__":
    sys.exit(main())
