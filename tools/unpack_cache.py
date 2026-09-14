"""unpack_cache.py -- the per-request cache of unpacked FP4 experts.

`tools/cb3_moe.py` owns the kernels; this file owns the POLICY and the bookkeeping, and it is
deliberately free of torch, triton and CUDA at import time so that `tools/test_unpack_cache.py`
can check the policy on any machine. The one place a GPU is touched is `timed`, which imports
torch itself.

See `PrefillUnpackCache` for what the policy is and why it is not an LRU.
"""

from __future__ import annotations

#: One packed-FP4 expert: 2 x (2304 x 2560 + 2304 x 160) + 5120 x 1152 + 5120 x 72 = 18.80 MB.
#: Same arithmetic as `engine/experts.py::EXPERT_BYTES` and `fp4_moe.ExpertArena.bytes_per_slot`,
#: written out so a budget can be taken without allocating an arena.
FP4_BYTES_PER_SLOT = 18_800_640
#: One CB3 expert, `tools/cb3_moe.py::CB3_BYTES_PER_SLOT`, repeated here for the same reason.
CB3_BYTES_PER_SLOT = 14_454_784

#: Estimated GPU time of ONE expert's CB3 -> FP4 unpack, used only to turn `unpack_hits` into a
#: saved-milliseconds figure in the stats. From NOTES 2026-09-11, one layer's kept set of 154
#: experts at T=2048: 74.52 ms through the unpack fallback against 55.31 ms for a same-sized FP4
#: arena, so 19.21 ms / 154 = 0.125 ms an expert. It is replaced by a measurement as soon as the
#: cache has timed `CALIB_BATCHES` real unpacks (`PrefillUnpackCache.timed`), and the stat is
#: labelled an estimate either way.
UNPACK_MS_PER_EXPERT = 0.125
#: How many unpack launches per request get a CUDA-event pair around them. Eight is enough to put
#: the per-expert cost within a few percent and costs two event records each.
CALIB_BATCHES = 8

ZERO_UNPACK_STATS = {"unpack_hits": 0, "unpack_misses": 0, "unpack_bytes_saved": 0,
                     "unpack_ms_saved": 0.0}


class PrefillUnpackCache:
    """Unpacked FP4 experts that survive across the chunks of ONE prompt.

    A CB3 expert cannot be fed to the prefill kernel directly, so `moe_forward_prefill` unpacks
    every expert a chunk touches back into packed FP4 codes and runs the FP4 kernel over them
    (NOTES 2026-09-12: this, not routing, is what prefill costs). The unpack is thrown away at the
    end of the call, so the NEXT chunk of the same prompt unpacks the same experts again -- a
    2,048-token chunk touches nearly every expert the router may pick, in all of layers 0..20, and
    a four-chunk prompt therefore pays the same unpack four times.

    This cache is a second region of the same FP4 scratch arena that is not overwritten between
    calls, plus a map `cb3 arena slot -> fp4 cache slot`.

    **Policy: a fixed layer window, filled on first touch, never evicted inside a request.**
    Chunk order is deterministic -- layers 0, 1, ... 20 for chunk 0, then the same layers again for
    chunk 1 -- and the working set of ONE layer is `ceil(keep * 384)` experts (139 at keep 0.36 =
    2.61 GB of FP4; 384 = 7.22 GB if nothing is pruned). A budget that cannot hold all the layers a
    chunk visits must therefore choose, and LRU is the worst possible choice here: this is the
    textbook sequential-scan pattern, so by the time chunk 1 asks for layer 0 again LRU has just
    evicted it as the oldest entry, and every single reference misses. Admitting on first touch and
    never evicting gives the opposite: the cache fills with layers 0..k-1 (plus a prefix of layer k
    when the budget does not divide evenly), those layers hit on every later chunk, and the layers
    past the window miss every time without ever disturbing the ones that hit. The hit rate is then
    simply `cached layers / layers per chunk`, and it degrades linearly with the budget instead of
    collapsing to zero.

    Admission is also switched OFF for the last chunk of a prompt and for the decoder replay,
    because nothing admitted there is ever read back: an admission costs an 18.8 MB write it would
    otherwise not pay.

    Correctness. The cache is keyed by ARENA SLOT, and an arena slot's contents change in exactly
    one place -- `CB3ArenaV2.load_slot` / `CB2ArenaV2.load_slot` -- which calls `invalidate`. In
    the pruned all-resident configurations this engine serves, that never happens after the warm
    start. In streaming mode (transient ring, LRU eviction) it happens constantly, every entry is
    invalidated before it can be reused, and the cache correctly degenerates to a no-op. Entries
    are not reclaimed one at a time; once half the cache is stale the whole map is flushed, which
    keeps the destination runs contiguous (the unpack kernel writes slot `out0 + b`).
    """

    def __init__(self, slots: int, bytes_per_unpack: int = CB3_BYTES_PER_SLOT + FP4_BYTES_PER_SLOT,
                 calib_batches: int = CALIB_BATCHES):
        self.slots = max(0, int(slots))
        #: what ONE unpack moves: a read of the packed expert plus a write of the FP4 one. 33.26 MB
        #: for CB3 (14.45 + 18.80), 28.79 MB for CB2.
        self.bytes_per_unpack = int(bytes_per_unpack)
        self.map: dict[int, int] = {}      # cb3 arena slot -> fp4 cache slot in [0, slots)
        self.next = 0                      # bump pointer into the cache region
        self.dead = 0                      # entries invalidated since the last flush
        self.admitting = False
        self.stats = dict(ZERO_UNPACK_STATS)
        self._chunks = 0
        self._chunk = -1
        self._calib = int(calib_batches)
        self._ev: list = []
        self._ev_used = 0
        self._ev_experts = 0
        self._ms_per_expert = None

    # -- lifecycle ---------------------------------------------------------
    def begin_request(self, n_chunks: int) -> None:
        """A new prompt. The map cannot outlive a request (see the class docstring), and the
        counters are per request so the stats epilogue reports this prompt."""
        self.flush()
        self.stats = dict(ZERO_UNPACK_STATS)
        self._chunks = max(1, int(n_chunks))
        self._chunk = -1
        self._ev_used = 0
        self._ev_experts = 0
        self._ms_per_expert = None
        self.admitting = False

    def begin_chunk(self) -> None:
        """Called once before each prefill chunk, and once more before the decoder replay."""
        self._chunk += 1
        self.admitting = self.slots > 0 and self._chunk < self._chunks - 1

    def end_request(self) -> None:
        self.admitting = False
        self.flush()

    def flush(self) -> None:
        self.map.clear()
        self.next = 0
        self.dead = 0

    def invalidate(self, slot: int) -> None:
        """The CB3 arena slot `slot` is being overwritten; anything unpacked from it is stale."""
        if self.map.pop(int(slot), None) is not None:
            self.dead += 1
            if self.slots and self.dead * 2 >= self.slots:
                self.flush()

    # -- the lookup a prefill call makes -----------------------------------
    def lookup(self, ids: list) -> tuple:
        """Split the arena slots this call needs into (resident src, resident dst, missing src)."""
        hit_src, hit_dst, fresh = [], [], []
        m = self.map
        for s in ids:
            d = m.get(s)
            if d is None:
                fresh.append(s)
            else:
                hit_src.append(s)
                hit_dst.append(d)
        self.stats["unpack_hits"] += len(hit_src)
        self.stats["unpack_misses"] += len(fresh)
        self.stats["unpack_bytes_saved"] += len(hit_src) * self.bytes_per_unpack
        return hit_src, hit_dst, fresh

    def admit(self, fresh: list) -> tuple:
        """Reserve a contiguous run of cache slots for as many of `fresh` as fit, first-touch-wins.
        Returns (admitted ids, first destination slot)."""
        if not self.admitting:
            return [], 0
        room = self.slots - self.next
        if room <= 0:
            return [], 0
        take = fresh[:room]
        dst0 = self.next
        self.next += len(take)
        for i, s in enumerate(take):
            self.map[s] = dst0 + i
        return take, dst0

    # -- the ms_saved estimate ---------------------------------------------
    def timed(self, fn, n_experts: int):
        """Run one unpack launch, timing the first `CALIB_BATCHES` of the request with CUDA events
        so `unpack_ms_saved` is this box's number rather than a constant from NOTES.

        torch is imported here, not at module scope: everything above this line is arithmetic and
        bookkeeping, and tools/test_unpack_cache.py exercises the policy on a machine that has no
        torch, no triton and no GPU."""
        if self._ev_used >= self._calib or n_experts <= 0:
            return fn()
        import torch  # noqa: PLC0415 - kept out of the import path on purpose
        while len(self._ev) <= self._ev_used:
            self._ev.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        a, b = self._ev[self._ev_used]
        self._ev_used += 1
        self._ev_experts += n_experts
        a.record()
        try:
            return fn()
        finally:
            b.record()

    def ms_per_expert(self) -> float:
        if self._ms_per_expert is not None:
            return self._ms_per_expert
        ms = 0.0
        if self._ev_used and self._ev_experts:
            try:
                for i in range(self._ev_used):
                    a, b = self._ev[i]
                    b.synchronize()
                    ms += a.elapsed_time(b)
            except Exception:  # noqa: BLE001 - an un-recorded event, e.g. an aborted request
                ms = 0.0
        self._ms_per_expert = (ms / self._ev_experts) if ms > 0 else UNPACK_MS_PER_EXPERT
        return self._ms_per_expert

    def summary(self) -> dict:
        """The four counters, for the engine's stats epilogue. `unpack_ms_saved` is an ESTIMATE:
        hits times the measured (or, failing that, the documented) cost of one unpack."""
        s = dict(self.stats)
        s["unpack_ms_saved"] = round(s["unpack_hits"] * self.ms_per_expert(), 1)
        s["unpack_cache_slots"] = self.slots
        s["unpack_cache_used"] = self.next
        return s
