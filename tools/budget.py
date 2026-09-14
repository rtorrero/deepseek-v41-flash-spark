"""budget.py -- what a topic selection costs on this box.

The engine refuses to start when the arena plus its working set will not fit
(engine/v41_engine.py, the pre-flight around `pack_scratch`), and finding that
out costs a three-minute load. This module answers the same question in a
millisecond, from the same arithmetic, so a configuration can be chosen before
it is paid for.

Three things are computed here.

MEMORY is exact. Every term below is either a shape out of the checkpoint's own
config or a constant the engine itself uses, and the total is compared against
`MemAvailable` the way the engine compares it.

COVERAGE is the interesting one. A keep-set is a cache policy learned from a
workload sample: per layer, only the top-N experts stay routable. Coverage is
the fraction of a topic's measured routing that lands on an expert that stayed.
It is the number that predicts whether generation holds together -- a topic the
trace never saw routes off the keep-set and the output degenerates, which is
exactly what a 0.03 coverage on markup did before the corpus was widened.

So the useful reading is not "how many topics" but "what is the weakest
selected topic's coverage, and what keep fraction does it need". Fewer topics
do not make a step faster on their own: step time is set by the bytes of the
experts a token activates, and that does not change. Fewer topics reach a given
coverage at a LOWER keep fraction, and a lower keep fraction is a smaller
arena -- which is where the memory, and the context window, come from.

THE TAIL is the third, and it exists because coverage stopped resolving anything
once the box was served on `saliency`: half a layer's saliency mass sits on one
expert, so a keep-set swap of 513 resident experts moved every bar by 0.005 and
flipped a generation gate from 6 of 10 to 3 of 10. `TopicIndex.tail_curves`
measures the same keep-set in PICKS rather than magnitude and reports what a
TOKEN loses to it -- how many of its six picks in a layer are not resident, and
how often all six are. Across the ten shipped profiles that spans 2.6x where the
coverage bar spans 1.03x. tools/tail_metric.py ranks both against the gates.

Which experts a selection keeps also depends on HOW the selected topics are
combined into one ranking -- the engine's DSV41_PRUNE_RANK. All three rules it
accepts are reimplemented here, because a screen that ranks by `sum` while the
engine ranks by `maxmin` reports coverage the engine will not deliver: over
{english, html, python, reasoning, css, javascript, typescript} at keep 0.36
that is english 0.522 by one rule and 0.676 by the other, on the same budget.
tools/test_budget_rank.py lifts the engine's own function out and holds the two
to the same keep-set.

No torch, no CUDA, no model load. numpy if it is there, plain Python if not.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field

try:
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None

# --- constants, each with where it comes from -------------------------------

N_LAYERS = 40
N_EXPERTS = 384          # routed experts per layer (config.json n_routed_experts)
N_ROUTED = N_LAYERS * N_EXPERTS   # 15,360
TOPK = 6                 # routed experts per token (config.json n_activated_experts)

# One expert as it sits in an arena slot.
#   fp4: 3 x (2304x2560 weights + 2304x160 UE8M0 scales) -- engine/experts.py EXPERT_BYTES
#   cb3: the 3-bit row-codebook layout -- tools/cb3_moe.py CB3_BYTES_PER_SLOT
EXPERT_BYTES = {"fp4": 18_800_640, "cb3": 14_454_784}

# The DSpark drafter's own experts: 3 MTP blocks x 128, always fully resident,
# always in the fp4 layout (engine/v41_engine.py loads them into dspark_arena).
DSPARK_BYTES = 3 * 128 * EXPERT_BYTES["fp4"]

# Everything that is not a routed expert: attention, shared experts, embeddings,
# the LM head, the Engram projections. Measured at load ("N GiB allocated after
# weights"), so it moves with DSV41_DENSE_FP4 and DSV41_HEAD_FMT.
DENSE_BYTES = {
    ("attn,wo_a", "fp8"): 7.61e9,   # measured 2026-09-12: "7.09 GiB allocated after weights"
    ("attn,wo_a", "bf16"): 8.94e9,  # + the bf16 head (129280x5120x2 = 1.32 GB)
    ("", "fp8"): 18.1e9,            # measured 2026-09-11 before the dense fp4 work
    ("", "bf16"): 19.4e9,
}
DENSE_DEFAULT = 7.61e9


def dense_key_from_env() -> tuple:
    """Which dense-weight figure applies, from the same two variables start.sh
    reads. Hardcoding the shipped pair understated resident memory by up to
    11.8 GB for anyone who changed either one in .env, which is enough to make
    the verdict wrong rather than merely imprecise."""
    groups = ",".join(sorted(g for g in os.environ.get("DSV41_DENSE_FP4", "attn,wo_a").split(",") if g))
    return (groups, os.environ.get("DSV41_HEAD_FMT", "fp8"))


# How several topics are combined into one ranking of the same budget. The
# engine's own three, with the engine's own default.
RANKS = ("sum", "max", "maxmin")
RANK_DEFAULT = "sum"


def rank_from_env() -> str:
    """The ranking rule the engine will use, from the variable it reads. A box
    whose .env says maxmin must not be shown sum's coverage: the two disagree by
    more than a tenth on the weakest topic, which is the whole number this tool
    exists to report. Returned as written, unvalidated, so that a typo is
    refused by the caller rather than silently served as the default."""
    return (os.environ.get("DSV41_PRUNE_RANK") or RANK_DEFAULT).strip() or RANK_DEFAULT


# WHICH measurement the histograms are, before any rule combines them.
# `counts` is how often each expert was picked; `saliency` is how much it
# contributed -- the summed gate weight x expert-output norm, REAP's criterion
# (Lasby et al., Cerebras, ICLR 2026, arXiv 2510.13999). Both are 384 numbers per
# layer and every rule above treats them the same, so this is orthogonal to
# RANKS: a coverage bar is one (source, rank) pair, and neither may differ from
# the engine's or the bar promises routing the server will not keep.
SOURCES = ("counts", "saliency")
SOURCE_DEFAULT = "counts"


def source_from_env() -> str:
    """The histogram family the engine will rank on (DSV41_PRUNE_SOURCE).
    Unvalidated, for the same reason as rank_from_env."""
    return (os.environ.get("DSV41_PRUNE_SOURCE") or SOURCE_DEFAULT).strip() or SOURCE_DEFAULT

# The KV and indexer caches are allocated for MAX_SEQ up front (engine/model.py
# Caches). Per token: for every kv_source_layer, one compressed-KV row of
# head_dim and one index row of index_head_dim, both bf16, at that layer's
# compression ratio.  ratios {2:2, 8:2, 14:2, 20:1}, head_dim 512, index 128:
#   (3 x 1/2 + 1) x (512 + 128) x 2 = 3,200 bytes per token.
KV_BYTES_PER_TOKEN = 3200
# The sliding-window rings do not scale with MAX_SEQ: RING 4096 x head_dim 512
# x bf16 x (40 layers + 3 MTP).
RING_DEFAULT = 4096
WINDOW_BYTES_PER_SLOT = 43 * 512 * 2      # 44,032 B, i.e. 44.0 MB per 1,000 slots
WINDOW_BYTES = WINDOW_BYTES_PER_SLOT * RING_DEFAULT


def ring_from_env(chunk: int | None = None) -> int:
    """DSV41_RING, or engine/model.py's default for the chunk in force. The ring
    has to be longer than window_size + the chunk (the window gather runs after
    the whole chunk is in the ring), so raising the chunk raises this too."""
    r = os.environ.get("DSV41_RING")
    if r:
        return int(r)
    return max(RING_DEFAULT, (chunk if chunk is not None else chunk_from_env()) + 512)


def window_bytes(ring: int | None = None) -> float:
    return WINDOW_BYTES_PER_SLOT * (ring_from_env() if ring is None else ring)

# What the warm start needs on top of the arena while it packs experts into it,
# and the floor the launcher keeps free. Both are the engine's own numbers.
PACK_SCRATCH_BYTES = {"cb3": 3e9, "fp4": 1e9}
KEEP_FREE_GB_DEFAULT = 6.0

# Prefill misses go through a small ring of slots instead of the LRU, so the
# arena has to hold the kept set PLUS that ring: engine/experts.py sets
# lru_slots = n_slots - transient_slots, and a kept set larger than lru_slots
# streams its tail from NVMe on every step -- which is exactly the property a
# fully resident keep-set exists to buy.
TRANSIENT_SLOTS_DEFAULT = 8

PREFILL_CHUNK_DEFAULT = 2048
N_INDEX_LAYERS = 8               # config.json index_source_layers

# What one prefill chunk needs on top of everything resident. This is the term
# that decides whether a configuration serves or gets killed, and it is
# measured, not derived: at MAX_SEQ 32768 and chunk 2048 an arena of 87 GB left
# 16.5 GB free and served; 98 GB left 5.5 GB and the memory watchdog killed the
# process on the first request, with MemAvailable at 0.4 GB.
#
#   2026-09-12 14:07  "FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB
#                      floor for 3.0 s"   (arena 98.0 GB, keep 0.44)
#
# Measured 2026-09-12 by prefilling prompts of 8k to 128k tokens in one load at
# max_seq 262144 and tracking the low-water mark of MemAvailable throughout:
#
#     context   prefill held
#         8k        7.3 GB
#        16k        7.4 GB
#        32k        7.5 GB
#        64k        7.5 GB
#       128k        9.2 GB
#
# Flat to 64k, then a step. A linear fit over the whole range gives 7.2 GB of
# chunk cost plus 15.1 KB per token of context, which over-states the flat
# region and is the safe direction to be wrong in. It also explains the kill
# that started all this: that configuration left 5.5 GB against a 7.5 GB need.
PREFILL_BYTES_PER_TOKEN = 7.2e9 / PREFILL_CHUNK_DEFAULT

# And how much that grows with the CONTEXT. The indexer's score tiles are shaped
# [chunk x compressed positions] and the compressed cache is half the sequence,
# so a longer context makes every chunk more expensive.
#
# This is the rate AT A 2,048-TOKEN CHUNK, and `prefill_bytes` scales it with the
# chunk, because the tile's first axis IS the chunk: a 4,096-token chunk pays
# 15.1 KB per context token twice. Pricing it as a constant is exactly how a
# 4,096-token chunk gets waved through and then killed at 32k.
PREFILL_BYTES_PER_CONTEXT_TOKEN = 15.1 * 1024

# What DSV41_PREFILL_KV_FP8=1 takes off the per-token term (engine/model.py,
# `_gather_kv_fp8`). At any T the bf16 attention path holds three buffers at the
# same moment -- the gathered window (window_size 128 x head_dim 512 x bf16 =
# 128 KB a token), the gathered compressed rows (index_topk 512 x 512 x bf16 =
# 512 KB) and the concatenation of both (640 KB) -- 1,280 KB a token. The fp8
# path gathers straight into one e4m3 buffer of 320 KB a token, so 960 KB a
# token goes away.
#
# That is the LIVE-byte saving. The 7.2 GB above is an allocator high-water mark
# and carries roughly a 2x multiplier over live bytes, which would make the
# saving ~2 MB a token -- so this constant claims the smaller of the two
# numbers on purpose. Being pessimistic here costs expert slots; being
# optimistic costs the process. tools/verify_prefill_fp8.sh measures it.
PREFILL_KV_FP8_SAVED_PER_TOKEN = 960 * 1024


def chunk_from_env() -> int:
    """DSV41_PREFILL_CHUNK, the same variable engine/model.py reads."""
    return int(os.environ.get("DSV41_PREFILL_CHUNK") or PREFILL_CHUNK_DEFAULT)


def kv_fp8_from_env() -> bool:
    """DSV41_PREFILL_KV_FP8. Off by default, and with it off the engine's
    prefill path is byte-for-byte what it is today."""
    return (os.environ.get("DSV41_PREFILL_KV_FP8") or "0") == "1"

# The watchdog kills the process below this, so a configuration has to leave the
# prefill reserve AND this on top of it, not one or the other.
WATCHDOG_FLOOR_GB = 2.5

# Resident bytes nothing above accounts for: the Engram tables' device buffers
# and row caches, the allocator's own retained blocks, the tokenizer, the
# process. Measured twice on a GB10 by comparing what this model predicted
# against what the machine reported once the server was up and settled:
#
#   arena 90 GB, 128k context : predicted 13.2 GB free, actual 9.6   -> 3.6
#   arena 60 GB, 256k context : predicted ~1.8 GB more than observed -> 1.8
#
# Take the larger. Being 3.6 GB pessimistic costs about 250 experts; being
# 3.6 GB optimistic hands out a configuration whose first request is killed.
UNMODELLED_RESIDENT_GB = 3.6

# Contexts that have been loaded and generated from on a GB10. Above the last
# one the KV arithmetic still holds, but the prefill path has not been run
# there: a 64k attempt tripped the memory watchdog at an arena that had room
# for the cache many times over, because the indexer's score tiles grow with
# the compressed cache and that term is not characterised yet. So the tool
# marks those lengths rather than predicting them.
VALIDATED_MAX_SEQ = 131072   # prefilled and measured at this length, 2026-09-12

GB = 1e9


# The prefill unpack cache (DSV41_PREFILL_UNPACK_CACHE_GB, default 0 = off).
# A CB3 expert has to be unpacked to packed FP4 before a prefill-sized kernel
# can read it, and the unpack is thrown away at the end of the call -- so chunk
# 1 of a prompt unpacks exactly the experts chunk 0 already unpacked. The cache
# is a region of the FP4 scratch arena that is NOT overwritten between chunks;
# it is persistent for the process, so it is a straight subtraction from what a
# prefill has to fit in.
#
# What a layer costs, at 18,800,640 B per FP4 expert:
#
#     experts kept per layer   a full layer of cache
#              384 (unpruned)          7.22 GB
#              154 (keep 0.40)         2.90 GB
#              139 (keep 0.36)         2.61 GB
#              123 (keep 0.32)         2.31 GB
#
# and a chunk visits layers 0..20 under `DSV41_SWA_REPLAY=1`, so holding every
# layer a chunk touches would be 21x that -- 55 GB at keep 0.36, which no
# configuration on this box has. The cache is therefore a fixed window of the
# first `budget / layer` layers (tools/unpack_cache.py::PrefillUnpackCache), and its
# hit rate is that ratio.
UNPACK_SLOT_BYTES = EXPERT_BYTES["fp4"]


def unpack_cache_gb_from_env() -> float:
    v = _env_gb("DSV41_PREFILL_UNPACK_CACHE_GB")
    return (v / GB) if v else 0.0


def unpack_cache_bytes(cache_gb: float = 0.0) -> float:
    """What the engine will actually take for the cache: whole experts only."""
    return (max(0.0, cache_gb) * GB // UNPACK_SLOT_BYTES) * UNPACK_SLOT_BYTES


def unpack_cache_layers(cache_gb: float, keep: float) -> float:
    """How many layers of one prompt's working set that budget holds."""
    per_layer = keep_n(keep) * UNPACK_SLOT_BYTES
    return unpack_cache_bytes(cache_gb) / per_layer if per_layer else 0.0


def prefill_bytes(max_seq: int = 32768, chunk: int = PREFILL_CHUNK_DEFAULT,
                  cache_gb: float = 0.0, kv_fp8: bool = False) -> float:
    """Peak transient memory of one prefill chunk -- the reserve a configuration
    must leave free, or the watchdog kills the server on the first request --
    plus the prefill unpack cache, which is not transient but is allocated on
    top of everything the resident total already names.

    Both of the transient terms are proportional to the chunk: the first by
    definition, the second because the indexer's score tile is
    [chunk x compressed positions]. `kv_fp8` (DSV41_PREFILL_KV_FP8) takes the
    fp8 gather's saving off the per-token term; `cache_gb`
    (DSV41_PREFILL_UNPACK_CACHE_GB) adds the persistent unpack cache on top.
    Kept identical to engine/v41_engine.py `prefill_reserve_bytes`;
    tools/test_budget.py fails if the two drift."""
    per_token = PREFILL_BYTES_PER_TOKEN - (PREFILL_KV_FP8_SAVED_PER_TOKEN if kv_fp8 else 0.0)
    return (chunk * per_token
            + max_seq * PREFILL_BYTES_PER_CONTEXT_TOKEN * (chunk / PREFILL_CHUNK_DEFAULT)
            + unpack_cache_bytes(cache_gb))


def kv_bytes(max_seq: int, ring: int | None = None) -> float:
    """Exact: the caches engine/model.py allocates up front for MAX_SEQ."""
    return max_seq * KV_BYTES_PER_TOKEN + window_bytes(ring)


# --- the box ----------------------------------------------------------------

@dataclass
class Host:
    name: str
    total_bytes: float
    available_bytes: float
    is_spark: bool
    cores: int = 0
    note: str = ""
    busy: str = ""      # a process already holding an arena, if there is one
    boxes: int = 1      # machines pooled by DSV41_BOXES; > 1 is sizing, not serving
    checked: float = 0.0    # when `busy` was last probed

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    @property
    def available_gb(self) -> float:
        return self.available_bytes / GB


def _meminfo() -> tuple:
    total = avail = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) * 1024
    except OSError:
        pass
    return total, avail


ARENA_SCRIPTS = ("v41_engine.py", "app.py", "expert_trace.py", "engram_rows.py")


def _busy(exclude_pid: int | None = None) -> str:
    """A process already holding an arena, if there is one. This box is
    single-tenant: two ~88 GB arenas wedge it past the point where sshd can
    fork (docs/gotchas.md).

    Read argv out of /proc rather than matching a pgrep pattern against whole
    command lines -- a shell *watching* for these names matches such a pattern
    and is not an engine. What counts is a Python interpreter whose script
    argument is one of them.
    """
    me = exclude_pid if exclude_pid is not None else os.getpid()
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return ""
    for pid in pids:
        if pid == me:
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        argv = [a for a in argv if a]
        if len(argv) < 2 or "python" not in os.path.basename(argv[0]):
            continue
        script = next((a for a in argv[1:] if os.path.basename(a) in ARENA_SCRIPTS), None)
        if script:
            return f"{os.path.basename(script)} (pid {pid})"
    return ""


def refresh(host: Host, busy_every: float = 4.0) -> Host:
    """Re-read what changes while the screen is open. MemAvailable moves on its
    own, so a budget checked against a startup reading is already stale;
    /proc/meminfo is two lines and costs nothing per frame. The process probe
    forks, so it is throttled."""
    if not (_env_gb("DSV41_HOST_TOTAL_GB") or _env_gb("DSV41_HOST_AVAIL_GB") or host.boxes > 1):
        total, avail = _meminfo()
        if total:
            host.total_bytes, host.available_bytes = total, avail
    now = time.monotonic()
    if now - host.checked >= busy_every:
        host.busy = _busy()
        host.checked = now
    return host


# The machine, overridable. The defaults come from /proc, but a keep-set has to
# be sized for the box it will run on, which is often not the box you are
# sitting at. These let you plan for one:
#
#   DSV41_HOST_TOTAL_GB   physical memory, in GB
#   DSV41_HOST_AVAIL_GB   memory free for the engine, in GB
#   DSV41_BOXES           how many such machines (see below)
#   DSV41_HOST_NAME       what to call it on screen
#
# DSV41_BOXES multiplies the memory and says so. It answers "what could two of
# these hold", which is a real question: the whole expert set is 222 GB in the
# 3-bit format, so two 121 GiB machines hold all of it and no keep-set is
# needed at all. What it does NOT do is make this engine serve across them --
# there is no pipeline or expert parallelism here, one process, one machine.
# Treat a BOXES > 1 answer as sizing for a system you would still have to build.
def _env_gb(name):
    v = os.environ.get(name)
    try:
        return float(v) * GB if v else None
    except ValueError:
        return None


def read_host() -> Host:
    """What this machine has, right now, unless the environment describes a
    different one. Linux reads /proc; anything else is a stand-in so the tool
    still runs where it is being edited."""
    total, avail = _meminfo()
    name, is_spark = platform.node(), False
    for p in ("/proc/device-tree/model", "/sys/devices/virtual/dmi/id/product_name"):
        try:
            m = open(p, "rb").read().decode("utf-8", "replace").strip("\x00 \n")
            if m:
                name = m
                break
        except Exception:  # noqa: BLE001
            pass
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=4).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    if gpu:
        name = gpu if gpu.lower() not in name.lower() else name
    is_spark = bool(re.search(r"GB10|DGX Spark|GX10", f"{name} {gpu}", re.I))
    note = "" if is_spark else "not a GB10 -- numbers are the model's, not this machine's"
    busy = _busy()
    if not total:  # not Linux: show the Spark so the arithmetic is still the real one
        total, avail = 130.6e9, 117.0e9
        note = "no /proc/meminfo here; showing a GB10's 121 GiB"

    t_over, a_over = _env_gb("DSV41_HOST_TOTAL_GB"), _env_gb("DSV41_HOST_AVAIL_GB")
    if t_over:
        total = t_over
        avail = a_over or t_over * (avail / total if total else 0.9)
    elif a_over:
        avail = a_over
    if t_over or a_over:
        note = "memory from the environment, not this machine"

    try:
        boxes = max(1, int(os.environ.get("DSV41_BOXES", "1")))
    except ValueError:
        boxes = 1
    if boxes > 1:
        total, avail = total * boxes, avail * boxes
        note = (f"{boxes} machines pooled: sizing only, this engine serves from one "
                f"(no pipeline or expert parallelism)")

    return Host(name=os.environ.get("DSV41_HOST_NAME") or name, total_bytes=total,
                available_bytes=avail, is_spark=is_spark, cores=os.cpu_count() or 0,
                note=note, busy=busy, boxes=boxes, checked=time.monotonic())


# --- topics -----------------------------------------------------------------

def _cumsum_desc(counts_by_layer, order_by_layer, topic_counts):
    """For one topic: curve[n] = routing mass of that topic captured by keeping
    the top-n experts of every layer. Pure Python, fine at 40x384."""
    curve = [0.0] * (N_EXPERTS + 1)
    for L in range(N_LAYERS):
        c = topic_counts[L]
        acc = 0.0
        for n, e in enumerate(order_by_layer[L], start=1):
            acc += c[e]
            curve[n] += acc
    return curve


def _normalised(counts):
    """One layer's counts of one topic, as fractions of their own sum.

    numpy where it is there, and not for speed: the engine divides by
    `np.asarray(c).sum()`, whose pairwise summation can differ from a sequential
    one in the last bit, and a keep-set that has to match the engine's expert
    for expert is not the place to be one ulp apart."""
    if _np is not None:
        c = _np.asarray(counts, dtype=_np.float64)
        tot = c.sum()
        return c / tot if tot > 0 else c
    s = sum(counts)
    return [x / s for x in counts] if s > 0 else list(counts)


def _total(values) -> float:
    return float(values.sum()) if _np is not None and hasattr(values, "sum") else float(sum(values))


def _desc(values) -> list:
    """Expert ids by descending value, ordered the way the engine orders them.

    `np.argsort(x)[::-1]`, because that is the call in engine/v41_engine.py and
    two experts of exactly equal mass have to be admitted in the same order
    here. Without numpy the tie goes to the lower id instead, which can only
    move experts whose counts are identical."""
    if _np is not None:
        return [int(e) for e in _np.argsort(_np.asarray(values, dtype=_np.float64))[::-1]]
    return sorted(range(len(values)), key=lambda e: (-values[e], e))


def _maxmin_order(per_norm: list, n_experts: int = N_EXPERTS) -> list:
    """One layer's experts in the order DSV41_PRUNE_RANK=maxmin admits them.

    Step for step the engine's `_maxmin_counts`: each slot goes to whichever
    selected topic currently has the least of its routing mass covered, and an
    expert admitted for one topic counts for every topic that also routes to it,
    so overlap is paid for once and the topics converge on a common coverage
    instead of a spread. `per_norm` is one normalised histogram per topic, in
    the order the topics are written into EXPERT_TOPICS -- two topics tied on
    coverage are served in that order by both.

    The whole order is returned, not a budget's worth, because the admission
    decision never reads the budget: the keep-set at ceil(frac * 384) experts is
    the first ceil(frac * 384) of this list, so one pass serves every point of a
    coverage curve and the curve stays monotone in the keep fraction.
    """
    n_t = len(per_norm)
    order = [_desc(p) for p in per_norm]
    ptr = [0] * n_t
    # A topic with no mass in this layer would otherwise be the least covered
    # forever and hand every slot to its argsort of zeros; it has nothing to ask
    # for, so it does not vote here.
    got = [0.0 if _total(p) > 0 else float("inf") for p in per_norm]
    if all(g == float("inf") for g in got):
        return _desc([0.0] * n_experts)      # what the engine's zero score vector cuts to
    admitted, seen = [], set()
    while len(admitted) < n_experts:
        t = min(range(n_t), key=lambda i: got[i])
        while ptr[t] < n_experts and order[t][ptr[t]] in seen:
            ptr[t] += 1
        if ptr[t] >= n_experts:
            # this topic has nothing left to ask for; take it out of the running
            got[t] = float("inf")
            if all(g == float("inf") for g in got):
                break
            continue
        e = order[t][ptr[t]]
        ptr[t] += 1
        seen.add(e)
        admitted.append(e)
        for u in range(n_t):
            got[u] += per_norm[u][e]
    return admitted


def topic_names(path: str, source: str = SOURCE_DEFAULT) -> list:
    """The topic names in a coverage file, without loading its histograms --
    find_stats compares every candidate in the checkout and there can be many.

    `source` because a file traced before the expert-output norms carries every
    `counts_<topic>` and no `saliency_<topic>`, and under --source saliency such
    a file has no usable topics at all: it must not win find_stats' count."""
    pre = source + "_"
    try:
        d = json.load(open(path))
        any_layer = next(iter((d.get("per_layer") or {}).values()), {})
        return sorted(k[len(pre):] for k in any_layer if k.startswith(pre))
    except Exception:  # noqa: BLE001
        return []


class TopicIndex:
    """The per-topic expert histograms in a coverage.json, and everything that
    can be derived from a selection of them without touching the model.

    `source` selects the family of histograms to read -- `counts_<topic>`
    (routing frequency) or `saliency_<topic>` (REAP's gate weight x expert-output
    norm, summed). One index reads one family, because everything below it is a
    ranking of one set of numbers and mixing the two in a single screen would
    show a coverage the engine cannot reproduce under either setting."""

    def __init__(self, path: str, source: str = SOURCE_DEFAULT):
        self.path = path
        self.source = source
        pre = source + "_"
        d = json.load(open(path))
        pl = d.get("per_layer") or {}
        any_layer = next(iter(pl.values()), {})
        self.topics = sorted(k[len(pre):] for k in any_layer if k.startswith(pre))
        self.counts = {}
        for t in self.topics:
            per = {}
            ok = True
            for L in range(N_LAYERS):
                v = pl.get(str(L), {}).get(pre + t)
                if v is None:
                    ok = False
                    break
                per[L] = [float(x) for x in v]
            if ok:
                self.counts[t] = per
        self.topics = [t for t in self.topics if t in self.counts]
        self.totals = {t: sum(sum(v) for v in self.counts[t].values()) for t in self.topics}
        # Every token routes to `n_activated_experts` experts in each of the 40
        # layers, so the histogram totals divide back to the tokens the trace
        # actually saw for that topic. A topic sampled thinly ranks noisily, and
        # its coverage bar is no more trustworthy than the sample under it.
        #
        # Only `counts` totals divide back that way: saliency's total is a sum of
        # magnitudes and carries no token count at all, so the sample size is
        # read out of the frequency histograms of the same file -- the same
        # trace, the same tokens, whichever family ranks them.
        #
        # The frequency histograms are kept whole and not only summed, because
        # they are the only thing in the file that counts PICKS. `tail_curves`
        # below measures a keep-set against them whichever family ranked it:
        # what breaks a generation is a token whose picks are not resident, and
        # a pick is a pick whatever magnitude it carried.
        self.picks = self.counts if source == "counts" else {}
        if source != "counts":
            for t in self.topics:
                per = {}
                ok = True
                for L in range(N_LAYERS):
                    v = pl.get(str(L), {}).get("counts_" + t)
                    if v is None:
                        ok = False
                        break
                    per[L] = [float(x) for x in v]
                if ok:
                    self.picks[t] = per
        counts_totals = {t: sum(sum(v) for v in self.picks[t].values()) if t in self.picks else 0.0
                         for t in self.topics}
        self.tokens = {t: int(round(counts_totals.get(t, 0.0) / (N_LAYERS * TOPK)))
                       for t in self.topics}
        self._cache: dict = {}
        self._tail: dict = {}
        # A topic's normalised histogram does not depend on what it is selected
        # with, and the profile screen ranks ten selections over the same
        # thirty-five topics, so it is normalised once per topic rather than
        # once per selection. The ranking itself is the expensive part and that
        # one cannot be shared: `maxmin` allocates the whole selection at once.
        self._norm: dict = {}

    THIN = 2000              # tokens below which a ranking is mostly noise

    def __bool__(self) -> bool:
        return bool(self.topics)

    def curves(self, selection: tuple, select: str = "uniform", only: tuple | None = None,
               rank: str = RANK_DEFAULT):
        """coverage curves for a selection: {topic: [384+1 floats]}, index n =
        keeping the top-n experts per layer. Computed once per selection, so a
        slider move is a lookup.

        `rank` is DSV41_PRUNE_RANK and it decides which experts those are, so it
        belongs in the cache key with the selection. `select` is in the key but
        changes nothing here: these curves are the per-layer (PRUNE_SELECT=
        uniform) cut, which is the one the engine takes for `maxmin` -- it
        refuses `global` with it -- and a cross-layer cut would need a different
        curve, not a different order."""
        key = (tuple(sorted(selection)), select, only, rank)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        # Sorted, because that is the order tune.py writes EXPERT_TOPICS in and
        # the order the engine then combines them in -- which `maxmin` can see,
        # since it breaks a tie between two equally covered topics by taking the
        # first of them.
        sel = sorted(t for t in selection if t in self.counts)
        if not sel:
            return {}
        if rank not in RANKS:
            raise ValueError(f"unknown rank {rank!r} ({' | '.join(RANKS)})")
        # the ranking the engine builds, per-layer-normalised counts either way
        for t in sel:
            if t not in self._norm:
                self._norm[t] = [_normalised(self.counts[t][L]) for L in range(N_LAYERS)]
        norm = {t: self._norm[t] for t in sel}
        combined, order = {}, {}
        for L in range(N_LAYERS):
            if rank == "maxmin":
                order[L] = _maxmin_order([norm[t][L] for t in sel])
                # The admission position is the score: the engine's own score
                # vector for this rule depends on the keep fraction (admitted
                # experts are lifted above 1.0) and a curve does not, and all
                # anything reads out of `combined` is the order it implies.
                acc = [0.0] * N_EXPERTS
                for i, e in enumerate(order[L]):
                    acc[e] = float(N_EXPERTS - i)
                combined[L] = acc
                continue
            if rank == "sum":
                combined[L] = [sum(norm[t][L][e] for t in sel) for e in range(N_EXPERTS)]
            else:   # "max": an expert that matters to any one topic is kept
                combined[L] = [max(norm[t][L][e] for t in sel) for e in range(N_EXPERTS)]
            # _desc, not a plain sort: a layer in which every selected topic is
            # silent scores 384 zeros, and the engine cuts THAT at the top N too
            # -- in np.argsort order, which is not the order a stable sort gives.
            order[L] = _desc(combined[L])
        out = {}
        # every topic in the file gets a curve, so an unselected one can be read
        # off too -- that is how you see what a selection costs the rest.
        # `only` narrows that when the caller just wants the selected ones,
        # which is ten times cheaper when ten selections are compared at once.
        for t in (only if only is not None else self.topics):
            tot = self.totals[t] or 1.0
            curve = _cumsum_desc(combined, order, self.counts[t])
            out[t] = [x / tot for x in curve]
        self._cache[key] = (out, order, combined)
        return self._cache[key]

    def coverage(self, selection: tuple, keep: float, select: str = "uniform",
                 rank: str = RANK_DEFAULT) -> dict:
        got = self.curves(selection, select, rank=rank)
        if not got:
            return {}
        curves = got[0]
        n = keep_n(keep)
        return {t: c[n] for t, c in curves.items()}

    def tail_curves(self, selection: tuple, select: str = "uniform", only: tuple | None = None,
                    rank: str = RANK_DEFAULT):
        """What the same keep-set costs a TOKEN rather than a topic's mass.

        Coverage answers "how much of this topic's routing stayed resident".
        That is a number about the corpus. What a generation trips over is a
        number about one token: the engine masks the router to the resident
        experts, so a token whose six picks in a layer are not all resident is
        computed with experts it did not ask for, and those substitutions are
        what compound into a redraft, a corrupted identifier, a loop.

        Coverage cannot see that, and under `saliency` it cannot see much of
        anything: half of a layer's saliency mass sits on one expert, so a
        keep-set swap of 513 experts moved every bar by 0.005 and flipped the
        gate (docs/keep-sets.md, 2026-09-14). The tail is where the tokens are.

        Returned per topic, each a curve indexed by experts kept per layer:

          cov_picks   the topic's PICKS that stayed resident. Same arithmetic
                      as `curves`, read off `self.picks` instead of whatever
                      family ranked the keep-set -- so a saliency keep-set is
                      still measured in picks, which is what a token spends.
          nonres      expected non-resident picks per token per layer, 0 to 6.
                      EXACT, not an estimate: `counts_<topic>` IS the histogram
                      of picks, so the mean of a per-token count is the mass
                      fraction times six, with no assumption about how the six
                      are distributed. Validated against the per-token arrays of
                      results/trace-full-20260910 (tools/test_tail_metric.py).
          all6        the rate at which all six picks of a token-layer are
                      resident -- the tail property itself -- ESTIMATED as
                      mean_L r_L^6, i.e. as if the six picks were drawn
                      independently from the topic's histogram. They are not:
                      experts co-fire, so the true rate is always HIGHER, never
                      lower. Measured against the per-token arrays of
                      results/trace-full-20260910 over twelve (topic, selection,
                      keep) points from 0.20 to 0.40, the estimate runs 6 % to
                      75 % low, worst on a topic the keep-set was not spent on,
                      and its rank correlation with the truth is 0.96. Read it
                      as a ranking statistic, not as a rate.
                      tools/test_tail_metric.py holds both of those.
          worst_layer min over layers of the resident pick fraction. Exact, and
                      the cheapest tail-sensitive number here: one bad layer is
                      enough, and a mean over forty hides it.

        Cached on the same key as `curves`, because a slider move must stay a
        lookup and this walks 40 x 384 per topic to build the curves.
        """
        key = (tuple(sorted(selection)), select, only, rank)
        hit = self._tail.get(key)
        if hit is not None:
            return hit
        got = self.curves(selection, select, only=only, rank=rank)
        if not got:
            return {}
        order = got[1]
        out = {}
        for t in (only if only is not None else self.topics):
            if t not in self.picks:
                continue
            # r[L][n] = the fraction of this topic's picks in layer L that the
            # top-n of that layer's admission order holds. One pass per layer.
            r = []
            for L in range(N_LAYERS):
                p = self.picks[t][L]
                tot = sum(p)
                row = [0.0] * (N_EXPERTS + 1)
                acc = 0.0
                for n, e in enumerate(order[L], start=1):
                    acc += p[e]
                    row[n] = acc
                for n in range(len(order[L]) + 1, N_EXPERTS + 1):
                    row[n] = acc      # a short admission order keeps its tail flat
                if tot > 0:
                    row = [x / tot for x in row]
                else:
                    # A topic with no pick at all in this layer has nothing to
                    # lose there. It cannot be counted as a miss, and counting
                    # it as a hit would flatter every other layer's average, so
                    # it does not vote: `None` drops it from the means below.
                    row = None
                r.append(row)
            live = [row for row in r if row is not None]
            if not live:
                continue
            nl = len(live)
            cov = [0.0] * (N_EXPERTS + 1)
            nonres = [0.0] * (N_EXPERTS + 1)
            all6 = [0.0] * (N_EXPERTS + 1)
            worst = [0.0] * (N_EXPERTS + 1)
            for n in range(N_EXPERTS + 1):
                s = 0.0
                s6 = 0.0
                mn = 1.0
                for row in live:
                    v = row[n]
                    s += v
                    s6 += v ** TOPK
                    if v < mn:
                        mn = v
                cov[n] = s / nl
                nonres[n] = TOPK * (1.0 - s / nl)
                all6[n] = s6 / nl
                worst[n] = mn
            out[t] = {"cov_picks": cov, "nonres": nonres, "all6": all6, "worst_layer": worst}
        self._tail[key] = out
        return out

    def tail(self, selection: tuple, keep: float, select: str = "uniform",
             rank: str = RANK_DEFAULT, only: tuple | None = None) -> dict:
        """`tail_curves` read at one keep fraction: {topic: {name: float}}."""
        got = self.tail_curves(selection, select, only=only, rank=rank)
        n = keep_n(keep)
        return {t: {k: v[n] for k, v in d.items()} for t, d in got.items()}

    def keep_for(self, selection: tuple, target: float, select: str = "uniform",
                 rank: str = RANK_DEFAULT) -> float | None:
        """Smallest keep fraction at which every selected topic reaches `target`.
        An empty selection means every topic, which is what the engine ranks on."""
        selection = tuple(selection) or tuple(self.topics)
        got = self.curves(selection, select, only=tuple(sorted(selection)), rank=rank)
        if not got:
            return None
        curves = got[0]
        sel = [t for t in selection if t in curves]
        for n in range(1, N_EXPERTS + 1):
            if all(curves[t][n] >= target for t in sel):
                return n / N_EXPERTS
        return None


def keep_n(keep: float) -> int:
    """Experts kept per layer at this fraction -- the engine's own rounding."""
    return max(6, math.ceil(keep * N_EXPERTS))


# --- the plan ---------------------------------------------------------------

@dataclass
class Plan:
    keep: float
    n_keep: int
    kept: int
    slots: int
    transient: int
    fmt: str
    max_seq: int
    arena: float
    dense: float
    dspark: float
    kv: float
    prefill: float
    scratch: float
    unpack_cache: float
    floor: float
    available: float
    chunk: int = PREFILL_CHUNK_DEFAULT
    kv_fp8: bool = False
    coverage: dict = field(default_factory=dict)
    selection: tuple = ()

    @property
    def resident_frac(self) -> float:
        return self.kept / N_ROUTED

    @property
    def resident(self) -> float:
        """Everything that stays in memory for the whole run, including what the
        line items above do not name (UNMODELLED_RESIDENT_GB)."""
        return self.arena + self.dense + self.dspark + self.kv + UNMODELLED_RESIDENT_GB

    # --- gate 1: the launcher's own pre-flight ------------------------------
    # engine/v41_engine.py refuses to start unless
    #     arena + pack_scratch + keep_free <= MemAvailable
    # measured after the dense weights are already resident. Reproduced here
    # against MemAvailable as it is now, so the dense term is explicit.
    @property
    def launch_need(self) -> float:
        # engine/v41_engine.py: floor = max(keep_free_gb, MAX_CHUNK * 5 MB).
        # Mirroring it exactly matters -- reading `keep_free_gb` alone made this
        # 4.2 GB more generous than the launcher's real margin at the default.
        # The prefill unpack cache is a persistent allocation the engine adds on
        # top of that max(), so `need_free` has it taken back out before the
        # comparison and it is added once, where the engine adds it.
        return (self.arena + self.scratch + self.unpack_cache + self.dense
                + max(self.floor, self.need_free - self.unpack_cache))

    @property
    def launch_slack(self) -> float:
        return self.available - self.launch_need

    # --- gate 2: what is left once it is up ---------------------------------
    @property
    def free_after_load(self) -> float:
        return self.available - self.resident

    @property
    def need_free(self) -> float:
        """A prefill chunk and its unpack cache, plus the floor the watchdog
        kills below. The cache is not transient, but it is allocated on top of
        everything `resident` names, so this is where it is paid for."""
        return self.prefill + WATCHDOG_FLOOR_GB

    @property
    def fits(self) -> bool:
        return self.launch_slack >= 0 and self.free_after_load >= self.need_free

    @property
    def verdict(self) -> str:
        # The engine's own gate lets a configuration start that the first
        # request then kills, because that gate does not know about the drafter
        # experts, the cache, or a prefill chunk. This one does.
        if self.launch_slack < 0 or self.free_after_load < self.need_free:
            return "over"
        if self.launch_slack < 3.0 or self.free_after_load < self.need_free + 3.0:
            return "tight"
        return "ok"

    @property
    def weakest(self):
        sel = {t: c for t, c in self.coverage.items() if t in self.selection}
        if not sel:
            return None, None
        t = min(sel, key=lambda k: sel[k])
        return t, sel[t]

    def max_arena(self) -> float:
        """Largest arena that both starts AND survives a prefill chunk."""
        launch = self.available - self.scratch - self.unpack_cache - self.dense - self.floor
        serve = (self.available - self.dense - self.dspark - self.kv
                 - UNMODELLED_RESIDENT_GB - self.need_free)
        return max(0.0, min(launch, serve))

    def max_keep(self) -> float:
        slots = self.max_arena() * GB / EXPERT_BYTES[self.fmt] - self.transient
        return max(0.0, slots / N_ROUTED)

    @property
    def everything_fits(self) -> bool:
        """Enough memory for every routed expert, so no keep-set is needed at
        all and the router is never restricted. Two GB10s get to 93 %; three
        clear it outright."""
        return self.max_keep() >= 1.0

def plan(host: Host, index: TopicIndex | None, selection, keep: float, max_seq: int,
         fmt: str = "cb3", select: str = "uniform", arena_gb: float | None = None,
         keep_free_gb: float = KEEP_FREE_GB_DEFAULT, dense_key=None,
         chunk: int | None = None, kv_fp8: bool | None = None,
         transient_slots: int = TRANSIENT_SLOTS_DEFAULT,
         rank: str = RANK_DEFAULT, cache_gb: float | None = None) -> Plan:
    # the engine's own rounding: ceil(keep * 384) experts in every layer
    dense_key = dense_key or dense_key_from_env()
    # The chunk, the fp8 switch and the unpack cache are ordinary .env keys the
    # engine reads, like DSV41_DENSE_FP4 above: a panel that prices a
    # 2,048-token chunk on a box whose .env says 4,096 understates the reserve
    # by 7 GB and its verdict cannot be trusted.
    chunk = chunk_from_env() if chunk is None else chunk
    kv_fp8 = kv_fp8_from_env() if kv_fp8 is None else kv_fp8
    cache_gb = unpack_cache_gb_from_env() if cache_gb is None else cache_gb
    if fmt != "cb3":
        cache_gb = 0.0   # an fp4 arena is already in the kernel's format; nothing is unpacked
    kept = keep_n(keep) * N_LAYERS
    slots = kept + transient_slots
    arena = (arena_gb * GB) if arena_gb else slots * EXPERT_BYTES[fmt]
    if arena_gb:
        slots = int(arena / EXPERT_BYTES[fmt])
        kept = min(kept, slots - transient_slots)
    kv = kv_bytes(max_seq, ring_from_env(chunk))
    cov = index.coverage(tuple(selection), keep, select, rank=rank) if index else {}
    return Plan(
        keep=keep, n_keep=keep_n(keep), kept=kept, slots=slots, transient=transient_slots,
        fmt=fmt, max_seq=max_seq,
        arena=arena / GB,
        dense=DENSE_BYTES.get(dense_key, DENSE_DEFAULT) / GB,
        dspark=DSPARK_BYTES / GB,
        kv=kv / GB,
        prefill=prefill_bytes(max_seq, chunk, cache_gb, kv_fp8) / GB,
        chunk=chunk, kv_fp8=kv_fp8,
        scratch=PACK_SCRATCH_BYTES[fmt] / GB,
        unpack_cache=unpack_cache_bytes(cache_gb) / GB,
        floor=keep_free_gb,
        available=host.available_gb,
        coverage=cov, selection=tuple(selection),
    )
