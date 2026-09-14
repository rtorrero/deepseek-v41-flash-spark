"""Checks on the cost model -- the point is that it agrees with what the box did.

Run: python3 tools/test_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want}{f' +-{tol}' if tol else ''})")
    if not ok:
        fails.append(name)


# --- the slot sizes are the arena's, not a rounded quote of it ---------------
check("fp4 bytes/expert", B.EXPERT_BYTES["fp4"], 18_800_640)
check("cb3 bytes/expert", B.EXPERT_BYTES["cb3"], 14_454_784)
try:
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from cb3_moe import CB3_BYTES_PER_SLOT  # needs torch; skipped where it is absent
    check("cb3 matches the kernel", B.EXPERT_BYTES["cb3"], CB3_BYTES_PER_SLOT)
except Exception as e:  # noqa: BLE001
    print(f"skip cb3_moe cross-check ({type(e).__name__})")

# --- against a real load: 2026-09-12, ARENA_GB=88, cb3 ----------------------
# "arena 88.0 GB = 6087 cb3 expert slots of 14.45 MB (40% of all routed experts)"
check("88 GB is 6087 cb3 slots", int(88e9 // B.EXPERT_BYTES["cb3"]), 6087)

# --- the KV formula against the two lengths that were measured --------------
check("KV at 32k", round(B.kv_bytes(32768) / 1e9 - B.WINDOW_BYTES / 1e9, 2), 0.10, 0.005)
check("KV at 128k", round(B.kv_bytes(131072) / 1e9 - B.WINDOW_BYTES / 1e9, 2), 0.42, 0.005)

# --- the launch gate is the engine's own ------------------------------------
# engine/v41_engine.py: arena + pack_scratch + keep_free <= MemAvailable, checked
# once the dense weights are resident. The box had MemAvailable 111.1 GB there
# and accepted an 88 GB arena; it would not have accepted 103.
host = B.Host("test", 130.6e9, 111.1e9 + 7.61e9, True)
# Since 2026-09-12 the engine's floor is max(keep_free_gb, MAX_CHUNK * 5 MB), so
# the 98 GB arena that used to pass this gate no longer does -- which is the
# point: it loaded, reported ready, and died on the first request.
for arena, want in ((84.0, True), (98.0, False), (103.0, False)):
    p = B.plan(host, None, (), 0.39, 32768, arena_gb=arena)
    check(f"arena {arena:.0f} GB accepted", p.launch_slack >= 0, want)

# --- the two configurations this box actually ran, 2026-09-12 ---------------
# Same box, same .env except the keep fraction. MemAvailable was 111.0 GB with
# the dense weights already resident, so 118.6 GB before them.
#   arena 87 GB -> served two 2,400-token generations, 0.0 GB read from NVMe
#   arena 98 GB -> "FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB
#                   floor for 3.0 s" on the FIRST request
# The engine's own pre-flight accepts both; it does not know about the drafter
# experts, the cache or a prefill chunk. The model here has to reject the second.
box = B.Host("gb10-20260912", 130.6e9, 111.0e9 + 7.61e9, True)
served = B.plan(box, None, (), 0.39, 32768)
killed = B.plan(box, None, (), 0.44, 32768)
check("keep 0.39 is offered", served.verdict != "over", True)  # 87 GB arena, 32k
check("keep 0.44 is refused", killed.verdict, "over")
check("  and the launcher refuses it too", killed.launch_slack < 0, True)
check("  free at 0.39 clears the prefill reserve", served.free_after_load > served.prefill, True)
check("  free at 0.44 does not", killed.free_after_load < killed.prefill, True)
# 0.42 until UNMODELLED_RESIDENT_GB was measured and subtracted; the ceiling is
# lower than the line items alone imply, which is the point of that constant
check("max keep on that box", round(served.max_keep(), 2), 0.40, 0.01)

# --- keep fraction -> slots -------------------------------------------------
p = B.plan(host, None, (), 0.39, 32768)
check("keep 39% kept experts", p.kept, 150 * 40)          # ceil(.39*384)=150 per layer
check("arena holds kept + transient ring", p.slots, 6008)
# the engine's OWN default ring is 400, not 8: sizing for 8 and running with
# 400 leaves 392 kept experts outside the LRU, streaming from NVMe every step
big = B.plan(host, None, (), 0.39, 32768, transient_slots=400)
check("a 400-slot ring is sized for", big.slots, 6400)
check("and costs more arena", round(big.arena - p.arena, 1), 5.7, 0.1)
check("keep 39% arena GB", round(p.arena, 1), 86.8, 0.05)
# max_keep is the largest fraction that both starts and survives a prefill
# chunk, so it is bounded by `fits`, not by the engine's own launch gate alone
mk = p.max_keep()
check("max keep fits", B.plan(host, None, (), mk - 0.003, 32768).fits, True)
check("just past max keep does not", B.plan(host, None, (), mk + 0.01, 32768).fits, False)

# --- the ARENA_GB the tool writes must actually hold the kept set -----------
# It is written as a whole number of GB, and the engine turns that back into
# slots by flooring. If the rounding ever went the other way the kept tail
# would stream from NVMe with nothing but the tok/s to say so.
import math as _math  # noqa: E402
bad = []
for _k in [round(0.02 * i, 2) for i in range(3, 31)]:
    for _ring in (8, 16, 400):
        _p = B.plan(host, None, (), _k, 32768, transient_slots=_ring)
        written = _math.ceil(_p.arena)                      # what env_for writes
        slots = int(written * B.GB // B.EXPERT_BYTES["cb3"])
        if slots - _ring < _p.kept:
            bad.append((_k, _ring, slots - _ring, _p.kept))
check("written ARENA_GB holds the kept set at every keep and ring", bad, [])

# --- coverage: a narrower selection is better served at the same budget -----
cov = os.path.join(ROOT, "results/keepsets/general/coverage.json")
if os.path.exists(cov):
    idx = B.TopicIndex(cov)
    if len(idx.topics) >= 2:
        a, b = idx.topics[0], idx.topics[1]
        alone = idx.coverage((a,), 0.39)[a]
        both = idx.coverage((a, b), 0.39)[a]
        check(f"{a} alone beats {a}+{b} at the same keep", alone > both, True)
        # and reaches a given coverage at a smaller budget
        k1 = idx.keep_for((a,), 0.85)
        k2 = idx.keep_for((a, b), 0.85)
        check("one topic needs a smaller keep than two", k1 < k2, True)
        print(f"     {a} alone {k1:.0%} vs {a}+{b} {k2:.0%} for 0.85 coverage")
        # coverage is monotone in the budget
        c = idx.curves((a, b))[0][a]
        check("coverage is monotone in keep", all(c[i] <= c[i + 1] + 1e-12 for i in range(384)), True)
        check("coverage reaches 1.0 at keep 100%", round(c[384], 3), 1.0, 0.001)
else:
    print("skip coverage checks (no keep-set in the checkout)")

# --- the dense figure must follow the environment, not a hardcoded pair -----
# DSV41_DENSE_FP4 and DSV41_HEAD_FMT are ordinary .env keys that tune.sh
# already sources. Pinning the shipped pair understated resident memory by up
# to 11.8 GB for anyone who changed either, which flips the verdict.
_save = {k: os.environ.get(k) for k in ("DSV41_DENSE_FP4", "DSV41_HEAD_FMT")}
os.environ["DSV41_DENSE_FP4"], os.environ["DSV41_HEAD_FMT"] = "", "bf16"
bf16 = B.plan(host, None, (), 0.39, 32768)
os.environ["DSV41_DENSE_FP4"], os.environ["DSV41_HEAD_FMT"] = "attn,wo_a", "fp8"
fp8 = B.plan(host, None, (), 0.39, 32768)
for k, v in _save.items():
    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
check("a bf16 head and dense attention cost more", round(bf16.dense - fp8.dense, 1), 11.8, 0.05)

# --- the ceiling must use this plan's ring, not the default one -------------
ring400 = B.plan(host, None, (), 0.39, 32768, transient_slots=400)
ring8 = B.plan(host, None, (), 0.39, 32768, transient_slots=8)
check("a 400-slot ring lowers the ceiling", ring400.max_keep() < ring8.max_keep(), True)
check("  by 392 slots", round((ring8.max_keep() - ring400.max_keep()) * B.N_ROUTED), 392, 1)

# --- the tool and the engine must reserve the same prefill headroom ---------
# If they drift, tune.sh advises configurations the engine refuses, or worse,
# ones it accepts and the watchdog then kills.
import re as _re  # noqa: E402
_src = open(os.path.join(ROOT, "engine/v41_engine.py")).read()
_m = _re.search(r"^PREFILL_BYTES_PER_TOKEN = ([0-9.e+]+) / 2048$", _src, _re.M)
check("the engine reserves a prefill chunk at all", bool(_m), True)
if _m:
    check("and at the same rate as this model", float(_m.group(1)) / 2048, B.PREFILL_BYTES_PER_TOKEN)
_m3 = _re.search(r"^PREFILL_BYTES_PER_CONTEXT_TOKEN = ([0-9.]+) \* 1024$", _src, _re.M)
check("the engine grows it with the context too", bool(_m3), True)
if _m3:
    check("  at the same rate", float(_m3.group(1)) * 1024, B.PREFILL_BYTES_PER_CONTEXT_TOKEN, 1)
_m4 = _re.search(r"^PREFILL_KV_FP8_SAVED_PER_TOKEN = ([0-9]+) \* 1024$", _src, _re.M)
check("the engine knows what fp8 saves", bool(_m4), True)
if _m4:
    check("  and by the same amount", float(_m4.group(1)) * 1024, B.PREFILL_KV_FP8_SAVED_PER_TOKEN)
# the context term has to carry the chunk factor in the engine too, or a 4,096
# chunk passes a gate sized for 2,048
check("the engine scales the context term with the chunk",
      "PREFILL_BYTES_PER_CONTEXT_TOKEN * (chunk / 2048)" in _src, True)
_m2 = _re.search(r"floor = max\(keep_free_gb \* 1e9, prefill_reserve\)", _src)
check("the engine takes the larger of the two floors", bool(_m2), True)
check("and adds the watchdog floor to its reserve",
      "DSV41_MEM_FLOOR_GB" in _src.split("prefill_reserve = (")[1][:400], True)

# --- a 4,096-token chunk has to be priced as a 4,096-token chunk -------------
# Both terms scale with it. Pricing only the first one understates the reserve
# by the whole context term, which at 32k is 0.5 GB and at 128k is 2.0 GB.
c2k = B.prefill_bytes(32768, 2048)
c4k = B.prefill_bytes(32768, 4096)
check("chunk 2048 at 32k is the measured 7.7 GB", round(c2k / B.GB, 1), 7.7, 0.05)
check("doubling the chunk doubles the whole reserve", round(c4k / c2k, 3), 2.0, 0.001)
check("  the chunk term doubled", round((4096 * B.PREFILL_BYTES_PER_TOKEN) / B.GB, 1), 14.4, 0.05)
check("  and so did the context term",
      round((c4k - 4096 * B.PREFILL_BYTES_PER_TOKEN) / B.GB, 2), 1.01, 0.01)

# --- what fp8 buys, and that it is claimed conservatively --------------------
# 128 KB (window) + 512 KB (compressed) + 640 KB (their concatenation) of bf16
# a token, against one 640 x 512 fp8 buffer.
_bf16 = (128 + 512) * 512 * 2 + (128 + 512) * 512 * 2
_fp8 = (128 + 512) * 512
check("the fp8 saving is the three bf16 buffers minus one fp8 one",
      B.PREFILL_KV_FP8_SAVED_PER_TOKEN, _bf16 - _fp8)
check("  which is under a third of the fitted per-token cost",
      B.PREFILL_KV_FP8_SAVED_PER_TOKEN < B.PREFILL_BYTES_PER_TOKEN / 3, True)
f4k = B.prefill_bytes(32768, 4096, kv_fp8=True)
check("fp8 at chunk 4096 costs less than bf16 at chunk 2048", f4k < c2k, False)
check("  but brings it back under the 16.5 GB that served", round(f4k / B.GB, 1), 11.4, 0.1)
check("fp8 does not change the context term",
      round((f4k - c4k + 4096 * B.PREFILL_KV_FP8_SAVED_PER_TOKEN) / B.GB, 6), 0.0, 1e-6)
check("and it is off unless asked for", B.prefill_bytes(32768, 2048, kv_fp8=False), c2k)

# --- the chunk must stay on a tile boundary ---------------------------------
# MM_TILE/ATTN_TILE pad the last tile of every GEMM; a chunk that overruns one
# pays ~30 % more iteration time for rows that only pad.
for _c in (512, 2048, 4096):
    check(f"chunk {_c} is a multiple of 128", _c % 128, 0)

# --- the ring has to follow the chunk ---------------------------------------
# The window gather runs after the whole chunk is in the ring, so RING must
# exceed window_size + chunk. At chunk 4096 the shipped 4096-slot ring is too
# small, and getting that wrong is silent: the first queries of a chunk read
# the KV the last ones wrote.
_msrc = open(os.path.join(ROOT, "engine/model.py")).read()
check("engine/model.py derives the ring default from the chunk",
      'os.environ.get("DSV41_RING", max(4096, MAX_CHUNK + 512))' in _msrc, True)
check("and refuses a ring that is too short at all",
      "RING < args.window_size + MAX_CHUNK" in _msrc, True)
check("the model prices the default ring the way this does",
      B.ring_from_env(2048), 4096)
check("  and grows it at chunk 4096", B.ring_from_env(4096), 4608)
check("a longer ring costs 44 MB per 1,000 slots",
      round(B.window_bytes(5096) - B.window_bytes(4096)) / 1e6, 44.0, 0.05)
check("the KV row at 32k is unchanged by the ring",
      round(B.kv_bytes(32768, 4096) - B.window_bytes(4096)), 32768 * B.KV_BYTES_PER_TOKEN)

# --- the panel must follow .env, not a hardcoded chunk ----------------------
_sv = {k: os.environ.get(k) for k in ("DSV41_PREFILL_CHUNK", "DSV41_PREFILL_KV_FP8", "DSV41_RING")}
os.environ["DSV41_PREFILL_CHUNK"], os.environ["DSV41_PREFILL_KV_FP8"] = "4096", "1"
os.environ.pop("DSV41_RING", None)
big = B.plan(host, None, (), 0.39, 32768)
for k, v in _sv.items():
    os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
small = B.plan(host, None, (), 0.39, 32768)
check("a 4096 chunk with fp8 reserves more than a 2048 bf16 one",
      big.prefill > small.prefill, True)
check("  and the longer ring lands in the KV row", round(big.kv - small.kv, 3), 0.023, 0.001)
check("the plan records what it priced", (big.chunk, big.kv_fp8), (4096, True))
check("and the default records the shipped pair", (small.chunk, small.kv_fp8), (2048, False))

# --- the prefill unpack cache ------------------------------------------------
# DSV41_PREFILL_UNPACK_CACHE_GB buys FP4 expert slots that survive between the
# chunks of one prompt. It is a persistent allocation, so a configuration that
# fitted without it does not automatically fit with it, and the tool has to say
# so before the box does.
check("cache slot is one FP4 expert", B.UNPACK_SLOT_BYTES, 18_800_640)
check("a 6 GB budget is whole experts only",
      B.unpack_cache_bytes(6.0), (6e9 // 18_800_640) * 18_800_640)
check("  which is 319 of them", int(B.unpack_cache_bytes(6.0) // 18_800_640), 319)
# one layer at keep 0.36 is ceil(.36*384) = 139 experts = 2.61 GB
check("a layer at keep 0.36 costs", round(B.keep_n(0.36) * B.UNPACK_SLOT_BYTES / B.GB, 2), 2.61, 0.01)
check("a layer unpruned costs", round(384 * B.UNPACK_SLOT_BYTES / B.GB, 2), 7.22, 0.01)
check("6 GB holds 2.3 layers at keep 0.36", round(B.unpack_cache_layers(6.0, 0.36), 1), 2.3, 0.05)
check("prefill_bytes grows by the cache",
      round((B.prefill_bytes(32768, 2048, 6.0) - B.prefill_bytes(32768, 2048)) / B.GB, 2),
      round(B.unpack_cache_bytes(6.0) / B.GB, 2), 0.01)
# the engine clamps rather than refuses, but the PLAN has to be honest: a keep
# that only just fitted must go tight or over once a cache is asked for
_no = B.plan(box, None, (), 0.39, 32768, cache_gb=0.0)
_yes = B.plan(box, None, (), 0.39, 32768, cache_gb=6.0)
check("the cache is charged to the prefill reserve",
      round(_yes.need_free - _no.need_free, 2), round(B.unpack_cache_bytes(6.0) / B.GB, 2), 0.01)
check("and to the launch gate too",
      round(_yes.launch_need - _no.launch_need, 2), round(B.unpack_cache_bytes(6.0) / B.GB, 2), 0.01)
check("keep 0.39 + a 6 GB cache no longer fits this box", _yes.verdict, "over")
check("  and the ceiling drops", _yes.max_keep() < _no.max_keep(), True)
check("an fp4 arena is never charged for a cache",
      B.plan(box, None, (), 0.30, 32768, fmt="fp4", cache_gb=6.0).unpack_cache, 0.0)
# and the engine must actually add it where this model says it does
_mc = _re.search(r"need = arena_gb \* 1e9 \+ pack_scratch \+ self\.unpack_cache_gb \* 1e9 \+ floor", _src)
check("the engine adds the cache to its own pre-flight", bool(_mc), True)
check("and clamps it to what is left rather than refusing",
      "clamped to" in _src and "UNPACK_SLOT_BYTES" in _src, True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
