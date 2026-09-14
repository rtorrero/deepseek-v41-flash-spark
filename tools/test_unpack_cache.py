"""The per-request unpack cache: the policy, and that it changes no number.

Run:  python3 tools/test_unpack_cache.py

Two halves, and the first one runs anywhere.

1. **The policy**, on `tools/unpack_cache.py` alone -- no torch, no triton, no GPU. It replays the
   access pattern prefill actually has (layers 0..20, then the same layers again for the next
   chunk) against a budget too small to hold it, and asserts the properties the design rests on:
   a fixed layer window forms, it survives to the next chunk, an LRU over the same trace would hit
   ZERO times, admission stops on the last chunk, and rewriting an arena slot invalidates it.

2. **The numbers**, on real experts through the real kernels. Skipped with a message wherever
   there is no CUDA or no checkpoint. The unpack is a deterministic byte transformation -- the CB3
   codes are a subset of the FP4 grid -- so an expert read out of the cache has to be bit-identical
   to one unpacked again on the spot, and the MoE output with the cache on has to be bit-identical
   to the output with it off. Bit-identical, not close: anything else means the cache is handing
   the kernel a slot that holds a different expert, which is the one failure mode that would not
   show up as a crash.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import unpack_cache as U  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


# =============================================================================
# 1. the policy
# =============================================================================
N_LAYERS = 21          # what one prefill chunk visits under DSV41_SWA_REPLAY=1
PER_LAYER = 139        # ceil(0.36 * 384): the experts a layer's keep-set leaves routable


def layer_slots(L: int) -> list:
    """The arena slots layer L's experts occupy -- distinct per layer, as they are in the engine."""
    return list(range(L * PER_LAYER, (L + 1) * PER_LAYER))


def replay(cache, n_chunks: int, n_layers: int = N_LAYERS) -> list:
    """Drive the cache through a whole prompt and return the hits per chunk."""
    cache.begin_request(n_chunks)
    per_chunk = []
    for _ in range(n_chunks):
        cache.begin_chunk()
        before = cache.stats["unpack_hits"]
        for L in range(n_layers):
            hit_src, _, fresh = cache.lookup(layer_slots(L))
            cache.admit(fresh)
        per_chunk.append(cache.stats["unpack_hits"] - before)
    return per_chunk


def lru_hits(budget_slots: int, n_chunks: int, n_layers: int = N_LAYERS) -> int:
    """What a least-recently-used cache of the same size would have got on the same trace."""
    from collections import OrderedDict
    lru: OrderedDict = OrderedDict()
    hits = 0
    for _ in range(n_chunks):
        for L in range(n_layers):
            for s in layer_slots(L):
                if s in lru:
                    hits += 1
                    lru.move_to_end(s)
                else:
                    if len(lru) >= budget_slots:
                        lru.popitem(last=False)
                    lru[s] = 1
    return hits


print("== policy: a fixed layer window, not an LRU ==")
# 6 GB is 319 FP4 experts, which is 2.29 of the 21 layers a chunk visits.
BUDGET = int(6e9 // U.FP4_BYTES_PER_SLOT)
check("6 GB is 319 expert slots", BUDGET == 319, f"got {BUDGET}")
check("  = 2.3 layers of 21", round(BUDGET / PER_LAYER, 1) == 2.3, f"{BUDGET / PER_LAYER:.2f}")

c = U.PrefillUnpackCache(BUDGET)
per_chunk = replay(c, 4)
print(f"     hits per chunk, 4 chunks, {BUDGET} slots: {per_chunk}")
check("chunk 0 cannot hit", per_chunk[0] == 0)
check("every later chunk hits the whole window", per_chunk[1:] == [BUDGET] * 3, str(per_chunk))
check("the window is exactly the budget", c.next == BUDGET, f"used {c.next}")
# and it is the FIRST layers, contiguously -- that is what makes it a window
window = sorted(c.map)
check("it is layers 0..1 plus a prefix of layer 2",
      window == list(range(BUDGET)), f"{window[:2]}..{window[-2:]}")

check("an LRU over the same trace hits zero times", lru_hits(BUDGET, 4) == 0,
      f"got {lru_hits(BUDGET, 4)}")
# ... and would, for any budget under the whole working set
ws = N_LAYERS * PER_LAYER
check("  and for any budget below the working set",
      all(lru_hits(b, 3) == 0 for b in (100, BUDGET, ws - 1)), f"working set {ws} slots")
check("  while the window scales with the budget",
      [replay(U.PrefillUnpackCache(b), 3)[2] for b in (PER_LAYER, 2 * PER_LAYER)]
      == [PER_LAYER, 2 * PER_LAYER])

print()
print("== policy: admission is off where nothing can read it back ==")
c = U.PrefillUnpackCache(BUDGET)
check("a one-chunk prompt admits nothing", replay(c, 1) == [0] and c.next == 0, f"used {c.next}")
c = U.PrefillUnpackCache(BUDGET)
c.begin_request(3)
c.begin_chunk(); check("chunk 0 of 3 admits", c.admitting)
c.begin_chunk(); check("chunk 1 of 3 admits", c.admitting)
c.begin_chunk(); check("the last chunk does not", not c.admitting)
c.begin_chunk(); check("nor does the decoder replay after it", not c.admitting)

print()
print("== policy: a rewritten arena slot is invalidated ==")
c = U.PrefillUnpackCache(BUDGET)
c.begin_request(4)
c.begin_chunk()
_, _, fresh = c.lookup(layer_slots(0))
c.admit(fresh)
check("layer 0 is cached", len(c.map) == PER_LAYER)
c.invalidate(layer_slots(0)[7])
hit_src, _, fresh = c.lookup(layer_slots(0))
check("the rewritten slot misses", len(hit_src) == PER_LAYER - 1 and len(fresh) == 1)
# streaming mode rewrites everything; the cache has to degenerate to a no-op, not to a wrong answer
c = U.PrefillUnpackCache(BUDGET)
c.begin_request(4)
c.begin_chunk()
for L in range(3):
    _, _, fresh = c.lookup(layer_slots(L))
    c.admit(fresh)
    for s in layer_slots(L):
        c.invalidate(s)          # the transient ring recycling the slot straight away
c.begin_chunk()
hits = sum(len(c.lookup(layer_slots(L))[0]) for L in range(3))
check("a fully recycled arena gives no hits at all", hits == 0, f"got {hits}")
check("  and the map is empty, not stale", len(c.map) == 0)

print()
print("== accounting ==")
c = U.PrefillUnpackCache(BUDGET)
replay(c, 4)
s = c.summary()
check("hits + misses is every lookup", s["unpack_hits"] + s["unpack_misses"] == 4 * N_LAYERS * PER_LAYER,
      f"{s['unpack_hits']} + {s['unpack_misses']}")
check("bytes saved is 33.26 MB per hit",
      s["unpack_bytes_saved"] == s["unpack_hits"] * (U.CB3_BYTES_PER_SLOT + U.FP4_BYTES_PER_SLOT))
check("ms saved falls back to the documented cost with no GPU to measure on",
      abs(s["unpack_ms_saved"] - s["unpack_hits"] * U.UNPACK_MS_PER_EXPERT) < 0.5,
      f"{s['unpack_ms_saved']} ms over {s['unpack_hits']} hits")
print(f"     4 chunks x 21 layers x {PER_LAYER} experts at a {BUDGET}-slot budget: "
      f"{s['unpack_hits']} hits / {s['unpack_hits'] + s['unpack_misses']} lookups "
      f"({s['unpack_hits'] / (s['unpack_hits'] + s['unpack_misses']) * 100:.1f} %), "
      f"{s['unpack_bytes_saved'] / 1e9:.1f} GB, ~{s['unpack_ms_saved'] / 1000:.2f} s")

# =============================================================================
# 2. the numbers -- CUDA only
# =============================================================================
print()
MD = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))


def _cuda_ready():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"no torch ({type(e).__name__})"
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        import triton  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"no triton ({type(e).__name__})"
    if not os.path.exists(os.path.join(MD, "model.safetensors.index.json")):
        return f"no checkpoint at {MD} (set MODEL_DIR)"
    return None


why = _cuda_ready()
if why:
    print(f"== numbers: SKIPPED, {why} ==")
else:
    print("== numbers: the cache must change nothing ==")
    import json

    import torch
    from safetensors import safe_open

    sys.path.insert(0, os.path.join(HERE, "..", "engine"))
    import cb3_moe as C3  # noqa: E402
    import fp4_moe as F4  # noqa: E402
    from codebook_sim import CodebookSim  # noqa: E402

    S = int(os.environ.get("EXPERTS", "24"))
    idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{MD}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")

    def build():
        a = C3.CB3ArenaV2(S, "cuda")
        a.sim = CodebookSim(3, "cuda")
        for e in range(S):
            p = f"layers.0.ffn.experts.{e}."
            a.load_slot(e, f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale"),
                        f.get_tensor(p + "w2.weight"), f.get_tensor(p + "w2.scale"),
                        f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale"))
        return a

    torch.manual_seed(0)
    T = 128                                   # above PREFILL_MIN_P, so this is the unpack path
    x = (torch.randn(T, F4.DIM, device="cuda") * 0.5).to(torch.bfloat16)
    slots = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(T)]).to(torch.int32)
    wgt = torch.rand(T, 6, device="cuda")

    off = build()
    y_off = C3.moe_forward_prefill(x, slots, wgt, off)

    on = build()
    # a budget deliberately too small for the experts this call touches, so the SAME call mixes
    # cached and freshly-unpacked experts in one pass -- the case a bug would hide in
    cache = C3.attach_unpack_cache(on, (S // 3) * C3.FP4_BYTES_PER_SLOT)
    check("the cache attached", cache is not None and 0 < cache.slots < S,
          f"{cache.slots if cache else 0} slots for {S} experts")
    cache.begin_request(4)
    ys = []
    for _ in range(4):                        # four chunks of the same prompt: chunk 0 fills it
        cache.begin_chunk()
        ys.append(C3.moe_forward_prefill(x, slots, wgt, on))
    st = cache.summary()
    check("chunk 0 filled the window and later chunks hit it", st["unpack_hits"] > 0, str(st))
    for i, y in enumerate(ys):
        d = float((y.float() - y_off.float()).abs().max())
        check(f"chunk {i}: bit-identical to the cache-off output", d == 0.0, f"max |delta| {d:.1e}")

    # the cached bytes themselves: a cached expert must be the bytes a re-unpack produces
    sc = on.fp4_scratch(C3.UNPACK_BATCH)
    base = cache.slots
    same = True
    for src, dst in sorted(cache.map.items()):
        sel = torch.tensor([src], dtype=torch.int32, device="cuda")
        C3._unpack_into(on, sel, sc, base)     # unpack it again, into the rotating region
        for t in (sc.w1, sc.s1, sc.w3, sc.s3, sc.w2, sc.s2):
            same &= bool(torch.equal(t[dst], t[base]))
    check("every cached expert equals a fresh unpack of the same slot, byte for byte", same)

    # and a rewritten arena slot must not be served from the cache
    victim = sorted(cache.map)[0]
    p = "layers.0.ffn.experts.1."
    on.load_slot(victim, f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale"),
                 f.get_tensor(p + "w2.weight"), f.get_tensor(p + "w2.scale"),
                 f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale"))
    check("rewriting an arena slot drops it from the cache", victim not in cache.map)
    cache.end_request()
    check("end_request empties the cache", len(cache.map) == 0 and cache.next == 0)

print()
print(f"{len(fails)} failed: {fails}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
