"""The escape hatch's decision rule: which masked-off pick is worth fetching, and what it costs.

Run: python3 tools/test_escape_rule.py

Same shape as tools/test_route_modes.py, for the same reason. The rule lives in three places --
`EscapeHatch` (engine/escape.py, the policy), the device scan in engine/fastdecode.py `_layer_a`,
and the eager one in engine/model.py `moe` -- and only the first can be imported without torch, a
GPU and a 510 GB checkpoint. So this file does three things.

1. A pure-numpy REFERENCE of the rule end to end: candidates, margins, admission, and the routing
   that comes out the other side. It says what an escape MEANS, independently of how torch spells
   it, and it is short enough to read against the engine line by line.

2. The shipped `EscapeHatch` itself, exercised against that reference. This is the real object the
   engine constructs, not a copy of it: the env parsing, the prefix rule, the K cap, the per-step
   budget and every counter `x_engine_stats` reports.

3. Regex pins on the engine source. The risk is not that the arithmetic is wrong today; it is that
   one of the three places drifts. `_reroute` in particular is a second copy of `_layer_a`'s
   routing tail, and prefill and decode disagreeing pick for pick is not a crash -- the verify step
   quietly rejects its own drafts.

The last group also pins the thing that matters most: with `DSV41_ESCAPE_K=0` -- the default, and
what env.example ships -- the engine builds no hatch, both mask lines are the ones
tools/test_route_modes.py already pins, and every escape site is behind an `is not None`.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.escape import DEFAULT_MARGIN, STEP_BUDGET_PER_K, EscapeHatch  # noqa: E402

MODEL = os.path.join(ROOT, "engine/model.py")
FAST = os.path.join(ROOT, "engine/fastdecode.py")
ENGINE = os.path.join(ROOT, "engine/v41_engine.py")
STORE = os.path.join(ROOT, "engine/experts.py")
ESCAPE = os.path.join(ROOT, "engine/escape.py")
ENV = os.path.join(ROOT, "env.example")
fails = []

ROUTE_SCALE = 2.5  # any positive constant; the engine's comes from the checkpoint
E, K = 384, 6      # experts per layer, router top-k


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


# ============================================================ the reference
def topk(v, k):
    """Indices of the k largest entries, largest first (same tie-break as tools/test_route_modes)."""
    return np.argsort(-v, kind="stable")[:k]


def substitute(scores, bias, resident, k):
    """The shipped routing: the mask hides everything not resident. Returns (idx, weights)."""
    idx = topk(np.where(resident, scores + bias, -np.inf), k)
    w = scores[idx]
    return idx, w / (w.sum() + 1e-20) * ROUTE_SCALE


def escape_rule(scores, bias, resident, k, n, margin_min):
    """Which non-resident experts this layer-step fetches, and why.

    scores    [E] the gate's sqrt(softplus(.)) scores -- strictly positive, and what the routed
              weights are made of
    bias      [E] the gate bias, added for the RANKING only and never to the weights
    resident  [E] bool, the engine's live `prune_mask[L]` (True = has an arena slot)
    n         how many candidates to look at, i.e. DSV41_ESCAPE_K

    Returns (cands [n], margins [n], admitted [<=n]).

    The candidates are the top-n NON-resident experts by the router's raw, unmasked logit -- the
    pick the model would have made had the keep-set not been in the way. Candidate j (1-based)
    would displace the j-th weakest of the resident top-k, and its margin is the share of this
    token's routed weight that would change hands:

        (scores[candidate] - scores[displaced]) / sum(scores[resident top-k])

    ... or -inf if the mask did not actually take it away: a candidate whose logit is below the
    pick it would displace is not in the router's unmasked top-k either, so fetching it would
    change nothing.

    Admission is the longest prefix whose margin clears `margin_min` -- a prefix and not a filter,
    because candidate j's margin is only the right number if the j-1 before it were taken too.
    """
    logits = scores + bias
    res = topk(np.where(resident, logits, -np.inf), k)         # resident top-k, strongest first
    den = scores[res].sum()
    cands = topk(np.where(resident, -np.inf, logits), n)       # non-resident, strongest first
    disp = res[::-1][:n]                                       # weakest first: what each displaces
    margins = np.where(logits[cands] > logits[disp],
                       (scores[cands] - scores[disp]) / den, -np.inf)
    adm = []
    for j in range(n):
        if margins[j] < margin_min:
            break
        adm.append(int(cands[j]))
    return cands, margins, adm


def route_after_escape(scores, bias, resident, k, admitted):
    """Routing once the admitted experts have an arena slot: the same mask, one bit richer."""
    live = resident.copy()
    for e in admitted:
        live[e] = True
    return substitute(scores, bias, live, k)


# ============================================================ the properties
rng = np.random.default_rng(20260914)


def a_case(n_resident=139):
    """scores/bias/resident for one token. 139 of 384 is the shipped budget at keep 0.36."""
    scores = np.sqrt(np.log1p(np.exp(rng.normal(size=E))))     # sqrt(softplus(.)) > 0
    bias = rng.normal(scale=0.3, size=E)
    resident = np.zeros(E, dtype=bool)
    resident[rng.choice(E, n_resident, replace=False)] = True
    return scores, bias, resident


# --- (a) the candidates are non-resident, and they are the router's own ranking
bad_res = bad_order = 0
for _ in range(400):
    scores, bias, resident = a_case()
    cands, margins, _ = escape_rule(scores, bias, resident, K, 3, 0.0)
    bad_res += int(resident[cands].any())
    want = topk(np.where(resident, -np.inf, scores + bias), 3)
    bad_order += int(not (cands == want).all())
check("every candidate is NON-resident", bad_res == 0, f"{bad_res} violations")
check("the candidates are the top of the UNMASKED router ranking", bad_order == 0,
      f"{bad_order} violations")

# --- (b) a candidate the mask did not displace is never fetched, at any margin
bad_nodisp = seen_nodisp = 0
for _ in range(600):
    scores, bias, resident = a_case()
    logits = scores + bias
    cands, margins, adm = escape_rule(scores, bias, resident, K, K, 0.0)
    res = topk(np.where(resident, logits, -np.inf), K)
    for j in range(K):
        if logits[cands[j]] <= logits[res[::-1][j]]:
            seen_nodisp += 1
            bad_nodisp += int(margins[j] != -np.inf or int(cands[j]) in adm)
check("a candidate that would not enter the top-k gets -inf and is never fetched",
      bad_nodisp == 0, f"{bad_nodisp} violations")
check("the cases above contained such candidates", seen_nodisp > 100, f"only {seen_nodisp}")

# --- (b2) admission is a PREFIX: it stops at the first refusal rather than skipping it
h_pref = EscapeHatch(3, 0.2)
check("select stops at the first candidate under the margin",
      h_pref.select([1, 2, 3], [0.9, 0.05, 0.9]) == [(1, 0.9)])

# --- (c) an admitted expert really does displace the pick the margin was computed against
bad_disp = bad_in = seen = 0
for _ in range(400):
    scores, bias, resident = a_case()
    cands, margins, adm = escape_rule(scores, bias, resident, K, 1, 0.05)
    if not adm:
        continue
    seen += 1
    res_before, _ = substitute(scores, bias, resident, K)
    res_after, _ = route_after_escape(scores, bias, resident, K, adm)
    bad_in += int(adm[0] not in res_after.tolist())            # the fetched expert is now routed
    gone = set(res_before.tolist()) - set(res_after.tolist())
    bad_disp += int(gone != {int(res_before[-1])})              # exactly the weakest, and only it
check("an admitted expert is in the new top-k", bad_in == 0, f"{bad_in} violations")
check("... and it displaces exactly the weakest resident pick", bad_disp == 0,
      f"{bad_disp} violations")
check("the cases above actually admitted something", seen > 100, f"only {seen}/400")

# --- (d) the margin is about routed WEIGHT, not about the router's ranking
# The ranking uses logits (scores + bias) and the weights use scores alone, so a candidate can
# outrank the pick it would displace and still be worth less. The hatch must not fetch it.
bad_sign = 0
for _ in range(600):
    scores, bias, resident = a_case()
    cands, margins, adm = escape_rule(scores, bias, resident, K, 2, 0.0 + 1e-12)
    for e in adm:
        j = int(np.flatnonzero(cands == e)[0])
        res = topk(np.where(resident, scores + bias, -np.inf), K)
        bad_sign += int(not scores[e] > scores[res[::-1][j]])
check("an admitted expert always carries MORE routed weight than the pick it displaces",
      bad_sign == 0, f"{bad_sign} violations")

# --- (e) the off path: K=0, and an unreachable margin, both leave routing exactly as it is
bad_off = bad_inf = 0
for _ in range(300):
    scores, bias, resident = a_case()
    base_idx, base_w = substitute(scores, bias, resident, K)
    _, _, adm0 = escape_rule(scores, bias, resident, K, 0, DEFAULT_MARGIN)
    i0, w0 = route_after_escape(scores, bias, resident, K, adm0)
    bad_off += int(adm0 != [] or not (i0 == base_idx).all() or not np.array_equal(w0, base_w))
    _, _, admI = escape_rule(scores, bias, resident, K, K, float("inf"))
    iI, wI = route_after_escape(scores, bias, resident, K, admI)
    bad_inf += int(admI != [] or not (iI == base_idx).all() or not np.array_equal(wI, base_w))
check("K=0 admits nothing and routing is the shipped routing, value for value", bad_off == 0,
      f"{bad_off} violations")
check("an unreachable margin does the same", bad_inf == 0, f"{bad_inf} violations")

# --- (f) how often it would fire at the shipped budget, at a few margins (not an assertion about
#         the model -- a sanity check that the default is a rare event and 0.0 is not)
rate = {}
for m in sorted({0.0, 0.05, DEFAULT_MARGIN, 0.20, 0.50}):
    fired = 0
    for _ in range(600):
        scores, bias, resident = a_case()
        _, _, adm = escape_rule(scores, bias, resident, K, 1, m)
        fired += int(bool(adm))
    rate[m] = fired / 600
print(f"     (reference fire rate per layer-step at keep 139/384, K=1: "
      + ", ".join(f"margin {m:.2f} -> {r:.2%}" for m, r in rate.items()) + ")")
check("the default margin fires less often than a margin of zero",
      rate[DEFAULT_MARGIN] < rate[0.0], f"{rate}")
check("a margin of zero fires on most layer-steps (the mask really does displace the top pick)",
      rate[0.0] > 0.5, f"{rate[0.0]:.2%}")


# ============================================================ the shipped EscapeHatch
# --- env parsing: off is the default and off means None
check("no DSV41_ESCAPE_K is off", EscapeHatch.from_env(env={}) is None)
check("DSV41_ESCAPE_K=0 is off", EscapeHatch.from_env(env={"DSV41_ESCAPE_K": "0"}) is None)
check("an empty DSV41_ESCAPE_K is off (start.sh sources .env with set -a)",
      EscapeHatch.from_env(env={"DSV41_ESCAPE_K": "", "DSV41_ESCAPE_MARGIN": "0.3"}) is None)
h = EscapeHatch.from_env(env={"DSV41_ESCAPE_K": "1"})
check("DSV41_ESCAPE_K=1 arms it at the default margin",
      h is not None and h.k == 1 and h.margin == DEFAULT_MARGIN)
h2 = EscapeHatch.from_env(env={"DSV41_ESCAPE_K": "2", "DSV41_ESCAPE_MARGIN": "0.05"})
check("DSV41_ESCAPE_MARGIN is read", h2.k == 2 and abs(h2.margin - 0.05) < 1e-12)
check("an empty DSV41_ESCAPE_MARGIN falls back to the default",
      EscapeHatch.from_env(env={"DSV41_ESCAPE_K": "1", "DSV41_ESCAPE_MARGIN": ""}).margin
      == DEFAULT_MARGIN)
for bad, why in (({"DSV41_ESCAPE_K": "yes"}, "not an integer"),
                 ({"DSV41_ESCAPE_K": "-1"}, "negative"),
                 ({"DSV41_ESCAPE_K": "7"}, "above the router's top-k"),
                 ({"DSV41_ESCAPE_K": "1", "DSV41_ESCAPE_MARGIN": "-0.2"}, "a negative margin"),
                 ({"DSV41_ESCAPE_K": "1", "DSV41_ESCAPE_MARGIN": "big"}, "not a float")):
    try:
        EscapeHatch.from_env(env=bad)
        ok = False
    except ValueError:
        ok = True
    check(f"{why} is refused, not quietly served as the default", ok)

# --- the per-step budget
h = EscapeHatch(2, 0.1)
check("the per-step budget is STEP_BUDGET_PER_K x K", h.step_budget == STEP_BUDGET_PER_K * 2)

# --- select(): the prefix rule, the K cap, the budget, and what each refusal is counted as
h = EscapeHatch(1, 0.25)
check("select admits a candidate over the margin", h.select([7], [0.30]) == [(7, 0.30)])
check("... and refuses one under it", h.select([7], [0.24]) == [])
check("a refusal on the margin is counted as one", h.stats["blocked_margin"] == 1)
h = EscapeHatch(1, 0.0)
check("K caps a layer-step", len(h.select([1, 2, 3], [0.9, 0.8, 0.7])) == 1)
check("... and the rest are counted against K", h.stats["blocked_k"] == 2)
h = EscapeHatch(2, 0.0)
took = sum(len(h.select([10 * s, 10 * s + 1], [0.9, 0.8])) for s in range(20))
check("the per-step budget caps a whole step", took == h.step_budget,
      f"{took} fetches, budget {h.step_budget}")
check("... and the refusals are counted against the budget", h.stats["blocked_budget"] > 0)
h.begin_step()
check("begin_step hands the budget back", h.step_left == h.step_budget)
check("everything it looked at is counted", h.stats["considered"] == 40)

# --- select() agrees with the reference, case for case
bad_sel = 0
for _ in range(400):
    scores, bias, resident = a_case()
    cands, margins, adm = escape_rule(scores, bias, resident, K, 2, 0.15)
    hh = EscapeHatch(2, 0.15)
    got = [e for e, _ in hh.select([int(c) for c in cands], [float(m) for m in margins])]
    bad_sel += int(got != adm)
check("the shipped select() admits exactly what the reference admits", bad_sel == 0,
      f"{bad_sel}/400 disagree")

# --- margins() is the reference's arithmetic. It is the formula alone: the -inf for a candidate
#     that displaces nothing is applied by the scan that knows the logits, not here.
bad_m = 0.0
for _ in range(200):
    scores, bias, resident = a_case()
    cands, _, _ = escape_rule(scores, bias, resident, K, 3, 0.0)
    res = topk(np.where(resident, scores + bias, -np.inf), K)
    disp = res[::-1][:3]
    want = (scores[cands] - scores[disp]) / scores[res].sum()
    got = EscapeHatch.margins(scores[cands], scores[disp], float(scores[res].sum()))
    bad_m = max(bad_m, float(np.abs(np.asarray(got) - want).max()))
check("margins() is the reference formula", bad_m < 1e-12, f"max error {bad_m:.2e}")
check("a zero denominator gives zero margins, not a division by zero",
      EscapeHatch.margins([1.0, 2.0], [0.0, 0.0], 0.0) == [0.0, 0.0])

# --- live bookkeeping and the accounting
h = EscapeHatch(1, 0.1)
h.admitted((3, 17), 4100, 0.4)
h.admitted((9, 200), 4101, 0.2)
h.record_fetch(18_874_368, 0.004)
h.record_fetch(18_874_368, 0.006)
check("live escapes are tracked by (layer, expert)", set(h.live) == {(3, 17), (9, 200)})
check("evicting one that is live reports True", h.evicted((3, 17)) is True)
check("evicting one that is not reports False", h.evicted((3, 17)) is False)
check("an eviction is counted", h.stats["evictions"] == 1)
rep = h.report(completion_tokens=200)
check("the report carries escapes per token", rep["escapes"] == 2 and rep["escapes_per_token"] == 0.01)
check("... bytes read", rep["escape_gb"] == round(2 * 18_874_368 / 1e9, 3))
check("... and ms spent, total and each", rep["escape_ms"] == 10.0 and rep["escape_ms_each"] == 5.0)
check("peak_live survives an eviction", rep["escape_peak_live"] == 2 and rep["escape_live"] == 1)
check("a request with no escapes reports no mean", EscapeHatch(1, 0.1).report(10)["escape_ms_each"] is None)
h.reset()
check("reset clears the live set and every counter",
      not h.live and h.report(1)["escapes"] == 0 and h.stats["fetch_s"] == 0.0)


# ============================================================ the runtime, lifted out
# `EscapeRuntime` lives in engine/v41_engine.py, which imports torch at module scope. Its
# admission/eviction bookkeeping does not touch torch at all, and that bookkeeping is the riskiest
# thing in the hatch: a router mask bit whose LUT entry has gone back to -1 is a gather off the end
# of the arena, which is how this repository has silently computed layers with the wrong experts
# before. So lift the class out with `ast` -- the way tools/test_budget_rank.py lifts the engine's
# ranking function -- and drive it with a fake mask, a fake LUT and a fake transient ring.
#
# The ring is a model of ExpertStore's, not the thing itself (the real one is pinned by regex
# below): a fixed list of slots, recycled in order, calling `on_transient_evict` for whatever the
# slot held.
import ast  # noqa: E402
import types  # noqa: E402

_tree = ast.parse(open(ENGINE).read())
_node = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "EscapeRuntime")
_ns = {"EscapeHatch": EscapeHatch, "torch": types.SimpleNamespace(Tensor=object)}
exec(compile(ast.Module(body=[_node], type_ignores=[]), ENGINE, "exec"), _ns)  # noqa: S102
EscapeRuntime = _ns["EscapeRuntime"]


class FakeMask(list):
    """One layer's 384 bools, indexable the way a torch bool tensor is."""


class FakeLUT:
    """(layer, expert) -> slot, -1 when the expert has none."""

    def __init__(self):
        self.t = {}

    def __setitem__(self, key, v):
        self.t[key] = v


class FakeRing:
    """ExpertStore's transient ring, modelled: `slots` slots recycled in order."""

    def __init__(self, slots=8):
        self.ring = list(range(4000, 4000 + slots))
        self.pos = 0
        self.holder = {}                 # slot -> key
        self.on_transient_evict = None
        self.reads = 0

    def escape_fetch(self, layer, expert):
        slot = self.ring[self.pos % len(self.ring)]
        self.pos += 1
        old = self.holder.pop(slot, None)
        if old is not None and self.on_transient_evict is not None:
            self.on_transient_evict(old)
        self.holder[slot] = (int(layer), int(expert))
        self.reads += 1
        return slot, 18_874_368, 0.004


def a_runtime(k=1, margin=0.0, slots=8, layers=40):
    masks = {L: FakeMask([False] * E) for L in range(layers)}
    store = FakeRing(slots)
    rt = EscapeRuntime(EscapeHatch(k, margin), masks, store, K)
    rt.attach_lut(FakeLUT())
    store.on_transient_evict = rt.on_evict
    return rt, masks, store


def _agree(rt, masks):
    """Every mask bit the hatch set has a real slot in the LUT, and nothing else does."""
    bits = {(L, e) for L, m in masks.items() for e in range(E) if m[e]}
    live = set(rt.h.live)
    lut_live = {key for key, v in rt.lut.t.items() if v != -1}
    return bits == live and lut_live == live


# ten decode steps, four escapes each -- 40 fetches through a ring of 8, so it wraps four times
rt, masks, store = a_runtime(k=1, slots=8)
for n_step in range(10):
    rt.begin_step()
    for L in range(4):
        rt.admit(L, [10 * n_step + L + 7], [0.5])
check("an admitted expert gets a mask bit and a LUT entry",
      bool(masks[3][100]) and rt.lut.t[(3, 100)] != -1)
check("the ring bounds how many escapes are live at once", len(rt.h.live) == 8,
      f"{len(rt.h.live)} live, ring of 8")
check("the mask and the LUT never disagree, however often the ring wraps", _agree(rt, masks))
check("every recycled escape was counted as an eviction", rt.h.stats["evictions"] == 32,
      f"{rt.h.stats['evictions']}")
check("and every fetch was a real read", store.reads == 40, f"{store.reads}")
rt.clear()
check("clear() takes every bit back out of the mask",
      not any(m[e] for m in masks.values() for e in range(E)))
check("... and every LUT entry back to -1", all(v == -1 for v in rt.lut.t.values()))
check("... and empties the live set", not rt.h.live)

# the per-step budget survives the round trip through the runtime
rt, masks, store = a_runtime(k=1, margin=0.2, slots=64)
rt.begin_step()
for L in range(40):
    rt.admit(L, [L + 7], [0.9])
check("the runtime stops fetching when the step budget is spent",
      store.reads == STEP_BUDGET_PER_K * 1, f"{store.reads} reads")
rt.begin_step()
rt.admit(0, [100], [0.9])
check("a new step may fetch again", store.reads == STEP_BUDGET_PER_K + 1)
rt.admit(1, [101], [0.19])
check("a candidate under the margin is not fetched by the runtime",
      store.reads == STEP_BUDGET_PER_K + 1)
check("the runtime says whether the layer has to be re-routed",
      rt.admit(2, [102], [0.9]) is True and rt.admit(3, [103], [0.01]) is False)


# ============================================================ pins on the engine source
model_src, fast_src = open(MODEL).read(), open(FAST).read()
engine_src, store_src = open(ENGINE).read(), open(STORE).read()
escape_src, env_src = open(ESCAPE).read(), open(ENV).read()

# --- off by default, everywhere it is read
check("engine/escape.py defaults DSV41_ESCAPE_K to 0",
      bool(re.search(r'env\.get\("DSV41_ESCAPE_K"\)\s*or\s*"0"', escape_src)))
check("K=0 returns no hatch at all", bool(re.search(r"if k == 0:\s*\n\s*return None", escape_src)))
check("the engine builds the hatch from the environment and nothing else",
      bool(re.search(r"self\.escape = EscapeHatch\.from_env\(topk=self\.args\.n_activated_experts\)",
                     engine_src)))
check("with no hatch there is no runtime either",
      bool(re.search(r"self\.escape_rt = None", engine_src)))
check("FastDecoder starts disarmed", bool(re.search(r"self\.esc = None", fast_src)))
check("the hatch is reported in the engine's config dict",
      bool(re.search(r'"escape_k": self\.escape\.k if self\.escape is not None else 0', engine_src)))

# --- the off path is the shipped path: the two mask lines tools/test_route_modes.py pins are
#     untouched, and every escape site is behind an `is not None`
for name, src in (("model.py", model_src), ("fastdecode.py", fast_src)):
    check(f"{name} still hard-masks the logits when substituting, unchanged",
          bool(re.search(r"not (?:self\.)?drop(?:_mode)?:\s*\n(?:\s*#[^\n]*\n)*"
                         r"\s*logits = logits\.masked_fill\(~pm\[L\], float\(\"-inf\"\)\)", src)))
check("model.py's escape call is guarded and decode-only",
      bool(re.search(r"if esc is not None and pruned and not drop and not prefill:\s*\n"
                     r"\s*esc\.consider\(L, logits, scores, k\)", model_src)))
check("... and it runs BEFORE the mask, on the raw router output",
      model_src.index("esc.consider(L, logits, scores, k)")
      < model_src.index('logits = logits.masked_fill(~pm[L], float("-inf"))'))
check("fastdecode.py's scan is guarded",
      bool(re.search(r"if self\.esc is not None and pruned:\s*\n\s*self\._escape_scan\(", fast_src)))
check("the armed LUT gather is nested inside that guard, so the disarmed path still resolves "
      "through the store",
      bool(re.search(r"self\._escape_between\(L\)\s*\n\s*if self\.lut is not None:\s*\n"
                     r"\s*self\.slots\.copy_\(self\.lut\[L\]\[self\.route_slot\]\)\s*\n\s*return",
                     fast_src)))
check("fastdecode.py's host decision is guarded",
      bool(re.search(r"if self\.esc is not None:\s*\n(?:\s*#[^\n]*\n)*\s*self\._escape_between\(L\)",
                     fast_src)))
check("the merged graph segments are kept for the disarmed engine and given up for the armed one",
      bool(re.search(r"if self\.lut is not None and GRAPH_SEGMENTS and self\.esc is None:", fast_src))
      and bool(re.search(r"if self\.lut is not None and self\.esc is None:\s*\n\s*g = torch\.cuda\.CUDAGraph\(\)",
                         fast_src)))
check("the store's eviction hook is None unless something sets it",
      bool(re.search(r"self\.on_transient_evict = None", store_src))
      and bool(re.search(r"if self\.on_transient_evict is not None:\s*\n\s*self\.on_transient_evict\(old\)",
                         store_src)))

# --- the decision: raw scores, non-resident candidates, row 0
check("the device scan masks OFF the resident experts, so every candidate is non-resident",
      bool(re.search(r"cand_lg, cand = raw\[0\]\.masked_fill\(live, float\(\"-inf\"\)\)\.topk\(n\)",
                     fast_src)))
check("the device scan reads the pre-mask logits",
      bool(re.search(r"raw = logits\b", fast_src))
      and fast_src.index("raw = logits")
      < fast_src.index('logits = logits.masked_fill(~pm[L], float("-inf"))'))
check("the margin's denominator is the renormalisation's own denominator",
      bool(re.search(r"den = wts\[0\]\.sum\(\)\.clamp_min\(1e-20\)", fast_src)))
check("the displaced pick is the j-th weakest resident pick",
      bool(re.search(r"m = \(scores\[0\]\[cand\] - wts\[0\]\.flip\(0\)\[:n\]\) / den", fast_src)))
check("both paths give a candidate that displaces nothing a margin of -inf",
      bool(re.search(r"torch\.where\(cand_lg > disp_lg, m, torch\.full_like\(m, float\(\"-inf\"\)\)\)",
                     fast_src))
      and bool(re.search(r"mg if cand_lg\[j\] > disp_lg\[j\] else float\(\"-inf\"\)", engine_src)))
check("the eager path takes the same three quantities",
      bool(re.search(r"cand_v, cand_i = logits\[0\]\.masked_fill\(m, float\(\"-inf\"\)\)\.topk\(n\)",
                     engine_src))
      and bool(re.search(r"disp = res_s\.flip\(0\)\[:n\]", engine_src))
      and bool(re.search(r"den = float\(res_s\.sum\(\)\)", engine_src)))

# --- the re-route is the same arithmetic as the router it replaces
renorm = r"wts = wts / \(wts\.sum\(dim=-1, keepdim=True\) \+ 1e-20\) \* a\.route_scale"
check("_reroute renormalises exactly as _layer_a does", len(re.findall(renorm, fast_src)) >= 2)
check("_reroute takes the same top-k of the same masked logits",
      bool(re.search(r"idx = self\.esc_logits\.masked_fill\(~pm\[L\], float\(\"-inf\"\)\)"
                     r"\.topk\(a\.n_activated_experts, dim=-1\)\[1\]", fast_src)))
check("_reroute gathers the weights from the same scores",
      bool(re.search(r"wts = self\.esc_scores\.gather\(1, idx\)", fast_src)))
check("_reroute writes the buffers layer B reads",
      bool(re.search(r"self\.route_idx\.copy_\(idx\)\s*\n\s*self\.route_w\.copy_\(wts\)", fast_src)))

# --- the fetch, the mask bit and the LUT entry
check("an escape is fetched into the TRANSIENT ring, never the LRU",
      bool(re.search(r"def escape_fetch\(self, layer: int, expert: int\)", store_src))
      and bool(re.search(r"slot = self\._transient_slot_for\(key\)", store_src)))
check("the mask bit and the LUT entry are written together",
      bool(re.search(r"m\[e\] = True\s*\n\s*if self\.lut is not None:\s*\n\s*self\.lut\[L, e\] = slot",
                     engine_src)))
check("... and withdrawn together when the slot is recycled",
      bool(re.search(r"self\.masks\[L\]\[e\] = False\s*\n\s*if self\.lut is not None:\s*\n"
                     r"\s*self\.lut\[L, e\] = -1", engine_src)))
check("the hatch is cleared between requests, so a gate's prompts stay independent",
      bool(re.search(r"if self\.escape_rt is not None:\s*\n(?:\s*#[^\n]*\n)*\s*self\.escape_rt\.clear\(\)",
                     engine_src)))
check("the per-step budget is reset once per decode step",
      bool(re.search(r"self\.escape_rt\.begin_step\(\)", engine_src)))

# --- the combinations the engine refuses
for pat, why in ((r"DSV41_ESCAPE_K needs a keep-set", "no keep-set"),
                 (r"does not combine with DSV41_PRUNE_MODE", "DSV41_PRUNE_MODE=drop"),
                 (r"TRANSIENT_SLOTS", "too few transient slots"),
                 (r"DSV41_ESCAPE_K needs the all-resident device slot LUT", "no device slot LUT")):
    check(f"the engine refuses to arm with {why}", bool(re.search(pat, engine_src)))

# --- the shipped default must not move: env.example may describe the switches, not set them
check("env.example documents DSV41_ESCAPE_K", "DSV41_ESCAPE_K" in env_src)
check("env.example documents DSV41_ESCAPE_MARGIN", "DSV41_ESCAPE_MARGIN" in env_src)
check("env.example leaves DSV41_ESCAPE_K unset, so the shipped behaviour is unchanged",
      not re.search(r"^\s*DSV41_ESCAPE_K\s*=", env_src, re.M))
check("env.example leaves DSV41_ESCAPE_MARGIN unset",
      not re.search(r"^\s*DSV41_ESCAPE_MARGIN\s*=", env_src, re.M))

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
