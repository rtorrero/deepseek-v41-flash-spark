"""test_tier_plan.py -- the three-tier planner, held against the box it was built from.

The strongest check available here is a cross-check against hardware that has been measured: the
DGX Spark profile, run through the same arithmetic and the same routing trace, has to land in the
regime the repo's own README reports for that box (a ~25% resident arena, ~0.9 GB of NVMe per
generated token, ~2.6 tok/s). It is not an identity -- the planner models three tiers, the Spark
has one plus an SSD -- but reproducing it from first principles is what makes the V100 plan worth
reading.

No GPU, no checkpoint. The routing trace is the one already in results/.

Usage:  python3 tools/test_tier_plan.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import budget as B  # noqa: E402
import tier_plan as T  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


TRACE = None
for cand in ("results/trace-union/stats/coverage.json",
             "results/trace-full-20260910/stats/coverage.json"):
    if os.path.exists(cand):
        TRACE = cand
        break


def test_arithmetic() -> None:
    print("[1] the slot arithmetic is exact and never over-allocates")
    slot = B.EXPERT_BYTES["fp4"]
    check("per_layer_slots honours the byte budget", T.per_layer_slots(100.6, slot) == int(100.6e9 // (slot * 40)),
          f"{T.per_layer_slots(100.6, slot)}")
    check("zero budget -> zero slots", T.per_layer_slots(0.0, slot) == 0)
    check("a negative budget cannot happen but would also be zero",
          T.per_layer_slots(-5.0, slot) == 0)

    h = T.PROFILES["v100"]
    p = T.plan(h, None)
    total = p.per_layer["vram"] + p.per_layer["ram"] + p.per_layer["streamed"]
    check("the three tiers account for all 384 experts per layer", total == B.N_EXPERTS, f"{total}")
    check("the VRAM line item is the sum of its terms",
          abs(p.budget["vram_used_by_rest"]
              - (B.DENSE_BYTES.get(B.dense_key_from_env(), B.DENSE_DEFAULT) / 1e9
                 + B.DSPARK_BYTES / 1e9 + B.kv_bytes(262_144) / 1e9 + B.WINDOW_BYTES / 1e9
                 + B.PACK_SCRATCH_BYTES["fp4"] / 1e9 + B.KEEP_FREE_GB_DEFAULT
                 + B.UNMODELLED_RESIDENT_GB)) < 1e-6,
          f"{p.budget['vram_used_by_rest']:.3f}")
    check("expert slots claimed fit the bytes available",
          (p.per_layer["vram"] * 40 * slot / 1e9) <= p.budget["vram_for_experts"] + 1e-9
          and (p.per_layer["ram"] * 40 * slot / 1e9) <= p.budget["ram_for_experts"] + 1e-9)


def test_uniform_prior() -> None:
    print("[2] with no trace, the masses are the slot shares and the plan says so")
    p = T.plan(T.PROFILES["v100"], None)
    check("masses sum to 1", abs(sum(p.mass.values()) - 1.0) < 1e-9, f"{sum(p.mass.values()):.4f}")
    check("it is not marked as traced", not p.traced)
    check("it warns that the ranking benefit is not modelled",
          any("no routing trace" in n for n in p.notes))
    share = p.per_layer["vram"] / B.N_EXPERTS
    check("VRAM's mass is its share of the 384 experts under the uniform prior",
          abs(p.mass["vram"] - share) < 1e-9, f"{p.mass['vram']:.4f} vs {share:.4f}")
    check("the prior's masses sum to 1 exactly", abs(sum(p.mass.values()) - 1.0) < 1e-12)


def test_masses_from_trace() -> None:
    print("[3] with a trace, the masses are the measured routing")
    cov = T.load_coverage(TRACE) if TRACE else None
    if cov is None:
        print("       (no trace in results/: skipped)")
        return
    layers = sorted(cov)[:B.N_LAYERS]
    # Hand-compute the expected masses for one allocation and compare.
    k_v, k_r = 133, 251
    v = sum(T._cov_at(cov[L], k_v) for L in layers) / len(layers)
    r = sum(T._cov_at(cov[L], k_v + k_r) - T._cov_at(cov[L], k_v) for L in layers) / len(layers)
    worst = min(T._cov_at(cov[L], k_v + k_r) for L in layers)
    p = T.plan(T.PROFILES["v100"], cov)
    check("VRAM mass == mean over layers of cov[133]", abs(p.mass["vram"] - v) < 1e-9, f"{p.mass['vram']:.4f}")
    check("RAM mass == mean of cov[384] - cov[133]", abs(p.mass["ram"] - r) < 1e-9, f"{p.mass['ram']:.4f}")
    check("the worst layer is the minimum, not the mean", abs(p.worst_layer_cov - worst) < 1e-9,
          f"{p.worst_layer_cov:.4f}")
    check("the trace is named in the output", p.traced)
    check("coverage curves are monotone (the trace's own invariant)",
          all(all(c[i] <= c[i + 1] + 1e-9 for i in range(len(c) - 1)) for c in cov.values()))


def test_v100_plan() -> None:
    print("[4] the 4x V100 plan")
    cov = T.load_coverage(TRACE) if TRACE else None
    p = T.plan(T.PROFILES["v100"], cov)
    check("every expert slot is resident", p.per_layer["streamed"] == 0, f"{p.per_layer}")
    check("nothing is streamed from NVMe", p.mass["nvme"] < 1e-9, f"{p.mass['nvme']:.4f}")
    check("worst-layer routing coverage is complete",
          p.worst_layer_cov > 0.999 if cov else True, f"{p.worst_layer_cov:.4f}")
    check("the CPU tier is the bottleneck, as expected on DDR4",
          p.ms["ram"] > p.ms["vram"], f"ram {p.ms['ram']:.1f} vs vram {p.ms['vram']:.1f} ms")
    check("a defensible token rate", 5.0 < p.tok_s() < 60.0, f"{p.tok_s():.1f} tok/s serialised")
    check("overlapping can only help", p.ms_token_overlapped <= p.ms_token + 1e-9,
          f"{p.ms_token_overlapped:.1f} <= {p.ms_token:.1f}")
    check("bytes per token are the routed bytes, not the whole model",
          abs(sum(p.gb.values()) - B.TOPK * B.N_LAYERS * p.slot_bytes / 1e9) < 1e-9,
          f"{sum(p.gb.values()):.2f} GB/token")


def test_spark_cross_check() -> None:
    print("[5] the same arithmetic against the measured box (the model's only calibration)")
    cov = T.load_coverage(TRACE) if TRACE else None
    if cov is None:
        print("       (no trace: skipped)")
        return
    p = T.plan(T.PROFILES["spark"], cov)
    tps = p.tok_s()
    nvme_gb = p.gb["nvme"]
    # Measured on that box (repo README): 2.64-2.71 tok/s, ~0.92 GB of NVMe per generated token,
    # a 25.6% resident arena with a 0.83 hit rate. The planner has a different tier structure
    # (it treats the unified pool as VRAM and the SSD as NVMe) so this is a regime check, not an
    # identity: if the planner claimed 20 tok/s or 0.1 GB/token for that box it would be lying.
    check("predicts the same order of magnitude as the measured 2.6-2.7 tok/s",
          1.2 <= tps <= 5.0, f"{tps:.2f} tok/s")
    check("predicts the same NVMe traffic regime as the measured 0.92 GB/token",
          0.5 <= nvme_gb <= 1.6, f"{nvme_gb:.2f} GB/token")
    check("predicts roughly the measured static coverage of that box",
          # The repo reports 0.748 static coverage at 4,000 resident slots, measured. At 125
          # slots/layer this planner says 0.739 of the routed mass, which is the same number from
          # the same trace -- the small gap is the per-layer allocation the arena actually uses
          # versus the global curve the 0.748 was read off.
          0.60 <= p.mass["vram"] <= 0.85, f"{p.mass['vram']:.3f} of the routed mass")
    check("predicts a roughly quarter-resident arena",
          0.15 <= p.per_layer["vram"] / B.N_EXPERTS <= 0.45,
          f"{p.per_layer['vram']} of {B.N_EXPERTS} slots per layer")
    check("most of the step is the SSD, which is the box's known bottleneck",
          p.ms["nvme"] > 3 * p.ms["vram"], f"nvme {p.ms['nvme']:.0f} vs vram {p.ms['vram']:.0f} ms")


def test_sensitivities() -> None:
    print("[6] sensitivities: more RAM, lower bits, slower GPU")
    cov = T.load_coverage(TRACE) if TRACE else None
    base = T.plan(T.PROFILES["v100"], cov)
    more = T.plan(T.Host(**{**vars(T.PROFILES["v100"]), "ram_gb": 512.0}), cov)
    check("more RAM never lowers the token rate", more.tok_s(True) >= base.tok_s(True) - 1e-9,
          f"{more.tok_s(True):.1f} vs {base.tok_s(True):.1f}")
    cb3 = T.plan(T.PROFILES["v100"], cov, slot_fmt="cb3")
    check("cb3 slots fit strictly more experts per layer",
          cb3.per_layer["vram"] > base.per_layer["vram"] and cb3.per_layer["ram"] < base.per_layer["ram"],
          f"vram {cb3.per_layer['vram']} vs {base.per_layer['vram']}")
    check("cb3 moves mass into the fast tier", cb3.mass["vram"] > base.mass["vram"],
          f"{cb3.mass['vram']:.3f} vs {base.mass['vram']:.3f}")
    # A GPU slower than the CPU tier stops being hidden by it: that is the crossover the split
    # lives or dies on, and it is worth asserting rather than assuming.
    slow = T.plan(T.Host(**{**vars(T.PROFILES["v100"]), "gpu_gbs": 10.0}), cov)
    check("a slower GPU slows the serialised rate", slow.tok_s() < base.tok_s(),
          f"{slow.tok_s():.1f} vs {base.tok_s():.1f}")
    check("a GPU slower than the CPU tier also slows the overlapped rate",
          slow.tok_s(True) < base.tok_s(True), f"{slow.tok_s(True):.1f} vs {base.tok_s(True):.1f}")
    fast = T.plan(T.Host(**{**vars(T.PROFILES["v100"]), "gpu_gbs": 1000.0}), cov)
    check("a much faster GPU stops helping once the CPU tier is the bottleneck",
          abs(fast.tok_s(True) - base.tok_s(True)) < 1e-9,
          f"{fast.tok_s(True):.1f} vs {base.tok_s(True):.1f}")
    tiny = T.plan(T.Host(vram_gb=40.0, ram_gb=64.0, nvme_gb=1000.0, cpu_gbs=20.0, gpu_gbs=300.0,
                         nvme_gbs=2.5, name="too small"), cov)
    check("a host that cannot hold the experts says so and streams the rest",
          tiny.per_layer["streamed"] > 0 and tiny.mass["nvme"] > 0,
          f"{tiny.per_layer['streamed']} slots/layer streamed")


def test_engram_and_render() -> None:
    print("[7] the Engram accounting and the report")
    rows = 384_006_168 + 384_016_682
    check("Engram bytes derive from the config's own row counts",
          abs(T.ENGRAM_BYTES - rows * (256 + 2)) < 1,
          f"{T.ENGRAM_BYTES / 1e9:.1f} GB")
    check("the derived figure is within a few percent of the 203 GB of the two shards that carry it",
          190e9 < T.ENGRAM_BYTES < 210e9, f"{T.ENGRAM_BYTES / 1e9:.1f} GB")
    p = T.plan(T.PROFILES["v100"], T.load_coverage(TRACE) if TRACE else None)
    text = T.render(p)
    for needle in ("VRAM budget", "RAM budget", "tok/s", "Engram", "serialised", "overlapped"):
        check(f"render() mentions {needle!r}", needle in text)


def main() -> int:
    print("test_tier_plan.py -- the three-tier planner\n")
    test_arithmetic()
    test_uniform_prior()
    test_masses_from_trace()
    test_v100_plan()
    test_spark_cross_check()
    test_sensitivities()
    test_engram_and_render()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
