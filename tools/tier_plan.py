"""tier_plan.py -- where the 15,360 expert slots live, and what a generated token costs.

The engine this tree comes from has one tier and one answer: 18,800,640 B per expert slot, all of
it either in the 121 GiB unified pool or streamed off NVMe. A 4x V100 host with 256 GB of DDR4 has
three tiers, and the interesting question stops being "does it fit" (the experts are 288.8 GB and
the box has 384 GB) and becomes "which experts go where, and what does that cost per token".

The cost model is the engine's own byte accounting, not a fitted curve:

    t_token = TOPK x N_LAYERS x slot_bytes x SUM over tiers of mass(tier) / bandwidth(tier)

where `mass(tier)` is the fraction of *routed slots* that land in that tier, and it comes from the
routing trace the repo already produces (`results/<name>/stats/coverage.json`: per layer, the
cumulative coverage of the top-k experts). Nothing here invents a hit rate: with the trace the
masses are measured, and without one the planner says so and falls back to a uniform prior, which
is the pessimistic end.

Memory follows tools/budget.py's terms exactly -- dense weights, the DSpark drafter, KV, the
sliding-window rings, the warm-start scratch, the keep-free floor -- so a plan here and a verdict
from `tools/budget.py` cannot disagree about the same host. The one term budget.py does not have is
the Engram tables, because on the Spark they are the checkpoint's problem and here they are the
NVMe tier's problem: 198 GB of FP8 hash tables, gathered ~12 KB per token, so they want page cache
rather than residence.

Usage:
    python3 tools/tier_plan.py                       # 4x V100 32 GB + 256 GB DDR4, measured CPU rate
    python3 tools/tier_plan.py --profile spark       # the same arithmetic against the single GB10
    python3 tools/tier_plan.py --cpu-gbs 70          # your host's measured number
    python3 tools/tier_plan.py --trace results/trace-union/stats/coverage.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import budget as B  # noqa: E402  (the repo's single source of memory arithmetic)

# The Engram tables, from the checkpoint's own config.json:
#   engram_num_embeddings = [384006168, 384016682] rows over two layers, engram_head_dim = 256,
#   fp8 e4m3 with one e8m0 scale byte per 128 values -> 258 B per row.
ENGRAM_ROWS = 384_006_168 + 384_016_682
ENGRAM_ROW_BYTES = 256 + 2
ENGRAM_BYTES = ENGRAM_ROWS * ENGRAM_ROW_BYTES  # ~198 GB

# How much of the Engram tables should be counted as page cache. The access pattern is 48 rows of
# 258 B per token out of a 384M-row table, so the hot set is small; the default is a compromise
# between trusting that and leaving room for the kernel's own page cache.
ENGRAM_CACHE_GB_DEFAULT = 24.0
# What the host OS and the rest of the process need on the RAM side. budget.py's KEEP_FREE_GB is
# the GPU-side equivalent of this number.
RAM_FLOOR_GB_DEFAULT = 12.0

# Measured here (docs/sm70-port.md): one Ryzen 5900X, AVX2, 12 threads, 29.1 GB/s.
CPU_GBS_MEASURED_RYZEN = 29.1
# Assumed GPU expert bandwidth, to be replaced by a measurement on the V100 host. The GB10 kernel
# reaches ~190-200 GB/s of a 273 GB/s copy ceiling; a V100 has 900 GB/s of HBM2, so this is
# deliberately conservative and flagged as an input rather than a result.
GPU_GBS_ASSUMED = 400.0
NVME_GBS_MEASURED_SPARK = 2.5


@dataclass
class Host:
    """The three tiers, in GB and GB/s. `cpu_gbs` is the one number that must be measured on the
    real host: `tools/bench_fp4_cpu.py` prints it."""

    vram_gb: float
    ram_gb: float
    nvme_gb: float
    cpu_gbs: float
    gpu_gbs: float
    nvme_gbs: float
    cpu_threads: int = 0
    name: str = "host"


@dataclass
class Plan:
    host: Host
    slot_fmt: str
    slot_bytes: int
    per_layer: dict[str, int] = field(default_factory=dict)     # vram/ram/streamed slots per layer
    mass: dict[str, float] = field(default_factory=dict)        # routed-slot fractions per tier
    worst_layer_cov: float = 0.0
    ms: dict[str, float] = field(default_factory=dict)          # per token, per tier
    gb: dict[str, float] = field(default_factory=dict)          # expert bytes per tier
    budget: dict[str, float] = field(default_factory=dict)      # GB available/needed per tier
    notes: list[str] = field(default_factory=list)
    traced: bool = False

    @property
    def ms_token(self) -> float:
        return sum(self.ms.values())

    @property
    def ms_token_overlapped(self) -> float:
        """Best case: the CPU tier and the GPU tier run at the same time, which is the whole
        reason to have two tiers. The NVMe reads are issued by the CPU, so they belong to its
        side of the max. This is a target for the integration, not a measurement."""
        return max(self.ms.get("vram", 0.0), self.ms.get("ram", 0.0) + self.ms.get("nvme", 0.0))

    def tok_s(self, overlapped: bool = False) -> float:
        t = self.ms_token_overlapped if overlapped else self.ms_token
        return 1000.0 / t if t > 0 else float("inf")


def per_layer_slots(budget_gb: float, slot_bytes: int, n_layers: int = B.N_LAYERS) -> int:
    if budget_gb <= 0:
        return 0
    return int(budget_gb * 1e9 // (slot_bytes * n_layers))


def load_coverage(path: str) -> dict[int, list[float]] | None:
    """results/<name>/stats/coverage.json -> {layer: cumulative coverage per kept-expert count}.

    Returns None rather than an empty dict when the file is missing, so the caller can say
    "no trace" instead of silently planning against zeros.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    per_layer = doc.get("per_layer") or {}
    out: dict[int, list[float]] = {}
    for k, v in per_layer.items():
        cov = v.get("cov") if isinstance(v, dict) else None
        if cov:
            out[int(k)] = cov
    return out or None


def _cov_at(cov: list[float], k: int) -> float:
    """Cumulative coverage of the top-`k` experts, from a curve indexed at k=1."""
    if k <= 0:
        return 0.0
    return cov[min(k, len(cov)) - 1]


def masses(coverage: dict[int, list[float]] | None, k_vram: int, k_ram: int,
           n_layers: int = B.N_LAYERS) -> tuple[dict[str, float], float]:
    """Fraction of routed slots served by each tier, averaged over layers, plus the worst layer's
    total coverage (the repo's discipline: a plan is judged by its weakest layer, not its mean)."""
    if coverage is None:
        # Uniform prior: with no trace, every expert is equally likely to be routed, so each tier
        # serves the share of experts it holds. This is the pessimistic end -- a trace always beats
        # it, because routing is skewed and the skewed experts are the ones the budget keeps.
        return ({"vram": k_vram / B.N_EXPERTS, "ram": k_ram / B.N_EXPERTS,
                 "nvme": max(0.0, (B.N_EXPERTS - k_vram - k_ram) / B.N_EXPERTS)},
                (k_vram + k_ram) / B.N_EXPERTS)
    layers = sorted(coverage)[:n_layers]
    v = r = s = 0.0
    worst = 1.0
    for L in layers:
        cov = coverage[L]
        c_v = _cov_at(cov, k_vram)
        c_r = _cov_at(cov, k_vram + k_ram)
        v += c_v
        r += c_r - c_v
        s += 1.0 - c_r
        worst = min(worst, c_r)
    n = max(1, len(layers))
    return {"vram": v / n, "ram": r / n, "nvme": s / n}, worst


def plan(host: Host, coverage: dict[int, list[float]] | None = None, slot_fmt: str = "fp4",
         max_seq: int = 262_144, engram_cache_gb: float = ENGRAM_CACHE_GB_DEFAULT,
         ram_floor_gb: float = RAM_FLOOR_GB_DEFAULT, keep_free_gb: float = B.KEEP_FREE_GB_DEFAULT,
         dense_key: tuple | None = None, report_keep: bool = True) -> Plan:
    slot_bytes = B.EXPERT_BYTES[slot_fmt]
    dense_key = dense_key or B.dense_key_from_env()
    dense_gb = B.DENSE_BYTES.get(dense_key, B.DENSE_DEFAULT) / 1e9
    dspark_gb = B.DSPARK_BYTES / 1e9
    kv_gb = B.kv_bytes(max_seq) / 1e9
    window_gb = B.WINDOW_BYTES / 1e9
    scratch_gb = B.PACK_SCRATCH_BYTES[slot_fmt] / 1e9
    unmodelled_gb = B.UNMODELLED_RESIDENT_GB

    # -------- VRAM: what is left for expert slots once everything else is paid for
    vram_used = dense_gb + dspark_gb + kv_gb + window_gb + scratch_gb + keep_free_gb + unmodelled_gb
    vram_experts_gb = max(0.0, host.vram_gb - vram_used)
    # -------- RAM: the Engram tables want page cache, the OS wants a floor, the rest is experts
    ram_experts_gb = max(0.0, host.ram_gb - ram_floor_gb - min(engram_cache_gb, ENGRAM_BYTES / 1e9))

    k_vram = min(B.N_EXPERTS, per_layer_slots(vram_experts_gb, slot_bytes))
    k_ram = min(B.N_EXPERTS - k_vram, per_layer_slots(ram_experts_gb, slot_bytes))
    k_stream = B.N_EXPERTS - k_vram - k_ram

    mass, worst = masses(coverage, k_vram, k_ram)
    passes = B.TOPK * B.N_LAYERS
    per_pass_bytes = slot_bytes

    def tier_ms(m: float, gbs: float) -> float:
        if m <= 0 or gbs <= 0:
            return 0.0
        return passes * m * per_pass_bytes / (gbs * 1e9) * 1e3

    res = Plan(host=host, slot_fmt=slot_fmt, slot_bytes=slot_bytes, traced=coverage is not None)
    res.per_layer = {"vram": k_vram, "ram": k_ram, "streamed": k_stream}
    res.mass = mass
    res.worst_layer_cov = worst
    res.ms = {"vram": tier_ms(mass["vram"], host.gpu_gbs),
              "ram": tier_ms(mass["ram"], host.cpu_gbs),
              "nvme": tier_ms(mass["nvme"], host.nvme_gbs)}
    res.gb = {"vram": mass["vram"] * passes * per_pass_bytes / 1e9,
              "ram": mass["ram"] * passes * per_pass_bytes / 1e9,
              "nvme": mass["nvme"] * passes * per_pass_bytes / 1e9}
    res.budget = {"vram_total": host.vram_gb, "vram_for_experts": vram_experts_gb,
                  "vram_used_by_rest": vram_used, "ram_total": host.ram_gb,
                  "ram_for_experts": ram_experts_gb, "engram_gb": ENGRAM_BYTES / 1e9,
                  "engram_cache_gb": min(engram_cache_gb, ENGRAM_BYTES / 1e9),
                  "experts_total_gb": B.N_LAYERS * B.N_EXPERTS * slot_bytes / 1e9}

    # Residence is the difference between this host and the Spark: say so explicitly.
    if k_stream == 0:
        res.notes.append(f"all {B.N_LAYERS * B.N_EXPERTS} expert slots are resident "
                         f"(VRAM holds {k_vram}/layer, DDR4 holds {k_ram}/layer): no expert byte is "
                         "streamed, which is the whole point of this host")
    else:
        res.notes.append(f"{k_stream * B.N_LAYERS} slots ({mass['nvme']:.1%} of routed slots) are "
                         f"streamed from NVMe: this is the Spark's regime, at {host.nvme_gbs} GB/s")
    if coverage is None:
        res.notes.append("no routing trace: masses assume every expert is equally likely, which "
                         "understates the benefit of ranking the hot set into VRAM")
    else:
        res.notes.append(f"masses from the trace; weakest layer covers {worst:.3f} of its routing")

    if report_keep and k_stream > 0:
        keep = (k_vram + k_ram) / B.N_EXPERTS
        res.notes.append(f"a keep-set of {keep:.1%} per layer would make it fully resident; the "
                         f"repo's tools/budget.py prices the routing a keep-set gives up")
    return res


def render(p: Plan) -> str:
    h = p.host
    out = [f"tier plan -- {h.name}",
           f"  tiers         {h.vram_gb:.0f} GB VRAM @ {h.gpu_gbs:.0f} GB/s | "
           f"{h.ram_gb:.0f} GB DDR4 @ {h.cpu_gbs:.1f} GB/s | {h.nvme_gb:.0f} GB NVMe @ {h.nvme_gbs:.1f} GB/s",
           f"  slot          {p.slot_fmt}, {p.slot_bytes / 1e6:.2f} MB  "
           f"({p.budget['experts_total_gb']:.1f} GB for all {B.N_LAYERS * B.N_EXPERTS})",
           ""]
    out.append(f"  VRAM budget   {h.vram_gb:.1f} GB total - {p.budget['vram_used_by_rest']:.2f} GB "
               f"(dense, drafter, KV, rings, scratch, floors) = {p.budget['vram_for_experts']:.1f} GB "
               f"-> {p.per_layer['vram']} slots/layer")
    out.append(f"  RAM budget    {h.ram_gb:.1f} GB total - {p.budget['engram_cache_gb']:.0f} GB Engram "
               f"page cache - floors = {p.budget['ram_for_experts']:.1f} GB "
               f"-> {p.per_layer['ram']} slots/layer")
    out.append(f"  NVMe          {p.budget['engram_gb']:.0f} GB Engram (gathered ~12 KB/token) + "
               f"{p.per_layer['streamed'] * B.N_LAYERS} streamed slots")
    out.append("")
    out.append(f"  {'tier':<8} {'mass':>7} {'GB/token':>9} {'ms/token':>9} {'tok/s alone':>12}")
    for t in ("vram", "ram", "nvme"):
        alone = 1000.0 / p.ms[t] if p.ms[t] > 0 else float("inf")
        out.append(f"  {t:<8} {p.mass[t]:>7.3f} {p.gb[t]:>9.2f} {p.ms[t]:>9.1f} {alone:>12.1f}")
    out.append(f"  {'total':<8} {sum(p.mass.values()):>7.3f} {sum(p.gb.values()):>9.2f} "
               f"{p.ms_token:>9.1f} {p.tok_s():>12.1f}")
    out.append("")
    out.append(f"  serialised     {p.ms_token:>7.1f} ms/token -> {p.tok_s():>6.1f} tok/s")
    out.append(f"  overlapped     {p.ms_token_overlapped:>7.1f} ms/token -> {p.tok_s(True):>6.1f} tok/s   "
               "(CPU expert work under GPU attention; a target, not a measurement)")
    for n in p.notes:
        out.append(f"  note: {n}")
    return "\n".join(out)


PROFILES = {
    # 4x V100 32 GB (31.8 GB usable per card), a server-class DDR4 host.
    "v100": Host(vram_gb=127.2, ram_gb=256.0, nvme_gb=2000.0, cpu_gbs=CPU_GBS_MEASURED_RYZEN,
                 gpu_gbs=GPU_GBS_ASSUMED, nvme_gbs=NVME_GBS_MEASURED_SPARK, cpu_threads=16,
                 name="4x V100 32 GB + 256 GB DDR4"),
    # The single GB10 the engine ships for: no separate RAM tier, so the expert bytes beyond the
    # pool budget are streamed. This profile exists to check the planner against a measured box.
    "spark": Host(vram_gb=121.0, ram_gb=0.0, nvme_gb=600.0, cpu_gbs=CPU_GBS_MEASURED_RYZEN,
                  gpu_gbs=190.0, nvme_gbs=NVME_GBS_MEASURED_SPARK, cpu_threads=20,
                  name="1x DGX Spark (121 GiB unified)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="v100")
    ap.add_argument("--vram", type=float, default=None, help="override VRAM GB")
    ap.add_argument("--ram", type=float, default=None)
    ap.add_argument("--cpu-gbs", type=float, default=None,
                    help="the host's measured CPU expert rate (tools/bench_fp4_cpu.py)")
    ap.add_argument("--gpu-gbs", type=float, default=None,
                    help="effective GPU expert bandwidth; measure it, do not trust the default")
    ap.add_argument("--nvme-gbs", type=float, default=None)
    ap.add_argument("--slot", choices=sorted(B.EXPERT_BYTES), default="fp4",
                    help="fp4 (18.80 MB) or cb3 (14.45 MB, lower traffic, quality cost documented)")
    ap.add_argument("--max-seq", type=int, default=262_144)
    ap.add_argument("--trace", default=None, help="results/<name>/stats/coverage.json")
    ap.add_argument("--engram-cache-gb", type=float, default=ENGRAM_CACHE_GB_DEFAULT)
    args = ap.parse_args()

    h = PROFILES[args.profile]
    h = Host(vram_gb=args.vram if args.vram is not None else h.vram_gb,
             ram_gb=args.ram if args.ram is not None else h.ram_gb,
             nvme_gb=h.nvme_gb,
             cpu_gbs=args.cpu_gbs if args.cpu_gbs is not None else h.cpu_gbs,
             gpu_gbs=args.gpu_gbs if args.gpu_gbs is not None else h.gpu_gbs,
             nvme_gbs=args.nvme_gbs if args.nvme_gbs is not None else h.nvme_gbs,
             cpu_threads=h.cpu_threads,
             name=h.name + ("" if (args.vram is None and args.ram is None and args.cpu_gbs is None
                                   and args.gpu_gbs is None) else " (overridden)"))

    trace = args.trace
    if trace is None:
        for cand in ("results/trace-union/stats/coverage.json",
                     "results/trace-full-20260910/stats/coverage.json"):
            if os.path.exists(cand):
                trace = cand
                break
    cov = load_coverage(trace) if trace else None
    p = plan(h, cov, slot_fmt=args.slot, max_seq=args.max_seq, engram_cache_gb=args.engram_cache_gb)
    if trace and cov:
        print(f"# routing trace: {trace} ({len(cov)} layers)\n")
    else:
        print("# no routing trace found: masses are the uniform prior\n")
    print(render(p))
    if args.gpu_gbs is None and args.profile == "v100":
        print(f"\n  The {GPU_GBS_ASSUMED:.0f} GB/s GPU rate is an input, not a measurement. Run the")
        print("  upstream tools/test_fp4_moe.py on the V100 once the portable decode is in and pass")
        print("  its number back through --gpu-gbs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
