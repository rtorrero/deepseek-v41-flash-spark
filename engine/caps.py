"""caps.py -- what this device can actually do, in one place.

The engine was written for one machine (a GB10: sm_121a, bf16 and FP4 tensor cores, 228 KB of
shared memory per SM, 48 SMs, unified memory) and every one of those properties is assumed
somewhere. Porting means the assumptions have to become queries, and this module is that query
surface: one place that answers "what dtype do the activations use here?", "is there a hardware
FP4 decode?", "how much shared memory does a block get?", "can a block of experts be computed on
the CPU next to the RAM they live in?" -- so that the call sites do not each grow their own
`if capability >= 8` and drift apart.

The two answers that matter for a V100 (sm_70):

  * **there is no bf16 hardware path before sm_80**. Volta has fp16 tensor cores and no bf16 at
    all, so `DSV41_DTYPE` resolves to fp16 and every bf16 site in the engine has to follow. That
    is deliverable 2 of the port and this module is where the decision lives, so the sweep has one
    definition to converge on instead of a scattered set of local choices.
  * **there is no FP4 decode instruction**. `cvt.rn.f16x2.e2m1x2` is sm_100a/120a/121a; on SM70 the
    decode is integer arithmetic (`tools/fp4_decode.py`, and `tools/fp4_decode_triton.py` for the
    Triton form). `has_fp4_decode()` is the switch, and the kernels should take the portable path
    when it is False rather than failing to assemble.

Nothing here is GPU-specific in a way that stops it running on a CPU-only box: with no CUDA device
the profile is `cpu`, the compute dtype is fp32, and the CPU expert tier is reported as available
if its library is built. That is what makes the module testable on the machine that wrote it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

# Set to "bf16", "fp16" or "fp32" to override the device's default. The engine reads this through
# `compute_dtype()`; nothing should call torch.bfloat16 directly.
ENV_DTYPE = "DSV41_DTYPE"
# "1"/"0" to force the CPU expert tier on or off when it is built and a GPU is present.
ENV_CPU_TIER = "DSV41_CPU_TIER"

# Shared memory and warp counts per architecture family, in bytes per SM. Used to sanity-check a
# launch config rather than to compute one: the GB10 triples in tools/fp4_moe.py were chosen by
# sweeping, and a different SM needs its own sweep, but a config that cannot even fit its tiles in
# shared memory is worth refusing before it is measured.
_SMEM_PER_SM = {
    70: 96 * 1024,      # V100: 96 KB carveout, 64 KB max per block
    75: 64 * 1024,      # Turing
    80: 163 * 1024,     # A100
    86: 99 * 1024,      # A10/3090-class
    89: 99 * 1024,      # L40/4090
    90: 227 * 1024,     # H100
    100: 227 * 1024,    # B200
    120: 227 * 1024,    # Blackwell consumer
}
_WARPS_PER_SM = {70: 4, 75: 4, 80: 4, 86: 4, 89: 4, 90: 4, 100: 4, 120: 4}

# (min capability, name) newest first, for reporting.
_ARCH_NAMES = (
    (120, "Blackwell (sm_120)"),
    (100, "Blackwell (sm_100)"),
    (90, "Hopper"),
    (89, "Ada"),
    (80, "Ampere"),
    (75, "Turing"),
    (70, "Volta"),
)


@dataclass(frozen=True)
class Profile:
    """The device's capabilities, as facts the engine can branch on."""

    kind: str                      # "cuda" | "cpu"
    name: str
    arch: str
    capability: tuple[int, int] | None
    compute_dtype: torch.dtype
    sm_count: int
    smem_bytes_per_sm: int
    warps_per_sm: int
    nvlink: bool
    cpu_expert_tier: bool
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ queries
    @property
    def is_sm70(self) -> bool:
        return self.capability is not None and self.capability[0] == 70

    def has_bf16(self) -> bool:
        """bf16 tensor cores. sm_80 and up; Volta and Turing have none, and triton will emulate
        a bf16 tl.dot with fp32 arithmetic (correct, several times slower, easy to mistake for a
        bandwidth problem)."""
        return self.capability is not None and self.capability[0] >= 80

    def has_fp8(self) -> bool:
        """Native fp8 in the tensor cores: sm_89 and up. On V100 the fp8 paths are conversions
        through fp16, which is what `--kv-cache-dtype fp8_e5m2` plus fp16 compute already does."""
        return self.capability is not None and self.capability[0] >= 89

    def has_fp4_decode(self) -> bool:
        """`cvt.rn.f16x2.e2m1x2`, i.e. the hardware FP4 nibble decode the upstream kernels use.
        Blackwell only -- sm_100a/120a/121a. When this is False the decode must be the integer
        version in tools/fp4_decode.py."""
        return self.capability is not None and self.capability[0] >= 100

    def max_smem_per_block(self) -> int:
        return max_smem_per_block(self.capability)

    def launch_warning(self, num_stages: int, num_warps: int, tile_bytes: int = 0) -> str | None:
        return launch_warning(self.capability, num_stages, num_warps, tile_bytes, self.arch)

    def describe(self) -> str:
        cap = f"{self.capability[0]}.{self.capability[1]}" if self.capability else "-"
        flags = []
        for label, ok in (("bf16", self.has_bf16()), ("fp8", self.has_fp8()),
                          ("fp4-decode", self.has_fp4_decode()), ("nvlink", self.nvlink),
                          ("cpu-tier", self.cpu_expert_tier)):
            flags.append(f"{label}={'yes' if ok else 'no'}")
        out = [f"{self.name} ({self.arch}, capability {cap}), compute dtype {str(self.compute_dtype).replace('torch.', '')}",
               f"  {self.sm_count} SMs, {self.smem_bytes_per_sm // 1024} KB shared/SM "
               f"({self.max_smem_per_block() // 1024} KB per block), {' '.join(flags)}"]
        out.extend(f"  - {n}" for n in self.notes)
        return "\n".join(out)


def default_compute_dtype(capability: tuple[int, int] | None) -> torch.dtype:
    """The dtype rule, as a pure function so it is testable without a GPU.

    bf16 where there are bf16 tensor cores (sm_80 and up), fp16 where there are only fp16 ones
    (Volta, Turing), fp32 with no device at all. This is the decision the whole fp16 sweep
    converges on, so it lives in one place and the test asserts it directly rather than trying to
    reach the CUDA branch of `detect` from a box that has no CUDA.
    """
    if capability is None:
        return torch.float32
    return torch.bfloat16 if capability[0] >= 80 else torch.float16


def max_smem_per_block(capability: tuple[int, int] | None) -> int:
    """What one block can actually request, in bytes. A V100 caps a block at 64 KB even though the
    SM has 96 KB, and a tile sized for a GB10 (up to 228 KB) will not launch at all."""
    if capability is None:
        return 0
    cc = capability[0]
    return min(_SMEM_PER_SM.get(cc, 48 * 1024), 64 * 1024) if cc <= 75 else _SMEM_PER_SM.get(cc, 227 * 1024)


def launch_warning(capability: tuple[int, int] | None, num_stages: int, num_warps: int,
                   tile_bytes: int = 0, arch: str = "this device") -> str | None:
    """A one-line reason a launch config cannot run here, or None.

    This does not replace a sweep: it catches the configurations that cannot run at all, which is
    the class of failure that otherwise shows up as a confusing compile error deep inside a Triton
    kernel. `num_warps` is accepted and unused -- occupancy is a tuning question, not a legality one.
    """
    if capability is None:
        return None
    need = tile_bytes * max(1, num_stages)
    limit = max_smem_per_block(capability)
    if tile_bytes and need > limit:
        return (f"{num_stages} stages of a {tile_bytes} B tile need {need} B of shared memory, "
                f"but {arch} allows {limit} B per block")
    return None


def _dtype_from_env() -> torch.dtype | None:
    want = (os.environ.get(ENV_DTYPE) or "").strip().lower()
    if not want:
        return None
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if want not in table:
        raise ValueError(f"{ENV_DTYPE}={want!r} is not one of {sorted(table)}")
    return table[want]


def _cpu_tier_available() -> bool:
    try:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        if os.path.join(root, "tools") not in sys.path:
            sys.path.insert(0, os.path.join(root, "tools"))
        import fp4_expert_cpu  # noqa: PLC0415
        return fp4_expert_cpu.kernels().available()
    except Exception:
        return False


def detect(device: int | None = None) -> Profile:
    """Probe the current device. Never raises for the absence of CUDA: a CPU profile is a
    supported answer, because everything that plans rather than runs has to work without one."""
    cpu_tier = _cpu_tier_available()
    forced = (os.environ.get(ENV_CPU_TIER) or "").strip()
    if forced in {"0", "false", "False"}:
        cpu_tier = False

    if not torch.cuda.is_available():
        return Profile(kind="cpu", name="cpu", arch="cpu", capability=None,
                       compute_dtype=_dtype_from_env() or torch.float32,
                       sm_count=os.cpu_count() or 1, smem_bytes_per_sm=0, warps_per_sm=0,
                       nvlink=False, cpu_expert_tier=cpu_tier,
                       notes=["no CUDA device: the GPU tiers cannot be planned against this box",
                              "the CPU expert tier is the only compute path here"])

    try:
        idx = torch.cuda.current_device() if device is None else device
        props = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        name = props.name
        sm_count = props.multi_processor_count
        nvlink = bool(getattr(props, "nvlink_domain_size", 1) and int(getattr(props, "nvlink_domain_size", 1)) > 1)
    except Exception as e:  # pragma: no cover - driver present but unhappy
        return Profile(kind="cuda", name=f"cuda:{device}", arch="unknown", capability=None,
                       compute_dtype=_dtype_from_env() or torch.float32, sm_count=0,
                       smem_bytes_per_sm=0, warps_per_sm=0, nvlink=False, cpu_expert_tier=cpu_tier,
                       notes=[f"could not query the device: {e}"])

    cc = cap[0]
    arch = next((label for minimum, label in _ARCH_NAMES if cc >= minimum), f"sm_{cc}")
    notes: list[str] = []
    default = default_compute_dtype(cap)
    if cc < 80:
        notes.append(f"{arch} has no bf16 hardware path: activations, KV and kernels must be fp16")
    if cc < 100:
        notes.append("no hardware FP4 decode: use tools/fp4_decode.py (integer decode), not "
                     "cvt.rn.f16x2.e2m1x2")
    if cc <= 75:
        notes.append(f"a block can request at most {min(_SMEM_PER_SM.get(cc, 48 * 1024), 64 * 1024) // 1024} KB "
                     "of shared memory: the GB10 launch configs in tools/fp4_moe.py must be re-swept")
    if cpu_tier:
        notes.append("CPU expert tier available: RAM-resident experts can be computed in place")
    chosen = _dtype_from_env() or default
    if chosen == torch.bfloat16 and not (cc >= 80):
        notes.append("WARNING: DSV41_DTYPE=bf16 on a device without bf16 tensor cores will be "
                     "emulated through fp32 and is several times slower")
    return Profile(kind="cuda", name=name, arch=arch, capability=cap, compute_dtype=chosen,
                   sm_count=sm_count, smem_bytes_per_sm=_SMEM_PER_SM.get(cc, 0),
                   warps_per_sm=_WARPS_PER_SM.get(cc, 4), nvlink=nvlink,
                   cpu_expert_tier=cpu_tier, notes=notes)


_CACHE: dict[int | None, Profile] = {}


def profile(device: int | None = None, refresh: bool = False) -> Profile:
    if refresh or device not in _CACHE:
        _CACHE[device] = detect(device)
    return _CACHE[device]


def compute_dtype(device: int | None = None) -> torch.dtype:
    """The dtype the engine's activations, caches and kernel outputs should use here.

    fp16 on anything older than Ampere, because Volta and Turing have fp16 tensor cores and no
    bf16 at all. Everything that currently says `torch.bfloat16` -- 239 sites across 12 files,
    listed in docs/sm70-port.md -- resolves through here.
    """
    return profile(device).compute_dtype


def state_dtype(device: int | None = None) -> torch.dtype:
    """Dtype for persistent state (KV cache, recurrent state, Engram rows held on device).
    Same as the compute dtype; separate name so that a future fp8 KV path has one place to change."""
    return compute_dtype(device)


def checkpoint_dtype(target: torch.dtype, source: torch.dtype) -> torch.dtype:
    """What a checkpoint tensor of dtype `source` becomes on this device.

    FP4 stays packed (it is storage, not arithmetic: `int8`/`uint8` with the decode done in the
    multiply either way). Everything else is cast to the compute dtype, which is where a bf16
    checkpoint turns into fp16 weights on Volta.
    """
    if source in (torch.uint8, torch.int8):
        return source
    return target


def summary(device: int | None = None) -> str:
    p = profile(device)
    return p.describe()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="what this device can do")
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()
    p = profile(args.device, refresh=True)
    print(p.describe())
    print(f"\n  compute dtype  {p.compute_dtype}")
    print(f"  state dtype    {state_dtype(args.device)}")
    print(f"  fp4 decode     {'hardware cvt.rn.f16x2.e2m1x2' if p.has_fp4_decode() else 'integer (tools/fp4_decode.py)'}")
    if p.capability is not None and not p.has_bf16():
        print(f"  launch note    64 KB per block, {p.warps_per_sm} warp schedulers: re-sweep "
              "tools/fp4_moe.py's _UP_CFG/_DOWN_CFG here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
