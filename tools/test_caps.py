"""test_caps.py -- the capability layer, which is where the fp16-not-bf16 decision lives.

No GPU required, and that is the design: the rules that decide dtype, FP4 decode availability and
shared-memory limits are pure functions of a compute capability, so they can be asserted on a box
that has never had a driver. `detect()` itself is exercised on whatever this machine is (a CPU
profile is a supported answer and has to stay one, because everything that *plans* rather than
*runs* -- tools/tier_plan.py, tools/budget.py -- has to work without a device).

Usage:  python3 tools/test_caps.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import caps as C  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def test_dtype_rule() -> None:
    print("[1] the dtype rule")
    check("sm_70 (V100) -> fp16, because Volta has no bf16 at all",
          C.default_compute_dtype((70, 0)) == torch.float16)
    check("sm_75 (Turing) -> fp16", C.default_compute_dtype((75, 0)) == torch.float16)
    check("sm_80 (A100) -> bf16", C.default_compute_dtype((80, 0)) == torch.bfloat16)
    check("sm_120 (GB10) -> bf16", C.default_compute_dtype((120, 0)) == torch.bfloat16)
    check("no device -> fp32", C.default_compute_dtype(None) == torch.float32)


def test_env_override() -> None:
    print("[2] DSV41_DTYPE overrides the rule, and a typo is refused")
    old = os.environ.get(C.ENV_DTYPE)
    try:
        os.environ[C.ENV_DTYPE] = "fp16"
        C.profile(refresh=True)
        check("DSV41_DTYPE=fp16 wins over the device default",
              C.compute_dtype() == torch.float16, f"{C.compute_dtype()}")
        os.environ[C.ENV_DTYPE] = "half"
        C.profile(refresh=True)
        check("'half' is accepted as a synonym", C.compute_dtype() == torch.float16)
        os.environ[C.ENV_DTYPE] = "bfloat16"
        C.profile(refresh=True)
        p = C.profile()
        if p.capability is not None and not p.has_bf16():
            check("bf16 on a non-bf16 device is allowed but warned about",
                  any("WARNING" in n for n in p.notes), "no warning emitted")
        else:
            print("       (no device or bf16-capable device: the warning path is not reachable here)")
        os.environ[C.ENV_DTYPE] = "float8"
        try:
            C.profile(refresh=True)
            check("an unknown DSV41_DTYPE raises", False, "no exception")
        except ValueError:
            check("an unknown DSV41_DTYPE raises", True)
    finally:
        if old is None:
            os.environ.pop(C.ENV_DTYPE, None)
        else:
            os.environ[C.ENV_DTYPE] = old
        C.profile(refresh=True)


def test_capability_gates() -> None:
    print("[3] the gates that decide which kernel a device gets")
    class Fake:
        def __init__(self, cc):
            self.capability = (cc, 0)

    rows = {70: (False, False, False, 64), 75: (False, False, False, 64),
            80: (True, False, False, 163), 89: (True, True, False, 99),
            90: (True, True, False, 227), 100: (True, True, True, 227),
            120: (True, True, True, 227)}
    for cc, (bf16, fp8, fp4, smem) in rows.items():
        f = Fake(cc)
        got = (C.Profile.has_bf16(f), C.Profile.has_fp8(f), C.Profile.has_fp4_decode(f),
               C.max_smem_per_block(f.capability) // 1024)
        check(f"sm_{cc}: bf16={bf16} fp8={fp8} fp4_decode={fp4} smem={smem}KB", got == (bf16, fp8, fp4, smem),
              f"{got}")


def test_launch_warning() -> None:
    print("[4] a launch config that cannot fit is refused before it is measured")
    # 3 stages of a 64 KB tile = 192 KB: fits a Hopper (227 KB), does not fit a V100 (64 KB). That
    # is the concrete class of bug this catches -- a GB10-tuned config carried to Volta.
    cfg = 64 * 1024
    warn70 = C.launch_warning((70, 0), 3, 4, cfg, "Volta")
    check("sm_70 refuses 3 stages of a 64 KB tile", warn70 is not None, warn70 or "")
    check("sm_90 accepts the same config", C.launch_warning((90, 0), 3, 4, cfg, "Hopper") is None)
    small = 16 * 1024  # 2 stages of 16 KB = 32 KB, inside Volta's 64 KB
    check("sm_70 accepts 2 stages of a 16 KB tile", C.launch_warning((70, 0), 2, 4, small) is None,
          C.launch_warning((70, 0), 2, 4, small) or "")
    check("even Hopper refuses 4 stages of a 128 KB tile",
          C.launch_warning((90, 0), 4, 4, 128 * 1024, "Hopper") is not None)
    check("no device -> no warning (there is nothing to launch on)",
          C.launch_warning(None, 4, 4, 128 * 1024) is None)


def test_checkpoint_dtype() -> None:
    print("[5] what a checkpoint tensor becomes on this device")
    check("packed FP4 stays packed (it is storage, not arithmetic)",
          C.checkpoint_dtype(torch.float16, torch.uint8) == torch.uint8
          and C.checkpoint_dtype(torch.float16, torch.int8) == torch.int8)
    check("a bf16 weight becomes the compute dtype (fp16 on Volta)",
          C.checkpoint_dtype(torch.float16, torch.bfloat16) == torch.float16)
    check("an fp8 weight also becomes the compute dtype",
          C.checkpoint_dtype(torch.float16, torch.float8_e4m3fn) == torch.float16)


def test_detect_here() -> None:
    print("[6] detect() on this machine")
    p = C.profile(refresh=True)
    check("returns a profile rather than raising", p is not None)
    check("kind is cuda or cpu", p.kind in ("cuda", "cpu"), p.kind)
    check("describe() renders", "compute dtype" in p.describe())
    d = p.describe()
    check("an sm<80 device reports the fp16 requirement and the missing FP4 decode",
          (p.capability is None or p.capability[0] >= 100)
          or ("fp16" in d and "no hardware FP4 decode" in d), d.replace("\n", " | "))
    tier = p.cpu_expert_tier
    expect = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fp4_cpu.so"))
    check("the CPU expert tier flag matches whether its library is built", tier == expect,
          f"flag={tier} library={expect}")
    if p.kind == "cpu":
        print(f"       (no CUDA device here: compute dtype {p.compute_dtype}, "
              f"CPU tier {'available' if tier else 'not built'})")


def main() -> int:
    print("test_caps.py -- capability and dtype policy\n")
    test_dtype_rule()
    test_env_override()
    test_capability_gates()
    test_launch_warning()
    test_checkpoint_dtype()
    test_detect_here()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {', '.join(FAILS)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
