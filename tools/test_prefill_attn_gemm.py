"""The prefill-only attention GEMM math type: DSV41_PREFILL_ATTN_GEMM.

Run: python3 tools/test_prefill_attn_gemm.py

Torch-free, so it runs on the CPU runner alongside the rest of the suite. The CUDA half -- the
three modes against each other on the real shapes, with error bounds -- is
`engine/test_attn_gemm_modes.py`, which self-skips where there is no GPU.

Three halves:

1. The parser, the alias and the prefill guard, exercised directly on the shipped module.
   `engine/prefill_attn_gemm.py` imports nothing but `contextlib` and `os`, so the object under
   test here is the one the engine loads, not a copy of it.

2. `math_scope` as a context manager, without torch: the fp32 and bf16 modes must be empty, must
   import nothing and must not so much as read a torch attribute -- that is what "the default is
   byte-identical" means at this level.

3. Regex pins on `engine/model.py`. What matters about this switch is not that it works -- it is
   a dtype and a context manager -- but that it is bf16 by default, stays inside the attention
   softmax, and stays PREFILL-ONLY. A mode that leaked into the decode path would change the
   drafted tokens the verify step is asked to accept and nothing would crash; the acceptance rate
   would just quietly fall. And a TF32 scope that was armed and never restored would do the same
   to every decode step after the prompt. So both guards are pinned mechanically.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import prefill_attn_gemm as AG  # noqa: E402
from engine.prefill_attn_gemm import ALIAS, DEFAULT, ENV_VAR, MODES, from_env, parse  # noqa: E402

MODEL = os.path.join(ROOT, "engine/model.py")
HOOK = os.path.join(ROOT, "engine/prefill_attn_gemm.py")
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


# ============================================================ the parser
check("the default is bf16", DEFAULT == "bf16" and MODES == ("fp32", "tf32", "bf16"))
check("unset means bf16", parse(None) == "bf16" and parse("") == "bf16")
check("the off spellings mean the default, bf16",
      all(parse(v) == "bf16" for v in ("off", "default", "none", " OFF ", "Default")))
check("each mode parses to itself", [parse(m) for m in MODES] == list(MODES))
check("case and whitespace are tolerated", parse("  TF32 ") == "tf32" and parse("BF16") == "bf16")
check("a mode that does not exist is refused",
      raises(parse, "fp16") and raises(parse, "tf16") and raises(parse, "1") and raises(parse, "on"))


def _msg(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


msg = _msg(parse, "fp16")
check("the refusal names the variable and lists the modes",
      ENV_VAR in msg and all(m in msg for m in MODES), msg)

# ============================================================ the environment and the alias
check("neither variable set is bf16", from_env({}) == "bf16")
check("the variable is read", from_env({ENV_VAR: "tf32"}) == "tf32")
check("the alias is read when the variable is unset", from_env({ALIAS: "bf16"}) == "bf16")
check("the variable wins over the alias",
      from_env({ENV_VAR: "tf32", ALIAS: "bf16"}) == "tf32")
check("an empty variable falls through to the alias",
      from_env({ENV_VAR: "  ", ALIAS: "bf16"}) == "bf16")
check("a bad value in the alias is refused too", raises(from_env, {ALIAS: "fp16"}))
check("the alias is the name the investigation started under",
      ALIAS == "DSV41_PREFILL_HC_GEMM" and ENV_VAR == "DSV41_PREFILL_ATTN_GEMM")

# ============================================================ the prefill guard
check("with neither variable set the module starts in bf16", AG.MODE == "bf16")
check("the default: prefill bf16, decode fp32", AG.mode_for(True) == "bf16" and AG.mode_for(False) == "fp32")
AG.MODE = "fp32"
check("fp32, every path is fp32", AG.mode_for(True) == "fp32" and AG.mode_for(False) == "fp32")
for m in ("tf32", "bf16"):
    AG.MODE = m
    try:
        check(f"{m}: prefill gets the mode", AG.mode_for(True) == m)
        check(f"{m}: DECODE stays fp32", AG.mode_for(False) == "fp32")
    finally:
        AG.MODE = "bf16"
check("the sweep leaves the module back in its shipped state", AG.MODE == "bf16")

# ============================================================ math_scope
check("torch is not loaded yet, so 'does not import torch' below means something",
      "torch" not in sys.modules)

entered = []
for m in ("fp32", "bf16"):
    with AG.math_scope(m):
        entered.append(m)
check("the fp32 and bf16 scopes are entered and left without torch",
      entered == ["fp32", "bf16"] and "torch" not in sys.modules)


class _FakeMatmul:
    allow_tf32 = False


class _FakeBackends:
    class cuda:  # noqa: N801  -- mirrors torch.backends.cuda.matmul
        matmul = _FakeMatmul


class _FakeTorch:
    backends = _FakeBackends
    precision = "highest"

    @staticmethod
    def get_float32_matmul_precision():
        return _FakeTorch.precision

    @staticmethod
    def set_float32_matmul_precision(p):
        _FakeTorch.precision = p
        # torch's own coupling: the precision string rewrites allow_tf32.
        _FakeMatmul.allow_tf32 = p in ("high", "medium")


sys.modules["torch"] = _FakeTorch
try:
    with AG.math_scope("tf32"):
        check("tf32 arms allow_tf32", _FakeMatmul.allow_tf32 is True)
        check("tf32 arms the precision string", _FakeTorch.precision == "high")
    check("tf32 restores allow_tf32 on the way out", _FakeMatmul.allow_tf32 is False)
    check("tf32 restores the precision string", _FakeTorch.precision == "highest")

    def _boom():
        with AG.math_scope("tf32"):
            raise RuntimeError("a chunk blew up")

    try:
        _boom()
    except RuntimeError:
        pass
    check("an exception inside the scope still restores both settings",
          _FakeMatmul.allow_tf32 is False and _FakeTorch.precision == "highest")

    _FakeMatmul.allow_tf32 = True
    _FakeTorch.precision = "high"
    with AG.math_scope("tf32"):
        pass
    check("a caller that already had tf32 on keeps it",
          _FakeMatmul.allow_tf32 is True and _FakeTorch.precision == "high")
    _FakeMatmul.allow_tf32 = False
    _FakeTorch.precision = "highest"

    for m in ("fp32", "bf16"):
        with AG.math_scope(m):
            pass
        check(f"the {m} scope touches no torch setting",
              _FakeMatmul.allow_tf32 is False and _FakeTorch.precision == "highest")
finally:
    del sys.modules["torch"]

# ============================================================ pins on engine/model.py
src = open(MODEL).read()

check("model.py imports the switch from the torch-free module",
      bool(re.search(r"^from engine import prefill_attn_gemm as AG", src, re.M)))
check("the mode is chosen from `prefill`, nowhere else",
      src.count("AG.mode_for(") == 1 and bool(re.search(r"mode = AG\.mode_for\(prefill\)", src)))
check("the operand dtype is bf16 only in the bf16 mode",
      bool(re.search(r'dt = torch\.bfloat16 if mode == "bf16" else torch\.float32', src)))
check("the tile loop runs inside the scope",
      bool(re.search(r"with AG\.math_scope\(mode\):\s*\n\s*for i in range\(0, T, B\):", src)))
check("math_scope is used exactly once, and only there", src.count("AG.math_scope(") == 1)
check("the two operands are widened with .to(dt), not with .float()",
      bool(re.search(r"qt, kvt, mt = q\[i:j\]\.to\(dt\), kv_all\[i:j\]\.to\(dt\), mask\[i:j\]", src)))
check("the score product is widened back so the softmax stays fp32",
      bool(re.search(r'scores = torch\.einsum\("thd,tnd->thn", qt, kvt\)\.float\(\) \* scale', src)))
check("the probabilities are cast to the operand dtype for the PV product",
      bool(re.search(r'torch\.einsum\("thn,tnd->thd", \(p / denom\)\.to\(dt\), kvt\)', src)))
check("_softmax_attn takes `prefill` and defaults it to False",
      bool(re.search(r"def _softmax_attn\(self, q, kv_all, mask, sink, prefill: bool = False\):", src)))
check("attention takes `prefill` and defaults it to False",
      bool(re.search(r"win_lo: int = 0, prefill: bool = False\):", src)))
check("attention passes it straight through to the softmax",
      bool(re.search(r"o = self\._softmax_attn\(q, kv_all, mask, w\.attn_sink, prefill\)", src)))
check("block hands attention its own prefill flag",
      bool(re.search(r"self\.attention\(y, w, L, S, sh, ring, freqs, mtp_extra, "
                     r"win_lo=win_lo, prefill=prefill\)", src)))
check("the DSpark drafter's blocks are still prefill=False",
      bool(re.search(r"self\.freqs_w, False,\s*\n\s*self\.W\.dspark_store", src)))

check("engine/prefill_attn_gemm.py is the only module that reads either variable from the environment",
      not any(re.search(r"environ[^\n]*(" + ENV_VAR + "|" + ALIAS + ")",
                        open(os.path.join(ROOT, d, f)).read())
              for d, f in (("engine", "model.py"), ("engine", "fastdecode.py"),
                           ("engine", "v41_engine.py"), ("tools", "v41_ref.py"))))
check("the decode path never mentions the switch",
      "AG." not in open(os.path.join(ROOT, "engine/fastdecode.py")).read()
      and "prefill_attn_gemm" not in open(os.path.join(ROOT, "engine/fastdecode.py")).read())
check("nothing sets allow_tf32 outside the hook",
      not re.search(r"allow_tf32", src)
      and not re.search(r"allow_tf32", open(os.path.join(ROOT, "tools/v41_ref.py")).read())
      and not re.search(r"allow_tf32", open(os.path.join(ROOT, "engine/fastdecode.py")).read()))
check("the hook itself imports torch only inside the tf32 branch",
      bool(re.search(r"if mode != \"tf32\":\s*\n\s*yield\s*\n\s*return\s*\n\s*import torch",
                     open(HOOK).read())))
check("the hook restores the precision string before allow_tf32",
      bool(re.search(r"finally:\s*\n\s*torch\.set_float32_matmul_precision\(prev_prec\)\s*\n"
                     r"\s*mm\.allow_tf32 = prev_allow", open(HOOK).read())))

# The arithmetic the finding rests on: ATTN_TILE query rows per tile, two products per tile,
# 21 encoder layers -> the profile's 672 calls of each kernel at a 2,048-token chunk.
tile = int(re.search(r"^ATTN_TILE = (\d+)", src, re.M).group(1))
check("ATTN_TILE is still 64, which is what makes the count 32 tiles at a 2,048-token chunk",
      tile == 64 and 2048 // tile == 32 and (2048 // tile) * 21 == 672)

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
