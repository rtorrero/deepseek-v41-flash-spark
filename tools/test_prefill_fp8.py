"""The prefill fp8 dequant switch: DSV41_PREFILL_FP8_DEQUANT.

Run: python3 tools/test_prefill_fp8.py

Torch-free, so it runs on the CPU runner alongside the rest of the suite. The numerics -- that
`fused` and `cached` produce the same bf16 bits as `FP8Weight.dequant()` -- are
`tools/test_fp8_dequant.py`, which needs torch and self-skips without it. Three halves here:

1. The parser and the mode helpers, exercised directly on the shipped `engine/prefill_fp8.py`,
   including the refusal of `scaled_mm` and the reason it carries.

2. The memory arithmetic of `cached`. It is the number `tools/budget.py` has to charge against the
   prefill reserve, so it is pinned to the shapes it was derived from rather than left as a
   comment: change a shape and this test says what the new reserve is.

3. Regex pins on `engine/model.py`, `tools/v41_ref.py` and `tools/fp8_linear.py`. What matters
   about this switch is not that it works -- it is one branch -- but that OFF is the path that
   shipped, byte for byte, and that `FP8Weight.dequant()` itself is untouched so the off path is
   literally the old code and not a re-derivation of it.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.prefill_fp8 import (ENV_VAR, MODES, cache_bytes, parse)  # noqa: E402

HOOK = os.path.join(ROOT, "engine/prefill_fp8.py")
MODEL = os.path.join(ROOT, "engine/model.py")
REF = os.path.join(ROOT, "tools/v41_ref.py")
FP8 = os.path.join(ROOT, "tools/fp8_linear.py")
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


def _msg(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


# ============================================================ the parser
check("unset means the shipped path, fused", parse(None) == "fused" and parse("") == "fused"
      and parse("default") == "fused")
check("the off spellings mean the shipped path",
      all(parse(v) == "off" for v in ("off", "0", "none", " OFF ", "Off")))
check("the two implemented modes parse", parse("fused") == "fused" and parse("cached") == "cached")
check("whitespace and case are tolerated", parse("  Fused ") == "fused" and parse("CACHED") == "cached")
check("an unknown mode is refused", raises(parse, "dequant") and raises(parse, "yes"))
check("the error names the variable and the modes",
      ENV_VAR in _msg(parse, "dequant") and "fused" in _msg(parse, "dequant"))

# scaled_mm is a name with a reason behind it, not a typo
check("scaled_mm is refused, not silently ignored", raises(parse, "scaled_mm"))
m = _msg(parse, "scaled_mm")
check("... and the refusal says why: the 32x32 scale layout",
      "32x32" in m and "_scaled_mm" in m and "fused" in m)
check("scaled_mm is still one of the documented spellings", "scaled_mm" in MODES)
check("MODES lists exactly what the parser knows", MODES == ("off", "fused", "cached", "scaled_mm"))

# ============================================================ the mode helpers
import engine.prefill_fp8 as PF  # noqa: E402

check("the module reads the environment once, at import",
      "MODE = parse(os.environ.get(ENV_VAR))" in open(HOOK).read())
check("with the variable unset the module starts fused", PF.MODE == "fused")
PF.MODE = "off"
check("off means neither branch is taken", not PF.enabled() and not PF.cached())

PF.MODE = "fused"
try:
    check("fused is enabled and not cached", PF.enabled() and not PF.cached())
    PF.MODE = "cached"
    check("cached is enabled and cached", PF.enabled() and PF.cached())
finally:
    PF.MODE = "fused"
check("the sweep leaves the module back in its shipped state", PF.MODE == "fused")

# ============================================================ what `cached` costs
# The five weights that are still FP8Weight under DSV41_DENSE_FP4=attn,wo_a, at the checkpoint's
# shapes, over the layers each of them lives on.
SHARED = 2 * (2304 * 5120) + 5120 * 2304          # w1, w3, w2 of one layer
WQ_B = 4096 * 1280                                 # indexer, one index-source layer
WKV = 25600 * 6144                                 # engram, one engram layer
MAIN_PROJ = 5120 * 15360                           # mtp.0, once

encoder = 21 * SHARED + 4 * WQ_B + 2 * WKV         # layers 0..20, index {2,8,14,20}, engram {1,14}
whole = 40 * SHARED + 8 * WQ_B + 2 * WKV + MAIN_PROJ

check("the encoder pass caches the weights of layers 0..20", cache_bytes(whole_prompt=False) == encoder * 2,
      f"{cache_bytes(whole_prompt=False)} != {encoder * 2}")
check("a whole prompt adds the decoder replay and the DSpark seed", cache_bytes(whole_prompt=True) == whole * 2,
      f"{cache_bytes(whole_prompt=True)} != {whole * 2}")
check("the encoder pass is 2.16 GB", abs(cache_bytes(whole_prompt=False) / 1e9 - 2.157) < 0.01,
      f"{cache_bytes(whole_prompt=False) / 1e9:.3f} GB")
check("a whole prompt is 3.70 GB", abs(cache_bytes(whole_prompt=True) / 1e9 - 3.701) < 0.01,
      f"{cache_bytes(whole_prompt=True) / 1e9:.3f} GB")
check("CACHE_BYTES is the whole-prompt high-water mark, which is what a reserve has to clear",
      PF.CACHE_BYTES == cache_bytes(whole_prompt=True))
check("the two engram wkv are 0.63 GB of that 2.16 -- the item nobody expects to be large",
      abs(2 * WKV * 2 / 1e9 - 0.629) < 0.01, f"{2 * WKV * 2 / 1e9:.3f} GB")
check("the 21 layers of shared experts are 1.49 GB of it",
      abs(21 * SHARED * 2 / 1e9 - 1.486) < 0.01, f"{21 * SHARED * 2 / 1e9:.3f} GB")

# ============================================================ pins on the engine
src = open(MODEL).read()
ref = open(REF).read()
fp8 = open(FP8).read()

check("model.py imports the hook from the torch-free module",
      bool(re.search(r"^from engine import prefill_fp8 as PF", src, re.M)))
check("model.py pushes the mode into v41_ref the way it pushes MM_TILE",
      bool(re.search(r"^R\.PREFILL_FP8_MODE = PF\.MODE$", src, re.M)))
check("v41_ref defaults to the shipped path on its own",
      bool(re.search(r'^PREFILL_FP8_MODE = "off"$', ref, re.M)))
check("dense() routes the M > 16 fp8 branch through the hook",
      bool(re.search(r"if x\.numel\(\) // x\.shape\[-1\] <= 16:\s*\n"
                     r"\s*return fp8_linear\(x, w\)\s*\n"
                     r"\s*return F\.linear\(x\.to\(torch\.bfloat16\), prefill_dequant\(w\)\)", ref)))
check("off calls FP8Weight.dequant() and nothing else",
      bool(re.search(r'if mode == "off" or fp8_dequant_fused is None:\s*\n\s*return w\.dequant\(\)', ref)))
check("FP8Weight.dequant() itself is untouched, so the off path IS the old code",
      bool(re.search(r"def dequant\(self\) -> torch\.Tensor:\s*\n"
                     r"\s*s = torch\.exp2\(self\.s\.float\(\) - 127\.0\)\s*\n"
                     r"\s*s = s\.repeat_interleave\(32, 0\)\[: self\.N\]\.repeat_interleave\(32, 1\)\[:, : self\.K\]\s*\n"
                     r"\s*return \(self\.w\.float\(\) \* s\)\.to\(torch\.bfloat16\)", fp8)))
check("the fused dequant is a Triton kernel in the module that owns FP8Weight",
      "@triton.jit" in fp8 and "def _fp8_dequant_kernel(" in fp8 and "def dequant_fused(" in fp8)
check("the cache is cleared when a prompt begins",
      bool(re.search(r"def begin_prompt\(self\):(?:.|\n)*?R\.prefill_fp8_cache_clear\(\)", src)))
check("... and again when the decoder replay ends, i.e. before the first decode step",
      bool(re.search(r"R\.prefill_fp8_cache_clear\(\)\s*\n"
                     r"\s*return logits, \(torch\.cat\(main_hiddens", src)))
check("... and after a full one-pass prefill forward, which has no later chunk to reuse it",
      bool(re.search(r"if prefill:\s*\n(?:\s*#[^\n]*\n)*\s*R\.prefill_fp8_cache_clear\(\)", src)))
check("engine/prefill_fp8.py is the only module that reads the variable out of the environment",
      not any(re.search(r"environ[^\n]*" + ENV_VAR, open(os.path.join(ROOT, d, f)).read())
              for d, f in (("engine", "model.py"), ("engine", "fastdecode.py"),
                           ("engine", "v41_engine.py"), ("tools", "v41_ref.py"),
                           ("tools", "fp8_linear.py"))))
check("the decode path is not touched: fastdecode never mentions the hook",
      "prefill_fp8" not in open(os.path.join(ROOT, "engine/fastdecode.py")).read())
check("nothing reaches torch._scaled_mm",
      "_scaled_mm" not in ref and "_scaled_mm" not in fp8)

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
