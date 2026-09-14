"""The fused-Sinkhorn-at-prefill switch: DSV41_PREFILL_FUSED_SINKHORN.

Run: python3 tools/test_prefill_sinkhorn.py

Torch-free, so it runs on the CPU runner alongside the rest of the suite. The numerics -- that the
fused kernel and the tiled torch path agree at prefill row counts -- are
`tools/test_hc_sinkhorn_prefill.py`, which needs CUDA and self-skips without it. Three halves:

1. The parser, exercised directly on the shipped `engine/prefill_sinkhorn.py`.

2. The launch arithmetic. The whole point of this switch is a launch count, so the count is
   computed here from the shapes rather than quoted from a doc: 698,880 launches per 2,048-token
   chunk against 42.

3. Regex pins on `engine/model.py`, `tools/v41_ref.py` and `engine/hc_sinkhorn.py`. What matters is
   that the switch is PREFILL-ONLY and that OFF is the tiled path that shipped. A fused Sinkhorn
   leaking into the un-graphed decode forward would move the DSpark verify block's numbers and
   nothing would crash; the acceptance rate would just quietly change. So the guard is pinned
   mechanically, the same way tools/test_prefill_topk.py pins the top-k hook.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.prefill_sinkhorn import ENV_VAR, fused_prefill, parse  # noqa: E402

HOOK = os.path.join(ROOT, "engine/prefill_sinkhorn.py")
MODEL = os.path.join(ROOT, "engine/model.py")
REF = os.path.join(ROOT, "tools/v41_ref.py")
KERNEL = os.path.join(ROOT, "engine/hc_sinkhorn.py")
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
check("unset means the tiled path", parse(None) is False and parse("") is False)
check("the off spellings mean the tiled path",
      all(parse(v) is False for v in ("0", "off", "no", "false", "none", "default", " OFF ")))
check("the on spellings mean the fused kernel",
      all(parse(v) is True for v in ("1", "on", "yes", "true", " On ", "TRUE")))
check("anything else is refused", raises(parse, "fused") and raises(parse, "2"))


def _msg(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


check("the error names the variable", ENV_VAR in _msg(parse, "2"))

# ============================================================ the guard
import engine.prefill_sinkhorn as PS  # noqa: E402

check("the module reads the environment once, at import",
      "ENABLED = parse(os.environ.get(ENV_VAR))" in open(HOOK).read())
check("with the variable unset the module starts off", PS.ENABLED is False)
check("off, neither phase is fused", not fused_prefill(True) and not fused_prefill(False))

PS.ENABLED = True
try:
    check("on, a prefill call is fused", fused_prefill(True) is True)
    check("on, a DECODE call is still the tiled path", fused_prefill(False) is False)
finally:
    PS.ENABLED = False
check("the sweep leaves the module back in its shipped state", PS.ENABLED is False)

# ============================================================ the launch arithmetic
# hc_split_sinkhorn is ~130 tiny torch kernels: the two sigmoid heads (~5 each), the softmax and
# the first column normalisation (~5), then 19 iterations of a row and a column normalisation at 3
# kernels each.
OPS_PER_CALL = 5 + 5 + 2 + 2 + 3 + 19 * 6
MM_TILE = 16
CHUNK = 2048
ENCODER_LAYERS = 21
CALLS_PER_LAYER = 2          # Model.block runs hc_mixes for the attention and for the FFN

tiles = -(-CHUNK // MM_TILE)
tiled = tiles * OPS_PER_CALL * CALLS_PER_LAYER * ENCODER_LAYERS
fused = CALLS_PER_LAYER * ENCODER_LAYERS

check("the tiled Sinkhorn is ~130 launches per call", 125 <= OPS_PER_CALL <= 135, str(OPS_PER_CALL))
check("a 2,048-token chunk tiles into 128 calls of it", tiles == 128)
check("which is ~700,000 launches per chunk", 690_000 <= tiled <= 710_000, f"{tiled}")
check("the fused kernel is one launch per hc_mixes: 42 per chunk", fused == 42)
check("the ratio is four orders of magnitude", tiled // fused > 10_000, f"{tiled // fused}x")

# ============================================================ pins on the engine
src = open(MODEL).read()
ref = open(REF).read()
ker = open(KERNEL).read()

check("model.py imports the hook from the torch-free module",
      bool(re.search(r"^from engine import prefill_sinkhorn as PS", src, re.M)))
check("model.py hands v41_ref the kernel decode already uses -- it does not grow a second one",
      bool(re.search(r"from engine\.hc_sinkhorn import hc_split_sinkhorn as _hc_fused", src))
      and bool(re.search(r"^\s*R\.HC_SINKHORN_FUSED = _hc_fused$", src, re.M)))
check("the fused kernel is imported defensively, so a torch-only checkout still runs",
      bool(re.search(r"try:(?:.|\n)*?from engine\.hc_sinkhorn import hc_split_sinkhorn as _hc_fused"
                     r"(?:.|\n)*?except Exception:", src)))
check("the call site is guarded by `prefill`",
      bool(re.search(r"fused_hc = PS\.fused_prefill\(prefill\)", src)))
check("both hc_mixes calls in block() take that one flag",
      src.count("fused=fused_hc") == 2 and src.count("fused_hc = PS.fused_prefill(prefill)") == 1)
check("v41_ref defaults to no fused kernel on its own",
      bool(re.search(r"^HC_SINKHORN_FUSED = None$", ref, re.M)))
check("hc_mixes defaults to the tiled path and needs BOTH the flag and the kernel",
      bool(re.search(r"def hc_mixes\(x: torch\.Tensor, hc_fn, hc_scale, hc_base, args: Args, "
                     r"fused: bool = False\):", ref))
      and bool(re.search(r"if fused and HC_SINKHORN_FUSED is not None:", ref)))
check("off falls through to the tiled torch Sinkhorn, unchanged",
      bool(re.search(r"return tiled_rows\(lambda t: hc_split_sinkhorn\(t, hc_scale, hc_base, args\.hc_mult,",
                     ref)))
check("the GEMM and the rsqrt above the Sinkhorn are still tiled either way",
      bool(re.search(r"xf = x\.flatten\(1\)\.float\(\)\s*\n"
                     r"\s*rsqrt = rms_rsqrt\(xf, args\.norm_eps\)\s*\n"
                     r"\s*mixes = mm\(xf, hc_fn\) \* rsqrt", ref)))
check("engine/prefill_sinkhorn.py is the only module that reads the variable out of the environment",
      not any(re.search(r"environ[^\n]*" + ENV_VAR, open(os.path.join(ROOT, d, f)).read())
              for d, f in (("engine", "model.py"), ("engine", "fastdecode.py"),
                           ("engine", "v41_engine.py"), ("tools", "v41_ref.py"),
                           ("engine", "hc_sinkhorn.py"))))
check("fastdecode still calls the kernel directly, untouched",
      bool(re.search(r"^from engine\.hc_sinkhorn import hc_split_sinkhorn$",
                     open(os.path.join(ROOT, "engine/fastdecode.py")).read(), re.M)))

# The kernel was written for decode; these are the two properties that make it legal at T = 2048.
check("the kernel is one program per row, so there is no batch or row limit to extend",
      bool(re.search(r"_hc_kernel\[\(max\(n, 1\),\)\]", ker)))
check("... and it guards the tail of the grid",
      bool(re.search(r"row = tl\.program_id\(0\)\s*\n\s*if row >= n_rows:\s*\n\s*return", ker)))

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
