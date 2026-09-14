"""The prefill-only top-k measurement hook: DSV41_PREFILL_TOPK_TEST.

Run: python3 tools/test_prefill_topk.py

Torch-free, so it runs on the CPU runner alongside the rest of the suite. Two halves, the same
shape tools/test_route_modes.py uses:

1. The parser and the override function, exercised directly. `engine/prefill_topk.py` imports
   nothing but `os`, so the real shipped object is under test here, not a copy of it.

2. Regex pins on `engine/model.py`. What matters about this hook is not that it works -- it is
   four lines -- but that it stays OFF and stays PREFILL-ONLY. A hook that leaked into the decode
   path would change the drafted tokens the verify step is asked to accept, and nothing would
   crash; the acceptance rate would just quietly fall. So the guard is pinned mechanically.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.prefill_topk import ENV_VAR, parse, prefill_topk  # noqa: E402

MODEL = os.path.join(ROOT, "engine/model.py")
HOOK = os.path.join(ROOT, "engine/prefill_topk.py")
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
check("unset means no override", parse(None) is None and parse("") is None)
check("the off spellings mean no override",
      all(parse(v) is None for v in ("off", "0", "none", "default", " OFF ", "Off")))
check("an integer in range is the override", [parse(str(k)) for k in (1, 3, 4, 6)] == [1, 3, 4, 6])
check("whitespace around the integer is tolerated", parse("  4 ") == 4)
check("a value above the checkpoint's k is refused", raises(parse, "7"))
check("zero is 'off', not an error", parse("0") is None)
check("a negative value is refused", raises(parse, "-1"))
check("a non-integer is refused", raises(parse, "yes") and raises(parse, "4.5"))
check("the default k the parser validates against is the caller's",
      parse("8", 8) == 8 and raises(parse, "8", 6))


def _msg(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


check("the error names the variable", ENV_VAR in _msg(parse, "7") and ENV_VAR in _msg(parse, "yes"))

# ============================================================ the override function
import engine.prefill_topk as PT  # noqa: E402

check("the module reads the environment once, at import",
      "OVERRIDE = parse(os.environ.get(ENV_VAR)" in open(HOOK).read())
check("with the variable unset the module starts with no override", PT.OVERRIDE is None)

# --- off: byte-identical, for every k a caller could pass
check("off returns the caller's k unchanged",
      all(prefill_topk(k) == k for k in range(1, 9))
      and all(prefill_topk(k, drop=True) == k for k in range(1, 9)))

# --- on
PT.OVERRIDE = 4
try:
    check("on returns the override", prefill_topk(6) == 4)
    check("the override never RAISES k", prefill_topk(3) == 3 and prefill_topk(4) == 4)
    check("on, drop mode is refused rather than silently mis-shaped",
          raises(prefill_topk, 6, True))
finally:
    PT.OVERRIDE = None
check("the sweep leaves the module back in its shipped state", PT.OVERRIDE is None)

# ============================================================ pins on engine/model.py
src = open(MODEL).read()

check("model.py imports the hook from the torch-free module",
      bool(re.search(r"^from engine import prefill_topk as PT", src, re.M)))
check("the call site is guarded by `prefill` and skips the DSpark drafter's own router",
      bool(re.search(r"if prefill and n_experts != 128:\s*\n\s*k = PT\.prefill_topk\(k, drop=drop\)",
                     src)))
check("... and it sits immediately before the router's topk",
      bool(re.search(r"k = PT\.prefill_topk\(k, drop=drop\)\s*\n"
                     r"\s*indices = logits\.topk\(k, dim=-1\)\[1\]", src)))
check("the hook is the ONLY place model.py mentions it", src.count("PT.prefill_topk") == 1)
check("engine/prefill_topk.py is the only module that reads the variable out of the environment",
      not any(re.search(r"environ[^\n]*" + ENV_VAR, open(os.path.join(ROOT, d, f)).read())
              for d, f in (("engine", "model.py"), ("engine", "fastdecode.py"),
                           ("engine", "v41_engine.py"), ("tools", "v41_ref.py"))))
check("the decode path's topk is untouched",
      bool(re.search(r"idx = logits\.topk\(a\.n_activated_experts, dim=-1\)\[1\]",
                     open(os.path.join(ROOT, "engine/fastdecode.py")).read())))

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
