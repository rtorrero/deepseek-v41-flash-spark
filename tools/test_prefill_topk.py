"""The prefill-only top-k switch: DSV41_PREFILL_TOPK / DSV41_PREFILL_FOLD.

Run: python3 tools/test_prefill_topk.py

Torch-free, so it runs on the CPU runner alongside the rest of the suite. Three halves, the same
shape tools/test_route_modes.py uses:

1. The parsers and the override function, exercised directly. `engine/prefill_topk.py` imports
   nothing but `os`, so the real shipped object is under test here, not a copy of it.

2. Regex pins on `engine/model.py`. What matters about this switch is not that it works -- the
   arithmetic is `tools/test_exfold.py`'s job -- but that it stays OFF, stays PREFILL-ONLY, and
   that the router's own top-k is still issued at the checkpoint's k so the fold knows which
   experts it is excluding. A switch that leaked into the decode path would change the drafted
   tokens the verify step is asked to accept, and nothing would crash; the acceptance rate would
   just quietly fall.

3. Pins that the old measurement-only variable is gone. `DSV41_PREFILL_TOPK_TEST` was the hook
   `tools/prefill_bound_test.py` used to ask whether prefill is compute-bound; it is, and this is
   what replaced it. Two mechanisms for the same thing is how a measurement ends up taken on a
   configuration nobody serves.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.prefill_topk import (ENV_FOLD, ENV_K, ENV_TABLE, TABLE_FILENAME,  # noqa: E402
                                 parse_fold, parse_k, prefill_topk, table_path)

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


# ============================================================ the k parser
check("unset means no override", parse_k(None) is None and parse_k("") is None)
check("the off spellings mean no override",
      all(parse_k(v) is None for v in ("off", "0", "none", "default", " OFF ", "Off")))
check("an integer in range is the override", [parse_k(str(k)) for k in (1, 3, 4, 6)] == [1, 3, 4, 6])
check("whitespace around the integer is tolerated", parse_k("  4 ") == 4)
check("a value above the checkpoint's k is refused", raises(parse_k, "7"))
check("zero is 'off', not an error", parse_k("0") is None)
check("a negative value is refused", raises(parse_k, "-1"))
check("a non-integer is refused", raises(parse_k, "yes") and raises(parse_k, "4.5"))
check("the default k the parser validates against is the caller's",
      parse_k("8", 8) == 8 and raises(parse_k, "8", 6))


def _msg(fn, *a):
    try:
        fn(*a)
    except ValueError as e:
        return str(e)
    return ""


check("the error names the variable", ENV_K in _msg(parse_k, "7") and ENV_K in _msg(parse_k, "yes"))


# ============================================================ the fold parser
# The default is the POINT of this variable: reducing the top-k without saying how to compensate
# is the arm the paper measures as the worse one, so an unset variable must not reach it.
check("unset, with a reduced k, defaults to exfold", parse_fold("", 4) == "exfold")
check("unset, with no reduction, is inert", parse_fold("", None) == "none")
check("the mode can be asked for explicitly",
      parse_fold("exfold", 4) == "exfold" and parse_fold("none", 4) == "none")
check("the spellings that mean 'just drop them' land on none",
      all(parse_fold(v, 4) == "none" for v in ("off", "drop", "0", " NONE ")))
check("an unknown mode is refused, not silently ignored", raises(parse_fold, "fold", 4))
check("the error names the variable", ENV_FOLD in _msg(parse_fold, "fold", 4))


# ============================================================ the table path
check("an explicit path wins", table_path("/x/y.npz", "/m") == "/x/y.npz")
check("otherwise the table sits next to the checkpoint",
      table_path("", "/m") == os.path.join("/m", TABLE_FILENAME))
check("with neither there is no path", table_path("", None) is None)
check("a ~ in the path is expanded",
      table_path("~/t.npz", None) == os.path.expanduser("~/t.npz"))


# ============================================================ the override function
import engine.prefill_topk as PT  # noqa: E402

check("the module reads the environment once, at import",
      "OVERRIDE = parse_k(os.environ.get(ENV_K)" in open(HOOK).read())
check("with the variable unset the module starts with no override", PT.OVERRIDE is None)
check("... and with no fold in force", PT.FOLD == "none")

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
      bool(re.search(r"k_pre = PT\.prefill_topk\(k, drop=drop\) if \(prefill and n_experts != 128\)"
                     r" else k", src)))
check("the router still issues the CHECKPOINT's k, so the fold knows what it excluded",
      bool(re.search(r"indices = logits\.topk\(k, dim=-1\)\[1\]", src))
      and "logits.topk(k_pre" not in src)
check("the reduction happens after the checkpoint's own renormalisation",
      src.index("weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale")
      < src.index("if k_pre < k:"))
check("... and before the slot lookup, so the kernel only ever sees the surviving routes",
      src.index("if k_pre < k:") < src.index("slots = lut[L][slot_idx]"))
check("the hook is the ONLY place model.py mentions it", src.count("PT.prefill_topk") == 1)
check("engine/prefill_topk.py is the only module that reads the variables out of the environment",
      not any(re.search(r"environ[^\n]*(" + ENV_K + "|" + ENV_FOLD + ")",
                        open(os.path.join(ROOT, d, f)).read())
              for d, f in (("engine", "model.py"), ("engine", "fastdecode.py"),
                           ("engine", "exfold.py"), ("tools", "v41_ref.py"))))
check("the decode path's topk is untouched",
      bool(re.search(r"idx = logits\.topk\(a\.n_activated_experts, dim=-1\)\[1\]",
                     open(os.path.join(ROOT, "engine/fastdecode.py")).read())))
check("the table variable is read once, by the engine, at start",
      open(os.path.join(ROOT, "engine/v41_engine.py")).read().count(f"PT.{'ENV_TABLE'}") == 1
      and ENV_TABLE not in src)


# ============================================================ the measurement hook is gone
for path in ("engine/model.py", "engine/prefill_topk.py", "engine/v41_engine.py",
             "tools/prefill_bound_test.py", "env.example", "docs/gemm-dispatch.md"):
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        check(f"{path} no longer mentions the retired measurement variable",
              "DSV41_PREFILL_TOPK_TEST" not in open(p).read())

print(f"\n{len(fails)} failed" if fails else "\nall ok")
sys.exit(1 if fails else 0)
