"""PRUNE_KEEP=auto: the keep fraction a context length can afford.

Run: python3 tools/test_keep_for_context.py

The thing being tested is a promise made to a launcher: `./start.sh` resolves
`auto` before it spends three minutes loading 80 GB, and whatever comes back is
what the box then runs for as long as it is up. So the numbers here are pinned
against a FAKE host -- the GB10's own memory model, not whatever machine this
runs on -- and the two that matter are the two the record measured: 0.36 holds a
filled 256k context, and a shorter context may hold more.

The host is 130.6 GB total / 117.0 GB available, which is what `budget.read_host`
shows for a 121 GiB GB10 when it cannot read /proc. Every expectation below is
that model's arithmetic, not a wish: at 64k, keep 0.40 leaves 9.0 GB free where
one prefill chunk behind a filled 64k context needs 10.7, which is the same 0.40
the watchdog killed 582 s into a 195k-token prefill (RESULTS.md 2026-09-13
22:50). 0.38 is the largest step that clears it.

No terminal, no GPU, no checkpoint.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402
import keep_for_context as K  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(name)


# The box, as the memory model describes it. Built by hand rather than read:
# this file must answer the same way on the box and on a laptop.
HOST = B.Host("gb10-test", 130.6e9, 117.0e9, True)


def keep(max_seq, **kw):
    return K.resolve(HOST, max_seq, K.AUTO, **kw).keep


# --- the ladder -------------------------------------------------------------
check("the steps are the ones tune.py slides along", (K.KEEP_STEPS[0], K.KEEP_STEPS[-1],
                                                      len(K.KEEP_STEPS)), (0.06, 0.6, 28))
check("0.36 is on it", 0.36 in K.KEEP_STEPS, True)

# --- what each context can afford -------------------------------------------
# A filled 256k context is the one the record measured, and it is the one number
# in this file that a change to the memory model must not be allowed to move
# quietly: 0.36 served it, 0.40 did not.
check("256k resolves to the fraction that held a filled 256k", keep(262144), 0.36)
check("128k", keep(131072), 0.38)
check("64k", keep(65536), 0.38)
check("32k", keep(32768), 0.38)

# Monotone in the context: a longer context can never afford MORE experts. Cheap
# to state, and the property that makes the answer trustworthy without a table.
ladder = [keep(n) for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144)]
check("never rises with the context", ladder == sorted(ladder, reverse=True), True)

# ... and every answer is a step, never a number between two of them
check("every answer is on the ladder", all(k in K.KEEP_STEPS for k in ladder), True)

# The step it lands on actually fits, and the step above it does not -- which is
# what "largest" has to mean or the resolution is just a constant.
for n in (32768, 262144):
    k = keep(n)
    above = K.KEEP_STEPS[K.KEEP_STEPS.index(k) + 1]
    check(f"{n // 1024}k: the resolved step fits",
          B.plan(HOST, None, (), k, n, fmt="cb3").verdict != "over", True)
    check(f"{n // 1024}k: the step above does not",
          B.plan(HOST, None, (), above, n, fmt="cb3").verdict != "over", False)

# --- the engine's own pre-flight margin is in it ----------------------------
r = K.resolve(HOST, 262144, K.AUTO)
check("the resolution leaves the launcher's pre-flight satisfied", r.plan.launch_slack >= 0, True)
check("  and a prefill chunk plus the watchdog floor on top of it",
      r.plan.free_after_load >= r.plan.need_free, True)
check("  the spare it reports is the tighter of the two",
      round(r.spare, 6) == round(min(r.plan.launch_slack,
                                     r.plan.free_after_load - r.plan.need_free), 6), True)
check("the line says what it resolved and why",
      r.line, "PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144 (fits with 3.3 GB spare)")

# --- KEEP_FREE_GB and the transient ring are honoured ------------------------
check("a bigger launcher floor cannot raise the answer",
      keep(32768, keep_free_gb=20) <= keep(32768), True)
check("a 400-slot transient ring costs the kept set its own slots",
      keep(32768, transient_slots=400) <= keep(32768), True)
check("fp4 slots are bigger, so fewer of them fit",
      keep(32768, fmt="fp4") < keep(32768, fmt="cb3"), True)

# --- ARENA_GB ---------------------------------------------------------------
# A pinned arena is not second-guessed: the answer is the most it holds.
slots, cap = K.arena_cap(81, "cb3", 8)
check("81 GB is the arena the 256k run was measured at", (slots, round(cap, 3)), (5603, 0.364))
check("and auto inside it lands on the fraction that run used",
      keep(262144, arena_gb=81), 0.36)
check("a small arena caps the keep below what memory would allow",
      keep(32768, arena_gb=60), 0.26)
r_small = K.resolve(HOST, 32768, K.AUTO, arena_gb=60)
check("  and says which arena did the capping",
      any("caps the keep" in n for n in r_small.notes), True)

# An arena too big for the context cannot be rescued by lowering the keep, since
# the arena is what is pinned. Say so instead of walking the ladder to the floor.
r_big = K.resolve(HOST, 262144, K.AUTO, arena_gb=90)
check("an over-budget pinned arena still answers with what it holds", r_big.keep, 0.40)
check("  reports that it does not fit", r_big.fits, False)
check("  and says the keep cannot fix it",
      any("the keep cannot fix it" in n for n in r_big.notes), True)

# --- a number is passed through untouched -----------------------------------
r_num = K.resolve(HOST, 262144, 0.44)
check("a pinned number is not resolved away", r_num.keep, 0.44)
check("  it is marked as pinned", r_num.pinned, True)
check("  and costed: 0.44 does not serve a filled 256k", r_num.fits, False)
check("  the line names it as pinned", r_num.line.startswith("PRUNE_KEEP=0.44 -> 0.44"), True)
r_ok = K.resolve(HOST, 32768, 0.36)
check("a pinned number that fits says so", (r_ok.pinned, r_ok.fits), (True, True))
check("  a string is a number too", K.resolve(HOST, 32768, "0.36").keep, 0.36)

# --- the gate floor is a warning, not a clamp -------------------------------
check("the floor is the smallest keep any gate was run at", K.GATED_FLOOR, 0.36)
tiny = B.Host("small-box", 60e9, 55e9, True)
r_tiny = K.resolve(tiny, 32768, K.AUTO)
check("a box that can only hold less is given less, not the floor",
      r_tiny.keep < K.GATED_FLOOR, True)
check("  with the warning that nothing that small has been gated",
      any("has been gated" in n for n in r_tiny.notes), True)
check("  and it is not clamped up to something the box cannot hold", r_tiny.fits, True)


# --- the CLI ----------------------------------------------------------------
def run(args, env=None):
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools/keep_for_context.py")] + args,
                          capture_output=True, text=True,
                          env={**os.environ, "DSV41_HOST_TOTAL_GB": "130.6",
                               "DSV41_HOST_AVAIL_GB": "117", **(env or {})})


r = run(["--max-seq", "262144"])
check("the CLI prints the line and nothing else on stdout",
      r.stdout.strip(), "PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144 (fits with 3.3 GB spare)")
check("  notes go to stderr, where a shell capturing the value will not eat them",
      r.stderr.startswith("#"), True)
check("  and it succeeds", r.returncode, 0)
check("--bare is the number alone, for a shell", run(["--max-seq", "262144", "--bare"]).stdout,
      "0.36\n")
check("--arena is the arena that keep needs", run(["--max-seq", "262144", "--arena"]).stdout,
      "81\n")
check("--arena at 32k is the bigger one 0.38 needs", run(["--max-seq", "32768", "--arena"]).stdout,
      "85\n")
check("the environment supplies the defaults ./start.sh has already sourced",
      run(["--bare"], env={"MAX_SEQ": "262144"}).stdout, "0.36\n")
check("a pinned PRUNE_KEEP in the environment is passed through",
      run(["--bare"], env={"MAX_SEQ": "32768", "PRUNE_KEEP": "0.31"}).stdout, "0.31\n")
check("a number that will not serve exits non-zero",
      run(["--max-seq", "262144", "--keep", "0.44"]).returncode, 1)
r = run(["--keep", "nonsense"])
check("a keep that is neither a number nor auto is refused", r.returncode, 2)
check("  by name", "must be a number or 'auto'" in r.stderr, True)
check("a keep outside (0, 1] is refused", run(["--keep", "1.5"]).returncode, 2)
check("a max-seq that is not a length is refused", run(["--max-seq", "0"]).returncode, 2)

# --- ./start.sh resolves it before it launches anything ----------------------
# The dry flag exists for exactly this: everything .env and the environment
# imply, resolved, with nothing started. ARENA_GB is passed so the check does
# not depend on what the .env of the box this runs on happens to pin.
env = {"DSV41_HOST_TOTAL_GB": "130.6", "DSV41_HOST_AVAIL_GB": "117",
       "PYTHON": sys.executable, "PRUNE_KEEP": "auto", "MAX_SEQ": "262144",
       "ARENA_GB": "81", "EXPERT_FORMAT": "cb3", "TRANSIENT_SLOTS": "8", "KEEP_FREE_GB": "6"}
sh = subprocess.run(["bash", os.path.join(ROOT, "start.sh"), "--print-env"],
                    capture_output=True, text=True, cwd=ROOT, env={**os.environ, **env})
check("start.sh --print-env starts nothing and exits cleanly", sh.returncode, 0)
check("  it prints the resolution as one line",
      "PRUNE_KEEP=auto -> 0.36 for MAX_SEQ=262144" in sh.stdout, True)
check("  and the value it will launch with is the resolved one",
      "PRUNE_KEEP=0.36" in sh.stdout, True)
check("  which is what the engine is actually handed",
      '"prune_keep": 0.36' in sh.stdout, True)
sh = subprocess.run(["bash", os.path.join(ROOT, "start.sh"), "--print-env"],
                    capture_output=True, text=True, cwd=ROOT,
                    env={**os.environ, **env, "PRUNE_KEEP": "0.31"})
check("a pinned PRUNE_KEEP is left exactly as written",
      "PRUNE_KEEP=0.31" in sh.stdout and "-> " not in sh.stdout, True)
sh = subprocess.run(["bash", os.path.join(ROOT, "start.sh"), "--print-env"],
                    capture_output=True, text=True, cwd=ROOT,
                    env={**os.environ, **env, "PRUNE_KEEP": "nought"})
check("a PRUNE_KEEP that is neither refuses to launch", sh.returncode, 1)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
