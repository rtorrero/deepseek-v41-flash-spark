#!/usr/bin/env python3
"""A kept set larger than the arena's LRU slots is refused unless the env says otherwise."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.keep_guard import ENV_ALLOW, check_fit  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    fails += 0 if ok else 1


check("a kept set that fits passes", check_fit(5560, 5603, {}) is None)
check("an exact fit passes", check_fit(5603, 5603, {}) is None)
m = check_fit(6160, 4142, {})
check("an overflow is refused with the counts and the remedies",
      m is not None and "6160" in m and "4142" in m and "ARENA_GB" in m and "PRUNE_KEEP" in m and ENV_ALLOW in m, str(m))
check("the env can allow the streaming tail", check_fit(6160, 4142, {ENV_ALLOW: "1"}) is None)
check("an unset or zero env does not", check_fit(6160, 4142, {ENV_ALLOW: "0"}) is not None
      and check_fit(6160, 4142, {ENV_ALLOW: ""}) is not None)
print("all ok" if not fails else f"{fails} failed")
sys.exit(1 if fails else 0)
