"""Which run in a GATE.md may stand as a profile's record.

A gate section is a dated measurement of one configuration, and a profile's
record is the newest FULL run on exactly its topics. Two things disqualify a
run from being that record:

  * `| only |` -- a filtered re-run says the prompts it ran pass and nothing
    about the ones it did not. The readers spot that row; `may_be_record` below
    weighs it.
  * a non-default DECODE CONTROL -- this file, `decode_controls`.

`may_be_record` is the whole rule, and every selector asks it rather than
spelling the disqualifiers out again: a caller holding only half of them agrees
with the others until the day a GATE.md first carries the half it does not know.

The second one arrived with tools/verify_think_controls.sh (2026-09-14), which
appends four runs to results/keepsets/{frontend,backend}/GATE.md taken with
`DSV41_THINK_BUDGET=2000` and, on two of them, `DSV41_THINK_REPEAT_BREAK=12`.
Those runs are the experiment: they say what capping the reasoning span does to
a keep-set, and they are worth reading against the baseline beside them. What
they are not is the profile's gate result, because the box does not serve with
them on. `./tune.sh`, `--print`, `results/keepsets/gates.json` and the article's
profile figure all report what a reader would get from the shipped engine, so
they must keep reading the newest run taken with the controls OFF.

The notion is deliberately generic rather than a list of two variables. The
gate card writer (tools/gate_profile.py `report`) can grow further control rows
-- `DSV41_ESCAPE_K` from tools/verify_escape.sh is the next one in line -- and a
new row must disqualify a run on the day it is first written, not on the day
someone remembers to add it here. So: any card row whose KEY reads as a decode
control ("... controls", "... hatch") and whose VALUE names a flag set to
anything other than zero/off counts, and a control row that cannot be parsed at
all counts too. Erring that way costs a profile its record until the parser is
taught the row -- visible, and correctable; erring the other way publishes an
experiment's numbers as the box's behaviour, which is neither.

No torch, no curses, no network: tools/tune.py, tools/tail_metric.py and the
article's figures_part3.py all import this.
"""
from __future__ import annotations

import re

# `| key | value |`, the shape every gate card row has.
_ROW = re.compile(r"^\|\s*([^|]*?)\s*\|\s*(.*?)\s*\|\s*$")

# `DSV41_THINK_BUDGET=2000`, with or without commas between assignments, and
# tolerating the backticks a shell-written section wraps its environment in.
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s,;)`]*)")

# A flag at one of these is the default, i.e. the control is not armed.
_OFF = {"", "0", "0.0", "off", "false", "none", "no", "unset", "default", "-", "—"}

# A control row with no assignment in it at all is the writer saying the control
# was off -- "off (neither DSV41_THINK_BUDGET nor DSV41_THINK_REPEAT_BREAK was
# set)" is what gate_profile.py writes. Recognised by its first word.
_OFF_HEAD = ("off", "none", "default", "no", "not", "unset", "absent", "-", "—")

# Keys the card writer uses today. `_is_control_key` also takes anything that
# ends in "controls" or "hatch", so a row added later needs no change here.
CONTROL_KEYS = {"reasoning-span controls", "decode controls", "escape hatch"}


def card_rows(lines) -> list:
    """Every `| key | value |` row of a gate card, in order, as (key, value).

    Stops at the first blank line after the card: the per-prompt table below it
    is the same shape and none of its rows is a card row.
    """
    out = []
    seen = False
    for ln in lines:
        m = _ROW.match(ln.rstrip())
        if not m:
            if seen:
                break
            continue
        key, value = m.group(1), m.group(2)
        if set(key) <= set("-: ") or not key:      # `|---|---|` and the `| | |` head
            continue
        seen = True
        out.append((key, value))
    return out


def _is_control_key(key: str) -> bool:
    k = key.strip().lower()
    return k in CONTROL_KEYS or k.endswith("controls") or k.endswith("hatch")


def _armed(value: str):
    """The flags this control row names that are NOT at their default.

    Returns a list (possibly empty), or None when the row names a control and
    cannot be read -- which the caller treats as armed.
    """
    pairs = _ASSIGN.findall(value or "")
    if pairs:
        return [f"{k}={v}" for k, v in pairs if v.strip().lower() not in _OFF]
    head = (value or "").strip().lower().lstrip("`*_ ")
    if not head or head.split()[0].strip(".,:;") in _OFF_HEAD:
        return []
    return None


def decode_controls(lines) -> list:
    """The non-default decode controls a gate card names, `KEY=value` each.

    Empty means the run was taken with the decode path the box ships with, which
    is the only kind of run that may be a profile's record.
    """
    out = []
    for key, value in card_rows(lines):
        if not _is_control_key(key):
            continue
        armed = _armed(value)
        if armed is None:
            out.append(f"{key.strip()}: {value.strip()}")
        else:
            out.extend(armed)
    return out


def default_decode(lines) -> bool:
    """True when this gate section was taken with default decode controls."""
    return not decode_controls(lines)


def may_be_record(section: dict) -> bool:
    """True when a parsed gate section may stand as a profile's record.

    `section` is what the two readers build -- tools/tune.py `read_gate` and
    tools/tail_metric.py `read_gate` -- carrying `filtered` for the `| only |`
    re-runs and `controls` for `decode_controls` above. BOTH disqualifiers are
    named here, in one place, so that a caller cannot hold half the rule.

    It could: the second disqualifier arrived (2026-09-14) while
    results/keepsets/{frontend,backend}/GATE.md still had no control card in
    them, so every copy of `not filtered` that had not been taught the new half
    went on agreeing with the selector by accident. The day
    tools/verify_think_controls.sh's four runs were committed, the stale copies
    started calling an experiment the newest record while `gate_for` -- rightly
    -- did not, and the disagreement surfaced as a test failure rather than as
    a wrong number on a screen only because a test happened to hold one of
    them. Adding a third disqualifier later must not need that luck.
    """
    return not section.get("filtered") and not section.get("controls")
