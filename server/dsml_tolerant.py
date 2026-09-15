"""The tolerant DSML tool-call parser: what runs after the checkpoint's strict parser has raised.

Torch-free and importable on its own, so the one thing it must get right is under test on a CPU
runner: a parameter VALUE runs to the parameter's closing tag, not to the first ``<``. The first
version stopped at ``<`` (``(?P<v>.*?)<``), which is fine for a search query and empties every
HTML or XML value at its first tag -- an opencode Write of a whole page came back as a call with
no usable arguments, twice in one evening (RESULTS.md 2026-09-15). The value's own closing tag
is unambiguous: the guard in server/tool_grammar.py masks the fullwidth bar after ``</`` inside a
value, so ``</｜DSML｜ parameter>`` cannot occur inside one.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

RE_INVOKE = re.compile(
    r'<｜DSML｜ invoke name="(?P<name>[^"]*)"\s*>?\n?(?P<body>.*?)(?=<｜DSML｜ invoke |</｜DSML｜ calls>|\Z)',
    re.DOTALL)
#: the value runs to the closing parameter tag; if the text was cut before one, to the end.
RE_PARAM_SPEC = re.compile(
    r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<s>true|false)"\s*>'
    r'(?P<v>.*?)(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter )|\Z)',
    re.DOTALL)
RE_PARAM_ATTR = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<v>.*?)"\s*>', re.DOTALL)


def loads_lenient(v: str):
    """JSON first; then a Python literal; then the raw string."""
    try:
        return json.loads(v)
    except Exception:  # noqa: BLE001
        pass
    import ast
    try:
        return ast.literal_eval(v)
    except Exception:  # noqa: BLE001
        return v


def parse_tolerant(text: str) -> List[dict]:
    """Best-effort DSML tool calls. Returns [] when nothing parses."""
    out: List[dict] = []
    for m in RE_INVOKE.finditer(text):
        name, body = m.group("name"), m.group("body")
        if not name:
            continue
        args: Dict[str, Any] = {}
        for pm in RE_PARAM_SPEC.finditer(body):
            k, is_str, v = pm.group("k"), pm.group("s"), pm.group("v")
            if k in args:
                continue
            args[k] = v if is_str == "true" else loads_lenient(v)
        consumed = set(args)
        for pm in RE_PARAM_ATTR.finditer(body):
            k, v = pm.group("k"), pm.group("v")
            if k and k not in consumed:
                args[k] = v
        out.append({"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return out
