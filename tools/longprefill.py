#!/usr/bin/env python3
"""longprefill.py -- one long, deterministic prompt, and what the server did with it.

Prefill work scales with the number of CHUNKS, not with the prompt: every chunk unpacks nearly
every expert of every layer it visits again (NOTES 2026-09-12). So anything that measures prefill
needs a prompt long enough to take several chunks, and the SAME prompt every time.

This builds one out of the checkout's own corpus (`corpus/sources`, `corpus/heldout_sources`) in a
fixed file order, cut to a character budget -- no network, no randomness, byte-identical on every
run and on every machine with this checkout. Then it sends it once, streaming, and reports:

  * time to first token, measured on the client, which is what a user waits
  * `prefill_s` and `prefill_tok_s` from the server's own `x_engine_stats`
  * the prefill unpack cache counters, when one is configured

    python3 tools/longprefill.py --print-only            # just the prompt, to a file
    python3 tools/longprefill.py --tokens 7000 --json out.json

Stdlib only: no torch, no model load, nothing to install.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib import error as urlerror
from urllib import request as urlrequest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed order, fixed files. Prose first so the prompt opens in a register the model will not
# mistake for an instruction, then code and configuration, then the technical prose. Changing this
# list changes the prompt, so any number measured with an older list is not comparable.
SOURCES = [
    "corpus/sources/prose/essay1.txt",
    "corpus/sources/prose/fiction1.txt",
    "corpus/sources/web/story.txt",
    "corpus/sources/web/notes.md",
    "corpus/sources/web/dashboard.html",
    "corpus/sources/web/styles.css",
    "corpus/sources/web/app.js",
    "corpus/sources/web/widget.jsx",
    "corpus/sources/web/config.yaml",
    "corpus/sources/web/schema.sql",
    "corpus/sources/prose/fiction2.txt",
    "corpus/sources/prose/dialogue1.txt",
    "corpus/heldout_sources/architecture.md",
    "corpus/heldout_sources/techreport_part2.txt",
    "corpus/sources/prose/fiction3.txt",
    "corpus/sources/prose/fiction4.txt",
]

# Measured on this corpus with this checkpoint's tokenizer: prose and code together come out near
# 4.0 characters a token. The tool reports the count the SERVER read back, so this only has to be
# close enough to land in the right chunk count.
CHARS_PER_TOKEN = 4.0

INSTRUCTION = (
    "Below is a collection of documents. Read all of them, then answer in one short "
    "paragraph: what do these documents have in common, and what is the single most "
    "technical one among them?\n\n"
)


def build_prompt(target_tokens: int) -> str:
    budget = int(target_tokens * CHARS_PER_TOKEN) - len(INSTRUCTION)
    parts, used = [], 0
    i = 0
    while used < budget:
        rel = SOURCES[i % len(SOURCES)]
        i += 1
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            if i > 4 * len(SOURCES):
                break
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            body = fh.read()
        head = f"\n\n===== document {len(parts) + 1}: {os.path.basename(rel)} =====\n\n"
        take = body[: max(0, budget - used - len(head))]
        if not take:
            break
        parts.append(head + take)
        used += len(head) + len(take)
        if i > 8 * len(SOURCES):
            break
    return INSTRUCTION + "".join(parts)


def post(url: str, payload: dict, api_key: str | None, timeout: int):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urlrequest.urlopen(urlrequest.Request(url, data=data, headers=headers), timeout=timeout)


def send(base: str, prompt: str, *, model: str, api_key: str | None, max_tokens: int,
         thinking: str, timeout: int) -> dict:
    """Stream one completion and return what it cost."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # docs/openai-api.md: `chat_template_kwargs.thinking` is the first thing consulted
        "chat_template_kwargs": {"thinking": thinking == "on"},
    }
    t0 = time.perf_counter()
    ttft = None
    text = []
    stats, usage = {}, {}
    with post(base.rstrip("/") + "/chat/completions", payload, api_key, timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("x_engine_stats"):
                stats = ev["x_engine_stats"]
            for ch in ev.get("choices") or ():
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    return {
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "wall_s": round(time.perf_counter() - t0, 3),
        "chars": sum(len(t) for t in text),
        "usage": usage,
        "stats": stats,
    }


def summarise(r: dict) -> str:
    st = r.get("stats") or {}
    u = r.get("usage") or {}
    rows = [
        ("prompt tokens", u.get("prompt_tokens") or st.get("prompt_tokens")),
        ("time to first token (client)", f"{r['ttft_s']} s" if r.get("ttft_s") else "-"),
        ("prefill_s (server)", st.get("prefill_s")),
        ("prefill_tok_s (server)", st.get("prefill_tok_s")),
        ("decode_tok_s", st.get("decode_tok_s")),
        ("unpack hits", st.get("unpack_hits")),
        ("unpack misses", st.get("unpack_misses")),
        ("unpack GB saved", st.get("unpack_gb_saved")),
        ("unpack ms saved (est.)", st.get("unpack_ms_saved")),
        ("unpack cache slots", st.get("unpack_cache_slots")),
        ("unpack cache used", st.get("unpack_cache_used")),
    ]
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k.ljust(w)}  {v}" for k, v in rows if v is not None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=7000, help="approximate prompt length")
    ap.add_argument("--url", default=os.environ.get("DSV41_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY"))
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="keep it small: this measures prefill, not decode")
    ap.add_argument("--thinking", choices=("on", "off"), default="off")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", help="write the whole result here")
    ap.add_argument("--prompt-out", help="write the prompt text here")
    ap.add_argument("--print-only", action="store_true", help="build the prompt and stop")
    a = ap.parse_args()

    prompt = build_prompt(a.tokens)
    if a.prompt_out:
        with open(a.prompt_out, "w", encoding="utf-8") as fh:
            fh.write(prompt)
    print(f"prompt: {len(prompt)} characters, ~{len(prompt) / CHARS_PER_TOKEN:.0f} tokens, "
          f"{prompt.count('===== document')} documents", file=sys.stderr)
    if a.print_only:
        if not a.prompt_out:
            sys.stdout.write(prompt)
        return 0

    model = a.model
    try:
        with urlrequest.urlopen(a.url.rstrip("/") + "/models", timeout=20) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
            if ids:
                model = ids[0]
    except (urlerror.URLError, OSError, ValueError, KeyError):
        pass

    try:
        res = send(a.url, prompt, model=model, api_key=a.api_key, max_tokens=a.max_tokens,
                   thinking=a.thinking, timeout=a.timeout)
    except (urlerror.URLError, OSError) as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 2
    res["prompt_chars"] = len(prompt)
    res["model"] = model
    print(summarise(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"  wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
