#!/usr/bin/env python3
"""prefill_probe.py -- one prompt through the server, the engine's own prefill and decode numbers out.

    python3 tools/prefill_probe.py results/prefill/prompt.txt            # prefill: TTFT and prefill tok/s
    python3 tools/prefill_probe.py --decode --max-tokens 200             # decode: tok/s and acceptance on a short prompt
    python3 tools/prefill_probe.py prompt.txt --repeat 2                 # the second run is the warm one

Prints one `PROBE {...}` JSON line per run: prompt_tokens, completion_tokens, ttft_s (wall clock to
the first streamed content byte, as a client sees it), prefill_s / prefill_tok_s / decode_tok_s /
accept_len_mean from `x_engine_stats`. Thinking is off: the prompt is the measurement and the
answer only has to start. It is the probe every prefill verify script carries inline, pulled out
so that several configurations can be compared with one identical harness -- a number from a
different harness is not comparable, however similar the prompt looks.
"""
import argparse
import json
import sys
import time
import urllib.request

DECODE_PROMPT = ("Write a short story of about 300 words about a lighthouse keeper who finds a "
                 "message in a bottle. Plain prose, no headings.")


def probe(base, prompt, max_tokens, timeout):
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "chat_template_kwargs": {"thinking": False}, "max_tokens": max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True}}
    try:
        with urllib.request.urlopen(base + "/models", timeout=30) as r:
            body["model"] = json.loads(r.read())["data"][0]["id"]
    except Exception:
        pass
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage, stats = {}, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.perf_counter() - t0
            usage = d.get("usage") or usage
            stats = d.get("x_engine_stats") or stats
    wall = time.perf_counter() - t0
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_s": round(ttft, 3) if ttft else None,
        "wall_s": round(wall, 2),
        "prefill_s": stats.get("prefill_s"),
        "prefill_tok_s": stats.get("prefill_tok_s"),
        "decode_tok_s": stats.get("decode_tok_s"),
        "accept_len_mean": stats.get("accept_len_mean"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt_file", nargs="?", help="the prompt, one file; omitted with --decode")
    ap.add_argument("--decode", action="store_true", help="a short prose prompt; the decode line is the number")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=None, help="64 for a prefill probe, 200 for --decode")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tag", default="", help="a label copied into every line")
    a = ap.parse_args()
    if a.decode:
        prompt = DECODE_PROMPT
        max_tokens = a.max_tokens or 200
    else:
        if not a.prompt_file:
            ap.error("a prompt file, or --decode")
        prompt = open(a.prompt_file, encoding="utf-8").read()
        max_tokens = a.max_tokens or 64
    for i in range(a.repeat):
        d = probe(a.url, prompt, max_tokens, a.timeout)
        d["run"] = i + 1
        if a.tag:
            d["tag"] = a.tag
        print("PROBE " + json.dumps(d), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
