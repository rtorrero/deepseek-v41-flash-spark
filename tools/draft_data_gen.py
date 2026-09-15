#!/usr/bin/env python3
"""draft_data_gen.py -- drive the server until it has written enough drafter training data.

The recorder (`engine/draft_record.py`, `DSV41_RECORD_DRAFT_DATA`) writes a shard per request while
the server decodes normally. This script is what makes it decode: it streams passages out of
`corpus/`, asks for a 150-200 token continuation of each with thinking off, and stops when the
server has settled `--tokens` tokens. Nothing here loads a model; it is one HTTP client and a
sampler, standard library only.

**Which text.** The drafter is weakest exactly where the server is slowest: 2.46 accepted tokens a
step on a long story against 5.07 on a single-file HTML page (RESULTS.md 4.3). So the mix is
prose-heavy by default -- `--prose-share 0.75` -- drawn from the registers the weak profiles are
made of (english, journalism, academic, translation, and the narrative/essay/dialogue passages in
`corpus/sources/prose`). The remaining quarter is code and markup, and it is not there for balance:
a head fine-tuned on prose alone is free to forget the markup register where it currently accepts
five tokens a step, and that register is where most of the tok/s is. A fine-tune that trades 5.07
for 3.5 is a loss however well the prose row moves.

`corpus/heldout_sources/` is never sampled. It is the held-out corpus every teacher-forced number in
RESULTS.md is measured on, and a drafter trained on it would make those numbers meaningless.

**How long.** At the served rate of ~17 tok/s on a batch-1 server, 300,000 tokens is
300000 / 17 = 17,600 s = **4.9 hours** of generation, plus ~0.6 s of prefill per request over
~1,700 requests: call it **5 hours**. The script prints that estimate before the first request and
refuses nothing -- but if the number is a surprise, it is better to see it at second zero. Progress
is written to a log and to a resume file after every request, so it can be stopped and restarted
(`--resume`) as often as the box needs.

Usage (on the box, against a running server):

    DSV41_RECORD_DRAFT_DATA=data/draft ./start.sh
    python3 tools/draft_data_gen.py --out data/draft --tokens 300000
    python3 tools/draft_data_gen.py --out data/draft --tokens 300000 --resume
    python3 tools/draft_data_gen.py --out data/draft --dry-run     # the plan, no HTTP

The `--out` directory is the recorder's directory: the driver writes its log and resume state
beside the shards so that one directory is the whole dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from urllib import error as urlerror
from urllib import request as urlrequest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")

# The served decode rate the wall-clock estimate is built on (RESULTS.md 4.3: 17.1 tok/s on a long
# story, 18.1 on an explanation -- the prose rows, which is what this mix mostly is).
TOK_S = 17.0
PREFILL_S = 0.6  # a ~200-token prompt at the measured 337 tok/s, plus request overhead

# register -> what a prompt of that register asks for. "prose" and "reasoning" are the weak
# workloads; "markup" and "code" are the ones the fine-tune must not lose.
PROSE = ("prose", "reasoning")
CODEY = ("markup", "code")

# file extension -> (register, fence label). Anything not here is skipped.
EXT = {
    ".txt": ("prose", ""), ".md": ("markup", "markdown"),
    ".html": ("markup", "html"), ".css": ("markup", "css"), ".xml": ("markup", "xml"),
    ".js": ("code", "javascript"), ".jsx": ("code", "jsx"), ".ts": ("code", "typescript"),
    ".py": ("code", "python"), ".go": ("code", "go"), ".rs": ("code", "rust"),
    ".java": ("code", "java"), ".cpp": ("code", "cpp"), ".sql": ("code", "sql"),
    ".sh": ("code", "bash"), ".yaml": ("code", "yaml"), ".yml": ("code", "yaml"),
    ".tex": ("code", "latex"), ".R": ("code", "r"), ".rb": ("code", "ruby"),
}

# Topic names that carry a deliberation register rather than a domain. fetch_topics.py writes them
# under these names and tune.py's profiles all carry them, because a request with thinking on writes
# in that register whatever its subject.
REASONING_TOPICS = ("reasoning", "reasoning_code", "reasoning_design", "reasoning_lang")


# =============================================================================
# sources
# =============================================================================

def _register_of(path: str, stem: str) -> tuple:
    ext = os.path.splitext(path)[1]
    reg, fence = EXT.get(ext, (None, ""))
    if reg is None:
        return None, ""
    if reg == "prose" and any(stem.startswith(r) for r in REASONING_TOPICS):
        return "reasoning", ""
    return reg, fence


def discover_sources(corpus_dir: str = CORPUS, topics_dir: str | None = None) -> list:
    """Every passage source the driver may draw from, as
    {topic, register, fence, path|text, chars}, sorted by (register, topic) so the plan is stable.

    Three kinds:
      * files under `corpus/sources/` -- the prose passages, the web/markup files, the code,
      * `<topics_dir>/<topic>.txt` -- what `corpus/fetch_topics.py` writes (english, journalism,
        academic, translation, the languages): the registers the weak profiles are made of,
      * `corpus/trace_corpus_v*.jsonl` -- the traced corpus, whose entries carry a category.

    `corpus/heldout_sources/` is excluded here, once, rather than at every call site.
    """
    out = []
    src = os.path.join(corpus_dir, "sources")
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d != "heldout_sources")
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            stem = os.path.splitext(fn)[0]
            reg, fence = _register_of(path, stem)
            if reg is None:
                continue
            rel = os.path.relpath(path, corpus_dir)
            if rel.startswith("heldout_sources"):
                continue
            topic = os.path.basename(dirpath) if os.path.basename(dirpath) != "sources" else stem
            try:
                chars = os.path.getsize(path)
            except OSError:
                continue
            if chars < 400:
                continue
            out.append({"topic": topic, "register": reg, "fence": fence, "path": path, "chars": chars})
    if topics_dir and os.path.isdir(topics_dir):
        for fn in sorted(os.listdir(topics_dir)):
            path = os.path.join(topics_dir, fn)
            stem = os.path.splitext(fn)[0]
            reg, fence = _register_of(path, stem)
            if reg is None or not os.path.isfile(path) or os.path.getsize(path) < 400:
                continue
            out.append({"topic": stem, "register": reg, "fence": fence, "path": path,
                        "chars": os.path.getsize(path)})
    for jl in sorted(f for f in os.listdir(corpus_dir) if f.startswith("trace_corpus") and f.endswith(".jsonl")):
        for i, line in enumerate(open(os.path.join(corpus_dir, jl))):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            text = _strip_chat(d.get("text", ""))
            if len(text) < 400:
                continue
            cat = d.get("category") or "general"
            reg = "code" if cat == "coding" else "prose"
            out.append({"topic": f"{jl.split('.')[0]}:{cat}", "register": reg,
                        "fence": "" if reg == "prose" else "", "text": text, "chars": len(text),
                        "path": f"{jl}#{i}"})
    out.sort(key=lambda s: (s["register"], s["topic"], s["path"]))
    return out


_CHAT_MARKS = re.compile(r"<[|｜][^>]{0,40}[|｜]>|</?think>")


def _strip_chat(text: str) -> str:
    """The traced corpus stores rendered chat turns. Only the body is a passage."""
    return _CHAT_MARKS.sub("", text).strip()


def read_source(src: dict) -> str:
    if "text" in src:
        return src["text"]
    with open(src["path"], "r", errors="replace") as f:
        return f.read()


# =============================================================================
# sampling
# =============================================================================

def register_pool(sources: list, registers) -> list:
    return [s for s in sources if s["register"] in registers]


def prose_behind(by_register: dict, prose_share: float) -> bool:
    """Which group the next request should come from, judged on SETTLED TOKENS, not requests.

    The first 150k-token run planned 75 % of its requests as prose and ended at 32 % prose by
    tokens: a prose continuation stops on its own after ~80 tokens while a code one runs to the
    cap, so a request share is not a token share. The drafter is trained on tokens, and the
    prose ones are the point, so the group that is behind its share gets the next request.
    """
    prose = sum(v for k, v in by_register.items() if k in PROSE)
    total = sum(by_register.values())
    if total == 0:
        return prose_share > 0
    return prose / total < prose_share


def plan_mix(sources: list, n: int, prose_share: float, seed: int) -> list:
    """`n` (topic, register) draws, deterministic in `seed`.

    Registers are drawn to the requested share, and WITHIN a register topics are taken round-robin
    from a shuffled order rather than uniformly at random: a uniform draw over a handful of topics
    leaves one of them with a third of its expected share over a few hundred requests, and the point
    of the mix is that no register is starved.
    """
    rng = random.Random(seed)
    prose = register_pool(sources, PROSE)
    codey = register_pool(sources, CODEY)
    if not prose or not codey:
        raise ValueError(f"need both registers: {len(prose)} prose/reasoning, {len(codey)} code/markup")
    order = {}
    for name, pool in (("prose", prose), ("code", codey)):
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        order[name] = idx
    n_prose = int(round(n * prose_share))
    picks = []
    for i in range(n):
        want_prose = i < n_prose
        pool, name = (prose, "prose") if want_prose else (codey, "code")
        picks.append(pool[order[name][(i if want_prose else i - n_prose) % len(pool)]])
    rng.shuffle(picks)
    return picks


def passage(text: str, rng: random.Random, min_words: int = 120, max_words: int = 240) -> str:
    """A contiguous slice of `text`, starting at a line boundary. Starting mid-sentence would teach
    the drafter a register the server never sees: a request always starts at something a writer
    wrote from the beginning of."""
    lines = [l for l in text.splitlines()]
    if not lines:
        return ""
    starts = [i for i, l in enumerate(lines) if l.strip()] or [0]
    i0 = rng.choice(starts)
    want = rng.randint(min_words, max_words)
    got, out = 0, []
    for l in lines[i0:]:
        out.append(l)
        got += len(l.split())
        if got >= want:
            break
    return "\n".join(out).strip()


def build_prompt(src: dict, text: str) -> str:
    if src["register"] in PROSE:
        return ("Continue this passage in the same voice and register. "
                "Write prose only, no commentary, no headings.\n\n" + text)
    fence = src["fence"] or ""
    return ("Continue this file from where it stops. Reply with the continuation only, "
            "in one code block, no explanation.\n\n"
            f"```{fence}\n{text}\n```")


def sample_key(src: dict, text: str) -> str:
    """Stable id of one drawn passage, so `--resume` never re-asks for the same continuation."""
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"{src['register']}/{src['topic']}/{h}"


# =============================================================================
# state and log
# =============================================================================

def state_path(out: str) -> str:
    return os.path.join(out, "draft_data_gen.state.json")


def load_state(out: str) -> dict:
    p = state_path(out)
    if not os.path.exists(p):
        return {"tokens": 0, "requests": 0, "seconds": 0.0, "done": [], "by_register": {}}
    with open(p) as f:
        st = json.load(f)
    st.setdefault("done", [])
    st.setdefault("by_register", {})
    return st


def save_state(out: str, st: dict):
    tmp = state_path(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, state_path(out))


def estimate(tokens: int, tok_s: float = TOK_S, per_request: int = 175) -> dict:
    """Wall clock for `tokens` settled tokens at the served batch-1 rate."""
    requests = max(1, int(round(tokens / per_request)))
    gen_s = tokens / tok_s
    total = gen_s + requests * PREFILL_S
    return {"requests": requests, "generate_s": gen_s, "total_s": total,
            "hours": total / 3600.0, "tok_s": tok_s}


# =============================================================================
# the run
# =============================================================================

def post_chat(base: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float,
              api_key: str | None, timeout: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "chat_template_kwargs": {"thinking": False}, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p}
    req = urlrequest.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    usage = d.get("usage") or {}
    st = d.get("x_engine_stats") or {}
    return {"completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "accept_len_mean": st.get("accept_len_mean"), "decode_tok_s": st.get("decode_tok_s"),
            "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/draft",
                    help="the recorder's directory (DSV41_RECORD_DRAFT_DATA); log and resume state "
                         "are written here too (default %(default)s)")
    ap.add_argument("--tokens", type=int, default=300000, help="stop after this many settled tokens")
    ap.add_argument("--url", default=os.environ.get("DSV41_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY"))
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prose-share", type=float, default=0.75,
                    help="fraction of requests drawn from the prose/reasoning registers "
                         "(default %(default)s; the rest is code and markup, so the head does not "
                         "forget the register it is already good at)")
    ap.add_argument("--min-tokens", type=int, default=150, help="shortest continuation asked for")
    ap.add_argument("--max-tokens", type=int, default=200, help="longest continuation asked for")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--topics-dir", default=os.environ.get("DSV41_TOPICS_DIR"),
                    help="a directory of <topic>.txt files from corpus/fetch_topics.py")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--resume", action="store_true", help="continue from the resume file in --out")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and stop, no HTTP")
    a = ap.parse_args()

    if not 0.0 <= a.prose_share <= 1.0:
        ap.error("--prose-share must be between 0 and 1")
    if a.min_tokens > a.max_tokens:
        ap.error("--min-tokens is above --max-tokens")
    a.url = a.url.rstrip("/")
    if not a.url.endswith("/v1"):
        a.url += "/v1"

    sources = discover_sources(CORPUS, a.topics_dir)
    per_request = (a.min_tokens + a.max_tokens) // 2
    est = estimate(a.tokens, per_request=per_request)
    picks = plan_mix(sources, est["requests"] * 2, a.prose_share, a.seed)  # 2x: short answers happen

    by_reg = {}
    for s in picks:
        by_reg[s["register"]] = by_reg.get(s["register"], 0) + 1
    print(f"sources: {len(sources)} ({len(register_pool(sources, PROSE))} prose/reasoning, "
          f"{len(register_pool(sources, CODEY))} code/markup)")
    print(f"plan: {a.tokens} tokens, ~{est['requests']} requests of {a.min_tokens}-{a.max_tokens} tokens, "
          f"mix {', '.join(f'{k} {v}' for k, v in sorted(by_reg.items()))}")
    print(f"estimate at {TOK_S:.0f} tok/s (batch-1 server): {est['generate_s'] / 3600:.1f} h of "
          f"generation, {est['hours']:.1f} h in total")
    if a.dry_run:
        for s in picks[:10]:
            p = s["path"]
            print(f"  {s['register']:<9} {s['topic']:<24} "
                  f"{os.path.relpath(p, ROOT) if os.path.isabs(p) else p}")
        return 0

    os.makedirs(a.out, exist_ok=True)
    st = load_state(a.out) if a.resume else {"tokens": 0, "requests": 0, "seconds": 0.0,
                                             "done": [], "by_register": {}}
    done = set(st["done"])
    if a.resume:
        print(f"resuming: {st['tokens']} tokens over {st['requests']} requests already recorded")

    logf = open(os.path.join(a.out, "draft_data_gen.log"), "a", buffering=1)
    logf.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} target {a.tokens} tokens, "
               f"prose_share {a.prose_share}, seed {a.seed}\n")
    rng = random.Random(a.seed ^ 0x5EED)
    t_start = time.perf_counter()
    # Two queues in plan order; which one the next request comes from is decided on settled tokens.
    queue = {"prose": [s for s in picks if s["register"] in PROSE],
             "code": [s for s in picks if s["register"] in CODEY]}
    extend_seed = a.seed + len(picks)
    try:
        while st["tokens"] < a.tokens:
            group = "prose" if prose_behind(st["by_register"], a.prose_share) else "code"
            if not queue[group]:  # that queue ran out; extend the plan deterministically
                more = plan_mix(sources, est["requests"], a.prose_share, extend_seed)
                extend_seed += len(more)
                queue["prose"] += [s for s in more if s["register"] in PROSE]
                queue["code"] += [s for s in more if s["register"] in CODEY]
                if not queue[group]:
                    continue
            src = queue[group].pop(0)
            text = passage(read_source(src), rng)
            if len(text) < 200:
                continue
            key = sample_key(src, text)
            if key in done:
                continue
            want = rng.randint(a.min_tokens, a.max_tokens)
            try:
                got = post_chat(a.url, a.model, build_prompt(src, text), want, a.temperature,
                                a.top_p, a.api_key, a.timeout)
            except (urlerror.URLError, OSError, ValueError) as e:
                logf.write(f"ERROR {key} {e}\n")
                print(f"  request failed ({e}); stopping. Re-run with --resume once the server is back.")
                break
            done.add(key)
            st["done"].append(key)
            st["tokens"] += got["completion_tokens"]
            st["requests"] += 1
            st["seconds"] += got["seconds"]
            st["by_register"][src["register"]] = st["by_register"].get(src["register"], 0) + got["completion_tokens"]
            save_state(a.out, st)
            logf.write(f"{time.strftime('%H:%M:%S')} {src['register']:<9} {src['topic']:<24} "
                       f"{got['completion_tokens']:>4} tok  accept {got['accept_len_mean']}  "
                       f"{got['decode_tok_s']} tok/s  {got['seconds']:.1f}s  {key}\n")
            if st["requests"] % 10 == 0 or st["tokens"] >= a.tokens:
                el = time.perf_counter() - t_start
                rate = st["tokens"] / max(el, 1e-9)
                left = max(a.tokens - st["tokens"], 0) / max(rate, 1e-9)
                print(f"  {st['tokens']:>7}/{a.tokens} tokens  {st['requests']:>5} requests  "
                      f"{rate:.1f} tok/s overall  ~{left / 3600:.1f} h left", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; state saved. Re-run with --resume.")
    finally:
        save_state(a.out, st)
        logf.close()

    print(f"\n{st['tokens']} tokens over {st['requests']} requests in "
          f"{(time.perf_counter() - t_start) / 3600:.2f} h")
    print("by register: " + ", ".join(f"{k} {v}" for k, v in sorted(st["by_register"].items())))
    print(f"shards in {a.out}; train with: python3 tools/train_mtp.py --data {a.out}")
    return 0 if st["tokens"] >= a.tokens else 1


if __name__ == "__main__":
    sys.exit(main())
