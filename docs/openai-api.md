# The OpenAI-compatible API

`server/app.py` is a standard-library HTTP front end — `http.server` plus `json`, no
FastAPI, no uvicorn. The engine is single-sequence, so **requests are served one at a
time**: concurrent callers wait on a lock, nothing is rejected and nothing is batched.

The authoritative reference for every flag and field is
[`server/README.md`](../server/README.md). This page is the operator's view: what a client
sees, and how to point a normal OpenAI client at it.

## Endpoints

| route | what it does |
|---|---|
| `GET /health` | `{"status","engine","busy","max_context","engine_config"}` — and `engine_config` is the point of it: `arena_gb`, `arena_slots`, `lru_slots`, `transient_slots`, `resident_expert_pct`, `max_seq`, `spec`, `trace_stats`, `kernel`, `act_quant`. A benchmark never has to be *told* how the server was started |
| `GET /v1/models`, `GET /v1/models/{id}` | one model card, id = `SERVED_MODEL_NAME` (`deepseek-v4.1-flash`) |
| `POST /v1/chat/completions` | the main one. `messages`, `tools`, `tool_choice`, `response_format`, `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`, `ignore_eos`, `no_repeat_ngram`, plus the thinking and reasoning-span controls below |
| `POST /v1/completions` | raw `prompt` (string or token ids). No chat template, no thinking parsing — the text comes back verbatim |
| `POST /v1/debug/prompt` | same body as chat; returns the rendered prompt, its token ids and the resolved thinking/effort/reasoning controls **without touching the engine**. On this recipe a prefill is minutes of NVMe streaming, so this is how the benchmark measures prompt lengths |

Rejected with HTTP 400: `n > 1`, images, `logprobs`. Errors are OpenAI-shaped
(`{"error": {"message","type","param","code"}}`); a failure mid-stream arrives as a
`data: {"error": ...}` event followed by `data: [DONE]`.

## Thinking and reasoning effort

V4.1's template is not a binary switch. Thinking is on or off, *and* there is an effort
budget from 1 to 100 that the encoder renders as a `Reasoning Effort: N` system prefix.
The server resolves both per request, first match wins:

1. `chat_template_kwargs.thinking` (bool)
2. `enable_thinking` (bool; top level, then inside `chat_template_kwargs`)
3. `reasoning.effort` / `reasoning_effort` (top level, then inside `chat_template_kwargs`):

   | value | thinking | effort |
   |---|---|---|
   | `none` | off | server default |
   | `low` | off | 50 |
   | `medium` | on | 60 |
   | `high` | on | 75 |
   | `xhigh` | on | 90 |
   | `max` | on | 100 |
   | integer 1–100, or a numeric string | on | that integer |

4. the server defaults: `DEFAULT_THINKING` (`off`) and `DEFAULT_EFFORT` (`75`) from
   [`env.example`](../env.example).

A boolean from steps 1–2 decides *whether* it thinks; an effort from step 3 still sets the
budget. So `enable_thinking: true` + `reasoning_effort: "low"` is thinking on at 50. An
effort with thinking off is harmless — the encoder only renders the prefix in thinking
mode.

That mapping exists because clients disagree about how to ask. A UI that sends
`reasoning_effort: "high"` and nothing else gets thinking; one that sends
`chat_template_kwargs: {"thinking": false}` gets chat mode whatever else it says.

## Reasoning-span controls

Two guards over the think block, **both off by default**, both ignored by a request with
thinking off (there is no reasoning span to guard).

| field | also accepted as | env default | what it does |
|---|---|---|---|
| `reasoning_budget` | `reasoning.max_tokens`, `chat_template_kwargs.reasoning_budget` | `DSV41_THINK_BUDGET` | tokens of reasoning after which the decode loop forces `</think>` and continues as the answer. `0` = off |
| `think_repeat_break` | `chat_template_kwargs.think_repeat_break` | `DSV41_THINK_REPEAT_BREAK` | n-gram length. Inside the think block only: when the last n tokens repeat a window already seen twice in this reasoning span, the tokens that continued the earlier copies are masked, so a third copy cannot be extended. `0` = off, `2`-`128` otherwise |

Why they exist: with thinking on, a run that passes this repo's generation gate deliberates
for about 4,000 reasoning **characters** and a run that fails deliberates for about 19,000. The
Backend keep-set at keep 0.36 answered all ten of its prompts correctly but restated itself
3-8 times on seven of them, at 18-38k reasoning characters; a whole-page prompt on the Frontend
keep-set spent 15,241 reasoning tokens on a "Maybe add X? Skip." loop and returned nothing
([`RESULTS.md`](../RESULTS.md) §5.4 and the 2026-09-14 addenda). The failure is not that the
model cannot answer, it is that it will not stop thinking.

The gate's length columns are **characters**; `reasoning_budget` is in **tokens**, and a token is
about four characters in this register (2,001 forced reasoning tokens = 7,953 characters, measured
2026-09-14). So the figures above are roughly 1,000 tokens for a run that passes, 4,800 for one
that fails and 4.5-9.5k for the ones that loop — divide a gate figure by ~4 before setting a
budget from it (`RESULTS.md`, correction of 2026-09-14 21:10). A reply says which it was:
`usage.completion_tokens_details.reasoning_tokens`, and `reasoning_budget_hit` only when the
budget is what ended the span.

The budget is enforced in the decode loop, not by truncating text afterwards: the close token
is masked into the next step (and on the speculative path the drafts are rejected so the step
emits it), which is why the answer that follows is a real continuation rather than a cut. It
is checked once per step, and a speculative step settles up to six tokens, so the span can
overrun the budget by a few tokens; `reasoning_tokens` reports the real length.

When the budget fires, the response says so — `usage.completion_tokens_details` carries
`reasoning_budget_hit: true` alongside `reasoning_tokens`, the field is absent otherwise, and
the server logs one line per request. `x_engine_stats.think_controls` reports what each
control did (`budget_forced`, `repeat_breaks`, `banned_tokens`, `mask_s`).

The loop breaker is deliberately not the answer-side `no_repeat_ngram`: an n-gram ban over an
answer was refuted here, because CSS repeats `px`, `0` and `}` legitimately and the ban
destroys it. Inside a think block there is no such register.

```jsonc
{"reasoning_effort": "high", "reasoning_budget": 8000, "think_repeat_break": 12}
```

## Streaming, and `reasoning_content`

Chat deltas put text generated **before** `</think>` in `reasoning_content` and text after
it in `content`. The marker itself is never emitted. Any client that knows the
DeepSeek/OpenAI reasoning convention — Open WebUI, LibreChat, the `openai` Python SDK
reading `delta.reasoning_content` — will show the thinking in its own pane without
configuration.

Other stream details worth knowing:

* Token bursts are detokenised incrementally and text is released only when the decode is
  stable, so a multi-byte UTF-8 character is never split across two deltas.
* `usage.completion_tokens_details.reasoning_tokens` counts the tokens before `</think>`, and
  carries `reasoning_budget_hit: true` beside it when a `reasoning_budget` ended the reasoning.
* `x_engine_stats` rides on non-stream responses and on the final SSE chunk (the one with
  `finish_reason`). It is `Engine.stats()` — hit rate, NVMe GB, engram rows, accepted
  length, the attention/MoE split — plus `server_completion_tok_per_s`. On a recipe that
  streams most of its weights, this is the difference between a number and an anecdote.
* Tool calls arrive in one delta at the end, `finish_reason: tool_calls`. The model's DSML
  block is parsed by the checkpoint's own `encoding.parse_message_from_completion_text`;
  namespaced tools come back as `namespace::name`. If the block is malformed or truncated
  the raw tail is returned as `content` rather than dropped.

## Open WebUI / any OpenAI client

There is no authentication in front of this server and the port is published on the
loopback interface only, so the client has to run on the same box or reach it through a
proxy you put there.

| setting | value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | anything non-empty; it is not checked |
| Model | `deepseek-v4.1-flash` (whatever `SERVED_MODEL_NAME` says) |
| Streaming | on |

In **Open WebUI**: Settings → Connections → add an OpenAI-compatible connection with that
base URL. The model appears by its served id. Its *Reasoning Effort* control sends a
top-level `reasoning_effort`, which lands on step 3 of the table above — so setting it to
`medium` or higher turns thinking on by itself, and `none`/`low` leaves chat mode alone.
Thinking text arrives as `reasoning_content` and renders in the collapsible pane.

Two expectations to set before anyone else uses it:

* **The first token can take minutes on a cold prompt.** Prefill misses almost every
  expert, and each miss is an 18.8 MB NVMe read. Raise any client-side request timeout to
  something in the tens of minutes.
* **One request at a time.** A second user does not get an error, they get a wait. This is
  a single-sequence engine on a single box; it is not a shared endpoint.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash",
  "messages": [{"role":"user","content":"Explain the transient ring in one paragraph."}],
  "reasoning_effort": "high",
  "max_tokens": 512,
  "stream": true,
  "stream_options": {"include_usage": true}
}'
```

Next: [benchmarking](benchmarking.md) · [gotchas](gotchas.md) · [architecture](architecture.md)
