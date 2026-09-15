# Generation gate — 2026-09-15 02:38

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code, reasoning_design |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code, reasoning_design — these topics were NOT gated |
| keep-set | not recorded — PRUNE_KEEP and the ranking pair were not in the environment of this run |
| reasoning-span controls | off (neither DSV41_THINK_BUDGET nor DSV41_THINK_REPEAT_BREAK was set) |

| prompt | thinking | finish | reasoning chars | reasoning tokens | answer chars | answer tokens | s | | why |
|---|---|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 9,634 | 2,431 | 2,309 | 495 | 134 | **FAIL** | repeat: reasoning loops 4x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' — the answer itself is sound (29 keys, anchored) |
| `css-card` | on | stop | 27,086 | 8,021 | 1,870 | 708 | 392 | **FAIL** | reasoning loops 4x on 'need maybe "custom properties on :root for its colours, ' |
| `en-explain` | on | stop | 11,517 | 2,709 | 1,572 | 354 | 196 | **FAIL** | repeat: reasoning loops 4x on '95% hit rate can leave a system slower than no cache at' — the answer itself is sound (2 paragraphs, 11 sentences) |
| `en-note` | on | stop | 14,509 | 3,185 | 1,022 | 217 | 198 | **FAIL** | repeat: reasoning loops 5x on 'within the error and latency thresholds for at least the' — the answer itself is sound (3 paragraphs, 6 sentences) |
| `html-page` | on | stop | 15,358 | 4,624 | 6,271 | 1,935 | 278 | PASS | 79 declarations, 22 functions, 0 empty rules |
| `js-debounce` | on | stop | 20,710 | 4,988 | 1,444 | 389 | 301 | **FAIL** | repeat: reasoning loops 4x on '= null; let lastargs = null; let lastthis = null; functi' — the answer itself is sound (12 callables) |
| `tech-explain` | on | stop | 6,635 | 1,410 | 1,522 | 316 | 130 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 7,884 | 2,115 | 1,346 | 373 | 119 | **FAIL** | repeat: reasoning loops 3x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 507 | 181 | 502 | 201 | 19 | PASS | says 0.05 |
| `reason-machines` | on | stop | 311 | 78 | 323 | 79 | 10 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 4x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' — the answer itself is sound (29 keys, anchored); `css-card` (on) reasoning loops 4x on 'need maybe "custom properties on :root for its colours, '; `en-explain` (on) repeat: reasoning loops 4x on '95% hit rate can leave a system slower than no cache at' — the answer itself is sound (2 paragraphs, 11 sentences); `en-note` (on) repeat: reasoning loops 5x on 'within the error and latency thresholds for at least the' — the answer itself is sound (3 paragraphs, 6 sentences); `js-debounce` (on) repeat: reasoning loops 4x on '= null; let lastargs = null; let lastthis = null; functi' — the answer itself is sound (12 callables); `ts-groupby` (on) repeat: reasoning loops 3x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 5.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
