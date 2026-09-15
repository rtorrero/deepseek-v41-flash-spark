# Generation gate — 2026-09-15 04:25

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
| `yaml-anchors` | on | stop | 32,427 | 7,797 | 1,022 | 303 | 426 | **FAIL** | repeat: reasoning loops 7x on 'block of environment variables and one healthcheck defin' — the answer itself is sound (24 keys, anchored) |
| `css-card` | on | stop | 9,972 | 2,744 | 2,245 | 781 | 175 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | stop | 11,859 | 2,707 | 1,047 | 214 | 188 | **FAIL** | repeat: reasoning loops 3x on 'a cache with a 95 % hit rate can leave a system' — the answer itself is sound (2 paragraphs, 6 sentences) |
| `en-note` | on | stop | 4,978 | 1,097 | 1,105 | 219 | 83 | **FAIL** | repeat: reasoning loops 3x on "rolling back this afternoon's release, what the symptom " — the answer itself is sound (3 paragraphs, 7 sentences) |
| `html-page` | on | stop | 19,215 | 5,715 | 7,130 | 2,178 | 349 | PASS | 81 declarations, 20 functions, 0 empty rules |
| `js-debounce` | on | stop | 21,572 | 5,253 | 1,736 | 479 | 308 | **FAIL** | repeat: reasoning loops 8x on 'timeoutid = null; const args = lastargs; const thisarg =' — the answer itself is sound (8 callables) |
| `tech-explain` | on | stop | 9,044 | 1,979 | 1,206 | 263 | 156 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 10,752 | 3,053 | 2,136 | 628 | 180 | **FAIL** | repeat: reasoning loops 5x on 't[]> { const groups: record<string, t[]> = {}; for (cons' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 1,455 | 501 | 619 | 225 | 36 | PASS | says 0.05 |
| `reason-machines` | on | stop | 412 | 105 | 440 | 113 | 13 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 7x on 'block of environment variables and one healthcheck defin' — the answer itself is sound (24 keys, anchored); `css-card` (on) missing prefers-color-scheme; `en-explain` (on) repeat: reasoning loops 3x on 'a cache with a 95 % hit rate can leave a system' — the answer itself is sound (2 paragraphs, 6 sentences); `en-note` (on) repeat: reasoning loops 3x on "rolling back this afternoon's release, what the symptom " — the answer itself is sound (3 paragraphs, 7 sentences); `js-debounce` (on) repeat: reasoning loops 8x on 'timeoutid = null; const args = lastargs; const thisarg =' — the answer itself is sound (8 callables); `ts-groupby` (on) repeat: reasoning loops 5x on 't[]> { const groups: record<string, t[]> = {}; for (cons' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 5.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
