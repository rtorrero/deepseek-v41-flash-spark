# Generation gate — 2026-09-15 13:06

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
| keep-set | PRUNE_KEEP=0.36, DSV41_PRUNE_RANK=maxmin, DSV41_PRUNE_SOURCE=saliency |
| reasoning-span controls | off (neither DSV41_THINK_BUDGET nor DSV41_THINK_REPEAT_BREAK was set) |

| prompt | thinking | finish | reasoning chars | reasoning tokens | answer chars | answer tokens | s | | why |
|---|---|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 2,894 | 737 | 1,817 | 465 | 57 | PASS | 32 keys, anchored |
| `css-card` | on | stop | 272 | 69 | 3,646 | 1,166 | 57 | PASS | 54 declarations, 0 empty rules |
| `en-explain` | on | stop | 6,549 | 1,481 | 1,894 | 425 | 122 | PASS | 2 paragraphs, 13 sentences |
| `en-note` | on | stop | 8,516 | 1,879 | 1,390 | 282 | 137 | PASS | 3 paragraphs, 5 sentences |
| `html-page` | on | stop | 5,021 | 1,330 | 3,511 | 1,074 | 114 | PASS | 51 declarations, 11 functions, 0 empty rules |
| `js-debounce` | on | stop | 22,532 | 5,278 | 1,505 | 394 | 322 | **FAIL** | repeat: reasoning loops 5x on 'let timerid = null; let lastargs = null; let lastthis = ' — the answer itself is sound (10 callables) |
| `tech-explain` | on | stop | 5,493 | 1,176 | 1,414 | 301 | 116 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | stop | 11,465 | 3,145 | 1,435 | 448 | 187 | **FAIL** | repeat: reasoning loops 5x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 818 | 259 | 413 | 172 | 20 | PASS | says 0.05 |
| `reason-machines` | on | stop | 546 | 140 | 281 | 67 | 12 | PASS | says 5 minutes |

**Verdict: FAIL** — 2 of 10 runs failed: `js-debounce` (on) repeat: reasoning loops 5x on 'let timerid = null; let lastargs = null; let lastthis = ' — the answer itself is sound (10 callables); `ts-groupby` (on) repeat: reasoning loops 5x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed)

10 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 0, repeat 2.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
