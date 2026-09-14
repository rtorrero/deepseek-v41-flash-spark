# Generation gate — 2026-09-15 00:18

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
| `yaml-anchors` | on | stop | 6,549 | 1,815 | 1,789 | 473 | 233 | **FAIL** | repeat: reasoning loops 6x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' — the answer itself is sound (27 keys, anchored) |
| `css-card` | on | stop | 6,829 | 2,085 | 2,134 | 774 | 242 | PASS | 42 declarations, 0 empty rules |
| `en-explain` | on | stop | 6,305 | 1,514 | 1,153 | 254 | 224 | PASS | 2 paragraphs, 11 sentences |
| `en-note` | on | stop | 8,762 | 1,871 | 1,061 | 214 | 268 | **FAIL** | repeat: reasoning loops 4x on 'persisted until rollback, which points to the release as' — the answer itself is sound (3 paragraphs, 6 sentences) |
| `html-page` | on | stop | 8,262 | 2,277 | 4,864 | 1,522 | 349 | PASS | 66 declarations, 18 functions, 0 empty rules |
| `js-debounce` | on | stop | 7,880 | 1,934 | 1,866 | 496 | 266 | **FAIL** | repeat: reasoning loops 3x on 'lastargs = undefined; result = fn.apply(thisarg, args); ' — the answer itself is sound (11 callables) |
| `tech-explain` | on | stop | 7,683 | 1,617 | 1,346 | 269 | 287 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | stop | 45,317 | 12,133 | 742 | 224 | 1436 | **FAIL** | repeat: reasoning loops 16x on 'must be a property of the element type whose value is a' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 392 | 144 | 578 | 223 | 38 | PASS | says 0.05 |
| `reason-machines` | on | stop | 497 | 131 | 399 | 110 | 28 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 6x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' — the answer itself is sound (27 keys, anchored); `en-note` (on) repeat: reasoning loops 4x on 'persisted until rollback, which points to the release as' — the answer itself is sound (3 paragraphs, 6 sentences); `js-debounce` (on) repeat: reasoning loops 3x on 'lastargs = undefined; result = fn.apply(thisarg, args); ' — the answer itself is sound (11 callables); `ts-groupby` (on) repeat: reasoning loops 16x on 'must be a property of the element type whose value is a' — the answer itself is sound (generic constrained, return typed)

10 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 0, repeat 4.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
