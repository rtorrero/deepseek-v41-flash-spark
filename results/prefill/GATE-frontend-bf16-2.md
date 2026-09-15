# Generation gate — 2026-09-15 12:43

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
| `yaml-anchors` | on | stop | 1,918 | 491 | 1,450 | 385 | 42 | PASS | 29 keys, anchored |
| `css-card` | on | stop | 7,471 | 2,197 | 1,702 | 663 | 126 | PASS | 35 declarations, 0 empty rules |
| `en-explain` | on | stop | 5,693 | 1,296 | 1,430 | 296 | 117 | PASS | 2 paragraphs, 9 sentences |
| `en-note` | on | stop | 9,453 | 2,076 | 1,145 | 231 | 143 | PASS | 3 paragraphs, 8 sentences |
| `html-page` | on | stop | 5,871 | 1,843 | 4,490 | 1,446 | 133 | PASS | 62 declarations, 11 functions, 0 empty rules |
| `js-debounce` | on | stop | 36,066 | 8,420 | 1,708 | 453 | 494 | **FAIL** | repeat: reasoning loops 6x on 'if (timerid !== null) { cleartimeout(timerid); timerid =' — the answer itself is sound (10 callables) |
| `tech-explain` | on | stop | 8,178 | 1,738 | 1,414 | 304 | 156 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 21,605 | 6,023 | 1,093 | 311 | 325 | **FAIL** | repeat: reasoning loops 9x on '```ts function groupby<t, k extends keyof t>( items: rea' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 453 | 152 | 518 | 217 | 20 | PASS | says 0.05 |
| `reason-machines` | on | stop | 436 | 113 | 485 | 112 | 14 | PASS | says 5 minutes |

**Verdict: FAIL** — 2 of 10 runs failed: `js-debounce` (on) repeat: reasoning loops 6x on 'if (timerid !== null) { cleartimeout(timerid); timerid =' — the answer itself is sound (10 callables); `ts-groupby` (on) repeat: reasoning loops 9x on '```ts function groupby<t, k extends keyof t>( items: rea' — the answer itself is sound (generic constrained, return typed)

10 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 0, repeat 2.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
