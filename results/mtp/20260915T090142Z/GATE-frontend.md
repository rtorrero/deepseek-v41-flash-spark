# Generation gate — 2026-09-15 11:36

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
| `yaml-anchors` | on | stop | 4,198 | 1,078 | 1,485 | 402 | 71 | PASS | 29 keys, anchored |
| `css-card` | on | stop | 19,722 | 5,931 | 1,683 | 588 | 336 | **FAIL** | reasoning loops 3x on 'var(--card-bg); color: var(--card-color); border: 1px so' |
| `en-explain` | on | stop | 9,196 | 2,201 | 1,377 | 306 | 161 | **FAIL** | repeat: reasoning loops 3x on 'with 95% hit rate can leave system slower than no cache ' — the answer itself is sound (2 paragraphs, 11 sentences) |
| `en-note` | on | stop | 21,683 | 4,947 | 1,172 | 246 | 306 | **FAIL** | repeat: reasoning loops 6x on 'to 12% within minutes after the deploy, while the previo' — the answer itself is sound (3 paragraphs, 9 sentences) |
| `html-page` | on | stop | 386 | 99 | 6,560 | 2,162 | 86 | PASS | 82 declarations, 20 functions, 0 empty rules |
| `js-debounce` | on | stop | 21,086 | 4,962 | 1,932 | 520 | 318 | **FAIL** | repeat: reasoning loops 4x on 'args); return result; } function debounced(...args) { la' — the answer itself is sound (13 callables) |
| `tech-explain` | on | stop | 7,327 | 1,654 | 1,833 | 417 | 154 | PASS | 2 paragraphs, 11 sentences |
| `ts-groupby` | on | length | 24,278 | 6,640 | 139 | 0 | 317 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on 'a property of the element type whose value is a string o' |
| `reason-bat-ball` | on | stop | 539 | 157 | 588 | 230 | 19 | PASS | says 0.05 |
| `reason-machines` | on | stop | 750 | 191 | 363 | 90 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `css-card` (on) reasoning loops 3x on 'var(--card-bg); color: var(--card-color); border: 1px so'; `en-explain` (on) repeat: reasoning loops 3x on 'with 95% hit rate can leave system slower than no cache ' — the answer itself is sound (2 paragraphs, 11 sentences); `en-note` (on) repeat: reasoning loops 6x on 'to 12% within minutes after the deploy, while the previo' — the answer itself is sound (3 paragraphs, 9 sentences); `js-debounce` (on) repeat: reasoning loops 4x on 'args); return result; } function debounced(...args) { la' — the answer itself is sound (13 callables); `ts-groupby` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on 'a property of the element type whose value is a string o'

8 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 1, corrupt 0, content 1, repeat 3.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
