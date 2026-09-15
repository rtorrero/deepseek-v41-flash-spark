# Generation gate — 2026-09-15 08:23

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
| `yaml-anchors` | on | stop | 3,681 | 947 | 1,843 | 479 | 71 | PASS | 20 keys, anchored |
| `css-card` | on | stop | 9,771 | 2,781 | 1,476 | 494 | 145 | **FAIL** | repeat: reasoning loops 3x on '@media (prefers-color-scheme: dark) { :root { --card-col' — the answer itself is sound (29 declarations, 0 empty rules) |
| `en-explain` | on | stop | 6,654 | 1,525 | 1,454 | 302 | 122 | PASS | 2 paragraphs, 9 sentences |
| `en-note` | on | stop | 4,714 | 1,007 | 1,154 | 234 | 82 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | length | 15,106 | 4,593 | 139 | 0 | 236 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 10x on 'wrong. correct code: ```js cells.foreach(c => { c.dis??=' |
| `js-debounce` | on | stop | 26,315 | 6,392 | 1,422 | 406 | 378 | **FAIL** | repeat: reasoning loops 4x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (12 callables) |
| `tech-explain` | on | stop | 6,809 | 1,431 | 1,604 | 333 | 130 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 16,574 | 4,557 | 1,240 | 352 | 250 | **FAIL** | repeat: reasoning loops 5x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 518 | 176 | 545 | 209 | 20 | PASS | says 0.05 |
| `reason-machines` | on | stop | 555 | 140 | 340 | 88 | 13 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `css-card` (on) repeat: reasoning loops 3x on '@media (prefers-color-scheme: dark) { :root { --card-col' — the answer itself is sound (29 declarations, 0 empty rules); `html-page` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 10x on 'wrong. correct code: ```js cells.foreach(c => { c.dis??='; `js-debounce` (on) repeat: reasoning loops 4x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (12 callables); `ts-groupby` (on) repeat: reasoning loops 5x on 'key must be a property of the element type whose value i' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 1, corrupt 0, content 0, repeat 3.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
