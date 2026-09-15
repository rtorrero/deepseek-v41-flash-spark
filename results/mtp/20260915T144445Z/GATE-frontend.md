# Generation gate — 2026-09-15 17:19

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
| `yaml-anchors` | on | stop | 27,210 | 6,488 | 985 | 287 | 347 | **FAIL** | repeat: reasoning loops 11x on 'yaml anchor and aliases, each service overriding a singl' — the answer itself is sound (28 keys, anchored) |
| `css-card` | on | stop | 3,699 | 1,022 | 1,514 | 502 | 76 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | stop | 14,002 | 3,236 | 1,671 | 339 | 244 | **FAIL** | repeat: reasoning loops 3x on 'a cache with a 95 % hit rate can leave a system' — the answer itself is sound (2 paragraphs, 9 sentences) |
| `en-note` | on | stop | 7,203 | 1,633 | 919 | 198 | 114 | **FAIL** | repeat: reasoning loops 3x on 'has to be true that the root cause is understood and the' — the answer itself is sound (3 paragraphs, 7 sentences) |
| `html-page` | on | stop | 9,043 | 2,458 | 4,799 | 1,478 | 181 | PASS | 56 declarations, 14 functions, 0 empty rules |
| `js-debounce` | on | stop | 19,173 | 4,577 | 1,375 | 373 | 267 | **FAIL** | repeat: reasoning loops 6x on 'const { context, args: callargs } = pending; pending = n' — the answer itself is sound (10 callables) |
| `tech-explain` | on | stop | 3,774 | 823 | 1,604 | 328 | 92 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 25,501 | 6,717 | 780 | 215 | 363 | **FAIL** | repeat: reasoning loops 8x on 'be a property of the element type whose value is a strin' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 630 | 228 | 576 | 228 | 25 | PASS | says 0.05 |
| `reason-machines` | on | stop | 980 | 248 | 297 | 68 | 19 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 11x on 'yaml anchor and aliases, each service overriding a singl' — the answer itself is sound (28 keys, anchored); `css-card` (on) missing prefers-color-scheme; `en-explain` (on) repeat: reasoning loops 3x on 'a cache with a 95 % hit rate can leave a system' — the answer itself is sound (2 paragraphs, 9 sentences); `en-note` (on) repeat: reasoning loops 3x on 'has to be true that the root cause is understood and the' — the answer itself is sound (3 paragraphs, 7 sentences); `js-debounce` (on) repeat: reasoning loops 6x on 'const { context, args: callargs } = pending; pending = n' — the answer itself is sound (10 callables); `ts-groupby` (on) repeat: reasoning loops 8x on 'be a property of the element type whose value is a strin' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 5.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
