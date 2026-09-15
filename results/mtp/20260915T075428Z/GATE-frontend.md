# Generation gate — 2026-09-15 10:37

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
| `yaml-anchors` | on | stop | 23,455 | 5,693 | 1,125 | 327 | 314 | **FAIL** | repeat: reasoning loops 5x on 'one block of environment variables and one healthcheck d' — the answer itself is sound (28 keys, anchored) |
| `css-card` | on | stop | 20,453 | 5,923 | 1,284 | 456 | 303 | **FAIL** | repeat: reasoning loops 3x on '#1d4ed8; --card-shadow: 0 1px 2px rgb(0 0 0 / 0.08); --c' — the answer itself is sound (28 declarations, 0 empty rules) |
| `en-explain` | on | stop | 15,607 | 3,813 | 1,289 | 297 | 264 | PASS | 2 paragraphs, 13 sentences |
| `en-note` | on | stop | 11,587 | 2,629 | 961 | 202 | 170 | **FAIL** | repeat: reasoning loops 3x on 'be "this afternoon\'s release" as "this afternoon\'s relea' — the answer itself is sound (3 paragraphs, 6 sentences) |
| `html-page` | on | stop | 25,273 | 7,563 | 5,662 | 1,622 | 351 | **FAIL** | repeat: reasoning loops 5x on '```html <!doctype html> <html lang="en"> <head> <meta ch' — the answer itself is sound (68 declarations, 16 functions, 0 empty rules) |
| `js-debounce` | on | stop | 6,653 | 1,584 | 1,402 | 389 | 103 | PASS | 13 callables |
| `tech-explain` | on | stop | 4,757 | 1,033 | 1,908 | 385 | 110 | PASS | 2 paragraphs, 12 sentences |
| `ts-groupby` | on | stop | 41,600 | 10,719 | 953 | 274 | 604 | **FAIL** | reasoning loops 13x on 'must be a property of the element type whose value is a' |
| `reason-bat-ball` | on | stop | 897 | 286 | 579 | 220 | 29 | PASS | says 0.05 |
| `reason-machines` | on | stop | 588 | 151 | 354 | 87 | 15 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 5x on 'one block of environment variables and one healthcheck d' — the answer itself is sound (28 keys, anchored); `css-card` (on) repeat: reasoning loops 3x on '#1d4ed8; --card-shadow: 0 1px 2px rgb(0 0 0 / 0.08); --c' — the answer itself is sound (28 declarations, 0 empty rules); `en-note` (on) repeat: reasoning loops 3x on 'be "this afternoon\'s release" as "this afternoon\'s relea' — the answer itself is sound (3 paragraphs, 6 sentences); `html-page` (on) repeat: reasoning loops 5x on '```html <!doctype html> <html lang="en"> <head> <meta ch' — the answer itself is sound (68 declarations, 16 functions, 0 empty rules); `ts-groupby` (on) reasoning loops 13x on 'must be a property of the element type whose value is a'

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 4.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
