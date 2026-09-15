# Generation gate — 2026-09-15 09:03

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
| `yaml-anchors` | on | stop | 3,119 | 849 | 1,875 | 499 | 71 | **FAIL** | repeat: reasoning loops 3x on 'environment: <<: *env # only this field is overridden fo' — the answer itself is sound (25 keys, anchored) |
| `css-card` | on | stop | 9,074 | 2,612 | 1,630 | 549 | 148 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | stop | 7,510 | 1,838 | 1,114 | 257 | 139 | PASS | 2 paragraphs, 11 sentences |
| `en-note` | on | stop | 7,853 | 1,722 | 965 | 193 | 120 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | stop | 15,378 | 4,695 | 3,743 | 1,217 | 244 | **FAIL** | repeat: reasoning loops 3x on '<title>tic-tac-toe</title> <style> * { box-sizing: borde' — the answer itself is sound (60 declarations, 16 functions, 0 empty rules) |
| `js-debounce` | on | stop | 44,045 | 10,843 | 1,694 | 467 | 604 | **FAIL** | repeat: reasoning loops 14x on '= lastargs; const thisarg = lastthis; lastargs = lastthi' — the answer itself is sound (8 callables) |
| `tech-explain` | on | stop | 5,507 | 1,282 | 1,411 | 319 | 121 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 44,927 | 12,568 | 942 | 274 | 668 | **FAIL** | repeat: reasoning loops 8x on 'must be a property of the element type whose value is a' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 1,079 | 315 | 638 | 239 | 29 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,164 | 305 | 327 | 84 | 24 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 3x on 'environment: <<: *env # only this field is overridden fo' — the answer itself is sound (25 keys, anchored); `css-card` (on) missing prefers-color-scheme; `html-page` (on) repeat: reasoning loops 3x on '<title>tic-tac-toe</title> <style> * { box-sizing: borde' — the answer itself is sound (60 declarations, 16 functions, 0 empty rules); `js-debounce` (on) repeat: reasoning loops 14x on '= lastargs; const thisarg = lastthis; lastargs = lastthi' — the answer itself is sound (8 callables); `ts-groupby` (on) repeat: reasoning loops 8x on 'must be a property of the element type whose value is a' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 4.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
