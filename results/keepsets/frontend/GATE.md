# Generation gate — 2026-09-13 00:55

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 1,561 | 2,001 | 56 | PASS | 32 keys, anchored |
| `css-card` | on | stop | 257 | 3,157 | 46 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | length | 67,513 | 0 | 738 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 68x on 'with no cache at all, every request goes to origin; if o' |
| `en-note` | on | stop | 9,451 | 1,293 | 145 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | stop | 22,188 | 7,307 | 419 | PASS | 93 declarations, 24 functions, 0 empty rules |
| `js-debounce` | on | stop | 36,653 | 1,066 | 571 | **FAIL** | reasoning loops 9x on 'wait) { let timer = null; let lastargs = null; let lastt' |
| `tech-explain` | on | stop | 9,046 | 1,422 | 176 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | length | 59,295 | 0 | 911 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 15x on 'must be a property of the element type whose value is a' |
| `reason-bat-ball` | on | stop | 1,141 | 548 | 32 | PASS | says 0.05 |
| `reason-machines` | on | stop | 751 | 381 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `css-card` (on) missing prefers-color-scheme; `en-explain` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 68x on 'with no cache at all, every request goes to origin; if o'; `js-debounce` (on) reasoning loops 9x on 'wait) { let timer = null; let lastargs = null; let lastt'; `ts-groupby` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 15x on 'must be a property of the element type whose value is a'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 17:19

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 19,876 | 1,027 | 260 | **FAIL** | reasoning loops 5x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' |
| `css-card` | on | stop | 21,820 | 2,450 | 333 | **FAIL** | reasoning loops 4x on 'dark) { :root { color-scheme: dark; --card-surface: #181' |
| `en-explain` | on | stop | 5,957 | 1,564 | 125 | PASS | 2 paragraphs, 13 sentences |
| `en-note` | on | stop | 27,870 | 1,171 | 345 | **FAIL** | reasoning loops 17x on 'fix must be proven to eliminate the symptom under the sa' |
| `html-page` | on | stop | 15,136 | 4,226 | 252 | PASS | 58 declarations, 16 functions, 0 empty rules |
| `js-debounce` | on | stop | 28,590 | 1,687 | 401 | **FAIL** | reasoning loops 6x on 'function debounce(fn, wait) { let timerid = null; let la' |
| `tech-explain` | on | stop | 5,725 | 1,462 | 112 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 57,644 | 1,023 | 833 | **FAIL** | reasoning loops 20x on 'function groupby<t, k extends groupablekey<t>>( items: r' |
| `reason-bat-ball` | on | stop | 860 | 528 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 636 | 445 | 16 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `yaml-anchors` (on) reasoning loops 5x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i'; `css-card` (on) reasoning loops 4x on 'dark) { :root { color-scheme: dark; --card-surface: #181'; `en-note` (on) reasoning loops 17x on 'fix must be proven to eliminate the symptom under the sa'; `js-debounce` (on) reasoning loops 6x on 'function debounce(fn, wait) { let timerid = null; let la'; `ts-groupby` (on) reasoning loops 20x on 'function groupby<t, k extends groupablekey<t>>( items: r'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 18:14

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 21,544 | 961 | 271 | **FAIL** | reasoning loops 6x on 'service_role: shared x-common-healthcheck: &common-healt' |
| `css-card` | on | stop | 4,694 | 1,792 | 87 | PASS | 39 declarations, 0 empty rules |
| `en-explain` | on | stop | 12,960 | 1,367 | 223 | PASS | 2 paragraphs, 10 sentences |
| `en-note` | on | stop | 6,851 | 1,044 | 118 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | stop | 4,201 | 3,613 | 103 | PASS | 48 declarations, 10 functions, 0 empty rules |
| `js-debounce` | on | stop | 10,788 | 1,756 | 169 | **FAIL** | reasoning loops 3x on 'function debounced(...args) { lastargs = args; lastthis ' |
| `tech-explain` | on | stop | 9,447 | 1,509 | 176 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | length | 60,064 | 0 | 843 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 17x on 'must be a property of the element type whose value is a' |
| `reason-bat-ball` | on | stop | 870 | 528 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 630 | 452 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 10 runs failed: `yaml-anchors` (on) reasoning loops 6x on 'service_role: shared x-common-healthcheck: &common-healt'; `js-debounce` (on) reasoning loops 3x on 'function debounced(...args) { lastargs = args; lastthis '; `ts-groupby` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 17x on 'must be a property of the element type whose value is a'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 01:22

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

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 5,919 | 2,341 | 105 | PASS | 31 keys, anchored |
| `css-card` | on | stop | 12,190 | 1,946 | 192 | PASS | 45 declarations, 0 empty rules |
| `en-explain` | on | stop | 5,099 | 1,035 | 100 | PASS | 2 paragraphs, 11 sentences |
| `en-note` | on | stop | 12,893 | 1,165 | 171 | **FAIL** | repeat: reasoning loops 4x on 'symptom was, and what has to be true before it goes out' — the answer itself is sound (3 paragraphs, 7 sentences) |
| `html-page` | on | stop | 6,982 | 4,152 | 141 | PASS | 62 declarations, 15 functions, 0 empty rules |
| `js-debounce` | on | stop | 15,284 | 1,820 | 230 | **FAIL** | repeat: reasoning loops 3x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (11 callables) |
| `tech-explain` | on | stop | 8,143 | 1,532 | 163 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 11,772 | 1,292 | 185 | **FAIL** | reasoning loops 4x on 'record<string, t[]> = {}; for (const item of items) { co' |
| `reason-bat-ball` | on | stop | 949 | 450 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 956 | 323 | 23 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 10 runs failed: `en-note` (on) repeat: reasoning loops 4x on 'symptom was, and what has to be true before it goes out' — the answer itself is sound (3 paragraphs, 7 sentences); `js-debounce` (on) repeat: reasoning loops 3x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (11 callables); `ts-groupby` (on) reasoning loops 4x on 'record<string, t[]> = {}; for (const item of items) { co'

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 2.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 20:21

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
| reasoning-span controls | DSV41_THINK_BUDGET=8000 |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 2,681 | 1,656 | 59 | PASS | 29 keys, anchored |
| `css-card` | on | stop | 4,623 | 3,100 | 109 | PASS | 50 declarations, 0 empty rules |
| `en-explain` | on | stop | 3,093 | 1,439 | 70 | PASS | 2 paragraphs, 14 sentences |
| `en-note` | on | stop | 10,354 | 921 | 161 | **FAIL** | repeat: reasoning loops 3x on '— i’m rolling back the release that went out this aftern' — the answer itself is sound (3 paragraphs, 6 sentences) |
| `html-page` | on | stop | 3,923 | 8,374 | 155 | **FAIL** | reasoning has a corrupted run 'a===b' in 'if (a && a===b && b===c) return a;'; answer loops 3x on 'html> <html lang="en"> <head> <meta charset="utf-8"> <me' |
| `js-debounce` | on | stop | 24,247 | 1,654 | 364 | **FAIL** | repeat: reasoning loops 5x on 'function debounced(...args) { lastthis = this; lastargs ' — the answer itself is sound (10 callables) |
| `tech-explain` | on | stop | 5,867 | 1,465 | 116 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | stop | 10,155 | 1,022 | 164 | **FAIL** | repeat: reasoning loops 4x on 'type keyofstringornumber<t> = { [k in keyof t]: t[k] ext' — the answer itself is sound (generic constrained, return typed) |
| `reason-bat-ball` | on | stop | 637 | 501 | 20 | PASS | says 0.05 |
| `reason-machines` | on | stop | 938 | 312 | 19 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `en-note` (on) repeat: reasoning loops 3x on '— i’m rolling back the release that went out this aftern' — the answer itself is sound (3 paragraphs, 6 sentences); `html-page` (on) reasoning has a corrupted run 'a===b' in 'if (a && a===b && b===c) return a;'; answer loops 3x on 'html> <html lang="en"> <head> <meta charset="utf-8"> <me'; `js-debounce` (on) repeat: reasoning loops 5x on 'function debounced(...args) { lastthis = this; lastargs ' — the answer itself is sound (10 callables); `ts-groupby` (on) repeat: reasoning loops 4x on 'type keyofstringornumber<t> = { [k in keyof t]: t[k] ext' — the answer itself is sound (generic constrained, return typed)

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 1, content 0, repeat 3.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 21:28

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
| reasoning-span controls | DSV41_THINK_BUDGET=2000 |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 6,822 | 2,178 | 120 | **FAIL** | repeat: reasoning loops 3x on '"share one block of environment variables and one health' — the answer itself is sound (28 keys, anchored) |
| `css-card` | on | stop | 6,989 | 3,621 | 151 | PASS | 46 declarations, 0 empty rules |
| `en-explain` | on | stop | 8,558 | 6,638 | 221 | **FAIL** | reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.'; answer loops 3x on 'backend request because they pay the cache lookup cost a' |
| `en-note` | on | stop | 8,928 | 949 | 130 | **FAIL** | repeat: reasoning loops 3x on 'but maybe "what has to be true before it goes out again"' — the answer itself is sound (2 paragraphs, 6 sentences) |
| `html-page` | on | stop | 6,473 | 5,199 | 167 | **FAIL** | 0 <script> blocks, wanted exactly 1 |
| `js-debounce` | on | stop | 8,659 | 1,577 | 131 | PASS | 8 callables |
| `tech-explain` | on | stop | 2,077 | 1,442 | 59 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 7,551 | 1,112 | 131 | PASS | generic constrained, return typed |
| `reason-bat-ball` | on | stop | 804 | 516 | 24 | PASS | says 0.05 |
| `reason-machines` | on | stop | 786 | 279 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 3x on '"share one block of environment variables and one health' — the answer itself is sound (28 keys, anchored); `en-explain` (on) reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.'; answer loops 3x on 'backend request because they pay the cache lookup cost a'; `en-note` (on) repeat: reasoning loops 3x on 'but maybe "what has to be true before it goes out again"' — the answer itself is sound (2 paragraphs, 6 sentences); `html-page` (on) 0 <script> blocks, wanted exactly 1

8 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 2, repeat 2.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 22:39

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
| reasoning-span controls | DSV41_THINK_BUDGET=2000, DSV41_THINK_REPEAT_BREAK=12 |

| prompt | thinking | finish | reasoning chars | reasoning tokens | answer chars | answer tokens | s | | why |
|---|---|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 7,999 | 2,003 | 1,760 | 504 | 131 | PASS | 31 keys, anchored [reasoning budget hit at 2,003 reasoning tokens] |
| `css-card` | on | stop | 6,845 | 2,002 | 1,853 | 691 | 122 | PASS | 36 declarations, 0 empty rules [reasoning budget hit at 2,002 reasoning tokens] |
| `en-explain` | on | stop | 6,857 | 1,681 | 1,426 | 326 | 133 | PASS | 2 paragraphs, 10 sentences |
| `en-note` | on | stop | 9,084 | 2,001 | 992 | 198 | 131 | **FAIL** | repeat: reasoning loops 3x on '"the symptom was that newly created records were missing' — the answer itself is sound (3 paragraphs, 6 sentences) [reasoning budget hit at 2,001 reasoning tokens] |
| `html-page` | on | stop | 6,450 | 2,005 | 11,020 | 3,203 | 203 | **FAIL** | 2 <style> blocks, wanted exactly 1 [reasoning budget hit at 2,005 reasoning tokens] |
| `js-debounce` | on | stop | 8,734 | 2,001 | 29,465 | 7,060 | 508 | **FAIL** | repeat: answer loops 4x on 'lastthis = lastargs = undefined; result = fn.apply(thisa' — the answer itself is sound (67 callables) [reasoning budget hit at 2,001 reasoning tokens] |
| `tech-explain` | on | stop | 8,350 | 1,821 | 1,352 | 277 | 150 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 7,175 | 2,004 | 31,755 | 8,746 | 601 | **FAIL** | repeat: answer loops 5x on '= { [k in keyof t]: t[k] extends string | number ?' — the answer itself is sound (generic constrained, return typed) [reasoning budget hit at 2,004 reasoning tokens] |
| `reason-bat-ball` | on | stop | 1,227 | 410 | 482 | 206 | 31 | PASS | says 0.05 |
| `reason-machines` | on | stop | 665 | 181 | 238 | 59 | 15 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `en-note` (on) repeat: reasoning loops 3x on '"the symptom was that newly created records were missing' — the answer itself is sound (3 paragraphs, 6 sentences) [reasoning budget hit at 2,001 reasoning tokens]; `html-page` (on) 2 <style> blocks, wanted exactly 1 [reasoning budget hit at 2,005 reasoning tokens]; `js-debounce` (on) repeat: answer loops 4x on 'lastthis = lastargs = undefined; result = fn.apply(thisa' — the answer itself is sound (67 callables) [reasoning budget hit at 2,001 reasoning tokens]; `ts-groupby` (on) repeat: answer loops 5x on '= { [k in keyof t]: t[k] extends string | number ?' — the answer itself is sound (generic constrained, return typed) [reasoning budget hit at 2,004 reasoning tokens]

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 3.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
