# Generation gate — 2026-09-15 11:01

| | |
|---|---|
| profile | Writing |
| topics | english, journalism, marketing, academic, translation, reasoning, reasoning_code |
| prompts | 8 runs over 8 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |
| keep-set | PRUNE_KEEP=0.36, DSV41_PRUNE_RANK=maxmin, DSV41_PRUNE_SOURCE=saliency |
| reasoning-span controls | off (neither DSV41_THINK_BUDGET nor DSV41_THINK_REPEAT_BREAK was set) |

| prompt | thinking | finish | reasoning chars | reasoning tokens | answer chars | answer tokens | s | | why |
|---|---|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 12,758 | 2,487 | 2,210 | 420 | 201 | **FAIL** | sentence repeated verbatim: 'Inclusion criteria are terminal merge/close, non-bot actor, ' |
| `en-explain` | on | stop | 25,418 | 6,637 | 1,149 | 286 | 433 | **FAIL** | repeat: reasoning loops 3x on 'hit rate is impressive, but the 5% misses and the cache ' — the answer itself is sound (2 paragraphs, 13 sentences) |
| `en-note` | on | stop | 11,066 | 2,385 | 1,179 | 231 | 160 | **FAIL** | repeat: reasoning loops 3x on 'that the root cause is understood and fixed, the fix has' — the answer itself is sound (3 paragraphs, 6 sentences) |
| `news-lede` | on | length | 11,949 | 2,653 | 139 | 0 | 152 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on 'left the facility without adequate power for its electiv' |
| `copy-landing` | on | stop | 15,463 | 3,432 | 482 | 92 | 233 | **FAIL** | repeat: reasoning loops 3x on 'query stops running, the index remains: it still occupie' — the answer itself is sound (4 paragraphs, 7 sentences) |
| `xl-en-fr` | on | length | 20,933 | 4,869 | 139 | 0 | 241 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 18x on '"nothing alerted" = "nothing alerted" = "nothing alerted' |
| `reason-bat-ball` | on | stop | 1,063 | 341 | 526 | 224 | 29 | PASS | says 0.05 |
| `reason-machines` | on | stop | 959 | 246 | 337 | 91 | 20 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 8 runs failed: `acad-abstract` (on) sentence repeated verbatim: 'Inclusion criteria are terminal merge/close, non-bot actor, '; `en-explain` (on) repeat: reasoning loops 3x on 'hit rate is impressive, but the 5% misses and the cache ' — the answer itself is sound (2 paragraphs, 13 sentences); `en-note` (on) repeat: reasoning loops 3x on 'that the root cause is understood and fixed, the fix has' — the answer itself is sound (3 paragraphs, 6 sentences); `news-lede` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on 'left the facility without adequate power for its electiv'; `copy-landing` (on) repeat: reasoning loops 3x on 'query stops running, the index remains: it still occupie' — the answer itself is sound (4 paragraphs, 7 sentences); `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 18x on '"nothing alerted" = "nothing alerted" = "nothing alerted'

5 of 8 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 2, corrupt 0, content 1, repeat 3.

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
