# Generation gate — 2026-09-15 12:10

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
| `acad-abstract` | on | stop | 9,677 | 1,892 | 1,876 | 349 | 146 | PASS | 4 paragraphs, 12 sentences |
| `en-explain` | on | stop | 11,270 | 2,660 | 1,712 | 398 | 203 | **FAIL** | repeat: reasoning loops 3x on 'with a 95 % hit rate can leave a system slower than' — the answer itself is sound (2 paragraphs, 14 sentences) |
| `en-note` | on | stop | 10,121 | 2,287 | 951 | 201 | 152 | **FAIL** | repeat: reasoning loops 3x on 'in 5xx responses on the checkout path, p99 latency cross' — the answer itself is sound (3 paragraphs, 8 sentences) |
| `news-lede` | on | stop | 19,711 | 4,104 | 880 | 156 | 285 | **FAIL** | repeat: reasoning loops 4x on 'list for a day, hospital officials said today, temporari' — the answer itself is sound (4 paragraphs, 7 sentences) |
| `copy-landing` | on | length | 63,555 | 16,000 | 0 | 0 | 520 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 160x on '"no superlatives you cannot support" means avoid superla' |
| `xl-en-fr` | on | length | 68,809 | 16,000 | 0 | 0 | 694 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 175x on 'began at 14:05, when a routine certificate rotation remo' |
| `reason-bat-ball` | on | stop | 1,101 | 365 | 651 | 235 | 32 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,133 | 287 | 441 | 127 | 24 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 8 runs failed: `en-explain` (on) repeat: reasoning loops 3x on 'with a 95 % hit rate can leave a system slower than' — the answer itself is sound (2 paragraphs, 14 sentences); `en-note` (on) repeat: reasoning loops 3x on 'in 5xx responses on the checkout path, p99 latency cross' — the answer itself is sound (3 paragraphs, 8 sentences); `news-lede` (on) repeat: reasoning loops 4x on 'list for a day, hospital officials said today, temporari' — the answer itself is sound (4 paragraphs, 7 sentences); `copy-landing` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 160x on '"no superlatives you cannot support" means avoid superla'; `xl-en-fr` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 175x on 'began at 14:05, when a routine certificate rotation remo'

6 of 8 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 2, guard 0, corrupt 0, content 0, repeat 3.

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
