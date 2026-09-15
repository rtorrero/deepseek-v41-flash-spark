# Generation gate — 2026-09-15 17:45

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
| `acad-abstract` | on | stop | 26,571 | 5,576 | 1,636 | 297 | 353 | **FAIL** | repeat: reasoning loops 3x on '"write the abstract and the first paragraph of the metho' — the answer itself is sound (4 paragraphs, 11 sentences) |
| `en-explain` | on | stop | 12,870 | 3,084 | 1,477 | 341 | 213 | **FAIL** | repeat: reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.' — the answer itself is sound (2 paragraphs, 15 sentences) |
| `en-note` | on | stop | 9,583 | 2,113 | 1,340 | 274 | 148 | **FAIL** | repeat: reasoning loops 3x on '"before it goes out again, it has to be true that..." go' — the answer itself is sound (3 paragraphs, 5 sentences) |
| `news-lede` | on | length | 79,696 | 16,000 | 0 | 0 | 685 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 97x on 'the hospital said it would resume the elective list once' |
| `copy-landing` | on | stop | 3,301 | 716 | 900 | 174 | 68 | PASS | 4 paragraphs, 12 sentences |
| `xl-en-fr` | on | length | 6,332 | 1,631 | 139 | 0 | 91 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 13x on '"the outage began at 14:05" = "the outage began at 14:05' |
| `reason-bat-ball` | on | stop | 502 | 172 | 526 | 216 | 18 | PASS | says 0.05 |
| `reason-machines` | on | stop | 787 | 208 | 302 | 81 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 8 runs failed: `acad-abstract` (on) repeat: reasoning loops 3x on '"write the abstract and the first paragraph of the metho' — the answer itself is sound (4 paragraphs, 11 sentences); `en-explain` (on) repeat: reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.' — the answer itself is sound (2 paragraphs, 15 sentences); `en-note` (on) repeat: reasoning loops 3x on '"before it goes out again, it has to be true that..." go' — the answer itself is sound (3 paragraphs, 5 sentences); `news-lede` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 97x on 'the hospital said it would resume the elective list once'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 13x on '"the outage began at 14:05" = "the outage began at 14:05'

6 of 8 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 1, guard 1, corrupt 0, content 0, repeat 3.

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
