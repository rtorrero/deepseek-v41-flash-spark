# Generation gate — 2026-09-15 18:44

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
| `acad-abstract` | on | stop | 9,673 | 1,866 | 2,086 | 384 | 148 | PASS | 5 paragraphs, 11 sentences |
| `en-explain` | on | stop | 12,272 | 2,703 | 2,052 | 439 | 189 | **FAIL** | repeat: reasoning loops 3x on 'the overhead paid on 100 requests. a miss can be even wo' — the answer itself is sound (2 paragraphs, 11 sentences) |
| `en-note` | on | stop | 4,521 | 1,015 | 1,312 | 276 | 90 | PASS | 3 paragraphs, 10 sentences |
| `news-lede` | on | length | 16,219 | 3,684 | 139 | 0 | 221 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on 'that\'s fine. i should also avoid using "according to" fo' |
| `copy-landing` | on | stop | 4,245 | 928 | 741 | 145 | 77 | PASS | 4 paragraphs, 9 sentences |
| `xl-en-fr` | on | length | 11,202 | 3,344 | 139 | 0 | 163 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on '"outage" = "outage" but "outage" = "outage" but french "' |
| `reason-bat-ball` | on | stop | 1,523 | 478 | 695 | 259 | 39 | PASS | says 0.05 |
| `reason-machines` | on | stop | 674 | 182 | 313 | 91 | 15 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 8 runs failed: `en-explain` (on) repeat: reasoning loops 3x on 'the overhead paid on 100 requests. a miss can be even wo' — the answer itself is sound (2 paragraphs, 11 sentences); `news-lede` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on 'that\'s fine. i should also avoid using "according to" fo'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on '"outage" = "outage" but "outage" = "outage" but french "'

6 of 8 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 2, corrupt 0, content 0, repeat 1.

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
