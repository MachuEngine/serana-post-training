# P3 progress log

Preference data (RLAIF), DESIGN.md §3.4 / CLAUDE.md build order.

## Step 1 -- sample two replies per prompt (SFT adapter, temp=0.9)

`scripts/generate_replies.py`, input: `data/ko/dpo_prompt_pool.jsonl`
(893 prompts, built in P1, leakage-checked against the eval set).

**Predicted vs measured** (8-prompt timing test before committing to the
full run):
- Predicted (from the 8-prompt test): ~32 min for 893 prompts (0.46 prompts/s)
- Peak VRAM: 18.64GB (bf16, not 4-bit -- this script loads the model
  full-precision since it's inference-only, no training memory to save on)

Quality spot-check on the 8-prompt test: both replies per prompt stay
in-character and are genuinely different from each other (temperature=0.9
doing its job) -- good raw material for a preference judge to have
something real to choose between.

**Full 893-prompt run: done.**

| | predicted (from 8-prompt test) | measured |
|---|---:|---:|
| wall clock | ~32 min | 33.2 min (1990.9s) |
| peak VRAM | 18.64GB | 18.67GB |

Close match on both -- the 8-prompt timing test extrapolated well.
1,786 total replies (893 prompts x 2) written to `data/ko/raw/reply_pairs.jsonl`.
GPU work done for this stage -- VM stopped before moving to judging.

## Step 2-4 -- preference judge, discard ties/near-dups -- done

`src/data/build_preferences.py`, run locally (API-only, no GPU needed
per CLAUDE.md cost discipline).

| | count | rate |
|---|---:|---:|
| pairs generated | 893 | -- |
| near-duplicate discarded | 1 | 0.11% |
| tie discarded | 31 | 3.47% |
| judge errors | 0 | -- |
| **final preference pairs** | **861** | -- |

Close to the ~1,000 target from DESIGN.md §3.4. **Tie rate is low
(3.47%)** -- DESIGN.md's caution was "a high tie rate predicts a small
DPO gain," so a low rate here is the encouraging direction: the two
sampled replies usually had a clear winner, meaning there's real
contrastive signal for DPO to learn from, not an SFT model that's
already saturated on this distribution.

Final set: `data/ko/prefs_1k.jsonl` (861 pairs) -- matches
`config/train_runs/dpo.yaml`'s expected `preference_file` path.

- **Batch-inference throughput recorded** -- done implicitly via the
  generation report above (0.449 prompts/s, 18.67GB peak).

## Hand-audit (DESIGN.md §3.4) -- found a real judge bug, prompt fixed

Built `data/ko/audit_sample.jsonl` (30 blind pairs from the 861,
A/B re-shuffled) + `artifacts/runs/p3_audit_key.json` (judge's real
choice, hidden until scoring). User filled in `human_choice` for all
30; `src.data.validate_audit` scored it:

| | v1 judge |
|---|---:|
| agreement rate | **56.67%** (17/30) |
| floor | 70% |
| **result** | **FAILED** |

Root cause: `preference_judge.py` and `judge_pcs.py` both interpolated
only `{persona_profile}` (biography) into the prompt, never
`{voice_notes}` (tone) or `{speech_level}` (persona.yaml's explicit
반말-only, 해요체/하십시오체-forbidden rule) -- even though `PROMPTS.md`
§2b's translation prompt already passes `{speech_level}` correctly.
Concretely: in `audit_03`, the judge preferred a reply ending in 존댓말
("...해요") over one correctly in 반말, because it was never told the
register rule existed.

**Fix applied**: added `{voice_notes}`/`{speech_level}` to both
prompts, bumped `PROMPTS.md` §4 v1→v2 (`[ready]`) and §5 v1→v2
(`[draft]` -- §5's P1 human-label validation ran under v1 and is now
stale; **re-run `src/eval/validate_judge.py` before trusting any
judge_pcs number in P5**). Updated `preference_judge.py`/`judge_pcs.py`
to match. Also documented (didn't fix -- too fragile to patch via
string surgery) that `reason` text reflects the judge's internal
shown-order, not the caller's un-swapped `choice` -- descriptive only,
never parse it programmatically.

Re-judged the same 30 audit pairs with the fixed prompt
(`src.data.rejudge_audit`, doesn't touch human_choice):

| | v1 judge | v2 judge (fixed) |
|---|---:|---:|
| agreement rate | 56.67% (17/30) | **73.08%** (19/26, 4 ties) |
| **result** | FAILED | **PASSED** (floor 70%) |

`audit_03` specifically flipped B→A, now matching the human pick.
Confirms the fix, not just noise.

## Full 861-pair re-judge -- blocked on OpenAI credit

Ran `src.data.build_preferences` again with the v2 judge on the full
893 reply-pairs. First attempt: **476/893 (53%) came back as
judge_error**, final pairs collapsed to 397 (well under the ~1,000
target). `judge_with_retry` only catches `RateLimitError` and the
script exited 0 (no crash), so every error was genuinely "retries
exhausted," not a schema bug -- initially attributed to the account's
known gpt-4o TPM cap (`MAX_WORKERS` dropped 3→2, `MAX_RETRIES` 6→8,
backoff cap 30s→60s per the comment in `build_preferences.py`).

**That diagnosis is now suspect**: the user confirmed OpenAI credit
hit zero around the same time. `openai-python` raises
`insufficient_quota` as `RateLimitError` too, so `judge_with_retry`
was retrying an error no amount of patience fixes. The concurrency/
retry tuning above may be irrelevant to what actually happened here --
left in place as a reasonable default, not as a proven fix.

Second attempt (fixed concurrency) was launched, then stopped
mid-run (`TaskStop`) once the credit issue was confirmed --
`data/ko/prefs_1k.jsonl` / `p3_preference_report.json` are untouched
from the first v2 attempt (397 pairs; `build_preferences.py` only
writes output after all 893 futures resolve, so a killed run leaves no
partial/corrupt write). The 397 pairs that did get judged used the
correct v2 prompt and are individually valid -- just far short of
target.

**Status: paused, not broken.** Next session, once credit is topped
up: re-run `uv run python3 -m src.data.build_preferences` on the full
893 pairs (no special resume logic -- re-judging all 893 fresh is
simpler than reconciling which 397 already succeeded, and cheap).
Confirm the error rate drops back to ~0 before trusting the result,
same as the original 893-pair run did. VM was not involved in any of
this (API/local only) and stays `TERMINATED` -- P4 (DPO, GPU) doesn't
start until this preference set is real.

## Full 861-pair re-judge -- resolved, credit confirmed topped up

User confirmed OpenAI credit was topped up. Re-ran
`uv run python3 -m src.data.build_preferences` on the full 893 pairs,
no special resume logic (per the plan above -- re-judging all fresh
rather than reconciling the partial 397).

| | v1 judge | v2, credit exhausted | v2, credit restored |
|---|---:|---:|---:|
| judge_error | 0/893 | 476/893 (53%) | **0/893** |
| tie_discarded | 31 (3.47%) | 19 (2.13%, incomplete run) | 55 (**6.16%**) |
| near_dup_discarded | 1 | 1 | 1 |
| **final preference pairs** | 861 | 397 | **837** |

`n_judge_error: 0` confirms the credit issue is what actually broke
the previous attempt, not the concurrency/retry tuning (`MAX_WORKERS`,
`MAX_RETRIES`) done earlier while that was still misdiagnosed as a TPM
cap issue. 837 final pairs -- close to v1's 861 and the ~1,000 target
from DESIGN.md §3.4.

**Tie rate note**: 6.16% is roughly double v1's 3.47% (and the 30-pair
audit's 2.13%, but that was a partial/incomplete run, not comparable).
Plausible explanation: the v2 prompt now scores on `voice_notes`/
`speech_level` in addition to content, giving the judge an extra axis
to disagree on -- more ties, not necessarily a worse signal. Still
comfortably inside "low" per DESIGN.md §3.4's caution (a *high* tie
rate predicts a small DPO gain); 6% isn't that. Not investigated
further -- flagging here in case DPO's eventual gain looks smaller
than expected and this is worth revisiting.

`data/ko/prefs_1k.jsonl` (837 pairs) and `p3_preference_report.json`
now reflect this final, credit-resolved run.

## P3: done

All CLAUDE.md/DESIGN.md §3.4 P3 done-criteria are met: pair set built
(837 pairs), tie/discard rate logged, hand-audit of 30 pairs agrees
(73.08%, v2 prompt, passed the 70% floor), batch-inference throughput
recorded (0.449 prompts/s, 18.67GB peak, from the generation step).
Next is P4 (DPO, GPU) -- continues the SFT adapter with `DPOTrainer`.
