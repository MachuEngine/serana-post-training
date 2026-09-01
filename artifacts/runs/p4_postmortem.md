# P4 postmortem: why DPO produced a null result

Companion to `p4_progress.md` (which logs *what happened* during the DPO
run) and `p5_progress.md` Stage 3 (which logs the eval comparison). This
document answers the one question a portfolio reviewer asks first: **DPO
didn't beat SFT, so why not?** No new GPU work: this is analysis of
artifacts that already exist (`run_report.json`, `p3_preference_report.json`,
`eval_*.json`, `results_quality.md`, `p3_audit_report.json`,
`judge_validation_report.json`).

One line: **the preference pairs carried almost no learnable signal, so
DPO training moved the policy by an amount indistinguishable from noise.
This is not circularity, and it is not a beta-tuning problem.**

---

## 1. The result

`results_quality.md`, applying DESIGN.md §4.5's rule (a difference is real
only if the 95% bootstrap CIs do not overlap):

| metric | SFT | DPO | verdict |
|---|---|---|---|
| PCS (judge) | 0.800 [0.633, 0.933] | 0.800 [0.633, 0.933] | identical |
| PRS (judge + regex) | 0.850 [0.700, 1.000] | 0.850 [0.700, 1.000] | identical |
| style similarity (embeddings) | 0.234 [0.216, 0.252] | 0.234 [0.218, 0.250] | identical |
| knowledge-boundary acc | 0.933 [0.833, 1.000] | 0.867 [0.733, 0.967] | DPO point estimate **lower**, CIs overlap |
| mean reply length | 23.36 | 24.14 | no change |
| distinct-2 | 0.587 | 0.590 | no change |

DPO shows **no CI-confirmed gain on any metric**, and the one metric that
moves at all (knowledge-boundary) moves *down*. The entire post-training
gain in this project is at the **B to SFT** step (reply length 150 to 23,
distinct-2 0.265 to 0.587, degeneration loops eliminated, knowledge-boundary
0.833 to 0.933). SFT to DPO adds nothing measurable.

---

## 2. The training log is the smoking gun

From `artifacts/lora/serana-dpo/run_report.json` (`log_history`),
beta=0.1, lr=5e-6, 1 epoch, grad_accum=8, 837 pairs (val 5% -> ~795 train
/ ~42 eval), 100 steps:

- **Per-step DPO loss never left ln(2) ~= 0.6931.** Step 5: 0.690. Step
  25: 0.691. Step 100: 0.692. The whole run sat at the loss value you get
  when the model assigns chosen and rejected equal probability, i.e. it
  never fit the training preferences at all.
  - (`final_train_loss: 0.3467` in the report is misleading: it is a
    HF-computed aggregate over the post-resume segment, not a per-step
    value. No logged step is below 0.68. Do not cite the 0.3467 number.)
- **Reward margins oscillate around zero with no trend.** Train
  `rewards/margins` bounces between -0.012 and +0.015 across all 100
  steps. Train `rewards/accuracies` averages ~0.51 (coin flip).
- **Held-out preference accuracy went *down* during training.**
  `eval_rewards/accuracies`: 0.690 (step 25) -> 0.690 (step 50) -> 0.595
  (step 75) -> 0.595 (step 100). And 0.595 = 25/42, which is within one
  binomial standard error (~0.076) of 0.50. The final "59.5%" is
  statistically indistinguishable from chance.
- **Held-out reward margin shrinks toward zero:** 0.0184 -> 0.0204 ->
  0.0070 -> 0.0042.
- grad_norm stayed in the 4.3-5.6 band (one 8.75 spike at the last step).
  No NaN, no divergence: this is not a broken run, it is an *inert* one.

### Why this rules out "beta was too high"

If beta were the problem (KL penalty too strong, policy pinned to the
reference), the **training** loss would still fall: the model would fit
the preference pairs it is shown, and only *transfer* to held-out data
would fail. Here the training loss itself never moved off ln(2). The
model could not fit the pairs even on the data it was trained on. That
points at the data, not the regularization strength.

**Conclusion: a beta sweep (0.05 / 0.1 / 0.3) would produce three
equally flat curves. Do not spend the ~8 GPU-hours on it.**

---

## 3. Root cause: three compounding factors

### 3a. Chosen and rejected are nearly the same text

`p3_progress.md` step 1: both replies per prompt were sampled from the
**same `serana-sft` adapter** at temperature 0.9 on the same prompt. The
SFT model's output distribution is narrow (that is what SFT did: it
collapsed reply length 150 -> 23 and made replies formulaic). Two samples
from a narrow distribution differ trivially. The judge can still pick a
"winner", but chosen minus rejected gives almost no gradient because the
two sequences are almost the same tokens.

Evidence: `eval_sft.json` vs `eval_dpo.json` replies are frequently
byte-identical (e.g. `ib01`, most `oob*` items) and never more than
cosmetically different. `run_report.json` shows `logps/chosen ~= -37.3`
vs `logps/rejected ~= -38.0` at eval time: the reference model itself
barely separates them.

The P3 log read the low tie rate (6.2%) as "real contrastive signal
exists". That was too optimistic: "not a tie, per the judge" is not the
same as "far enough apart for the policy to learn from".

### 3b. The preference judge agrees with humans only ~70%

`p3_audit_report.json`: the v1 judge agreed with the human on **17/30 =
56.7%** of audit pairs and **failed** the 70% floor. After fixing a real
prompt bug (v1 never told the judge about the 반말-only register rule:
see `PROMPTS.md` §4 v1->v2), the v2 judge reached **19/26 = 73.08%** ,
barely over the floor, on a 26-pair sample. A labeler that is right ~70%
of the time on a 2-way choice injects a large fraction of noise labels
directly into the DPO training target. (Separately, the *eval* judge
`judge_pcs` v2 scored Spearman 0.7338 against the 50 human labels: also
"passes, not strong".)

### 3c. Small data, one epoch, tiny learning rate

837 pairs, ~795 after the val split, 1 epoch, 100 steps, lr 5e-6. DPO in
the literature typically uses thousands to tens of thousands of pairs.
Even with a clean signal this is a small nudge: `rewards/chosen` stays in
+-0.01 the whole run, i.e. the policy log-probs barely move from the
reference.

---

## 4. What this is NOT

- **Not circularity.** CLAUDE.md's circularity guard anticipates "DPO
  wins on judge-scored metrics but nothing else". That did not happen:
  DPO did not win on PCS or PRS either (0.800 = 0.800, 0.850 = 0.850). If
  circular reinforcement were inflating the result, the judge metrics
  would have risen while the regex/embedding metrics stayed flat. Every
  channel is flat. This is the opposite failure mode: no effect at all,
  not a fake effect. Worth stating explicitly in the writeup, because it
  is the reverse of what the guard was built to catch.
- **Not a broken pipeline.** `p4_progress.md` documents three real bugs
  found and fixed on the first DPO run (conversational-format mismatch,
  the `train.dpo.*` override path not being read, eval-batch-size OOM).
  The run that produced these numbers is the post-fix run: lr 5e-6, 1
  epoch, 100 steps, all as intended. The null is a property of the data,
  not a leftover bug.
- **Not degeneration.** Both DPO guards pass: mean reply length 24.1 (vs
  SFT 23.4), distinct-2 0.590 (vs 0.587). No length blowup, no repetition
  collapse.

---

## 5. The null is corroborated across independent signals

This matters because a single flat metric could be a measurement
artifact. Here every independent channel agrees:

| signal | produced by | SFT -> DPO |
|---|---|---|
| DPO training loss | the optimizer | never left ln(2) |
| held-out preference accuracy | the reference model | 0.69 -> 0.60 (down, near chance) |
| P4 smoke test (3 prompts, hand-read) | a human | Q1 byte-identical to SFT; Q3 regression unchanged |
| PCS | LLM judge (rubric A) | 0.800 -> 0.800 |
| style similarity | ko-sroberta embeddings (no LLM) | 0.234 -> 0.234 |
| distinct-2 / reply length | pure string metrics | unchanged |
| PRS failure breakdown | regex rule-checks (no LLM) | identical ({exits_role: 1, none: 2}) |
| knowledge-boundary acc | LLM judge (rubric B) | 0.933 -> 0.867 (down) |

Five of these do not involve an LLM judge at all. They all say the same
thing. This is a robust null, not an artifact.

---

## 6. What would have been needed (and why we are not doing it)

To give DPO a real chance at showing a delta:

1. **Higher-contrast pairs.** Sample one reply from `serana-sft` and one
   from base **B** (or from a deliberately degraded prompt), so chosen
   and rejected are genuinely far apart. This is a P3 redo.
2. **A better preference judge.** A stronger model, an explicit
   checklist rubric, and discarding low-confidence pairs.
3. **More pairs.** 3k-10k rather than 837. Another data-pipeline pass.

All three are out of scope for this portfolio (CLAUDE.md scope
discipline: the experiment is deliberately minimal, and a DPO beta sweep
/ data-scale sweep are explicitly cut). The honest move is to **report
the null with its cause**, not to keep spending until DPO wins, which
CLAUDE.md names directly.

---

## 7. Recommended framing for the README / blog

> DPO produced a clean, corroborated null: no CI-confirmed change versus
> SFT on any quality metric, and its one visible movement (knowledge
> boundary) was slightly negative. The cause is visible in the training
> log: per-step DPO loss never left ln(2), meaning the model never fit
> the preference pairs even on the training set. The pairs carried almost
> no learnable signal because both the chosen and the rejected reply were
> sampled from the same already-narrow SFT distribution and differed only
> cosmetically, and the AI judge labeling them agreed with human
> annotators only about 70% of the time. This is not circularity, which
> would have inflated the judge-scored metrics (it did not), and it is
> not a KL-strength problem, which would still have moved the training
> loss (it did not). It is the honest ceiling of RLAIF when the policy
> you sample from has already converged and the judge is noisy. The
> post-training gain in this project lives entirely at the B to SFT step.
