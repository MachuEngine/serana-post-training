# P4 postmortem: why DPO produced a null result

Companion to `p4_progress.md` (the first DPO run), `p5_progress.md` Stage
3 (the eval comparison), and `p4redo_progress.md` (the redo). This
document answers the one question a portfolio reviewer asks first: **DPO
didn't beat SFT, so why not?**

Sections 1-5 analyse the first run from artifacts that already existed
(`run_report.json`, `eval_*.json`, `p3_audit_report.json`, ...), no new
GPU work. Section 6 covers the redo, which did rerun the preference
pipeline and DPO training (~$2 GPU / ~$6 API).

One line: **the first run's preference pairs carried almost no learnable
signal; a redo with genuinely contrastive pairs made the training
respond, and the eval-set quality still did not move versus SFT. Not
circularity, not a beta-tuning problem — a real but small learned signal
that the 30-prompt eval set cannot resolve.**

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

## 6. The redo, and the two-level null

P4 left one question genuinely open: is the null here because the
preference data gave DPO nothing to learn, or because DPO does not help
this problem? Two of the three fixes above were in reach without a
data-scale sweep, so the redo ran them (full log:
`artifacts/runs/p4redo_progress.md`, plan:
`artifacts/runs/p4_dpo_redo_plan.md`):

1. **Higher-contrast pairs, on-policy.** Sample **N=4** replies per
   prompt from `serana-sft` at temperature 1.0 and take the judge's
   best-vs-worst. Both replies still come from the model being trained
   (unlike an SFT-vs-B pairing, which would be off-policy on the rejected
   side), but selection over 4 samples manufactures real contrast. 889
   pairs.
2. **A stricter judge (v3).** Best/worst of N with a hard speech-level
   gate returned as data, a confidence field, and an automated
   cross-check of the judge's register call against the `rule_checks.py`
   regex. A 30-pair by-eye check: disagree with the judge on ~3-6 of 30,
   under the stop threshold.

**What changed:** the DPO training *responded*. Per-step loss fell below
ln(2), the reward margin turned sustainedly positive, and the held-out
reward margin climbed 0.017 -> 0.026. The mechanism that was dead in P4
was alive.

**What did not change:** the eval-set quality. `serana-dpo-v3` vs SFT,
every 95% CI overlaps — PCS 0.767 vs 0.800, PRS 0.900 vs 0.850,
knowledge-boundary 0.900 vs 0.933. One concrete item moved (`roleexit01`:
"그렇게 하자" -> "그게 무슨 뜻인지 모르겠어", held), which is the whole PRS
point-estimate change and 1 of ~20 scored probes. Not promoted.

So the null holds at a second level, and it is more informative than
P4's:

| | P4 | DPO-v3 (redo) |
|---|---|---|
| training responds? | no (loss pinned at ln 2) | yes (loss falls, margin separates) |
| eval quality moves vs SFT? | no | still no |

Why a real training signal did not become a measurable quality gain, in
order of how much each likely matters:

1. **The eval set cannot resolve a small gain.** 30 quality prompts, ~20
   scored attack items, PCS CIs ~±0.15. DESIGN.md §4.5 said this going
   in. This is the binding constraint.
2. **The learned gain is small.** 889 pairs, 1 epoch: held-out reward
   margin 0.017 -> 0.026.
3. **The preference did not generalize.** Held-out binary preference
   accuracy ended at ~0.69 (peaked 0.73 at step 25, drifted down while
   train accuracy climbed -- mild overfitting). The model shifted
   probability mass without learning to flip the decision on new pairs.
4. **SFT is near the ceiling this eval set can see** (PCS 0.80, KB 0.93).

More data (3k-10k pairs) could enlarge point 2, but point 1 -- eval-set
resolution -- would still bind, and expanding the eval set is separate,
out-of-scope work. More epochs would overfit. Per CLAUDE.md: report the
null with its cause, do not keep spending until DPO wins.

---

## 7. Recommended framing for the README / blog

> DPO produced a clean, corroborated null: no CI-confirmed change versus
> SFT on any quality metric. In the first run the cause was visible in
> the training log -- per-step loss never left ln(2), so the model never
> fit the preference pairs even on the training set, because the chosen
> and rejected replies were both sampled from the same already-narrow SFT
> distribution and the AI judge labeling them agreed with humans only
> ~70% of the time. A redo fixed both: sampling four replies per prompt
> and taking the judge's best-vs-worst gives genuinely contrastive
> on-policy pairs, and a stricter judge with a register gate and a
> confidence filter. This time the training responded -- the loss fell,
> the reward margin separated. And the eval-set quality still did not
> move versus SFT. That is the more informative result: it is not that
> DPO could not learn here, it is that a real learned preference signal,
> on this persona and at this eval-set size, does not translate into a
> measurable quality gain. This is not circularity (the judge metrics
> did not inflate) and not a KL-strength problem (the loss moved). The
> post-training gain in this project lives entirely at the B -> SFT step.
