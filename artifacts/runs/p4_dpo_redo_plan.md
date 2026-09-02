# P4 DPO redo — plan

**Status: executed (2026-09-01 to 09-02).** This document is the plan as
written before the run. What actually happened, step by step, is in
`p4redo_progress.md`; the outcome (a "two-level null" — training
responded, eval quality did not move) is in `p4_postmortem.md` §6. The
plan held up well: measured cost ~$2 GPU / ~$6 API vs the ~$1.5 / ~$5–6
predicted; the only surprises were a 2× slow generation-throughput miss
and flash-attn missing from the serving VM.

**Why:** P4's DPO run was a corroborated null (`artifacts/runs/p4_postmortem.md`).
Two causes were in reach without rebuilding the whole data pipeline:

1. **The two candidate replies were nearly identical.** P3 sampled both
   from the SFT model at temperature 0.9, and that model's output
   distribution is narrow, so the "chosen" and "rejected" reply differed
   only cosmetically. DPO had almost no gradient to work with — per-step
   loss never left ln(2).
2. **The preference judge agreed with humans only ~73%.** A 2-way choice
   labeled right ~73% of the time means a large share of the training
   labels were noise, and the old prompt gave no way to tell a confident
   label from a coin flip.

**Not attempted here:** scaling to several thousand pairs (a new prompt
pool + leakage re-check + ~5x the judge cost — out of scope per CLAUDE.md,
which cuts a data-scale sweep). A β grid sweep. PPO.

---

## What changed since the first draft

The first draft paired one SFT reply with one reply from the earlier
CPT-only adapter. The user and the code review both flagged that as
**off-policy on the rejected side** — the CPT model is not the model being
trained, so DPO would be pushing the policy away from text it already
rarely produces, which changes the claim from "what does the DPO stage
buy" to "sharpen the CPT→SFT gap".

**Revised approach: sample N replies from the SFT model itself and take
the best-vs-worst pair.** Both replies still come from the model being
trained (on-policy, the textbook setup). The contrast comes from
*selection over N samples* rather than hoping two random draws differ.
N=4 at a higher temperature.

Best-and-worst-of-N (not a full ranking) because LLM judges get less
reliable as the option count grows — position bias compounds, rankings
turn intransitive. Asking only for the two extremes, where the judgement
is clearest, plus a confidence filter, keeps that in check at N=4.

---

## Success criteria (define "done" before starting)

Either outcome is a valid, reportable result:

1. **A real gain:** the retrained DPO adapter beats SFT on at least one
   metric that does **not** come from an LLM judge (the PRS regex check,
   style similarity, or knowledge-boundary accuracy), with 95% CIs that
   do not overlap (`DESIGN.md` §4.5). A non-judge signal is required so
   the result clears the circularity guard.
2. **A stronger null:** same flat result as P4, now with on-policy
   best-of-N pairs and a better judge — which confirms the postmortem's
   conclusion instead of leaving it open. Stop for good.

Training-health checks (must pass to trust either outcome):

- DPO training loss drops **below ln(2) ≈ 0.693** and stays there
  (P4's never did — that was the tell).
- The reward-margin curve trends positive across the run.
- Held-out preference accuracy exceeds 0.65 (P4 ended at 0.595, chance).

---

## The steps

### 1. Generate the reply groups

`scripts/generate_reply_groups.py` — sample 4 replies per prompt from
`serana-sft` at temperature 1.0, for the existing 893-prompt DPO pool
(`data/ko/dpo_prompt_pool.jsonl`, already leakage-checked in P1). Writes
`data/ko/raw/reply_groups_v3.jsonl`.

- Peak VRAM: `generate()` expands to `batch_size * n` concurrent decode
  streams. Default `--batch-size 4` with `--n 4` = **16 streams**, the
  same count as P3's 2-reply / batch-8 job that measured **18.7 GB** — so
  the prediction carries over. Raising either number needs its own VRAM
  check first (CLAUDE.md rule 5).
- Predicted wall clock: P3 did 893×2 replies (16 streams) in ~33 min.
  893×4 replies at the same 16-stream batching is ~2x the decode work at
  roughly half the prompts-per-step ≈ **~70–80 min ≈ 1.2–1.3 GPU-hr**.
- Provisioning: **on-demand** (short job, no need to risk a preemption redo).
- Run it first with `--limit 8` to confirm the timing and the peak VRAM
  before the full pass.

### 2. Judge each group and build the preference set

`python3 -m src.data.build_preferences --in data/ko/raw/reply_groups_v3.jsonl
--out data/ko/prefs_v3.jsonl --report artifacts/runs/p4redo_preference_report.json`

The v3 judge (`PROMPTS.md` §4) returns, per group: the best reply, the
worst reply, which replies break 반말, and a confidence. A group is
dropped when the judge errors or returns a wrong-shape response, best ==
worst, the best reply breaks 반말, best and worst are near-identical, or
confidence is not high/medium.

- API only, no GPU. Predicted **~$4–5** (893 calls, ~1.2k input + ~120
  output tokens each, GPT-4o rates).
- Writes `data/ko/prefs_v3.audit.json` — 30 judged groups (all 4 replies,
  the best/worst picks, the register flags, the reason) for a by-eye
  check.

### 3. Sanity-check the judge by eye

Read `data/ko/prefs_v3.audit.json`. For each of the 30 groups, decide
whether you agree with the judge's best pick and worst pick. Tally the
disagreements.

This is a lighter check than P3's, by choice: P3 ran a formal
human-labeled audit with an automated 70% agreement floor
(`validate_audit.py`). That needed fresh human labels on the new
best-of-N pairs, which we are not producing. The trade is deliberate,
and the manual tally below stands in for the floor.

**Decision points — stop and report if any of these:**
- The final pair count in the report is under ~300 (not enough to train).
- You disagree with the judge on more than ~9 of the 30 groups (mirrors
  P3's 70% floor).
- The confidence buckets in the report are mostly `low` (the N replies
  were not far enough apart).

Any of these means "the SFT model's outputs are too uniform for
preference optimization to have material to work with" — update
`p4_postmortem.md` and stop.

### 4. Retrain DPO

New config `config/train_runs/dpo_v3.yaml` — a copy of `dpo.yaml` with
`preference_file: data/ko/prefs_v3.jsonl` and `output_adapter:
artifacts/lora/serana-dpo-v3`. Hyperparameters unchanged from P4 (beta
0.1, lr 5e-6, 1 epoch, grad_accum 8, seq_len 1024, adapter-toggle
reference). The original `dpo.yaml` and `serana-dpo` adapter are left
untouched until this run either wins (then promote) or produces a null
(then keep the original and document).

- Pair count after the filters: uncertain, likely **400–600**.
- Predicted: ~50–75 steps at the measured **27 s/step** ≈ **25–35 min**.
  Peak VRAM ~15 GB. Spot, ~$0.20. Preemption risk low on a sub-hour job;
  checkpoint/resume is already wired.

**Decision point:** if the training loss is still sitting at ln(2) after
30 steps, **stop the run.** Same null as P4, now confirmed on on-policy
best-of-N pairs. Update `p4_postmortem.md` and stop.

Conditional: one extra run at beta 0.05, **only if** the reward-margin
curve rose during the beta 0.1 run (signal is there but the KL penalty
damped it). One run, not a sweep.

### 5. Evaluate

Run the P5 evaluation path (`scripts/generate_eval_replies.py` +
`score_eval_replies.py`) on `serana-dpo-v3`. ~30 min GPU + ~$0.5 API (80
judge calls). Regenerate `results_quality.md` with the v3 adapter as a
fourth row.

### 6. Write up

Update `p4_postmortem.md` with the outcome. If the headline changed,
update the README and `DESIGN.md` §3.4.

---

## Budget (predicted, from measured P2–P5 numbers)

| step | GPU-hr | GPU $ | API $ | wall |
|---|---:|---:|---:|---|
| 1. generate 893×4 replies (on-demand) | ~1.2–1.3 | ~$0.90 | – | ~2 h incl. provisioning |
| 2. judge 893 groups | 0 | 0 | ~$4–5 | ~1 h + API turnaround |
| 3. by-eye check | 0 | 0 | 0 | ~30 min (human) |
| 4. retrain DPO (+ optional beta 0.05) | ~0.5–1.0 | ~$0.20–0.40 | – | ~1 h |
| 5. evaluate | ~0.5 | ~$0.35 | ~$0.5 | ~1 h |
| **total** | **~2–2.5** | **~$1.5** | **~$5–6** | **~1 day** with buffers |

- GCP credit: not a constraint (GPU spend to date is ~$10–20 of $300).
- OpenAI: $15.25 available, this needs ~$6. Check the balance again right
  before step 2 (P3 hit zero mid-run once).

---

## Stop conditions (CLAUDE.md: cut, don't raise the cap)

- Final pair count under ~300, or the by-eye check shows a bad judge → stop.
- Training loss still at ln(2) after 30 steps → stop.
- GPU spend reaches **$3** or GPU time reaches **3 GPU-hr** → stop.
- Wall clock reaches **2 days** → stop.
- Any OOM is a config and a prediction failure — log both, do not reach
  for a bigger GPU.

---

## Risks

| risk | likelihood | mitigation |
|---|---|---|
| After filtering, under 300 pairs left | medium | step 3 decision point stops cleanly; the low count is itself the finding |
| N=4 still too uniform, loss stays at ln(2) | medium | step 4 decision point stops at ~30 steps, ~$0.10 spent |
| Judge less reliable on 4 options than 2 | low–medium | only best/worst asked (not a ranking); shuffled order; confidence filter; by-eye check in step 3 |
| Spot preemption on step 4 | medium | job is under an hour; on-demand fallback ~$0.40 |
| OpenAI balance runs out mid-judge | low | check balance right before step 2; it is one ~$5 batch |
| A gain that shows up only in the LLM judge (circularity) | low | success criterion 1 requires a non-judge signal |
