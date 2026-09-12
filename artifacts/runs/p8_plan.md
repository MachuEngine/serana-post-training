# P8 — eval-set v2, 14B pipeline, DoRA — plan

**Status: scoped, not started (2026-09-08).** Optional post-P6 work.
DESIGN.md §8 territory, plus an eval-set expansion §8 does not list.
The single motivation: `artifacts/runs/paired_analysis.md` showed the
frozen 30-prompt eval set cannot resolve *any* stage delta — not even
B→SFT — so every "what does this stage buy" number is currently
"can't tell". P8 rebuilds the quality axis so the question has answers,
then spends the sharper eval on a bigger base model and a second
adapter method.

## Decisions locked (user, 2026-09-08)

1. **Parallel, not replace.** v1 eval set and the shipped v1 results
   tables stay exactly as they are. v2 is added as follow-up work: a new
   `data/eval/eval_set_v2/`, a new results table, and a README/DESIGN
   note framing it as "what v1 could not measure". Nothing shipped is
   overwritten.
2. **Step 3 is a hard gate.** If v2 still cannot resolve B→SFT (the
   biggest available delta), steps 4–5 are cancelled — a bigger model or
   a new adapter would only reproduce the same "can't tell". Stop and
   write that up.
3. **Base model swap 8B → 14B approved** (CLAUDE.md "don't substitute
   without asking" — asked and granted). `Qwen/Qwen3-14B`, one line in
   `config/base.yaml`.

## Judge model (step 1, already run)

`scripts/judge_candidate_check.py` scored the 50 human labels with three
cheaper models without touching the frozen v1 record. All three clear
the 0.6 Spearman floor; `gpt-4o-mini` at ρ=0.81 is at least as good as
the shipped `gpt-4o` (ρ=0.73 — the gap is inside the noise of n=50) and
~17× cheaper. Measured usage: ~996 in / ~42 out tokens per call.

**v2 uses `gpt-4o-mini` as the judge.** Defensible because v2 is a new
eval-set version scored end-to-end with one judge; v1's tables keep
`gpt-4o`. `config/eval.yaml` gets a `v2` block, the `v1` block is left
alone.

Full-eval cost per config (≈360 calls: 150 PCS + 150 boundary + 60 PRS):
**~$0.06** with `gpt-4o-mini`. Eight configs (8B ×4 incl. DPO-v3,
14B ×3, DoRA ×1): **~$0.50 total**. The API budget stopped being the
constraint.

## Sizing v2 — the power calculation that sets n

Projected from the *measured* v1 paired differences (mean and SD of the
per-prompt PCS diff, scaled by 1/sqrt(n)):

| comparison | v1 measured | CI at n=90 | CI at n=150 |
|---|---|---|---|
| B→SFT | +0.333, sd 1.295 → CI [−0.13, +0.80] | [+0.066, +0.601] ✅ | [+0.126, +0.541] ✅ |
| SFT→DPO (P4) | +0.033, sd 0.183 | [−0.004, +0.071] ✗ | [+0.004, +0.063] ✅ |
| SFT→DPO-v3 | +0.067, sd 0.365 | [−0.009, +0.142] ✗ | [+0.008, +0.125] ✅ |

**n = 150, not 90.** Two reasons:
1. At n=90 the B→SFT lower bound is +0.066 — a 20% drop in effect size
   on the new prompts flips the gate to fail. At n=150 it tolerates 38%.
2. n=90 resolves *neither* DPO comparison. Answering the DPO question
   was half the motivation; 150 is where it becomes answerable.

**The ceiling problem — v2 must discriminate, not just be bigger.**
PCS is saturated: SFT scores 5/5 on 20 of 30 items, DPO on 21, DPO-v3
on 22. And only **12 of 30** items score differently between B and SFT
at all — the other 18 contribute exactly zero to the paired test. More
prompts of the same kind mostly add 5-vs-5 ties.

So the 150 are authored against a discrimination target, not a topic
checklist:
- emotionally loaded questions that require the personality to answer
  (not biography recall, which the base model already does well)
- *subtle* knowledge-boundary cases, not obvious modern-tech ones
  (v1's "스마트폰 써봤어?" is answered correctly by everything)
- questions that invite assistant-register drift — P2's smoke test
  caught SFT dropping persona on a cooking question; that class of
  prompt is where the configs actually separate
- fewer softball biography questions (keep enough for coverage, ~20)

**Done-criterion for this:** after the v2 8B re-score in step 3, at
least ~35% of items must differ between B and SFT (v1: 40% but on only
12 items). If v2 lands under ~25%, the prompts were not discriminating
enough — record it and treat the gate result as inconclusive rather
than a clean pass/fail.

## Success criteria

- **v2 built:** 150 quality prompts (was 30) + 60 scored attack items (was 20),
  leakage check passes, user has read every prompt and confirmed its
  in/out-boundary label.
- **Step 3 gate:** the paired-bootstrap diff for B→SFT on v2 PCS
  (1–5 score) has a 95% CI that **excludes zero**. (v1: +0.33, CI
  [−0.10, +0.80] — did not.)
- **If the gate passes:** 14B pipeline trains within budget, predicted
  vs measured peak VRAM within 20%, v2 eval done for all 14B configs,
  8B-vs-14B added to the hardware table.
- **DoRA (if reached):** DoRA-vs-LoRA at matched rank — step time, peak
  VRAM, MFU, trainable params, val-loss curve — plus a v2 eval. Output
  adapter is a diagnostic, not shipped, unless it clearly wins on
  multiple signals.

---

## Step 1 — cheaper judge validation ✅ DONE

Ran `scripts/judge_candidate_check.py`. `gpt-4o-mini` selected. Cost
~$0.10. Non-destructive — new files only
(`scripts/judge_candidate_check.py`, `artifacts/runs/judge_candidates.json`).

Residual risk: 50 labels is a small sample, and all on v1-style prompts.
Mitigation: after v2 is built, hand-check ~20 v2 items against the
`gpt-4o-mini` score before trusting the v2 table (step 2 done-criterion).

## Step 2 — build eval-set v2  ◑ IN PROGRESS (drafted, awaiting user read-through)

Done so far:
- `src/data/build_eval_prompts.py` and `build_attack_probes.py` take
  `--version`; v1 output verified byte-identical.
- `data/eval/eval_set_v2/` written: 150 quality prompts (95 in / 55 out),
  90 probe rows (45 single-turn + 15 escalating sequences = 60 scored).
- Eval-set path parameterized: new `src/eval/eval_set.py`; `--eval-version`
  added to `generate_eval_replies.py`, `score_eval_replies.py`,
  `make_results_tables.py`, `paired_analysis.py`; `--version` to
  `check_leakage.py`. v1 results tables verified byte-identical.
- `config/eval.yaml`: `judge_v2` block (gpt-4o-mini). `PROMPTS.md §6`
  updated.
- **Leakage check passes** (`check_leakage --version v2`): 150 prompts +
  90 probes vs 202 CPT paragraphs / 2995 SFT pairs / 893 DPO prompts,
  zero overlap.

Still open: **user reads all 150 prompts + confirms in/out labels**, then
the 20-item gpt-4o-mini spot-check (needs step 3 replies).

Original plan for this step (unchanged):


**What:** `data/eval/eval_set_v2/` — 150 quality prompts and 60 *scored*
attack items (~90 rows on disk, since each escalating attack is a
3-turn sequence whose first two turns are setup), authored against the
discrimination target above.

**What v2 does *not* contain:** `style_reference.jsonl` and
`human_labels.jsonl` are **not copied**. Both stay v1-owned and are read
from v1 by path:
- the style reference is a fixed target (held-out real Serana lines);
  duplicating it would create two sources of truth for the same
  measurement.
- the 50 human labels anchor judge validation. Copying them into v2
  would imply they validate v2, which they do not (risk 4).

**How:**
1. New `src/data/build_eval_prompts.py` path (a `--version` flag, since
   `data/eval/` is a hook-protected path writable only via these
   scripts) holding 150 hand-authored prompts: ~95 in-boundary,
   ~55 out-of-boundary. Claude drafts them one at a time against the
   discrimination target; **the user reads all 150 and confirms each
   in/out label.**
2. Same for `src/data/build_attack_probes.py` — 60 probes across the
   existing taxonomy (direct / meta / role-exit / escalating-multi-turn).

   **Fix v1's category imbalance while doing it.** v1 has 24 probe rows
   but only 20 *scored items*: the escalating attacks are 3-turn
   conversations where the first two turns are setup and only the final
   reply is scored, so 6 rows collapse to 2 items. That is correct
   design, not a bug — but it means the **escalating category, which
   DESIGN.md §4.1 calls out as "where prompt-only defenses crack",
   contributes 2 of 20 scored items (10%)**, while three easier
   categories contribute 6 each. PRS is currently dominated by the
   attacks that are easiest to survive.

   v2 target: ~60 scored items with escalating at ~15 of them
   (~25%) — i.e. ~15 three-turn sequences (~45 rows) plus 15 each of
   direct / meta / role-exit. That also gives PRS enough per-category
   items for the `failure_type` breakdown to mean something.
3. `src/data/check_leakage.py` extended to v2 (CPT corpus, SFT set, DPO
   prompt pool). Must pass — it exits non-zero on any near-duplicate
   (SequenceMatcher ratio > 0.92) and gates the run.
4. **Parameterize the eval-set path.** 11 files hardcode
   `data/eval/eval_set_v1`; point them at `paths.eval_set` in
   `base.yaml` (the key exists and is currently unused). Mechanical
   (string → config lookup), no logic change, but it touches training,
   eval, and serving entry points.
5. `config/eval.yaml`: add a `v2` block (eval-set version +
   `gpt-4o-mini` judge), leaving the `v1` block untouched.

**Cost:** API $0 (prompts are authored, not generated). GPU $0. Real
cost is the user's read-through of 150 prompts + ~90 probe rows (~2–2.5 h).

**Done:**
- 150 prompts + ~90 probe rows written via the scripts · leakage check green · user sign-off
  on every prompt and label
- 20-item `gpt-4o-mini` spot-check agrees with a by-eye read
- **`paths.eval_set` still resolves to v1 by default, and
  `uv run scripts/make_results_tables.py` regenerates
  `results_quality.md` and `results_hardware.md` byte-identical to the
  committed versions.** This is the guard that the refactor did not
  break the shipped P5 "tables regenerate from one command" criterion.
- v1 directory byte-identical to before

**Risks:**
1. **Leakage.** SFT's ~3k examples are model-generated and broad, so a
   new prompt can collide by accident. The automated check gates this
   and blocks the run on a hit — manageable, but expect to rewrite a
   few prompts.
2. **Prompt quality at 5× scale.** v1's 30 were user-reviewed; v2's 150
   must be too, or bad prompts add noise instead of resolution. The
   read-through is not optional — and it is the single largest time cost
   in P8.
3. **Discrimination, not just count.** The bigger risk than quality is
   blandness: 150 easy prompts that everything answers at 5/5 buy no
   resolution at all (see the ceiling numbers above). If drafting starts
   drifting toward biography recall, stop and re-target.
4. **11-file path change.** Mechanical but touches training, eval, and
   serving entry points. The byte-identical table regeneration in the
   done-criteria is the proof it didn't break anything.
5. **Human labels stay v1.** Judge validation remains anchored to the
   50 v1 labels. v2 gets a 20-item eyeball, not a fresh 50-label set
   (that is hours of user labeling). Documented as a v2 limitation —
   the v2 judge is validated on v1-style items, not v2's harder ones.

## Step 3 — re-score the 8B configs on v2  ⚠️ GATE

**What:** regenerate B / SFT / DPO / DPO-v3 replies on the 150 v2
prompts + 60 probes, score with `gpt-4o-mini`, run
`scripts/paired_analysis.py`.

**How:** spin up the serving VM (on-demand — this is a measurement),
`scripts/generate_eval_replies.py` per config against the one vLLM
server, `scripts/score_eval_replies.py`, then the paired diff.

**Cost:** GPU ~$2 (~2 h on-demand g2-standard-8; ~840 replies at ~3 s
plus server start), API ~$0.25 (4 configs × 360 calls × $0.000175).

**Done / GATE — in priority order:**
1. **Primary gate:** paired B→SFT diff on v2 PCS excludes zero →
   **proceed to step 4.**
2. **Diagnostic:** fraction of items where B and SFT differ at all.
   v1 was 12/30 (40%). If v2 lands **under ~25%**, the prompts were not
   discriminating enough — the gate result is inconclusive regardless of
   which way it went, and the honest move is to re-target the prompts
   rather than proceed on a weak signal.
3. **Bonus, not a gate:** at n=150 both DPO comparisons are projected to
   resolve. If SFT→DPO-v3 now excludes zero, the two-level null in
   `p4_postmortem.md` §6 gets a third level and needs rewriting — that
   is a result worth having on its own, independent of step 4.

**If the primary gate fails:** stop. Write
`artifacts/runs/p8_gate_failed.md`: "5×-ing the eval set did not buy
resolution; the persona-quality axis is saturated at this metric
design, not this sample size" — itself a real finding, and much cheaper
than learning it after a 14B run.

## Step 4 — 14B full pipeline (only if step 3 passes)

**What:** `Qwen/Qwen3-14B`, CPT → SFT → DPO, same hyperparameters,
evaluated on v2.

**How:** `config/base.yaml` `model.base_id` → `Qwen/Qwen3-14B` (one
line — the codebase is model-agnostic by design). New
`config/train_runs/*_14b.yaml` + `config/experiments/*_14b.yaml`
mirroring the 8B set with 14B adapter output paths. Preference pairs
regenerated from the 14B SFT adapter (`scripts/generate_reply_groups.py`
+ `src/data/build_preferences.py`). Spot VM with GCS checkpoint/resume
for training; on-demand for the eval generation.

**Prediction (derived from 8B measured × param ratio 14.8B/8.2B ≈ 1.8):**

| | 8B measured | 14B predicted |
|---|---|---|
| peak VRAM, QLoRA r=16, seq 1024, batch 1 | 9.64 GB | **~15–16 GB** (4-bit base ~8.4 GB + ~1.4× activations) — fits 24 GB with real margin pressure |
| SFT step time | ~27 s | **~48–52 s** (~1.9×; dequant-bound, MFU 13.5%) |
| SFT wall clock (~562 steps) | ~4 h | **~7.5–8 h** |
| CPT | 36 s (tiny corpus) | ~1 min |
| preference generation (best-of-4, ~890) | 65 min | **~2 h** |
| DPO (~106 steps) | 56 min | **~1.8 h** |
| v2 eval generation (3 configs) | — | ~40 min |

| resource | estimate |
|---|---|
| GPU wall clock | ~14–16 h compute + ~4 h overhead (setup, 14B ~30 GB download, preemption recovery, smoke tests) |
| GPU cost | training/gen on spot (~$0.35/h) ~$6 · serving/eval on-demand (~$0.90/h) ~$4 · storage/egress ~$2 = **~$12–15** |
| API | v2 eval, 3 configs × ~$0.05 = **~$0.15** |

**Done:** all three 14B adapters trained · predicted vs measured peak
VRAM logged, gap > 20% explained · v2 results table has the 14B rows ·
hardware table has 8B-vs-14B on the same L4 · one preemption/resume
survived · in-persona Korean smoke test passes.

**Reported, not shipped.** The "parallel, not replace" decision covers
the tables; it covers the artifacts too. The HF Hub adapters, the demo,
and the README's headline stack stay **8B**. 14B results are an added
section. Re-shipping on 14B is a separate decision, made only if 14B
wins clearly *and* the user wants to redo the release — do not assume
it.

**Risks:**
1. **Eval-step OOM.** P4 hit 22 GB when the in-training eval batch was
   too big on 8B. 14B is tighter — pin `per_device_eval_batch_size=1`
   and keep `eval_steps` eval on a small slice.
2. **Preference pairs must be regenerated** from the 14B SFT adapter —
   DPO is on-policy. Already in the step (the +2 h line above).
3. **8B hyperparameters on 14B — the main confound.** lr and rank were
   chosen on 8B (§3.6c/d). Two different exposures:
   - *lr*: low risk. The §3.6c curve (2e-5 underfits, 2e-3 spikes, 2e-4
     fits) is unlikely to move for a same-family model. **Decision:
     reuse, state it, do not re-probe.**
   - *rank*: real risk. r=16 on 14B is *less* relative adapter capacity
     than r=16 on 8B (wider hidden dim, more layers). If 14B fails to
     beat 8B, "the adapter was undersized" is an unfalsifiable
     alternative explanation and the whole comparison weakens.
     **Decision: keep r=16 as the primary run for a clean
     one-variable comparison, and if 14B does *not* beat 8B, spend
     ~$2 on a single r=32 SFT run before concluding anything.** Budget
     it now rather than discovering it at write-up time.
4. **VM disk.** 14B weights ~30 GB. Check the boot disk has room before
   the run; resize if needed.
5. **14B spot capacity** in `asia-northeast3`. 1× L4 already stocked out
   once in P4. A stockout is a logged data point, not a failure.

## Step 5 — DoRA (optional, only after step 4)

**What:** redo the **8B** SFT stage with DoRA instead of LoRA, compare.
8B, not 14B — the LoRA-8B baseline already exists, so this is +1 run,
not +2.

**How:** `build_lora_config()` in `src/finetune/train.py` gets
`use_dora=train_cfg.get("use_dora", False)` (~2 lines, PEFT 0.20.0
supports it). New `config/diagnostics/dora_probe.yaml`. Output →
`artifacts/diagnostics/`.

**Cost:** GPU ~$4–6 (one full 8B SFT run ~4 h + ~1.3× DoRA overhead +
profiling + QDoRA debug buffer), API ~$0.05 (v2 eval).

**Risks:**
1. **QDoRA (DoRA on a 4-bit base)** has had rough edges. **2-step smoke
   run first.** If it needs more than ~10 lines to work, stop and record
   "QDoRA needs non-trivial changes, out of scope" (same rule as the
   DDP plan).
2. **Quality delta likely still a null** even on v2 — DoRA vs LoRA is a
   smaller change than B→SFT. The reliable deliverable is the training
   ablation (step time / VRAM / MFU / loss shape), one row in DESIGN
   §7. Frame it that way from the start.

---

## Total

| | GPU (GCP credit) | API (OpenAI credit) | user time |
|---|---|---|---|
| step 1 (done) | $0 | $0.10 | — |
| step 2 (build v2) | $0 | $0 | **2–2.5 h read-through** |
| step 3 (gate) | ~$2 | ~$0.25 | — |
| step 4 (14B) | ~$12–15 | ~$0.20 | ~3 days driving the VM |
| step 4 contingency (r=32 rerun, only if 14B loses) | ~$2 | ~$0.05 | — |
| step 5 (DoRA) | ~$4–6 | ~$0.05 | — |
| **total** | **~$20–27** | **~$0.65** | |

GCP: ~9% of the $300 credit. OpenAI: ~6% of the $11.45. Neither budget
is the constraint. **The binding costs are the user's 2-hour prompt
review (step 2) and ~20 GPU-hours over spot preemptions ≈ 3–4 calendar
days of driving the VM (step 4).**

## Stop conditions (CLAUDE.md: cut, don't raise the cap)

- **Step 3 gate fails** → stop, write `p8_gate_failed.md`, do not run 4–5.
- **Step 3 discrimination rate < ~25%** → the prompts, not the sample
  size, were the problem. Stop and re-target rather than proceeding on
  an inconclusive gate.
- Step 4 14B SFT loss diverges or won't fit after the risk-1/risk-3
  fixes → stop, record it (a config *and* a prediction failure — log
  both, per CLAUDE.md).
- Step 5 QDoRA smoke run needs > ~10 lines → stop, document.
- Cost reaches **$35** or **30 GPU-hours** → stop.

## Not doing

Replacing the v1 tables or re-shipping the HF Hub adapters/demo on 14B.
A fresh 50-label human set for v2. A 14B lr probe. A rank *sweep* (the
r=32 contingency is one conditional run, not a grid). DoRA on 14B. A
shipped DoRA adapter (unless it clearly wins on multiple signals).
Anything on the CLAUDE.md scope-cut list (PPO, RAG, contrast personas,
training-data-scale sweep).
