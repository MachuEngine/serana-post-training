# §8 extensions -- prep log

Prep for three of the DESIGN.md §8 optional extensions, done with **zero
GPU spend** so the actual runs are a supervised kickoff, not a blind
overnight cron. Same style as `p1_progress.md`..`p6_progress.md`.

Status as of this entry: **plumbing built and locally smoke-tested. No
GPU run started.** The kickoff needs an L4 VM (see the stockout risk
below) and, per CLAUDE.md, an explicit predicted-vs-measured pass — which
is what the tables here set up.

---

## 1. DPO β sweep (DESIGN.md §8) -- ready to run

**What it answers:** does the β=0.1 choice for `serana-dpo` matter? Run
β ∈ {0.05, 0.3} against the *same* SFT adapter and eval set, compare.
The shipped β=0.1 already showed no CI-confirmed quality gain over SFT
(P5); if 0.05 and 0.3 also don't, that closes off "you just picked a bad
β" as an objection and **strengthens the headline finding** rather than
just re-confirming it.

### Files added (all committed, no GPU touched)

| Path | Role |
|---|---|
| `config/train_runs/dpo_beta005.yaml`, `dpo_beta030.yaml` | training configs; inherit `dpo.yaml`, override only `train.dpo.beta` + `train.output_adapter` |
| `config/experiments/dpo_beta005.yaml`, `dpo_beta030.yaml` | serving/eval view; `serve_up.py` auto-discovers them, eval scripts run unchanged (CLAUDE.md single-pipeline rule) |
| `scripts/run_beta_sweep.py` | VM-side orchestration: train both β → serve → generate raw eval → write `beta_sweep_report.json`. Preemption-safe (done-β skip + checkpoint resume). `--dry-run` verified locally. |

Kept **out** of `make_results_tables.py`'s `CONFIGS` list on purpose:
§8 sweeps get their own table, not a row in §6.1 (DESIGN.md §3.6 rule,
applied to §8 too).

### Pre-run prediction (CLAUDE.md working principle #5)

Baseline: the real `serana-dpo` run (β=0.1) measured **~26-27s/step,
100 steps, ~46 min wall (2 segments / 1 preemption), ~15GB peak VRAM,
~$0.25** (`p4_progress.md`). β does not change tensor shapes, step count,
or memory — only the loss value — so per-β training cost ≈ the β=0.1
baseline.

| | predicted | why |
|---|---|---|
| peak VRAM per run | **13-16 GB** (expect ~15) | identical to β=0.1 — same batch/seq/rank/ref-toggle |
| step time | **~26-30 s/step** | same as measured β=0.1; small variance only |
| wall clock, 2 training runs | **~1.5-2.0 h** | 2 × ~46 min + model-load/boot overhead |
| eval generate, 2 configs | **~0.5-1.0 h** | 54 prompts each through vLLM; ~2-12 s/reply (P5) + FlashInfer cold start |
| **total GPU wall** | **~2.5-3.5 h** | DESIGN.md §8 budgeted ~8 GPU-h (conservative) |
| **cost** | **~$1.0-1.5** | spot g2-standard-8 ~$0.33/h; well inside §9.2's P4 line (~$4, already had "one β retry" headroom — this is two) |

Judge-API scoring cost is separate and small (~140 calls, same as one P5
config), runs locally.

### Run procedure (supervised kickoff, then walk away)

Because the VM is a **manual scp'd tarball, not git** (`p4_progress.md`)
and its service account is `devstorage.read_only` (no GCS write, no
compute API), the flow is:

1. Provision / start a Spot g2-standard-8 + 1×L4 in `asia-northeast3-a`
   or `-b` (whichever has capacity). Reuse the P4/P5 disk if it still
   exists — `serana-sft` must be on it.
2. `scp` the repo tarball over (same as P4), incl. the new `config/` and
   `scripts/run_beta_sweep.py`. `md5`-check `src/finetune/train.py`
   matches local.
3. `nohup python3 scripts/run_beta_sweep.py --stop-when-done > ~/beta_sweep.log 2>&1 &`
   - `--stop-when-done` runs `sudo shutdown -h +1` at the end (works
     without compute-API scope; VM → TERMINATED, compute billing stops).
   - Watch the first ~5 min: model loads, β=0.05 training starts, step
     count shows `0/100`. Then disconnect.
4. Morning: if the VM auto-stopped, start it (disk persists), or if it's
   already down, `gcloud compute scp` these off it:
   - `artifacts/lora/serana-dpo-beta{005,030}/`
   - `artifacts/runs/raw_eval_dpo_beta{005,030}.json`
   - `artifacts/runs/beta_sweep_report.json`
5. Locally (API, no GPU):
   `uv run python3 scripts/score_eval_replies.py --config dpo_beta005 dpo_beta030`
6. Compare `eval_dpo_beta005.json` / `eval_dpo_beta030.json` against the
   existing `eval_dpo.json` (β=0.1) and `eval_sft.json`. Write up as a
   small standalone table in the blog/appendix — **not** in the §6.1
   tables.
7. Delete the sweep adapters from the VM disk if not keeping them;
   `--stop-when-done` already handled the VM.

### Risk: L4 stockout

P4 hit a **region-wide L4 Spot stockout** in `asia-northeast3` (both
zones a and b) that blocked progress for multiple sessions
(`p4_progress.md`). This is the real reason the sweep is **not** a blind
overnight job: if the VM never provisions, the night is wasted (though
$0 is spent — safe). Kick it off when a VM actually comes up and you can
watch the first few minutes.

---

## 2. Brand-tone demo (DESIGN.md §8) -- ready, ~zero GPU

**What it shows:** the same base model, no adapter, no weight change —
four brand-voice system prompts produce four recognizably different,
in-session-consistent tones. The foil: that's what a *prompt* buys;
multi-turn persistence and an adversarially-robust knowledge boundary
are what the *training* bought (the PRS/PCS tables).

### Files added

| Path | Role |
|---|---|
| `demo/brand_prompts.yaml` | 4 brand voices (support bot / radio DJ / API docs / Gen-Z social) + 5 domain-neutral test prompts |
| `demo/brand_tone.py` | loads base model (mirrors `smoke_test.py`), runs voice × prompt grid, `--markdown` for the appendix, optional `--adapter` for the "brand prompt vs persona adapter" curiosity |

### Local smoke test -- done

Ran `demo/brand_tone.py --base-id Qwen/Qwen3-0.6B --max-new-tokens 60`
on the M5. Plumbing works end to end; voices are clearly differentiated
even at 0.6B (support bot = terse 존댓말, API docs = dry, Gen-Z = emoji +
short). Output quality is 0.6B-mediocre (some repetition, hallucinated
dates) — expected; the real run is on the 8B.

### To produce the real artifact

Needs an 8B forward-pass box for ~2 min of generation (20 completions).
Cheapest: piggyback on the β-sweep VM while it's already up —
`uv run python3 demo/brand_tone.py --markdown > artifacts/diagnostics/brand_tone.md`
before `--stop-when-done` fires. Or run it in a short on-demand session.
No training, no eval-set, no judge.

---

## 3. Multi-turn length check (DESIGN.md §8) -- DRAFT

**What it answers:** does the persona drift as the conversation gets
longer? Measure PCS / style-sim / register-breaks at turn 1 vs 3 vs 10.

### Files added

| Path | Role |
|---|---|
| `data/multiturn/multiturn_probes_v1.jsonl` | 3 in-boundary conversations × 10 turns (mother/Soul Cairn, trust arc, living as a vampire) |
| `scripts/generate_multiturn_replies.py` | VM-side: replay each conversation feeding history turn-to-turn, capture every turn's reply |
| `scripts/score_multiturn.py` | local/API: judge_pcs + rule checks + style-sim at each checkpoint depth → `results_multiturn.md` drift table |

### Why it's a DRAFT, not ready

- The probe set is at `data/multiturn/`, **not** the frozen
  `data/eval/`, and is **not** wired into `src/data/check_leakage.py`.
  Training is already frozen so there's no live leakage risk, but
  graduating this means: user-review the 3 conversations, move them under
  `data/eval/` with a version bump, add them to the leakage guard.
- The 30 turns are hand-written first drafts. Read them before spending
  GPU time — a bad probe turn (e.g. one that accidentally invites an
  out-of-boundary answer) would confound the drift signal.
- Depth checkpoints default to 1/3/10; n=3 per depth is small — the
  bootstrap CIs will be wide. Fine for a "does the line slope down"
  read, not for a precise number.

### Cost when it graduates

Eval-only. Generate = 3 conv × 10 turns × N configs on the VM (~a few
min per config). Score = ~90 judge calls locally. DESIGN.md §8 budgeted
"a few GPU-h" — dominated by VM boot, not compute. Run it on the same VM
as the β sweep.

---

## What to review in the morning

1. This doc's β-sweep prediction table — sanity-check the numbers before provisioning.
2. `demo/brand_prompts.yaml` — are these the 4 voices worth showing?
3. `data/multiturn/multiturn_probes_v1.jsonl` — read the 30 turns; flag any that could confound.
4. Decide: run the β sweep now (needs a VM + ~3 GPU-h + watching the first 5 min), or defer.
