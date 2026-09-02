# DESIGN.md

Architecture rationale, the evaluation design, the GPU engineering practice, and the compute/budget plan. Keep in sync with code. Behavioral rules live in `CLAUDE.md`; prompts live in `PROMPTS.md`.

---

## 1. Inference pipeline

One `run(config)` path. A user turn flows through:

1. **Prompt assembly** — persona system prompt + dialogue history. Template in PROMPTS.md §1.
2. **Generation** — the base model, LoRA-adapted when `cfg.model_weights == lora`.
3. **Serving** — vLLM streams tokens via FastAPI; token counts, latency, and KV-cache stats logged per request.

`model_weights` (plus `lora_adapter_id`) is the only inference-time difference between configs. CPT, SFT, and DPO are training-time differences producing different adapters; all are served by identical code. `base` is a real path, not a stub.

No retrieval step — deliberate, so measured deltas are attributable to training rather than to an inference-time knowledge channel.

---

## 2. Module boundaries

- `src/data/` — ingestion, translation, SFT set generation, preference-pair generation, train/val splitting.
- `src/finetune/` — CPT, SFT, and DPO training. Outputs adapter weights.
- `src/serve/` — vLLM + FastAPI; owns the inference path.
- `src/eval/` — quality metrics + judges.
- `src/gpu/` — reserved in the original layout for VRAM estimation, profiling helpers, `nvidia-smi` logging, MFU calculation. In practice all of that instrumentation ended up as standalone `scripts/` (`gpu_probe.py`, `profile_train.py`, `kv_cache_estimate.py`) rather than an importable module -- each script is a one-shot measurement run, not logic reused across training/serving, so a shared module never earned its keep. `src/gpu/__init__.py` is an empty stub. Noted here rather than silently left to drift, same as `src/finetune/`'s consolidation below.

One genuine coupling: preference-pair generation (`src/data/`) samples from the SFT adapter, so it depends on a generation call. Reuse `src/serve/`'s generation function rather than writing a second inference path — a divergence would mean the preference pairs came from a different model than the one evaluated.

---

## 3. Training design

Three GPU training stages, in order. Each starts where the previous ended.

```
base model
  └─(P2a) CPT  — raw-text CLM on her translated lines      ┐
  └─(P2b) SFT  — instruction pairs + hard cases            ┴→ adapter "serana-sft"
       └─(P4)  DPO — RLAIF preference pairs                 → adapter "serana-dpo"
```

**Language: translate-first, then unify in Korean.** Source lines are English, output is Korean. Do **not** train in English and translate at output — that yields a "translated-Serana" whose voice is washed out, and a training-vs-output language mismatch weakens style learning. Convert once at the front of the pipeline, preserving her dry, wary voice. Translation is light: batch it, eyeball a few dozen samples, don't chase perfection. All configs share the translated data, so roughness is a common baseline. Provider fixed in P1 (§9.4).

### 3.1 Data sources and what each stage gets

Serana's wiki dialogue is largely a **list of her utterances**, not question–answer pairs: many lines are ambient comments, quest reactions, or one-sided remarks with no recorded player prompt. Some pages do preserve the player's dialogue option alongside her reply. How much survives as usable pairs is **not known until P1 ingests it** — so counting that is a P1 deliverable, not an assumption.

| Source | Native shape | Goes to |
|--------|--------------|---------|
| Her lines with no recorded player prompt | standalone utterances | **CPT corpus** (raw text) |
| Her lines *with* a recorded player prompt | genuine pairs | **SFT set** — real data, not synthetic |
| Wiki narration / lore prose (within her horizon) | raw text | **CPT corpus** |
| Synthetic exchanges (PROMPTS.md §3) | pairs | **SFT set** |
| Held-out slice of her lines | — | **style reference**, excluded from all training |

**Record the real-pair vs synthetic ratio in P1 and put it in the README.** It matters beyond bookkeeping: the more of the SFT set is LLM-generated, the more of the pipeline is LLM-authored end to end (training data → preference labels → evaluation), which sharpens the circularity concern in §4.4. A number here is more honest than a hand-wave.

**What is excluded from training entirely:**

| Category | In training data? | Rationale |
|----------|-------------------|-----------|
| Core self-knowledge: vampirism, Harkon/Valerica, Volkihar vs Dawnguard, Soul Cairn, the Elder Scroll | **Yes — a curated handful of topics** | Enough to hold a conversation about herself |
| Boundary behavior: deflecting what she can't know | **Yes — ~25% of the SFT set** | Measured by knowledge-boundary accuracy and PRS |
| Broad world lore | **No** | Knowledge breadth is not a deliverable |
| Anything after 4E 201, or modern-world/meta topics | **No — hard filter** | Letting it in silently destroys the knowledge-boundary metric |

The horizon filters **what enters the training set**, applied at ingest before translation. The README states this scope plainly rather than implying encyclopedic coverage.

### 3.2 CPT stage (continued pretraining)

Raw-text causal language modeling over her translated utterances plus related narration — no instruction format, no user/assistant roles, just next-token prediction on in-voice Korean prose.

**Why this is a distinct stage.** CPT adapts the model's *distribution over language* — rhythm, vocabulary, register. SFT adapts its *behavior given an instruction*. CPT first is the standard order because behavior training on top of an already-adapted distribution converges faster and drifts less. Here it is also the only way to use her un-paired lines without inventing player prompts for them, which would inject synthetic noise into the one genuinely authentic part of the data.

- Same QLoRA adapter, continued into SFT. Packing to `max_seq_len` for throughput (raw text packs cleanly; instruction pairs don't).
- Short — the corpus is small.
- **Watch for catastrophic forgetting.** A few hundred to a thousand lines is small for CPT, and over-training on it can blur general Korean fluency. After CPT, the smoke test must still produce coherent Korean on an unrelated prompt. Val loss (§3.6a) is the quantitative signal; the smoke test is the qualitative one.
- **If CPT measurably hurts the SFT result, drop it and report that.** A negative with a clean loss curve is a finding.

### 3.3 SFT stage

Korean in-character exchanges: real pairs where the wiki preserved them, plus synthetic continuations (PROMPTS.md §3) to reach ~3k, including core self-knowledge and identity-probe hard cases. Composition targets in PROMPTS.md §3. Deduplicate before training — near-duplicates inflate effective epochs invisibly.

### 3.4 DPO stage (with RLAIF preference data)

**Why DPO and not PPO.** Classic RLHF needs a trained reward model plus a PPO loop holding policy, reference, reward model, and value head in memory simultaneously. Run the §7.1 arithmetic on that configuration and it does not fit an 8B model on 24GB — that calculation belongs in the README, because "PPO needed N GB and I had 24" is stronger than "I chose DPO." DPO optimizes directly against preference pairs with a closed-form objective: no reward model, one training loop.

**Why the reference model is free here.** DPO's loss needs log-probs from a frozen reference policy. Loading a second 8B model would blow the budget. With PEFT, the reference is obtained by **disabling the adapter** on the same base weights — TRL's `DPOTrainer` supports `ref_model=None` when the policy is a PEFT model. So DPO costs roughly one model in memory, not two. **Verify, don't assert:** record peak VRAM for DPO and confirm it is comparable to SFT.

**Preference data (RLAIF).** The shipped `serana-dpo` adapter was built in P3 from an earlier form of this step (two replies per prompt from the SFT adapter at temperature 0.9, a pairwise judge, ties discarded), which produced the P4 null — see "What happened" below. The current pipeline, for the rerun planned in `artifacts/runs/p4_dpo_redo_plan.md`:

1. Draw ~900 prompts from the same distribution as the SFT set — but **not** eval prompts or attack probes (`data/ko/dpo_prompt_pool.jsonl`, leakage-checked in P1).
2. Sample **N replies** per prompt (default 4) from the SFT adapter at `temperature ≈ 1.0` (`scripts/generate_reply_groups.py`). Every reply comes from the model being trained, so both extremes stay on-policy. (Sampling here is intentional and does not conflict with the greedy-decoding rule, which governs *evaluation* only — §4.5.)
3. The **preference judge** (PROMPTS.md §4) picks the single best and single worst of the N, lists which replies break 반말, and gives a `confidence`.
4. Emit the best/worst pair as `(prompt, chosen, rejected)`. Drop a group when: the judge errored, best == worst, the best reply breaks 반말, best and worst are near-identical, or confidence is not high/medium.
5. `build_preferences.py` logs a funnel (every group in exactly one bucket), the confidence distribution, a length-guard metric, and an automated cross-check of the judge's register call against the `rule_checks.py` regex.
6. **By-eye check ~30 pairs** before training (`--audit-sample`). Disagree with the judge on more than ~9 of 30 → fix the prompt before spending GPU time. P3 ran a formal human-labeled audit with a 0.7 agreement floor; the rerun trades that for the manual check plus the automated register cross-check, since fresh human labels on best-of-N pairs are not being produced.

**Key hyperparameters** (`config/`, §5.3): `beta` (KL strength — the main knob), learning rate (~an order of magnitude below SFT), epochs (1–2; DPO overfits preference sets quickly).

**Known failure modes**, both reported in the results table:
- **Length inflation.** DPO reliably learns "longer = preferred" if the labeler has any length bias. Track mean reply length per config.
- **Degeneration / mode collapse.** Replies get repetitive or lapse into a generic "persona voice." Check smoke-test outputs by eye.

If DPO is worse than SFT, report it. Do not tune until it wins.

**What happened (P4).** DPO produced a corroborated null: no CI-confirmed gain on any quality metric, and knowledge-boundary accuracy trended slightly down. Root cause in `artifacts/runs/p4_postmortem.md`: per-step DPO loss never left ln(2) (the model never fit the pairs even on the training set), because the pairs carried almost no learnable signal: chosen and rejected were both sampled from the same narrow SFT distribution, and the preference judge agreed with humans only ~70% of the time. Not circularity (the judge-scored metrics did not inflate), not a beta problem (the training loss did not move).

**The redo (`artifacts/runs/p4redo_progress.md`).** The rerun that addresses both causes — best-of-4 on-policy pairs and the v3 judge above — was run (`serana-dpo-v3`, 889 pairs, ~$2 GPU / ~$6 API). This time the training responded: loss fell below ln(2), the reward margin separated to +0.026 on held-out. The eval-set quality still showed no CI-confirmed gain over SFT (PCS 0.767 vs 0.800, PRS 0.900 vs 0.850, all CIs overlap), so `serana-dpo-v3` is **not promoted** — the shipped adapters and tables stay P3/P4. The two-level null (training responds, quality does not move) is the more informative result; see `p4_postmortem.md` §6. Still out of scope: scaling to several thousand pairs, a beta grid sweep, PPO.

### 3.5 Training knobs under a 24GB budget (L4)

These live in `config/`, not in code. **They are starting points, not answers** — §3.6 is how they get chosen, §7.2 is how their hardware cost gets quantified.

| Knob | Starting value | Why |
|------|---------------|-----|
| Quantization | 4-bit NF4, double-quant, bf16 compute | ~5GB for an 8B base (~4.5GB at 7B, scaled); L4 supports bf16 natively |
| Gradient checkpointing | on | trades step time for a large activation cut — **quantified in §7.2** |
| Max sequence length | 1024 | persona turns are short; activation memory is linear in this |
| Per-device batch size | 1 (DPO processes chosen+rejected, so ~2× activations) | |
| Gradient accumulation | 16 (CPT/SFT) / 8 (DPO) | keeps effective batch reasonable at batch size 1 |
| Optimizer | `paged_adamw_8bit` | optimizer state is what actually causes the OOM |
| Attention | FlashAttention-2 | supported on Ada — **lift quantified in §7.2** |
| LoRA rank | r=16 | **chosen in §3.6d, not assumed** |
| Learning rate | 2e-4 (SFT/CPT) | **chosen in §3.6c, not assumed** |
| DPO reference model | PEFT adapter toggle (`ref_model=None`) | avoids a second resident model (§3.4) |

- **Each stage continues the previous adapter.** Shared hyperparameters keep their values, or stage deltas measure hyperparameter changes.
- **Fallback ladder if 8B does not fit or is too slow** — one-line config changes: `Qwen3-8B → Qwen3-4B → Qwen3-1.7B`. **All configs move together.** Record which model produced the §6.1 tables, in the tables.
- **Checkpoint/resume is required.** Spot VMs; save adapter + optimizer state to GCS every N steps and resume on restart. Log preemption/resume events and the resumed loss curve showing continuity.

### 3.6 Choosing hyperparameters, and seeing them fail

Copying the defaults above and reporting a run that worked demonstrates that a library works — not that you can operate it. The practices below make the config values *chosen* and make the three canonical training failures (divergence, underfitting, overfitting) things you have actually watched happen rather than read about.

**They are diagnostics, not experiments.** They inform `config/`; they get no row in the §6.1 results tables. **Total budget: ~3 GPU-hours, ~$2, hard cap.** If the probes overrun, cut (d) first, then (c). Most of the value is in (a) and (b), which cost nothing.

#### (a) Validation loss, in every run — free

Split 5% off the CPT corpus and 5% off the SFT set as a **val split**. This is separate from the eval set in §4.1: the eval set measures persona quality with judges and metrics; the val split measures optimization health with loss.

Log train and val loss on the same axis for every run. Two things follow at zero cost:
- **Overfitting becomes visible in runs you are doing anyway** — train loss falling while val loss turns upward.
- **Epoch count stops being a guess.** The fixed `num_epochs: 3` becomes a ceiling with an early-stopping signal underneath it.

DPO's analogue is the reward-margin curve plus implicit accuracy on a held-out preference slice — already logged in §3.4. Split 5% of the preference pairs for it.

#### (b) Local failure sandbox — $0, before any GPU spend

**Divergence, underfitting, and overfitting are properties of the optimization, not of model size.** They appear on a 0.6B model exactly as they do on an 8B one. So they get induced locally, on the M5 with `Qwen3-0.6B` via the tiny-model config path — and they get done in **P1, before the first paid GPU hour**, so you can read a loss curve before you are paying for one.

Four deliberate breaks, each a few minutes:

| Break | How | What to watch |
|-------|-----|---------------|
| **Divergence** | learning rate × 10–100 | gradient norm blows up first, then loss spikes and goes NaN or flat-high. Learning that grad-norm leads loss is the point |
| **Underfitting** | r=2, lr × 0.1, 1 epoch | train loss barely moves; generations are still recognizably the base model, not the persona |
| **Overfitting** | 200 examples × 20 epochs | train loss → near zero while val loss turns up; generations start reciting training lines verbatim |
| **Data leak** | deliberately put 5 eval prompts into the training file | val loss suspiciously low, model reproduces held-out lines. Now you know what a leak *looks* like — and §4.6's guard stops being paperwork |

Save all four curves as artifacts. They are blog-post material and they are the honest basis for answering "what do you check first when training goes wrong."

#### (c) Learning-rate probe on the real model — ~1 GPU-h, ~$1

Three short SFT runs (~100 steps each) at lr ∈ {2e-5, 2e-4, 2e-3}, everything else fixed. Pick from the curves. This is not an exercise — it is how the value in §3.5 should be chosen in the first place.

Record the chosen value **and the two rejected curves**. A config file that shows only the winner hides the reasoning.

#### (d) LoRA rank comparison — ~2 GPU-h, ~$1

r ∈ {4, 16, 64} at the chosen lr. Compare on: val loss, trainable-parameter count, peak VRAM, step time, and a 10-prompt eyeball read of the outputs.

Deliberately **not** the full eval set — that would cost a complete eval pass per rank and turn a diagnostic into an experiment. The question is narrow: *was r=16 necessary, or would r=4 have done?* Val loss plus a smoke read answers it well enough to justify the config value, and "I checked, and 4 was enough / 16 was needed" is the sentence worth being able to say.

Cut this first if the budget tightens.

---

## 4. Quality evaluation

Four metrics, two DPO guards, one judge validation. Hardware metrics are in §7. Live values in `config/eval.yaml`.

### 4.1 Eval set definition

| Component | Size | Purpose |
|-----------|-----:|---------|
| In-boundary prompts | ~18 | She should answer, from the curated topic set (§3.1) |
| Out-of-boundary prompts | ~12 | She should deflect |
| Attack probes | 24 | PRS — 6 each across direct / meta / role-exit / escalating |
| Human-labeled pairs | 50 | Judge validation (§4.4) |
| Style reference | held-out slice of her translated lines | Style similarity reference |

- Distinct from the **val split** (§3.6a): the val split is loss-based optimization monitoring, this is judge- and metric-based quality measurement. Keep them separate; a val split used as an eval set stops being an honest early-stopping signal.
- The style reference is held out **before** any generation, and excluded from the CPT corpus too — easy to miss, since CPT ingests raw text rather than pairs.
- Frozen as `eval_set_vN`. Never overlaps CPT text, SFT data, the val split, or the DPO prompt pool.
- 24 probes is deliberate: enough for a per-category breakdown, small enough to hand-write carefully.
- 50 human labels is the floor for a usable Spearman estimate and the most tedious manual step. Do it once, in P1, in one sitting.

### 4.2 Persona quality metrics

| Metric | Definition | Computation | Direction |
|--------|-----------|-------------|-----------|
| **PCS** | 1 − (violations / total utterances) | (a) rule checks for hard constraints, then (b) `judge_pcs` for soft traits. Union. | 0–1, higher better. **Primary.** |
| **PRS** | fraction of attack probes survived without breaking character | Rule check (regex on "AI / language model / system prompt") ∪ `judge_robustness` over the 24 probes | 0–1, higher better |
| **Style similarity** | closeness to her original lines | BERTScore(generated, held-out reference) + perplexity under a reference LM | higher BERTScore / lower Δppl better |
| **Knowledge-boundary accuracy** | (correct in-boundary answers + correct deflections) / total | in/out label from the eval set; rule + judge | 0–1, higher better |

Expected shape: **B → SFT** should move style similarity and PRS most (a prompt saying "don't admit you're an AI" is a request; trained refusal is a disposition). **SFT → DPO** should sharpen *inconsistent* cases — borderline slips, weak deflections — so watch PCS and PRS rather than style similarity, which SFT largely saturates.

Distinctiveness is **cut** (needs contrast personas). The overfitting risk it guarded is watched via the val split (§3.6a) and qualitatively in smoke tests.

#### 4.2.1 Persona robustness — why it's first-class

An NPC that admits "I'm an AI" the moment a user pushes has broken the product; the same failure in a production chatbot is a prompt-injection leak.

- **Attack taxonomy:** *direct*, *meta*, *role-exit*, *escalating multi-turn* (3 turns). Category 4 is where prompt-only defenses crack — keep it despite the token cost.
- **Design constraint:** holding character must not require answering out-of-boundary questions. Both goals are "stay inside the persona."
- **Judge honesty:** `judge_robustness` scores only whether character held.
- **Report the `failure_type` breakdown**, not just PRS. *Which way* each config breaks is where the post-training story is most legible.

### 4.3 DPO guards (in the main table)

| Guard | Definition | Why |
|-------|-----------|-----|
| **Mean reply length** | mean output tokens per config | DPO's classic failure: the labeler mildly prefers longer replies, so the model learns verbosity |
| **Degeneration check** | manual read of ~20 smoke-test replies per config + distinct-n repetition rate | Catches mode collapse that metrics miss |

Publishing these unprompted is the point — it shows you knew the failure mode before it happened.

### 4.4 Judge validation and the circularity guard

- Score the 50 human-labeled pairs with `judge_pcs`; report **Spearman and exact / ±1 agreement**.
- If Spearman < ~0.6, revise the judge prompt before trusting any judge-backed number.
- Judge model and prompt version logged with every run. Never switch mid-experiment.

**The circularity problem.** An LLM labels the preference pairs that train DPO, and an LLM scores the output — and per §3.1, much of the SFT data is LLM-generated too. If the same rubric does the training and the scoring, DPO's gain on judge-scored metrics is partly self-fulfilling. Three required mitigations:

1. **Separate prompts, separate rubrics.** `preference_judge` picks the best and worst of N candidate replies; `judge_pcs` makes an *absolute* rating with a violation call.
2. **Corroborate with a non-judge signal.** DPO's gain must show up in at least one of: the PRS regex check, style similarity (no LLM judgement), or the 50 human labels. A gain visible *only* to the eval judge is reported as unconfirmed.
3. **Say it in the README**, alongside the real-pair-vs-synthetic ratio from §3.1.

### 4.5 Deciding whether a difference is real

- Report every quality metric as **mean ± 95% CI** (bootstrap over eval prompts, ≥ 1,000 resamples). With ~30 prompts the CIs will be wide — that is the honest picture at this eval-set size.
- Call a difference **real only if the 95% CIs don't overlap**. Otherwise "no measurable difference," which is itself a finding.
- **Eval decoding is greedy** (`temperature: 0.0`) — deterministic, single-run, no seed budget. Separate from preference-pair generation (§3.4), which samples deliberately.
- Quantization, `max_model_len`, `max_num_seqs`, and the base model are shared constants across configs.
- A metric at a suspicious extreme is treated as a leak/bug until verified against §4.6 — and you will know what that looks like, having induced one in §3.6b.

### 4.6 Leakage guard

1. Hold out the eval/style-reference slice **first**, before corpus assembly and any generation. Split the val slice at the same time.
2. Held-out lines never enter the CPT corpus and are never passed to `synth_dialogue` as examples.
3. **Attack probes are never SFT hard cases or DPO prompts.** Hard cases *teach*; probes *measure*. Overlap makes PRS a memorization test.
4. **The DPO prompt pool excludes every eval prompt** — easy to get wrong since both come from the same distribution. Assert it explicitly.
5. Before each training run, assert no eval prompt or reference line appears in the training file (exact + near-duplicate check).

---

## 5. Config schema

### 5.1 Layout

```
config/
├── base.yaml            # shared defaults (paths, model ids, train, serving, compute, gpu)
├── data_sources.yaml
├── persona.yaml         # persona_profile/voice_notes/glossary/speech_level/self_knowledge -- every PROMPTS.md prompt's {persona_*} vars (added P1; every prompt already assumed "comes from config/" but no file existed until then)
├── eval.yaml            # eval-set version, metric params, thresholds, judge models
├── experiments/         # serving-time configs
│   ├── b.yaml · sft.yaml · dpo.yaml
├── train_runs/
│   ├── cpt.yaml · sft.yaml · dpo.yaml
└── diagnostics/         # §3.6 probes — never produce shipped adapters
    ├── tiny_local.yaml  # Qwen3-0.6B on the M5, for the failure sandbox
    ├── lr_probe.yaml
    └── rank_probe.yaml
```

`diagnostics/` configs inherit from `base.yaml` like any other, so the failure sandbox exercises the real code path rather than a toy script. Their outputs go to `artifacts/diagnostics/` and are never used as a serving adapter.

### 5.2 The whole surface configs differ on

```yaml
# config/experiments/dpo.yaml
name: dpo                     # label only — pipeline must NOT branch on this
inherits: ../base.yaml
model_weights: lora
lora_adapter_id: serana-dpo
```

```yaml
# config/train_runs/dpo.yaml
name: serana-dpo
inherits: ../base.yaml
train:
  method: dpo                          # cpt | sft | dpo — selects the trainer
  init_adapter: artifacts/lora/serana-sft
  data:
    preference_file: data/ko/prefs_1k.jsonl
  dpo:
    beta: 0.1
    learning_rate: 5.0e-6
    num_epochs: 1
    grad_accum_steps: 8
  output_adapter: artifacts/lora/serana-dpo
```

```yaml
# config/diagnostics/tiny_local.yaml — the §3.6b sandbox base
name: tiny-local
inherits: ../base.yaml
model:
  base_id: Qwen/Qwen3-0.6B
train:
  method: sft
  output_adapter: artifacts/diagnostics/tiny
# each break overrides one or two fields from the CLI:
#   divergence   → train.learning_rate: 2.0e-2
#   underfitting → train.lora_r: 2, train.learning_rate: 2.0e-5
#   overfitting  → train.data.limit: 200, train.num_epochs: 20
#   data leak    → train.data.inject_eval_prompts: 5
```

### 5.3 base.yaml (shared constants)

```yaml
model:
  base_id: Qwen/Qwen3-8B   # swap = one line (fallback ladder in §3.5)
generation:
  temperature: 0.0                     # EVAL decoding: greedy, deterministic (§4.5)
  max_tokens: 512
preference_gen:                        # P3 only — deliberately not greedy (§3.4)
  temperature: 0.9
  n_samples: 2
  n_prompts: 1000
train:                                 # QLoRA on 1x L4 24GB (§3.5)
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16
  gradient_checkpointing: true
  max_seq_len: 1024
  per_device_batch_size: 1
  grad_accum_steps: 16
  num_epochs: 3                        # ceiling; early stop on val loss (§3.6a)
  learning_rate: 2.0e-4                # provisional — set from the §3.6c probe
  optim: paged_adamw_8bit
  attn_implementation: flash_attention_2
  lora_r: 16                           # provisional — confirmed by the §3.6d probe
  ref_model: null                      # DPO reference via PEFT adapter toggle (§3.4)
  val_split: 0.05                      # §3.6a — train/val loss on the same axis
  eval_steps: 25                       # how often val loss is computed
  log_grad_norm: true                  # divergence shows here before it shows in loss
  early_stopping_patience: 3           # in eval_steps units; null to disable
  checkpoint_every_steps: 50
  checkpoint_uri: gs://<bucket>/artifacts/lora
  data:
    cpt_file: data/ko/cpt_corpus.txt
    sft_file: data/ko/sft_3k.jsonl
gpu:                                   # §7 — instrumentation is config-driven too
  log_nvidia_smi: true
  nvidia_smi_interval_s: 1
  profile_steps: 20                    # torch.profiler window, after warmup
  record_peak_vram: true
  vram_estimate_tolerance: 0.20        # gap that triggers a written explanation
eval:
  style_embedding_id: jhgan/ko-sroberta-multitask
serving:
  engine: vllm
  max_model_len: 4096                  # justified by the §7.4 KV-cache arithmetic
  max_num_seqs: 8
  gpu_memory_utilization: 0.90
  quantization: awq                    # null = bf16; both measured in §7.4
  enable_lora: true
compute:
  provider: gcp
  region: asia-northeast3              # Seoul — kr-west hard constraint
  zone: asia-northeast3-b
  train_machine: g2-standard-8         # 1x NVIDIA L4 24GB
  train_provisioning: spot
  serve_machine: g2-standard-8
  serve_provisioning: on-demand        # Spot would corrupt latency numbers
paths:
  eval_set: data/eval/eval_set_v1
  gcs_bucket: gs://<bucket>
```

### 5.4 Rules

- **Capability flags only.** Inference reads `cfg.model_weights` / `cfg.lora_adapter_id`; training reads `cfg.train.method`. Never `if cfg.name == "dpo"`.
- **Never override `serving`, `compute`, or `gpu` in an experiment file.**
- **Diagnostics configs never produce a shipped adapter.** They write to `artifacts/diagnostics/` and are excluded from `lora_adapter_id` resolution.
- **Reproducibility.** Log the resolved config and hash with every run; both results tables are keyed by it. Also log machine type, provisioning mode, wall-clock, **predicted and measured peak VRAM**, step time, GPU utilization, train/val loss curves, grad-norm, epochs/steps, and estimated cost. For DPO additionally: β, tie/discard rate, reward-margin curve.
- **Thresholds are variables** — judge-correlation floor, TTFT target, VRAM tolerance, early-stopping patience, all in config.
- **Secrets never here.**

---

## 6. Repo layout

```
.
├── CLAUDE.md · DESIGN.md · PROMPTS.md · README.md
├── pyproject.toml · .env.example
├── config/                   # §5
├── data/
│   ├── raw/                  # ingested dialogue (English)
│   ├── ko/                   # cpt_corpus.txt + sft_3k.jsonl + prefs_1k.jsonl (+ val splits)
│   └── eval/                 # frozen eval set + human labels + attack probes + style reference
├── src/
│   ├── data/                 # ingest, translate, sft gen, preference gen, splitting
│   ├── finetune/             # train.py -- one entry point, train.method (cpt|sft|dpo) dispatches; extends the §1 single-pipeline rule to training instead of three near-duplicate scripts (device-aware model/LoRA loading, val split, run-report are identical across all three methods)
│   ├── serve/                # vLLM + FastAPI (the run(config) path)
│   ├── eval/                 # quality metrics, judges, results tables
│   └── gpu/                  # empty in practice -- see §2, this instrumentation lives in scripts/ instead
├── scripts/                  # one-command runners (incl. gpu_probe.py, GCE up/down)
├── artifacts/
│   ├── lora/                 # shipped adapters
│   ├── diagnostics/          # §3.6 curves and throwaway adapters
│   └── runs/                 # run logs, profiler traces (mirrored to GCS)
└── demo/                     # Gradio app
```

### 6.1 The two results tables

**Quality** — three rows:
`config | PCS | PRS | style sim | knowledge-boundary acc | mean reply length`
Each quality cell carries its 95% CI. A second small table breaks PRS down by `failure_type`.

**Hardware** — per training stage and per serving config:
`stage/config | predicted VRAM | measured peak VRAM | step time (or TTFT p50/p95) | GPU util % | MFU % | throughput | cost`

Headers state **base model, quantization, GPU (1× L4 24GB), and driver/CUDA versions**.

Companion figures: a grouped bar chart of the four quality metrics with mean reply length on a secondary axis; the throughput-vs-concurrency curve from §7.4; and the train/val loss curves for each stage. The §3.6 diagnostic curves belong in the blog post and an appendix, **not** in these tables — they explain how the config was chosen, they are not results.

---

## 7. GPU engineering practice

A deliverable, not a support function. The rule from CLAUDE.md — **predict, then measure** — is operationalized here. Almost none of it needs extra GPU time: it is instrumentation attached to runs that happen anyway. Budget impact ~5 GPU-hours (§9.2), separate from §3.6's ~3.

### 7.1 VRAM accounting: predict before you run

Before the first training run of each stage, write the arithmetic in the run plan. `src/gpu/vram_estimate.py` implements it; the training script records `torch.cuda.max_memory_allocated()` at the end. Gaps beyond `gpu.vram_estimate_tolerance` require a written explanation.

| Term | Estimate |
|------|----------|
| Base weights (4-bit NF4) | ≈ params × 0.5 bytes + quantization constants (double-quant reduces these) |
| LoRA parameters (bf16) | per target module: `r × (d_in + d_out) × 2 bytes`, summed over modules and layers |
| Gradients | only for LoRA params — this is the whole point of the method |
| Optimizer state | `paged_adamw_8bit`: ~2 states × 1 byte per **trainable** param, paged to host on pressure |
| Activations | ~`batch × seq × hidden × layers × k`; gradient checkpointing replaces the layer term with roughly its square root at ~30% step-time cost |
| CUDA context + fragmentation | a fixed few hundred MB plus allocator slack; measure once and carry the number |

Deliverables:
- A predicted-vs-measured VRAM table for CPT, SFT, DPO, and serving.
- **The PPO counterfactual.** Run the same arithmetic for policy + frozen reference + reward model + value head and show the number exceeds 24GB. This converts "I chose DPO" into "PPO needed N GB and I had 24."
- **The DPO reference-model check.** Measured DPO peak ≈ SFT peak is the proof of the §3.4 claim.
- Note that r appears in this arithmetic, so the §3.6d rank comparison and this section share a number — report them together.

### 7.2 Knob ablation: quantify the defaults

Four short runs (~50 steps each, ~30 minutes total):

| Run | Gradient checkpointing | FlashAttention-2 | Records |
|-----|:---:|:---:|---|
| A | off | off | step time, peak VRAM |
| B | **on** | off | the memory/time trade |
| C | off | **on** | attention's memory and speed lift |
| D | **on** | **on** | the shipped config |

The interview value is being able to say *"gradient checkpointing cost me X% step time and bought Y GB, which is what made batch size N possible."* If a run OOMs, that is a result — record the step it died at. Run once, during P2, on the SFT stage.

### 7.3 Utilization, profiling, and MFU

**`nvidia-smi` logging** at 1 Hz through every training run, capturing utilization, memory, power draw, and clocks. Look for sustained utilization well below 100% (input pipeline or sync stalls) and clock throttling under load.

**`torch.profiler`** over `gpu.profile_steps` steps after warmup, exported as a trace. Answer one question in writing: *is this workload memory-bandwidth-bound or compute-bound, and how do the top kernels support that?* For 4-bit QLoRA the dequantization path usually dominates.

**MFU.** Achieved FLOPs ≈ `6 × N_params × tokens / wall_seconds` for forward+backward; divide by the L4's published bf16 dense peak (read the datasheet, don't guess). Expect a low number — 4-bit dequant overhead, batch size 1, and gradient checkpointing all suppress it. **The low number is fine; the explanation is the deliverable.**

### 7.4 Serving-side GPU work (P5)

**KV-cache arithmetic, then verification.** Per-token cache cost: `2 (K and V) × layers × kv_heads × head_dim × dtype_bytes`. Qwen3 uses grouped-query attention, so `kv_heads < attention heads` — which is exactly why this must be computed rather than assumed. Multiply by `max_model_len × max_num_seqs` to justify those config values against the VRAM left after weights, then compare against the KV blocks vLLM reports allocating.

**Throughput vs concurrency sweep.** Tokens/sec and TTFT p50/p95 at concurrency 1, 2, 4, 8, 16 on one config. The knee — where throughput stops rising and latency climbs — justifies `max_num_seqs`, and the plot is the most legible artifact of the serving work.

**Quantization comparison.** Serve the merged model as bf16 and as AWQ. Report VRAM, TTFT, throughput, and PCS for both. Worth stating the asymmetry: training used 4-bit NF4 (bitsandbytes, memory-optimized for training) while serving uses AWQ (throughput-optimized for inference) — same idea, different tools for different jobs.

**Adapter overhead.** LoRA adds a per-token forward-pass cost. All configs share an identical prompt shape, so this is the only real serving-side variable. A near-zero delta is a legitimate finding — *"post-training bought quality at no serving cost."*

### 7.5 Spot preemption as GPU ops

Preemption is guaranteed on Spot, so treat recovery as a designed capability. Deliverables: the preemption event in the run log, the resumed loss curve showing continuity across the gap, and the wall-clock overhead of the restart.

### 7.6 What this section deliberately excludes

Custom CUDA/Triton kernels, FSDP/DeepSpeed sharding, tensor parallelism. The first is weeks of work for a signal §7.2–§7.3 already provide; the other two are unnecessary when a 4-bit 8B fits one card. If multi-GPU is wanted, plain DDP on 2× L4 (§8) is the honest version.

---

## 8. Optional extensions (only after P6 lands)

None are deliverables. Listed with real cost.

| Extension | Cost | Value |
|-----------|------|-------|
| **2× L4 DDP scaling run** (`g2-standard-24`, spot, one short SFT run) | ~3 GPU-h, ~$3 | Multi-GPU done honestly: report scaling efficiency (speedup ÷ 2) and where the gap goes. Recommended first |
| **DPO β sweep** (0.05 / 0.1 / 0.3) | +2 runs, ~8 GPU-h | Shows you understand the KL-strength tradeoff |
| **CPT as a fourth eval config** | +1 eval pass, ~5 GPU-h | Isolates CPT's contribution instead of folding it into SFT |
| **Multi-turn length check** (1 / 3 / 10 turns) | Eval-only, a few GPU-h | Persona drift as context grows |
| **Brand-tone demo** (prompt-level, no adapter) | ~zero GPU | Makes the transfer legible. Qualitative only |
| **vLLM Multi-LoRA serving** | A few GPU-h | Incremental VRAM per adapter is megabytes, not gigabytes |

If credit or time is tight at P6, ship without all of them.

---

## 9. Compute & budget plan (GCP $300 credit)

The whole build runs on the Free Trial credit, on one 24GB GPU in Seoul.

Why L4: Seoul (`asia-northeast3`) offers G2 (L4) in zones `-a` and `-b`, A2 (A100 40GB) in `-b`, and A3 Edge (H100) with limited capacity. A100/H100 rates would consume the credit in a few dozen hours.

### 9.1 Account prerequisites (P0 blockers)

1. **Upgrade the Free Trial billing account to Paid.** GPUs cannot be attached and quota increases cannot be requested on a non-billable trial account. Credit carries over and expires 90 days from signup.
2. **Request GPU quota in `asia-northeast3`:** `NVIDIA_L4_GPUS` and `PREEMPTIBLE_NVIDIA_L4_GPUS`, ≥1 each (≥2 if the §8 DDP run is planned). Approval can take a day; file on day one.
3. **Create budget alerts** at 50 / 80 / 100% of the credit.
4. **Create a single-region GCS bucket** in `asia-northeast3` for checkpoints, datasets, run logs, and profiler traces.
5. **Build the VM image once:** Deep Learning VM base + project deps, then snapshot the boot disk. Record driver and CUDA versions — they belong in the results table header.

### 9.2 GPU-hour budget

Verify rates before P2.

| Phase | Machine | Provisioning | Est. hours | Est. cost |
|-------|---------|--------------|-----------:|----------:|
| P0–P1 (ingest, translate, SFT set, eval set, **local failure sandbox §3.6b**) | n2-standard-4 (CPU) / local M5 | on-demand | ~17 | ~$4 |
| P2 (CPT + SFT, incl. reruns and preemptions) | g2-standard-8 | spot | ~16 | ~$5 |
| P2 diagnostics (**§3.6c lr probe + §3.6d rank probe**) | g2-standard-8 | spot | ~3 | ~$2 |
| P2 GPU instrumentation (§7.2 ablation, §7.3 profiling) | g2-standard-8 | spot | ~3 | ~$1 |
| P3 (preference-pair generation) | g2-standard-8 | spot | ~6 | ~$2 |
| P4 (DPO, incl. one β retry) | g2-standard-8 | spot | ~12 | ~$4 |
| P5 (serving + eval, 3 configs; incl. §7.4 sweeps and AWQ comparison) | g2-standard-8 | on-demand | ~24 | ~$22 |
| P6 (demo, HF upload) | g2-standard-8 | mixed | ~6 | ~$6 |
| GCS + persistent disk + egress | — | — | — | ~$6 |
| **Total** | | | | **~$52–68** |

The §3.6 diagnostics add ~$2 and the failure sandbox adds $0 (local). Judge/translation API cost is separate (§9.4), low tens of dollars. Comfortably inside the credit.

### 9.3 Cost discipline rules

- **The instance is stopped when not computing.** Persistent disks bill while stopped — small boot disk, artifacts in GCS.
- **CPU/API/local work never runs on the GPU instance.** Ingestion, translation, SFT-set generation, judge-API calls, and the §3.6b failure sandbox all run on CPU or the M5. Exception: preference-pair generation needs the GPU to sample from the SFT adapter — batch it, then shut down.
- **Diagnostics are capped.** §3.6 has a hard ~3 GPU-hour / ~$2 ceiling; overrunning it means cutting a probe, not raising the cap.
- **Spot for training, batch generation, and diagnostics; on-demand for measurement.** Preemption is fine where checkpoint/resume exists and fatal for latency benchmarks.
- **No managed training service.** Raw GCE — the point is to have driven the GPU directly.
- **Every run log records** machine type, provisioning mode, wall-clock, predicted and measured peak VRAM, step time, GPU utilization, train/val loss, epochs/steps, and estimated cost.
- **`scripts/` owns VM lifecycle.** One command up, one down.

### 9.4 Judge / translation provider — decide once, in P1

`translate_persona`, `synth_dialogue`, `preference_judge`, and both eval judges call an external LLM. One decision covers all:

- **Keep GPT-4o (OpenAI).** Not covered by the GCP credit; total is low tens of dollars, paid out of pocket. No re-validation needed.
- **Switch to Gemini on Vertex AI (`asia-northeast3`).** Covered by the credit, keeps text in-region. But changing the judge invalidates any prior §4.4 correlation result.

The credit does not cover third-party generative models offered as a managed API on Google Cloud; only Google's own models qualify.

Fix it at the start of P1, log the model id with every run, and **never change it mid-experiment**. The same provider sits in the training loop (SFT data, preference labels) *and* the measurement loop, so a mid-project switch would break both the data's consistency and the eval's comparability.
