# CLAUDE.md

Behavioral guide + project context for Claude Code. Read at the start of every session.

**Companion docs** (single source of truth for their area — don't duplicate their content here):
- `DESIGN.md` — architecture, evaluation design, **hyperparameter selection and failure diagnostics (§3.6)**, **GPU engineering practice (§7)**, and the compute/budget plan.
- `PROMPTS.md` — every LLM prompt, versioned.

Rule: when the pipeline changes, update `DESIGN.md` in the same change; when a prompt changes, update `PROMPTS.md`. This file stays light.

---

## Working principles (Karpathy-derived)

These take precedence over speed. For trivial one-liners, skip the ceremony; for anything else, follow them.

1. **Think before coding.** State assumptions explicitly; if a request is ambiguous, ask instead of guessing. Surface confusion, inconsistencies, and tradeoffs out loud. When two reasonable designs exist, show both with the tradeoff, then recommend one.
2. **Simplicity first.** Minimum code that solves the task — 50 lines beat 200. No speculative abstractions. Clean up dead code and unused imports.
3. **Surgical changes.** Never modify files outside the task. Never restyle untouched code. Confirm scope before editing more than 3 files. No drive-by refactoring.
4. **Goal-driven execution.** Before a multi-step task, state the plan and the success criteria. Each piece has a verification step. Define "done" before starting, then loop until it passes.
5. **Predict, then measure.** Before any GPU run: state the expected peak VRAM, step time, and dollar cost — *derived*, not guessed. After the run: record the actual and explain any gap over ~20%. "I ran it and it worked" is not an engineering result; "I predicted 18.4GB, measured 19.1GB, and the delta was fragmentation" is. See DESIGN.md §7.
6. **A copied hyperparameter is not a chosen hyperparameter.** Learning rate, LoRA rank, and epoch count come from the probes in DESIGN.md §3.6, and the rejected curves are kept alongside the chosen value. A config that shows only the winner hides the reasoning.

**Before finishing any task:** diff contains only requested changes · assumptions were stated or questions asked · code is simple on the first pass · no out-of-scope files touched · plan + success criteria were stated for multi-step work · **any GPU work has predicted-vs-measured numbers in the run log**.

---

## What this project is

An end-to-end **post-training stack** — prompt → CPT → SFT → DPO — applied to the **persona-consistency problem in general-purpose LLMs**, built and measured entirely on **one 24GB GPU**. The test bed is **Serana, an NPC from The Elder Scrolls V: Skyrim** (Dawnguard).

Why Serana: a large existing dialogue footprint (UESP + Fandom wiki), a sharply defined personality, and a natural knowledge boundary — she is a ~4,000-year-old vampire asleep for millennia, so she plausibly knows nothing of recent events or the modern world. That boundary is what makes the knowledge-boundary and persona-robustness metrics ("does she stay in character when told she's an AI") clean to measure.

**Two questions it answers, with numbers:**

> **1. Model:** Prompt → CPT → SFT → DPO — what does each stage of the post-training stack actually buy on the same persona and the same eval set?
>
> **2. Hardware:** What does each stage *cost* on a single L4 — in VRAM, step time, utilization, and dollars — and how close does a careful engineer get to the hardware's ceiling?

The second question is not decoration. A portfolio that reports quality metrics without hardware numbers reads as "I called a training library"; one that reports predicted-vs-measured VRAM, an MFU figure with an explanation, and a throughput/latency curve reads as someone who can be trusted with a GPU budget.

**Techniques exercised end to end:** QLoRA · **CPT** (continued pretraining, raw-text CLM) · **SFT** (supervised fine-tuning) · **RLAIF** (AI-generated preference labels) · **DPO** (reward-model-free preference optimization) · **GPU engineering** (VRAM accounting, profiling, knob ablation, quantized serving, throughput tuning, Spot preemption recovery) · **training diagnostics** (induced divergence, underfitting, overfitting, and leakage; lr and rank probes).

PPO-based RLHF is deliberately excluded (see the scope table). Being able to explain *why* DPO instead of PPO — backed by the VRAM arithmetic showing PPO doesn't fit — is a better signal than having attempted both badly.

This is a **portfolio build, not academic work.** Do not use "research," "study," or "thesis" in code, comments, docs, or commits. Frame everything as engineering: experiment, implementation, measurement, result.

### Scope discipline (read this before proposing anything)

The main risk is scope growth. The experiment is deliberately **minimal**: three serving configs, four quality metrics, three training runs. GPU engineering and training diagnostics are added as **instrumentation and short probes on runs that already happen** — which is why together they cost ~$4 rather than doubling the project.

**Deliberately out of scope. Do not propose these:**

| Cut | Why |
|-----|-----|
| Retrieval / RAG of any kind | A separate project. An inference-time knowledge channel would confound what training buys. |
| **PPO / reward-model RLHF** | Policy + reference + reward model + value head resident at once doesn't fit 8B on 24GB — and the arithmetic proving that is itself a deliverable (DESIGN.md §7.1). |
| Contrast personas (other NPCs) | Triples the data pipeline to earn one metric. |
| A data-scale sweep (100/500/1k/3k) | Replaced by the CPT→SFT→DPO axis at the same cost. |
| Custom CUDA / Triton kernels | Weeks of work for a signal §7.2–§7.3 already provide. |
| FSDP / DeepSpeed ZeRO sharding | 4-bit 8B fits one L4; sharding would be theater. Plain DDP on 2× L4 (DESIGN.md §8) is the honest multi-GPU demo. |
| A full hyperparameter grid search | §3.6 buys the *judgement* with three short probes. A grid buys marginal accuracy at many times the cost. |
| Broad Elder Scrolls world lore | Knowledge breadth is not a deliverable. |
| Multi-seed eval runs | Eval decoding is greedy, so seeds change nothing. Free cut. |

If a task seems to need one of these, say so and stop.

### Where the data actually comes from

Source→destination routing, the exclusion table, and the real-pair-ratio requirement: **DESIGN.md §3.1** (single source — don't restate here). Language policy (translate-first, unify in Korean): **DESIGN.md §3**.

---

## The experiment: B, SFT, DPO

Three configs (B / SFT / DPO), one `run(config)` pipeline — full description in DESIGN.md §1 and §3.

**Each stage must be justified by its own delta.** If DPO doesn't beat SFT on any metric, that is a reportable finding — say so plainly rather than tuning until it wins.

**Single-pipeline rule.** One `run(config)` entry point; the config toggles `model_weights: base | lora` plus which adapter to load. All configs run identical inference code.

Enforcement:
- Never branch on the config name. `if config_name == "dpo"` is a design violation.
- Adding a fourth config must require zero new inference code.
- `model_weights: base` is a real code path, not a stub.
- Base model, quantization, and all serving/decoding parameters are shared constants. If the model is downsized (DESIGN.md §3.5), **all configs move together.**
- **Each training stage continues the previous adapter.** Shared hyperparameters keep their values, or stage deltas measure hyperparameter changes.
- **Diagnostic runs (DESIGN.md §3.6) never produce a shipped adapter.** They write to `artifacts/diagnostics/` and get no row in the results tables.

---

## Stack (fixed; don't substitute without asking)

7-8B-class open model (Qwen3-8B default; id in `config/`, never in code) · `LoRA/QLoRA` (PEFT) · **`TRL`** for `SFTTrainer` and `DPOTrainer` · `vLLM` + `FastAPI` serving · `ko-sroberta-multitask` embeddings **for evaluation only** · custom persona metrics + LLM-as-Judge · `Gradio` demo on HF Spaces · **`GCP Compute Engine G2` (1× NVIDIA L4, 24GB) in `asia-northeast3` (Seoul)** · **MacBook M5 + `Qwen3-0.6B`** for the local failure sandbox and plumbing tests.

**GPU tooling** (part of the stack, not optional): `torch.profiler` · `nvidia-smi dmon` logging · `torch.cuda.max_memory_allocated()` in every training script · gradient-norm logging · `FlashAttention-2` · `AWQ` for serving quantization · vLLM's KV-cache and throughput reporting.

Deliberately **not** in the stack: ChromaDB, LangChain, retrieval of any kind, Ragas, Phoenix, PPO/`PPOTrainer`, a reward model, DeepSpeed, custom kernels, hyperparameter-search frameworks (Optuna/Ray Tune — §3.6's three probes are the scoped alternative).

**Model-agnostic by design.** No hardcoded model id anywhere; swapping the base model is a one-line config change. This is also what makes the local sandbox possible — the same code path runs 0.6B on the M5 and 8B on the L4.

**Hard constraint:** compute and data stay in `kr-west`. No code path ships data outside the region except text-only calls to the judge/translation API. (`kr-west` = GCP `asia-northeast3`.)

**VRAM is the binding constraint — and the subject matter.** Every VRAM-affecting decision must be justified in arithmetic, before the run (DESIGN.md §7.1). If something does not fit, shrink the model via `config/`. **An OOM is a config problem and a prediction failure — log both the fix and why the estimate was wrong.** Do not reach for a bigger GPU.

**Budget constraint (GCP $300 welcome credit).** Binding on design:

- Billing must be **upgraded to Paid** and L4 quota granted in `asia-northeast3` before any P2 work. Credit carries over but expires 90 days from signup.
- Before any GPU run, state estimated wall-clock hours and dollar cost.
- **Diagnostics are capped at ~3 GPU-hours / ~$2** (DESIGN.md §3.6). Overrunning means cutting a probe, not raising the cap.
- Training and batch generation run on **Spot** with GCS checkpoint/resume. Preemption is expected — recovering cleanly is a logged deliverable.
- Serving and latency measurement run on **on-demand** VMs.
- The VM is stopped whenever it is not actively computing. Artifacts live in GCS.
- Budget alerts at 50 / 80 / 100% of credit are a P0 exit criterion.

Full prerequisites and the GPU-hour budget are in **DESIGN.md §9**.

**IP / licensing note.** (1) Wiki text (UESP, Fandom) is CC BY-SA — attribute per `config/data_sources.yaml`. (2) The Elder Scrolls / Serana IP belongs to Bethesda/ZeniMax. **Non-commercial engineering portfolio**; state in the README that the IP belongs to its owners.

---

## Evaluation

**Quality** — PCS, PRS, style similarity, knowledge-boundary accuracy · two DPO guards (mean reply length, degeneration check) · judge validation against 50 human labels, which must pass before any judge-backed number is trusted.

**Hardware** — per training stage: predicted vs measured peak VRAM, step time, GPU utilization, MFU. Per serving config: TTFT p50/p95, throughput, KV-cache footprint, cost per 1k requests.

**Optimization health** — train/val loss on the same axis for every run, plus grad-norm. This is a *val split* (5% of training data), deliberately separate from the eval set. It makes overfitting visible in runs that happen anyway and turns the epoch count into a ceiling with an early-stopping signal underneath.

**Circularity guard.** An LLM generates much of the SFT data, labels the preference pairs that train DPO, and scores the results. Mitigations: the preference judge and eval judge are **separate prompts with different rubrics** (PROMPTS.md §4 vs §5); DPO's gain must be corroborated by a signal the eval judge doesn't produce (PRS regex check, style similarity, or the human labels); and the README states the limitation alongside the real-pair-vs-synthetic ratio.

---

## Repo conventions

- Python 3.12+, `uv` for env/deps, `ruff` for lint/format.
- Config in `config/*.yaml` — no hardcoded paths, model names, or hyperparameters in scripts. Schema in DESIGN.md §5; repo layout in §6.
- Serving configs differ **only** in `model_weights` and `lora_adapter_id`.
- Every experiment reproducible from one command; log the resolved config + hash, machine type, provisioning mode, wall-clock, **predicted and measured peak VRAM**, step time, **train/val loss and grad-norm**, and estimated cost, per run.
- Secrets via `.env` (never committed).
- Leakage guard: eval prompts, attack probes, and the style reference never appear in CPT text, SFT data, the val split, or preference pairs — verify before each training run (DESIGN.md §4.6).
- Local dev (MacBook M5, Qwen3-0.6B): plumbing, config branching, smoke tests, and the §3.6b failure sandbox. **Local timings are never reported as GPU results** — different memory architecture and serving stack. Loss *shapes* from the sandbox are valid to discuss; loss *values* and speeds are not comparable.

---

## Build order

Seven phases. **Every GPU phase's criterion includes its measurements** — a phase that produced a working artifact but no numbers is not done.

- **P0 Scaffold** — `uv` project, repo skeleton, config schema (incl. `config/diagnostics/`), `.env.example`, **plus the GCP prerequisites in DESIGN.md §9.1**. *Done:* `run(config)` executes end-to-end with stubs · an L4 can be provisioned in Seoul · `scripts/gpu_probe.py` reports device properties, available VRAM, and a matmul throughput sanity check on the real L4.
- **P1 Data + local diagnostics** — ingest UESP + Fandom dialogue, **count how many lines survive as genuine pairs vs how many are un-paired** (DESIGN.md §3.1), translate to Korean once, hold out the eval slice and the val split, build the CPT corpus and the ~3k SFT set. Build the eval set (30 prompts + 24 attack probes + 50 human labels). Fix the judge/translation provider (DESIGN.md §9.4). **Then run the §3.6b failure sandbox locally on 0.6B** — induce divergence, underfitting, overfitting, and a deliberate data leak, and save all four curves. CPU/local only, no GPU spend. *Done:* both corpora built · real-pair ratio recorded · horizon filter applied · eval set frozen · leakage check passes · **four failure curves saved and readable**.
- **P2 CPT + SFT** — QLoRA, two stages, Spot G2 VM, GCS checkpoint/resume. *Done:* adapter trains within budget · **lr chosen from the §3.6c probe, with the two rejected curves kept** · **§3.6d rank comparison recorded** · **train/val loss curves logged for both stages** · predicted vs measured peak VRAM recorded · **§7.2 knob ablation table filled in** · `torch.profiler` trace and an MFU figure with a written explanation · survives one preemption-resume cycle · passes an in-persona Korean smoke test.
- **P3 Preference data (RLAIF)** — sample two replies per prompt from the SFT adapter at `temperature > 0`, preference judge picks, emit ~1k pairs. Discard ties and near-identical pairs. *Done:* pair set built · tie/discard rate logged · hand-audit of ~30 pairs agrees · batch-inference throughput recorded.
- **P4 DPO** — continue the SFT adapter with `DPOTrainer`, reference via PEFT adapter toggle. *Done:* trains within budget · **measured peak VRAM confirms the no-second-model claim** · reward-margin curve and held-out preference accuracy logged · smoke test shows no degeneration or length blowup.
- **P5 Serving + Eval** — vLLM sized from the KV-cache arithmetic (§7.4), FastAPI streaming, then quality metrics + judge validation + hardware metrics across all three configs. *Done:* one endpoint serves any config · KV-cache budget predicted then verified against vLLM's reported blocks · throughput-vs-concurrency sweep plotted · AWQ vs bf16 compared on VRAM, TTFT, throughput, and PCS · both results tables regenerate from one command.
- **P6 Ship** — Gradio demo on HF Spaces, adapters to HF Hub, README with both tables and the hardware stated, blog post. The §3.6 diagnostic curves go in the blog post and an appendix, not the results tables. *Done:* a stranger can open the demo and reproduce from the repo on a single 24GB GPU.

**Optional, only after P0–P6 land** (DESIGN.md §8): 2× L4 DDP scaling run · DPO β sweep · CPT as a fourth eval config · multi-turn length check · brand-tone demo · vLLM Multi-LoRA serving.

---

## When in doubt

Touching >3 files or changing the pipeline interface → confirm the plan first. Adding a dependency → check the stack covers it; reject anything retrieval-, PPO-, sharding-, or custom-kernel-shaped. A metric looking too good (e.g., 100%) → suspect a leak/bug — and you will know what a leaked loss curve looks like, having induced one in P1. DPO winning on judge-scored metrics but nothing else → suspect circularity, not a real gain. **Any GPU change → state predicted VRAM, step time, and cost before running, and record the actual after.** An OOM is a config problem *and* a failed prediction — fix both. A diagnostic probe running long → cut it, don't raise the cap. "Let's add another config / persona / a reward model / a grid search" → re-read the scope discipline table.
