# serana-post-training

> "Serana" and The Elder Scrolls are property of Bethesda/ZeniMax. This is a non-commercial engineering portfolio project, not an official product, and is not affiliated with Bethesda/ZeniMax.

<img width="253" height="180" alt="image" src="https://github.com/user-attachments/assets/d9c149da-3c4a-47a5-88ec-ad031ca12dcc" />

An end-to-end **post-training stack** — prompt → CPT → SFT → DPO — applied to the **persona-consistency problem in general-purpose LLMs**, built and measured entirely on **one 24GB GPU** (NVIDIA L4). The test bed is **Serana**, an NPC from *The Elder Scrolls V: Skyrim* (Dawnguard): a large existing dialogue footprint, a sharply defined personality, and a natural knowledge boundary (a ~4,000-year-old vampire asleep for millennia, who plausibly knows nothing of the modern world).

**Two questions, with numbers:**

1. **Model** — Prompt → CPT → SFT → DPO: what does each stage actually buy on the same persona and the same eval set?
2. **Hardware** — what does each stage *cost* on a single L4 (VRAM, step time, throughput, dollars), and how close does a careful engineer get to the hardware's ceiling?

Full design rationale, hyperparameter-selection method, and the compute budget: [`DESIGN.md`](DESIGN.md). Every LLM prompt used, versioned: [`PROMPTS.md`](PROMPTS.md). Behavioral rules and build order: [`CLAUDE.md`](CLAUDE.md).

**Adapters**: [`machu8/serana-sft`](https://huggingface.co/machu8/serana-sft) · [`machu8/serana-dpo`](https://huggingface.co/machu8/serana-dpo) (LoRA, require `Qwen/Qwen3-8B` as base)
**Demo**: `demo/app.py` — a Gradio app (one input, three responses: B/SFT/DPO side by side) built for HF Spaces' free ZeroGPU tier, not yet deployed there (ZeroGPU currently requires a HF PRO subscription or a community grant). Run it yourself on any CUDA machine: `pip install -r demo/requirements.txt && python demo/app.py`.

---

## Headline finding

**DPO shows no statistically-confirmed quality gain over SFT on any metric measured** — PCS, PRS, knowledge-boundary accuracy, style similarity, mean reply length, and distinct-2 all have overlapping 95% CIs between the SFT and DPO rows below. This isn't a single disappointing run: it's corroborated by three independent signals —

1. DPO's own training metrics were weak (held-out preference accuracy 59.5%, barely above chance; tiny reward margins).
2. A hand-read smoke test found DPO didn't fix the one regression SFT had (a boundary case where the trained model drops persona framing).
3. This full CI-backed evaluation pass shows no metric where DPO's confidence interval clears SFT's.

Shipped anyway — the honest result of the pipeline, not the result that was tuned for. See `artifacts/runs/p4_progress.md` and `p5_progress.md` for the full trail.

---

## Results — Quality

Base model: `Qwen/Qwen3-8B` · bf16 · 1× NVIDIA L4 24GB, `asia-northeast3` (Seoul) · driver 580.173.02, CUDA 12.9 (training) / CUDA 13.0 (serving, via a later `vllm` install) · 30 in/out-of-boundary prompts + 24 attack probes, greedy decoding, 95% bootstrap CI (≥1000 resamples).

| config | PCS | PRS | style sim | knowledge-boundary acc | mean reply length | distinct-2 |
|---|---|---|---|---|---|---|
| B (base + prompt) | 0.733 [0.567, 0.900] | 0.850 [0.650, 1.000] | 0.293 [0.283, 0.303] | 0.833 [0.700, 0.967] | 150.6 [121.8, 181.8] | 0.265 |
| SFT | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.216, 0.252] | 0.933 [0.833, 1.000] | 23.4 [21.0, 25.7] | 0.587 |
| DPO | 0.800 [0.633, 0.933] | 0.850 [0.700, 1.000] | 0.234 [0.218, 0.250] | 0.867 [0.733, 0.967] | 24.1 [21.6, 26.7] | 0.590 |

- **PCS** (persona consistency score) and **PRS** (persona robustness score, held under 24 direct/meta/role-exit/escalating attack probes) are both rule-check ∪ LLM-judge unions — a reply only counts as a violation/break if *either* signal catches it.
- With ~30 quality prompts and ~20 scored attack probes, most CIs are wide. Real, CI-confirmed differences: B is far more verbose (150 tokens vs ~23-24) — the "info-dump vs. concise in-voice reply" pattern trained out by SFT. B also scores *higher* on style similarity than SFT/DPO, opposite DESIGN.md's predicted direction — most likely a low-discriminative-power artifact of the embedding metric on this small a reference set (values cluster in a narrow 0.21–0.30 band regardless of input) rather than a real style regression; flagged, not smoothed over.
- Full per-category PRS breakdown and per-prompt data: `artifacts/runs/results_quality.md`, `eval_*.json`.

## Results — Hardware

| stage/config | predicted VRAM | measured peak VRAM | step time / TTFT p50,p95 | MFU % | throughput | cost |
|---|---|---|---|---|---|---|
| CPT (training) | – | 9.64 GB | 36.2s (tiny corpus) | – | – | ~$0 |
| SFT (training, r=16, lr=2e-4) | – | 9.64 GB | predicted 20–40min → measured 4h12m* | 13.5% | – | ~$5 |
| DPO (training, resumed after 1 Spot preemption) | 13–16 GB | ~15 GB | predicted 34–38s/step → measured ~27s/step | – | – | ~$0.25 |
| Serving KV-cache (max_model_len=4096, max_num_seqs=8) | 4.50 GB | 3.17 GB default / 4.87 GB to fully utilize | – | – | – | – |
| SFT via LoRA, bf16, concurrency=8 | – | – | p50=0.213s p95=0.422s | – | 80.6 tok/s | – |
| SFT merged, bf16, concurrency=8 | 15.27 GB | 15.36 GB weights | p50=0.247s p95=0.839s | – | 75.4 tok/s | – |
| SFT merged, **AWQ**, concurrency=8 | 3.82 GB | 5.8 GB weights | p50=0.091s p95=0.529s | – | **183.8 tok/s** | – |

\* SFT's wall-clock miss (a real predicted-vs-measured gap, not hidden): `grad_accum_steps` wasn't overridden for the real run, so each logged "step" was 16 micro-batches, not 1 — root-caused live, fixed for the resumable-checkpoint version. See `artifacts/runs/p2_progress.md`.

**Throughput-vs-concurrency knee, exactly at the configured `max_num_seqs=8`**: throughput scales near-linearly 1→8 (13→24→45→81 tok/s), then completely flatlines at 16 (80.9 tok/s) while TTFT p50 explodes 10x (0.213s → 2.268s). The config value is validated by data, not assumed.

**AWQ vs bf16** (same merged weights, quantization isolated from the adapter question): 2.44× throughput, 2.7× faster TTFT p50, **no PCS loss** (0.767 vs 0.800, CIs overlap). Total VRAM used stays ~19.3–19.5GB either way — `gpu_memory_utilization=0.9` is a target, not a cap, so AWQ's weight savings (15.36GB → 5.8GB) get redirected almost entirely into **4× more KV-cache capacity** (23,056 → 92,656 tokens), not lower memory used. Stated that way deliberately, since the flatter "AWQ uses less memory" claim would be inaccurate.

**LoRA adapter overhead**: ~7% between LoRA-on-base and the fully-merged model at matched concurrency — within run-to-run noise at this sample size. Reads as the near-zero overhead DESIGN.md anticipated: post-training bought quality at effectively no serving cost.

Full predicted-vs-measured trail for every GPU phase, including two real environment bugs found and fixed mid-run (a `flash-attn`/torch ABI break, Qwen3's thinking-mode token budget): `artifacts/runs/p2_progress.md` … `p5_progress.md`.

---

## Data composition & the circularity guard

- **51.4%** of ingested wiki dialogue lines (UESP + Fandom, CC BY-SA) survived as genuine `(player line, reply)` pairs; the rest are standalone utterances (CPT corpus) or excluded by the horizon filter (nothing after 4E 201 / modern-world topics).
- The final ~3k SFT set is **7.7% real pairs, 92.3% synthetic** (GPT-4o-generated, matched to the real data's voice). This ratio matters beyond bookkeeping: the more of the pipeline is LLM-authored end to end (SFT data → DPO preference labels → eval scoring), the sharper the circularity concern below.
- **Circularity guard**: the preference judge (pairwise, trains DPO) and the eval judge (absolute rating, scores results) are deliberately separate prompts with different rubrics (`PROMPTS.md` §4 vs §5), validated separately against 50 hand-scored human labels (Spearman 0.73, floor 0.6). Any DPO gain would need to show up in a non-judge signal (the PRS regex check, style similarity, or the human labels) to be trusted — moot here, since DPO didn't show a gain to begin with.

---

## Stack

`Qwen/Qwen3-8B` · QLoRA (PEFT) · `TRL` (`SFTTrainer`, `DPOTrainer`) · `vLLM` (OpenAI-compatible server, multi-adapter) + `FastAPI` · `ko-sroberta-multitask` (eval embeddings only) · custom persona metrics + LLM-as-judge (GPT-4o) · `AWQ` (serving quantization) · `Gradio` on HF Spaces (ZeroGPU) · GCP Compute Engine G2 (1× L4) in `asia-northeast3`. PPO/reward-model RLHF deliberately excluded — the VRAM arithmetic for policy+reference+reward+value simultaneously resident doesn't fit 8B on 24GB (`DESIGN.md` §7.1); that calculation is itself part of the deliverable.

## Reproducing

Everything needed to reproduce end to end — config schema, build order, GPU-hour budget, prerequisites — is in `DESIGN.md` and `CLAUDE.md`. Runs on one 24GB GPU. `HARNESS_ENGINEERING.md` documents the guardrails (`.claude/hooks/`) used to keep an AI coding agent inside the project's scope and region constraints while building this.
