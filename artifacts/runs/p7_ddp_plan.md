# 2x L4 DDP scaling run — plan

**Status: blocked on quota (increase request submitted 2026-09-02).**
DESIGN.md §8 optional extension ("2x L4 DDP scaling run ... the honest
multi-GPU demo"). Judge-free — GCP credit only, no OpenAI API.

The project's L4 quota is capped at **1 GPU** in both places that gate a
2x L4 instance: `NVIDIA_L4_GPUS` (asia-northeast3) and `GPUS_ALL_REGIONS`
(global), the same two-layer cap flagged in P0. `g2-standard-24` (2x L4)
cannot be created until both are raised to >= 2. A quota-increase request
is in; GPU increases on a $300-trial project are often slow or denied,
so this may not clear. The DDP code changes (`src/finetune/train.py`,
committed) are ready either way — if the quota clears, this is a quick
execution; if not, the honest portfolio note is "the multi-GPU demo was
scoped, coded, and blocked by a hard L4 quota of 1", which is itself a
real constraint (the same class as P4's L4 stockout).

## Goal

Measure what a second L4 buys on SFT training: speedup, parallel
efficiency (speedup / 2), and where the missing efficiency goes
(gradient all-reduce over NCCL). Adds one row to DESIGN.md §7's
hardware story. This is not about making training faster — the 1-GPU
SFT run is already fine — it is about demonstrating multi-GPU
competence with real numbers.

DDP, not FSDP/ZeRO sharding: the 4-bit 8B model fits one L4, so sharding
the model across GPUs would be theater (DESIGN.md §8). DDP keeps a full
model copy per GPU and splits the batch.

## Success criteria (define "done" first)

1. A 1-GPU vs 2-GPU table: step time, samples/s, tokens/s, MFU, peak
   VRAM per GPU, and all-reduce as a % of step time (from `torch.profiler`).
2. **Equivalence check:** global batch held at 16 in both runs (1-GPU:
   `per_device=1, grad_accum=16`; 2-GPU: `per_device=1, grad_accum=8,
   world_size=2`). The 2-GPU loss curve for the first ~50 steps should
   overlay the 1-GPU one — proof it is the same training, run faster,
   not a different (larger-batch) experiment.
3. A written explanation of the efficiency gap (why ~1.8x, not 2x).

## Setup

Both runs on **one `g2-standard-24` (2x L4, on-demand)** in
`asia-northeast3-a`. Same machine for both so only the GPU count varies:

- **1-GPU baseline:** `CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py
  --config config/train_runs/sft.yaml --set train.max_steps=60`
- **2-GPU DDP:** `torchrun --nproc_per_node=2 scripts/train.py --config
  config/train_runs/sft.yaml --set train.max_steps=60 --set
  train.grad_accum_steps=8`

`max_steps=60` — first ~10 steps are warmup, the remaining ~50 give a
stable step-time average. No full 534-step run. Output adapters go to
`artifacts/diagnostics/` and are discarded.

`torch.profiler` on a short window (5-6 steps) of each run to capture
the NCCL all-reduce time; extends the P2 §7.2 profiling method.

## Code changes needed (`src/finetune/train.py`)

DDP with 4-bit QLoRA needs each process to load the full model onto its
own GPU. Minimal, backward-compatible (`LOCAL_RANK` unset -> single-GPU
path unchanged):

1. `run()`: read `LOCAL_RANK` from env; if set and CUDA,
   `torch.cuda.set_device(local_rank)` before loading the model.
2. `load_model_and_tokenizer()`: pass `device_map={"": local_rank}` to
   `from_pretrained` when quantized under DDP.
3. Guard the `run_report.json` write to the main process only
   (`local_rank <= 0`), so rank 1 does not race/clobber it.

~15 lines. If this turns out not to be enough (bnb + DDP has known rough
edges), see the stop condition below.

## Prediction

| | value |
|---|---|
| peak VRAM / GPU | ~15 GB (per_device batch stays 1, full model copy per GPU — same as 1-GPU) |
| 1-GPU, 60 steps | ~25 min (27-28 s/step, from P2/P4) |
| 2-GPU, 60 steps | ~14 min (expected speedup ~1.8x) |
| provisioning | on-demand (measurement run, DESIGN.md §9.3) |
| cost | g2-standard-24 ~$1.4/hr x ~1.5 hr (both runs + profiler + teardown) ≈ **~$2-3 GCP credit** |
| API | **$0** |

## Risks

1. **4-bit QLoRA + DDP device placement** — the main risk. The three
   code changes above are the expected fix; bnb + DDP can still surface
   edge cases (gradient-checkpointing reentrancy, `find_unused_parameters`).
   Mitigation: a 2-step DDP smoke run first.
2. **2x L4 stockout** in `asia-northeast3` — 1x L4 already hit this in
   P4. `g2-standard-24` may be harder to get.
3. `torch.cuda.max_memory_allocated()` reports rank 0 only — fine, it is
   representative (both ranks do identical work).

## Stop conditions (CLAUDE.md: cut, don't raise the cap)

- The 2-step DDP smoke run fails and the fix is more than ~10 further
  lines -> stop, document "4-bit QLoRA under DDP needs non-trivial code
  changes, out of scope for a demo".
- No `g2-standard-24` capacity after a few retries -> stop, record the
  stockout (itself a data point, same as P4).
- Cost reaches **$5** or **3 GPU-hr** -> stop.

## Not doing

A shipped adapter (output is discarded). FSDP / ZeRO sharding. Any
judge or eval-set run. A full 534-step training run.
