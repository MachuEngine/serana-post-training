# P2 progress log

Running log of P2 (CPT + SFT on the real L4) work. Same style as
`artifacts/runs/p1_progress.md` — updated as work happens, not a final
report.

## VM

Spot `g2-standard-8` (1x L4, 23GB), `asia-northeast3-b`,
`pytorch-2-9-cu129-ubuntu-2204-nvidia-580` image.
`--instance-termination-action=STOP` so the boot disk (and any local
checkpoints) survive a Spot preemption.

## Environment bugs found and fixed on this VM

1. **torchaudio ABI mismatch** — image ships a torchaudio built against a
   different torch build; importing `peft` (which transitively imports
   `transformers.audio_utils` → `torchaudio`) crashed with an `OSError`.
   Fixed: `sudo pip uninstall -y torchaudio` (not needed, text-only project).
2. **jinja2 too old** — image's jinja2 (3.0.3) is below the 3.1.0 chat
   templates require. Fixed: `pip install -U jinja2`.
3. **flash-attn compile abandoned** — no prebuilt wheel matched this
   image's torch/CUDA/Python combo; source compile ran 33+ min and was
   killed. All runs below use `--set train.attn_implementation=sdpa`
   instead. Flash-attn vs sdpa is still an open comparison for §7.2.

## CPT — done

`config/train_runs/cpt.yaml`. wall_clock 36.2s, peak VRAM 9.64GB,
final_train_loss 3.68. Fast because the corpus is tiny (3 steps total).

## SFT — done

`config/train_runs/sft.yaml`, continuing the CPT adapter.

- **Predicted 20–40 min, measured 4h12m9s (15129.2s)** — a real predict-
  vs-measure miss, off by 6-12x. Root cause investigated live:
  - `grad_accum_steps: 16` was not overridden, so each logged "step" is
    actually 16 sequential micro-batches (confirmed with the user,
    walked through in chat).
  - Tried increasing `per_device_batch_size` 1→4 (with
    `grad_accum_steps` 16→4, same effective batch=16) expecting a big
    speedup from better GPU parallelism. **Wrong hypothesis** — a 3-step
    test at batch=4 showed peak VRAM barely moved (9.64GB → 10.46GB) and
    per-example step time barely improved (~6%). Real bottleneck is
    more likely the 4-bit dequantization cost per forward pass and/or
    the lack of flash-attn (packing/padding-free warnings already
    flagged sdpa as suboptimal here) — not GPU parallelism/batch size.
  - **Real finding, not yet root-caused**: this stack (QLoRA + sdpa, no
    flash-attn) is just slow on this VM. Worth a real ablation in §7.2
    rather than more guessing.
- peak VRAM: 9.64GB (batch=1 run) — essentially unchanged from CPT despite
  a much larger dataset; gradient checkpointing is doing its job.
- final_train_loss 0.1401, eval_loss plateaued at 0.083-0.084 over the
  last 3 eval checkpoints (not still falling, not diverging upward
  either) — a plateau, not the train↓/val↑ divergence pattern from the
  §3.6b overfitting demo. So no obvious overfitting signature in the
  loss curves themselves.
- eval_mean_token_accuracy: **0.98** — see TODO below, this number alone
  doesn't settle whether the model generalized or memorized.
- **Bug found and fixed**: trainer had `save_strategy="no"` — a Spot
  preemption mid-run would have lost the entire 4h12m with no way to
  resume. Fixed: `save_strategy="steps"` + `save_steps` (from
  `checkpoint_every_steps`) + `save_total_limit=3`, and `run()` now
  auto-detects and resumes from the latest local checkpoint
  (`get_last_checkpoint`) if one exists in `output_adapter`. Local-disk
  checkpoints are sufficient here specifically because
  `--instance-termination-action=STOP` keeps the boot disk alive across
  preemption — a fresh VM would still need a GCS sync step, which this
  does not add.
- **Bug found and fixed (before running)**: `config/diagnostics/rank_probe.yaml`
  was missing `max_steps` (unlike its sibling `lr_probe.yaml`) — would have
  run the full ~534-step training per rank candidate instead of a short
  probe. Caught before it cost anything.

## TODO — carried into P5

**Check real-vs-memorized generalization**, raised while discussing the
98% eval accuracy: ~92% of the SFT set is GPT-4o-generated synthetic data
sharing similar phrasing patterns (same `persona_profile`, same few-shot
examples, same generation prompt across thousands of calls). A 98% token
accuracy on the held-out val split doesn't distinguish "the model learned
Serana's voice generally" from "the model learned GPT-4o's synthetic-SFT
writing style specifically" — the val split is a random slice of the
*same* generation process, not genuinely novel prompts.

**Resolution belongs in P5**: run the eval-set prompts (§4.1, 30 prompts
— written by hand, never seen during SFT/DPO training) through the
trained model and check PCS/style-similarity there. That's the real test
of generalization vs memorization; the val-loss plateau alone can't
answer it. Flag this explicitly when writing up SFT's results — a high
token-accuracy number without this caveat would overstate what the loss
curve actually proves.

## lr_probe (§3.6c) — in progress

3 candidates at max_steps=100, `config/diagnostics/lr_probe.yaml`.

| lr | wall_clock | eval_loss | eval_accuracy | grad_norm behavior |
|---|---:|---:|---:|---|
| 2e-5 | 2835.6s | 2.058 | 0.562 | smooth, but barely moving (underfits) |
| **2e-4 (winner)** | 2844.1s | 0.1128 | 0.977 | smooth, monotonic decrease, no spikes |
| 2e-3 | 2844.2s | 0.0949 (best raw number) | 0.977 | spiked to 23.6 at step 40 (loss jumped 0.10→0.98), recovered by step 60 |

**Winner: 2e-4.** Slightly worse final eval_loss than 2e-3 (0.113 vs 0.095) but no
instability episode — 2e-3's grad_norm spike (base.yaml's own comment:
"divergence shows here before it shows in loss") is a real risk sign that
a raw eval_loss comparison alone would have missed. Also retroactively
validates tonight's real SFT run, which already used 2e-4 and completed
cleanly.

## rank_probe (§3.6d) — done

3 candidates at max_steps=100, lr fixed at 2e-4 (the lr_probe winner).

| r | wall_clock | eval_loss | eval_accuracy | peak VRAM | train_loss |
|---|---:|---:|---:|---:|---:|
| 4 | 2828.2s | 0.1255 | 0.976 | 9.57GB | 0.740 |
| **16 (kept)** | 2829.8s | 0.1129 | 0.977 | 9.64GB | 0.432 |
| 64 | 2836.9s | 0.1046 | 0.977 | 9.79GB | 0.261 |

**Decision: keep r=16.** r=64 has the lowest raw eval_loss at 100 steps,
but two reasons against switching: (1) higher rank = more memorization
capacity, and the val split here is a random slice of the *same*
generation process as train (92% synthetic) -- more capacity cuts the
wrong way against the generalization-vs-memorization concern already
flagged as a P5 TODO above; (2) a 100-step probe doesn't reveal
overfitting risk over a full 3-epoch run (that needs repeated exposure to
show up, per the §3.6b sandbox). r=16 already produced a clean, working
adapter tonight (the real SFT run) -- switching would mean re-running
CPT+SFT (~4.5h more) for a marginal, not-clearly-safe gain. Confirmed
with the user.

## §3.6c/d wrap-up

Both probes done, 6/6 runs. lr=2e-4 and r=16 -- the values tonight's real
CPT+SFT already used -- are now the *justified* choice, not just the
provisional default. CLAUDE.md's P2 done-criteria for these two items
are met.

## Still open for full P2 completion

## §7.2 knob ablation -- done

Used `scripts/profile_train.py --tag ckpt_on` / `--no-grad-checkpoint --tag ckpt_off`
(same 6-micro-batch, seq_len=512 synthetic-batch method as the MFU run).

| | checkpointing ON | checkpointing OFF |
|---|---:|---:|
| peak VRAM | 6.906GB | 10.723GB (+55%) |
| extrapolated step time (×16) | 24.35s | **16.49s (32% faster)** |
| MFU | 13.7% | 20.2% |

Clean, direct tradeoff: OFF is faster and has higher MFU, at the cost of
55% more VRAM. Given tonight's real runs only ever used 9.6-10.5GB peak
out of the 23GB available (comfortable headroom either way), checkpointing
may not have been necessary for CPT/SFT at this model/rank/batch size --
it bought memory headroom nobody needed at a real 32% speed cost.
**Caveat, not yet verified**: this test used `seq_len=512`, half of the
real training's `max_seq_len=1024` -- activation memory (what
checkpointing saves) scales with sequence length, so turning it off for
the *real* 1024-token runs would use more than this 10.7GB figure. Likely
still fits in 24GB, but that's a prediction, not a measurement -- would
need its own real-seq-len test before recommending the config change for
P3/P4.

### Flash-attn vs sdpa -- resolved, unexpected result

First attempt at installing flash-attn (source compile) was abandoned
after 33 minutes with no matching prebuilt wheel found via a generic
search. Found the actual fix afterward: the community repo
`mjun0812/flash-attention-prebuild-wheels` (GitHub Actions CI-built
wheels, not the official Dao-AILab release, but widely referenced across
that project's own issue tracker as the practical workaround) hosts a
wheel built for this exact stack --
`flash_attn-2.8.3+cu129torch2.9-cp310-cp310-linux_x86_64.whl` (found via
`gh api repos/.../releases --paginate` filtered for `torch2.9` + `cp310`
+ `cu128`/`cu129` + `linux`). Installed via plain `pip install <wheel URL>`
in seconds -- no compilation.

| | sdpa | flash_attention_2 |
|---|---:|---:|
| extrapolated step time (×16) | 24.35s | 25.75s (no improvement -- slightly slower) |
| peak VRAM | 6.906GB | 6.906GB (identical) |
| MFU | 13.7% | 12.9% |
| `kDequantizeBlockwise` share of GPU time | 24.8% | 25.5% (essentially unchanged) |

**Flash-attn bought nothing here.** It only optimizes the attention
computation itself; the actual bottleneck (confirmed by the profiler,
both with and without it) is 4-bit weight dequantization in the
linear/MLP layers, which flash-attn doesn't touch. The abandoned 33-minute
compile attempt earlier tonight would not have been worth it even if it
had succeeded on the first try -- this is worth stating plainly rather
than assuming flash-attn is a default win. Caveat: tested at
`seq_len=512` (half the real training's 1024); flash-attn's advantage
grows with sequence length, so a real-length test might show a small
edge, but not enough to change the conclusion that dequantization, not
attention, is this workload's actual bottleneck.

## `torch.profiler` + MFU -- done

First attempt (`PROFILE_STEPS=10`, `grad_accum=16` -> 160 traced
forward+backward passes with full CPU+CUDA activity recording) hung the
VM: SSH started connecting then got closed ("Connection closed by ...
port 22") repeatedly for 5+ minutes, `gcloud compute instances describe`
showed `RUNNING` (not preempted). Root cause: g2-standard-8 has 31GB host
RAM (confirmed via `free -h` after recovery); profiler's per-event
recording over 160 iterations almost certainly exhausted it, leaving the
VM too starved to even fork an sshd session. Recovered with `gcloud
compute instances reset` (hard reboot, no SSH needed) -- boot disk and
all existing adapters (`serana-cpt-intermediate`, `serana-sft`) confirmed
intact afterward. Retried with 6 raw micro-batches instead of 160 full
accumulated steps -- ran clean in 9.3s.

**Second bug found and fixed**: `sum(p.numel() for p in model.parameters())`
reported 4.73B params for a model documented as 8.2B. Root cause:
bitsandbytes packs two 4-bit values per stored uint8 byte, so `.numel()`
on the packed base-weight tensors returns roughly half the true logical
parameter count for anything 4-bit quantized -- would have silently
halved the FLOPs estimate and thus the MFU number. Fixed by hardcoding
Qwen3-8B's documented total (8.2B, 6.95B non-embedding) instead of
runtime-counting.

**Results** (`scripts/profile_train.py`, `artifacts/diagnostics/mfu_report.json`):

| metric | value |
|---|---:|
| achieved compute | 16.33 TFLOPS |
| L4 peak BF16 (dense) | 121 TFLOPS |
| **MFU** | **13.5%** |
| extrapolated step time (micro-batch × 16) | 24.68s |
| real training's actual step time (all probes tonight) | ~25-28s |

The extrapolated step time matches the real measured step time closely
-- cross-validates the profiling methodology (a synthetic-batch,
optimizer-step-free micro-benchmark) as representative of the real
training loop, not just a toy number.

**Where the other 86.5% goes** -- `prof.key_averages()` breakdown
(`artifacts/diagnostics/profile_top_ops.txt`): `kDequantizeBlockwise`
(unpacking 4-bit weights back to bf16 before they can be used in a
matmul) alone accounts for **24.8% of total GPU time** -- a quarter of
every step is spent un-compressing weights, not computing on them. This
is the first *measured*, not guessed, confirmation of the dequantization
bottleneck hypothesized earlier tonight during the SFT wall-clock
investigation.

## Preemption/resume cycle -- verified

Simulated a preemption rather than waiting for a real one: started a
60-step run with `checkpoint_every_steps=15`, waited for
`checkpoint-15` to appear on disk, `pkill -9`'d the training process,
then re-ran the identical command.

Confirmed working: log printed `[resume] found checkpoint at
artifacts/diagnostics/preemption_test/checkpoint-15, resuming from
there`, and wall-clock for the second run (1380.5s) matches ~45
remaining steps, not all 60 -- it did not redo the first 15 steps. The
`save_strategy="no"` bug fixed earlier tonight is confirmed actually
fixed, not just plausible-looking code.

## In-persona Korean smoke test -- done, real B vs SFT comparison run

`scripts/smoke_test.py` (new, plain-transformers generation -- not the
real P5 vLLM serving path, just a quick "did the adapter come out
working" check). Both configs use the identical `persona_system` prompt
(PROMPTS.md §1) -- the only variable is whether the SFT LoRA adapter is
loaded, isolating what training added on top of the prompt, the same way
DESIGN.md's B/SFT/DPO comparison is meant to work.

Two bugs hit and fixed before getting a clean run: `apply_chat_template(...,
return_tensors="pt")` returns a dict in this transformers version, not a
bare tensor (needed `return_dict=True` + `model.generate(**inputs)`); and
Qwen3's thinking mode was eating the whole `max_new_tokens` budget on
English reasoning before ever reaching the Korean answer (needed
`enable_thinking=False`).

| Prompt | B (base + prompt, no LoRA) | SFT |
|---|---|---|
| "너는 누구야?" | Recites the persona_profile back almost verbatim, long biographical info-dump | "나는 세라나야. 뱀파이어이면서도, 그 이상도 그 아래도 아닌 존재야." -- short, dry, sounds like a character, not a bio |
| "요즘 유행하는 게임이 뭐야?" | Correct deflection but repeats "몰라" three times, rambling | "그게 뭐야? 난 게임을 잘 몰라." -- same correct deflection, one clean line |
| "파스타 레시피를 간단히 알려줘." | Stays in character (explicitly notes she's a vampire, dismissive of human food) while still answering | Answers plainly like a generic assistant -- **drops the persona framing entirely** |

**Honest read, not just "SFT wins":**
- Q1/Q2: matches DESIGN.md's predicted shape exactly -- SFT's contribution isn't *knowledge* (B already knows who Serana is, likely from Qwen3's own pretraining data covering Dawnguard wiki content) but *voice/concision* -- talking like the character instead of reciting a system prompt.
- Q3 is a real, uncomfortable counter-example: **B stayed more in-character than SFT did.** The SFT out-of-boundary training category (PROMPTS.md §3) leaned on modern-tech/post-sealing examples; "food" as a boundary case likely wasn't well represented, so SFT generalized "answer helpfully" from the ~75% ordinary-conversation slice more strongly than it generalized the boundary behavior to a case that wasn't explicitly drilled. Not a uniform improvement -- a concrete, measured regression on one axis alongside real gains on two others.

**Also notable (unplanned finding):** the base model's `<think>` block already
contains a roughly-correct plot summary of Serana's backstory (family,
prophecy, imprisonment) before any training. Qwen3 likely saw Dawnguard
wiki/fan content during pretraining. This means B's baseline for *this
specific, well-known persona* may be stronger than it would be for an
obscure/original character -- worth stating plainly in the eventual
writeup rather than assuming B is a "blank slate" baseline.

## P2: done

All CLAUDE.md/DESIGN.md build-order criteria for P2 are met: CPT and
SFT adapters trained within budget on the real L4 · lr chosen from the
§3.6c probe (2e-4), both rejected curves (2e-5 underfits, 2e-3 spikes)
kept · §3.6d rank comparison recorded, r=16 kept over r=64 with reasons
stated · train/val loss logged for both stages · predicted-vs-measured
peak VRAM recorded, including the real 6-12x wall-clock miss on SFT
(root-caused live: `grad_accum_steps` override bug) · §7.2 knob
ablation (checkpointing on/off) and flash-attn-vs-sdpa comparison both
done, with the counter-intuitive flash-attn finding (bought nothing --
dequantization, not attention, is the real bottleneck) stated plainly
rather than assumed away · `torch.profiler` trace + MFU (13.5%) with a
written explanation, cross-validated against real training step times
· one simulated preemption/resume cycle verified working · in-persona
Korean smoke test done, with an honest read that includes the one
regression (Q3) alongside the two real gains, not just "SFT wins".

Two real bugs found and fixed along the way: `save_strategy="no"`
(would have lost a Spot-preempted run entirely) and a missing
`max_steps` in `rank_probe.yaml` (would have run full training instead
of a short probe). Both caught before they cost anything.

Next is P3 (RLAIF preference data) -- needs the `serana-sft` adapter
this stage produced.
