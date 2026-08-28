# P4 progress log

DPO (DPOTrainer, continuing the SFT adapter), DESIGN.md §3.4 / CLAUDE.md
build order. Same style as `p1_progress.md`/`p2_progress.md`/`p3_progress.md`.

## Pre-run prediction (CLAUDE.md working principle #5)

Inputs: `data/ko/prefs_1k.jsonl` (837 pairs, val_split 5% -> train ~795),
`per_device_batch_size=1`, `grad_accum_steps=8` (dpo.yaml) -> effective
batch 8 -> ~100 steps for 1 epoch. `lora_r=16`, `beta=0.1`, `lr=5e-6`,
`max_seq_len=1024`, `ref_model=None` (adapter-toggle reference, no
second resident model).

- **Predicted peak VRAM: ~13-16GB.** Base: SFT measured 9.64GB at
  `r=16`/`seq_len=1024`/`batch=1` (fixed weight+optimizer overhead
  ~4-5GB, activation portion ~4.6-5.6GB). DPO concatenates chosen+
  rejected per micro-batch (policy forward acts like batch=2), so
  activation portion roughly doubles -> ~4-5GB fixed + ~9-11GB
  activation = 13-16GB. Comfortably under 24GB.
- **Predicted step time: ~34-38s/step, ~100 steps -> ~55-65 min wall
  clock.** SFT micro-batch ~1.56-1.75s (25-28s / grad_accum 16). DPO
  per micro-batch = policy fwd+bwd on batch=2 (~6 units, using a 1:2
  fwd:bwd cost ratio) + reference fwd-only on batch=2, no backward
  (~2 units) = 8 units vs SFT's 3-unit baseline -> ~2.67x ->
  4.2-4.7s/micro-batch x grad_accum 8 ~= 34-38s/step.
- **Predicted cost: ~$0.30-0.35** for one run (spot g2-standard-8 ~=
  $0.33/hr, derived from DESIGN.md §9's P3 budget line ~6h/~$2). Well
  inside DESIGN.md §9's P4 line (~12h/~$4, already includes headroom
  for one beta retry).

## Blocked: zone stockout

VM `serana-p2-train` (Spot g2-standard-8, `asia-northeast3-b`,
`TERMINATED` but disk intact via `--instance-termination-action=STOP`)
would not start:

```
gcloud compute instances start serana-p2-train --zone=asia-northeast3-b
```
-> `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / reason: `stockout` --
no g2-standard-8 + 1x L4 capacity available in `asia-northeast3-b`
right now. Retried 3x (2026-08-27), same result each time. Not a quota
or billing problem (those were already resolved in P0 -- see
[[gcp-project-provisioning-serana]]) and not a config/prediction
failure -- genuine capacity unavailability on GCP's side.

**User decision: wait and retry later**, rather than migrating to
another zone (`asia-northeast3-a`/`-c` would need a disk snapshot +
new VM + re-verifying the installed deps/adapters, per CLAUDE.md
surgical-change discipline this is more machinery than the problem
needs unless stockout persists) or switching to on-demand (would
violate DESIGN.md §9's "training runs on Spot" principle without
checking in first).

**Update 2026-08-27: stockout cleared, VM started, run attempted.**

VM code lives at `~/serana` on the VM (plain directory, not git --
this repo has no push-based sync mechanism; `data/ko/prefs_1k.jsonl`
and any local code changes must be scp'd over manually each session).
Adapters `serana-cpt-intermediate`/`serana-sft` confirmed intact on
the VM's disk. Local `config/`, `src/`, `scripts/`, `data/` synced via
a tarball + scp + extract (careful not to touch `artifacts/lora/`).

**Two real bugs found and fixed in `src/finetune/train.py`'s DPO path**
(the code's own comment flagged this path as "not exercised" before
P4 -- both surfaced on the very first real run):

1. **Conversational format mismatch.** `prompt` was built as a chat-
   message list but `chosen`/`rejected` were passed as raw strings.
   TRL's `DPOTrainer` detects list-valued `prompt` as conversational
   and does `example["prompt"] + example["chosen"]` (list concat) --
   crashed with `TypeError: can only concatenate list (not "str") to
   list`. Fixed: wrap `chosen`/`rejected` as single-turn message lists
   (`[{"role": "assistant", "content": ...}]`) too.
2. **`dpo.yaml`'s hyperparameter overrides never applied.** `common_args`
   read `learning_rate`/`grad_accum_steps`/`num_epochs` from the
   top-level `train.*` keys, but `dpo.yaml` nests its DPO-specific
   values under `train.dpo.*` (a different key path, by design --
   see the file's own header comment). Without an explicit override,
   DPO silently trained at SFT/base's `lr=2e-4` (40x DPO's intended
   `5e-6`) for 3 epochs instead of 1 -- exactly what DESIGN.md §3.4
   warned against ("DPO overfits preference sets quickly"). Symptom
   that caught it: step count showed `0/150`, not the predicted `~100`.
   Fixed: `dpo_args` now explicitly overrides those three keys from
   `dpo_cfg` before constructing `DPOConfig`.

After fix #2, re-launched -- confirmed `0/100` steps (matches
prediction), peak VRAM 15,076 MiB (within the 13-16GB prediction),
GPU util 98%. 25 steps in (~11 min, ~26.8s/step, a bit faster than the
predicted 34-38s/step), **hit a third real bug**:

3. **Eval OOM at `eval_steps=25`.** `common_args` set
   `per_device_train_batch_size` explicitly but never
   `per_device_eval_batch_size`, which silently defaulted to HF's `8`.
   DPO's eval forward pass needs (chosen+rejected) x (policy+reference)
   = 4 full-vocab (~152k) logits tensors per example; at batch=8 that
   tried to allocate 8.51 GiB on top of ~19GB already in use, past the
   ~22GB usable ceiling. Training steps themselves were fine at 15GB
   (matching the pre-run prediction) -- only eval spiked. Fixed:
   `per_device_eval_batch_size` now explicitly tracks
   `per_device_batch_size`, for all three methods (cpt/sft/dpo), not
   just DPO -- SFT/CPT likely got lucky on memory headroom rather than
   being immune to the same bug.

Re-launched again after fix #3. No checkpoint had been written yet
either time (`save_steps=50`, both crashes happened before step 50),
so each restart was a clean re-run from step 0, not a resume -- cheap.

**Then: real Spot preemption**, confirmed via
`gcloud compute operations list` (`compute.instances.preempted` at
2026-08-27T08:00:49-07:00) -- not a bug, not our config, GCP reclaimed
the L4. `--instance-termination-action=STOP` kept the boot disk (and
`serana-sft`/`serana-cpt-intermediate`) intact, matching the design.
VM went to `TERMINATED` as expected.

**Immediate restart attempt hit the same `ZONE_RESOURCE_POOL_EXHAUSTED`
stockout as the original P4 blocker** -- plausible, since preemption
usually means GCP wanted that exact capacity back, so it's likely
still in demand right after. Unknown whether training had gotten past
step 50 (a checkpoint) before the preemption; not yet confirmed since
the VM won't start to check.

**Status: blocked on stockout again, immediately after a real
preemption.** Next session: retry
```
gcloud compute instances start serana-p2-train --zone=asia-northeast3-b --project=project-9a113a17-2211-4c9a-ae7
```
(this session's SSH note: direct port-22 SSH timed out from this
network -- `--tunnel-through-iap` on both `gcloud compute ssh` and
`gcloud compute scp` worked reliably instead; occasional transient
`255`/broken-pipe errors on individual calls were not real failures,
just retry). Once up, first check
`ls ~/serana/artifacts/lora/serana-dpo/` for a `checkpoint-*` dir
before assuming a fresh run is needed -- `run()`'s resume logic
(`get_last_checkpoint`) picks it up automatically if one exists. Then:
```
cd ~/serana && nohup python3 scripts/train.py --config config/train_runs/dpo.yaml > ~/serana/dpo_train.log 2>&1 &
```
Local `src/finetune/train.py` now has all three fixes above --
confirm the VM's copy matches (`md5`/`diff`) before trusting a resume,
since the VM's code is a manual copy, not git-tracked. If stockout
persists across several more sessions, revisit the zone-migration
option (still same `asia-northeast3` region, kr-west constraint
intact).

**Update 2026-08-28: zone migration attempted, region-wide stockout confirmed.**

Retried `asia-northeast3-b` start several more times across the
session (all `ZONE_RESOURCE_POOL_EXHAUSTED`). User approved migrating
to `asia-northeast3-a` (the only other zone in-region with L4 hardware
at all -- `asia-northeast3-c` has no L4 accelerator type, confirmed via
`gcloud compute accelerator-types list`).

Migration performed:
1. `gcloud compute disks snapshot serana-p2-train --zone=asia-northeast3-b --snapshot-names=serana-p2-train-migrate-a` -- done, disk was 150GB.
2. `gcloud compute disks create serana-p4-train-a --zone=asia-northeast3-a --source-snapshot=serana-p2-train-migrate-a --type=pd-balanced` -- done, disk exists in zone a now (adapters/deps intact, unverified by boot yet).
3. `gcloud compute instances create serana-p4-train --zone=asia-northeast3-a ...` (machine-type g2-standard-8, accelerator nvidia-l4:1, spot, instance-termination-action=STOP, same service account/scopes as the original VM, disk=serana-p4-train-a as boot) -- **also hit `ZONE_RESOURCE_POOL_EXHAUSTED` in zone a.**

**Conclusion: this is a region-wide L4 stockout, not zone-b-specific.**
Migrating zones did not help. `asia-northeast3-c` was never a candidate
(no L4 SKU there at all) -- the region genuinely has only 2 usable
zones for this workload, and both are out of stock together.

**Leftover state to clean up eventually:**
- `serana-p4-train-a` disk (150GB, `pd-balanced`) sits idle in zone a,
  costs storage ($ per GB-month) with no VM using it yet. Not deleted
  yet -- worth reusing if zone a clears up (`gcloud compute instances
  create serana-p4-train --zone=asia-northeast3-a ...` -- exact
  command below -- would attach it directly, no need to re-snapshot).
- `serana-p2-train-migrate-a` snapshot -- cheap, keep as a recreate
  source regardless of which zone ends up with capacity.
- Original `serana-p2-train` VM (zone b, `TERMINATED`) and its disk
  are untouched and still the primary path if zone b clears first.

**Status: blocked on a region-wide GCP L4 stockout affecting both
usable zones (`a` and `b`) in `asia-northeast3`.** Next session: retry
both
```
gcloud compute instances start serana-p2-train --zone=asia-northeast3-b --project=project-9a113a17-2211-4c9a-ae7
```
and (if b still fails)
```
gcloud compute instances create serana-p4-train --zone=asia-northeast3-a --project=project-9a113a17-2211-4c9a-ae7 \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --disk=name=serana-p4-train-a,boot=yes,auto-delete=no \
  --provisioning-model=SPOT --instance-termination-action=STOP --no-restart-on-failure --maintenance-policy=TERMINATE \
  --network=default --subnet=default \
  --service-account=65588351586-compute@developer.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/pubsub,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append
```
Whichever comes up first wins -- no need to keep trying the other
after one succeeds. If this drags on for many sessions, worth asking
the user whether an out-of-region fallback (breaking the kr-west
convenience constraint, not the hard data-residency one -- confirm
which this actually is before proposing) or on-demand (not Spot, cost
tradeoff, breaks DESIGN.md §9's "training runs on Spot" default) is
worth discussing explicitly, rather than continuing to retry silently.

## P4: done (2026-08-28)

`asia-northeast3-a` came up on a later retry (region-wide stockout
cleared for zone a specifically -- zone b stayed exhausted). VM
`serana-p4-train` created from the migrated disk, `checkpoint-50` from
the pre-preemption run was present and valid (`trainer_state.json`
confirmed `global_step: 50`), `run()`'s resume logic picked it up
automatically -- no wasted compute from the preemption beyond the ~22
min it took to reach step 50 the first time.

**Resumed run finished clean**, no OOM at any of the eval checkpoints
(25/50/75/100 -- the eval-batch-size fix held). Full run report at
`artifacts/lora/serana-dpo/run_report.json`.

| | predicted | measured |
|---|---:|---:|
| peak VRAM | 13-16GB | **~15GB** (nvidia-smi, live); `torch.cuda.max_memory_allocated()` in `run_report.json` reports 9.956GB for just the resumed segment -- narrower metric (PyTorch-tracked allocations only, resets per process launch), not the OOM-relevant figure. nvidia-smi's ~15GB is the one that matches the pre-run prediction and the one that mattered for the eval-time OOM debugging. |
| step time | 34-38s/step | ~26-27s/step (a bit faster than predicted, consistent across both the original and resumed segments) |
| total wall clock (both segments) | ~55-65 min | ~22 min (to step 50, pre-preemption) + 24.1 min (step 50-100, resumed) = **~46 min** |
| cost | ~$0.30-0.35 | ~46 min of actual GPU compute at spot g2-standard-8 rate (~$0.33/hr) ≈ **~$0.25**, plus the zone-migration overhead (snapshot + idle disk time) -- all well inside DESIGN.md §9's P4 line (~12h/~$4) |

**Final metrics** (`artifacts/lora/serana-dpo/run_report.json`):
`final_train_loss: 0.3467`, final eval: `eval_loss 0.6912`,
`eval_rewards/accuracies 0.5952`, `eval_rewards/margins 0.004224`.

**Honest read, not just "trains successfully":** per-step DPO loss
hovered near ln(2)≈0.693 for essentially the entire run (both before
and after the preemption), and reward accuracy (59.5%) is only
modestly above chance (50%) with a small margin (0.0042). This is
weak evidence that DPO differentiated chosen/rejected only mildly in
100 steps / 1 epoch over 795 pairs -- not a strong preference signal,
though not degenerate either (no NaN, no divergent loss, grad_norm
stayed in the 4-9 range throughout).

**Smoke test** (`scripts/smoke_test.py --adapter artifacts/lora/serana-dpo`,
same 3 prompts used for P2's B-vs-SFT comparison): no degeneration, no
repetition, no length blowup -- passes the P4 done-criterion.
Comparing to P2's SFT smoke-test output:
- Q1 ("너는 누구야?"): byte-identical to SFT's answer.
- Q2 (games, out-of-boundary): DPO added one extra in-character clause
  ("봉인되기 전에 하던 게임이랑은 많이 달라졌겠지") -- a small, plausible
  improvement.
- Q3 (pasta recipe): **DPO did not fix** the regression P2 flagged for
  SFT -- still answers plainly like a generic assistant, drops the
  persona framing. Same failure mode carried through unchanged.

Consistent with the weak reward-margin signal above: DPO nudged
behavior slightly, didn't transform it. **The real test is P5's
eval-set comparison** (PCS, style similarity, knowledge-boundary
accuracy across B/SFT/DPO) -- these smoke-test/training-metric
observations are early signal, not the final verdict. Per CLAUDE.md:
"if DPO doesn't beat SFT on any metric, that is a reportable finding."

**Adapter shipped**: `artifacts/lora/serana-dpo/` (adapter_config.json,
adapter_model.safetensors, tokenizer files, run_report.json,
training_args.bin) downloaded to local repo and uploaded to
`gs://serana-post-training-ann10266/artifacts/lora/serana-dpo/` (the
VM's service account only had `devstorage.read_only` scope, so the
upload ran from local, not the VM). `checkpoint-50`/`checkpoint-100`
resume checkpoints were downloaded locally too (`du -sh` ~362MB total
including both) but not uploaded to GCS -- resume state, not the
shipped artifact.

VM `serana-p4-train` (zone a) stopped after upload/verification.

**Cleanup not yet done, flagged for the user rather than done
unprompted (destructive/cost-bearing):**
- Original `serana-p2-train` VM (zone b, `TERMINATED`) + its disk --
  now redundant (its adapters are also on the zone-a disk that
  actually finished the job). Deleting would stop its storage cost.
- `serana-p4-train-a` disk snapshot lineage
  (`serana-p2-train-migrate-a` snapshot + the zone-a disk now attached
  to `serana-p4-train`) -- the working copy, keep.
- Local `checkpoint-50`/`checkpoint-100` dirs under
  `artifacts/lora/serana-dpo/` -- fine to keep locally (small), but
  don't commit large safetensors to git; check `.gitignore` covers
  `artifacts/lora/*/checkpoint-*` before any commit.

Next: P5 (serving + eval) -- this is where DPO's actual gain (or lack
of one) over SFT gets a real answer.
