# RUNBOOK.md

How to provision a GCP L4 VM, get the repo and adapters onto it, run a
training or generation job, and tear it down — without leaving the GPU
billing on.

- **Why** each command / gotcha exists → `TROUBLESHOOTING.md` (sections A
  "GPU provisioning" and B "VM environment"). This file is the ordered
  procedure; that file is the incident history.
- Budget and provisioning policy → `DESIGN.md §9`.
- Commands the project has actually used → `LEARNING.md §14` (gitignored
  study notes; this file is the committed operational version).

Rule of thumb from `DESIGN.md §9.3`: **training and batch generation on
Spot; latency measurement on on-demand; the VM is stopped whenever it is
not actively computing.**

---

## 0. Prerequisites (one-time, on your laptop)

```bash
gcloud auth login                 # browser login to the Google account
gcloud config set project project-9a113a17-2211-4c9a-ae7
gcloud components install beta    # some IAP flags live under `beta`
```

`gcloud compute ssh` needs an SSH key; it generates one on first use and
uploads it to the project. If SSH later fails with a key error, delete
`~/.ssh/google_compute_engine*` and let it regenerate.

You do **not** put `OPENAI_API_KEY` on the VM, ever — judging/scoring is
local-only (`CLAUDE.md` cost discipline: "CPU/API/local work never runs
on the GPU instance"). The VM only needs a Hugging Face token, and only
if a model or adapter repo is private.

---

## 1. Project values (fill these into the `<...>` below)

| placeholder | value |
|---|---|
| `<PROJECT>` | `project-9a113a17-2211-4c9a-ae7` |
| `<ZONE>` | `asia-northeast3-b` (fallback `-a`; `-c` has **no** L4) |
| `<BUCKET>` | `gs://serana-post-training-ann10266` |
| `<MACHINE>` | `g2-standard-8` (1× L4 24 GB, 31 GB host RAM) |
| `<IMAGE_FAMILY>` | `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` |
| `<IMAGE_PROJECT>` | `deeplearning-platform-release` |
| repo | `https://github.com/MachuEngine/serana-post-training.git` |

Billing is in **KRW**. Budget alerts are set at 50/80/100 % of ₩400,000
(≈ the $300 credit). Check spend: **Console → Billing → Reports**.

---

## 2. VM lifecycle

### 2.1 See what exists

```bash
gcloud compute instances list
```

- A row with `STATUS: TERMINATED` is a **stopped** VM — reuse it (§2.2).
  Its disk still holds the repo, the Python env, the base-model cache,
  and any adapters. Reusing saves ~30 min of setup and a 16 GB download.
- No rows → create one (§2.4).

### 2.2 Start a stopped VM

```bash
gcloud compute instances start <VM> --zone=<ZONE>
# GPU billing starts here (~$0.71/hr Spot, ~$0.90/hr on-demand for g2-standard-8)

gcloud compute instances describe <VM> --zone=<ZONE> --format="value(status)"
# wait for RUNNING (1-2 min)
```

### 2.3 Stop a VM (do this the moment a job finishes)

```bash
gcloud compute instances stop <VM> --zone=<ZONE>
```

`stop` halts CPU+GPU billing; the disk stays (small storage cost) so the
next `start` is instant. **Never `delete`** unless you mean to lose the
disk — artifacts must be in `<BUCKET>` first.

### 2.4 Create a fresh VM

On-demand (measurement runs, or when Spot capacity is gone):

```bash
gcloud compute instances create <VM> \
  --zone=<ZONE> \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB
```

Spot (training, batch generation — preemption is expected and fine with
checkpoint/resume):

```bash
gcloud compute instances create <VM> \
  --zone=<ZONE> \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --no-restart-on-failure \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB
```

Line meanings:
- `--accelerator` — attach 1× L4.
- `--maintenance-policy=TERMINATE` — **required** for GPU VMs (GPUs can't
  live-migrate).
- `--instance-termination-action=STOP` (Spot) — on preemption the VM
  stops instead of being deleted, so the disk (and your checkpoint)
  survives. `TROUBLESHOOTING.md A2`.
- `--boot-disk-size=150GB` — 8B base ~16 GB, 14B base ~30 GB, plus
  adapters, plus the pytorch image itself. 150 GB is comfortable; go
  200 GB for the 14B pipeline.
- 250 GB disk for 14B if you keep both 8B and 14B caches.

If creation fails with `ZONE_RESOURCE_POOL_EXHAUSTED` or a stockout:
wait and retry, or try `--zone=asia-northeast3-a`. Do **not** fall back
to a region outside `asia-northeast3` — `kr-west` is a hard data-residency
constraint. `TROUBLESHOOTING.md A1/A2`.

### 2.5 Spot → on-demand (or vice versa)

GCP will not flip `provisioningModel` in place. Delete the instance with
the disk preserved, then recreate on the same disk:

```bash
gcloud compute instances delete <VM> --zone=<ZONE> --keep-disks=boot
gcloud compute instances create <VM> --zone=<ZONE> --disk=name=<VM>,boot=yes \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE   # add --provisioning-model=SPOT for the other direction
```

`TROUBLESHOOTING.md A3`. Data loss: zero.

---

## 3. SSH and file transfer

`--tunnel-through-iap` is **required** on this network — direct port-22
times out (`TROUBLESHOOTING.md A5`). Intermittent `255` / broken-pipe is
not a real failure; retry.

```bash
# interactive shell
gcloud compute ssh <VM> --zone=<ZONE> --tunnel-through-iap

# one-off remote command
gcloud compute ssh <VM> --zone=<ZONE> --tunnel-through-iap --command="nvidia-smi"

# laptop -> VM
gcloud compute scp <local_path> <VM>:~/serana/<path> --zone=<ZONE> --tunnel-through-iap

# VM -> laptop (directories need --recurse)
gcloud compute scp --recurse <VM>:~/serana/artifacts/runs ./artifacts/ --zone=<ZONE> --tunnel-through-iap
```

For anything that should survive the VM being deleted, use GCS instead of
scp:

```bash
# on the VM
gcloud storage cp -r artifacts/lora/<adapter> <BUCKET>/artifacts/lora/
gcloud storage cp artifacts/runs/<file> <BUCKET>/artifacts/runs/
# pull back on the laptop
gcloud storage cp -r <BUCKET>/artifacts/lora/<adapter> artifacts/lora/
```

---

## 4. First-time setup on a fresh VM

Skip this entirely if you reused a stopped VM (§2.2) — it's already done.

```bash
gcloud compute ssh <VM> --zone=<ZONE> --tunnel-through-iap
```

Then on the VM:

```bash
# 1. repo
git clone https://github.com/MachuEngine/serana-post-training.git ~/serana
cd ~/serana
# for P8: git fetch && git checkout p8-eval-v2

# 2. Python env -- the image has Python + CUDA + a system torch already
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # uv AND vllm land here; non-interactive
                                            # SSH does not pick it up (LEARNING.md §14)
uv sync

# 3. known image-bundle conflicts (TROUBLESHOOTING.md B1/B2) -- do these once
sudo pip uninstall -y torchaudio            # peft imports it transitively; ABI-mismatched, unused
pip install -U jinja2                        # image ships 3.0.3; chat templates need >= 3.1.0

# 4. Hugging Face -- only if a model/adapter repo is private
huggingface-cli login                        # paste a read token

# 5. adapters (base model auto-downloads on first use)
mkdir -p artifacts/lora
# from GCS (serana-dpo-v3 is local/GCS only -- never promoted to the Hub):
gcloud storage cp -r <BUCKET>/artifacts/lora/serana-dpo-v3 artifacts/lora/
# serana-sft / serana-dpo are on the Hub and load by id if configured that way,
# or pull them the same way if you keep them on GCS.

# 6. sanity
python3 scripts/gpu_probe.py                 # device props, free VRAM, matmul throughput
```

**Assume the first run of any code path hits 2-3 bugs in a row**
(`TROUBLESHOOTING.md`, recurring pattern). Check `torch.__version__`
after any large `pip install` — vLLM in particular has silently upgraded
torch and broken a compiled extension's ABI (`TROUBLESHOOTING.md B4`).

---

## 5. Running a job on the VM

Long jobs must survive the SSH session dropping:

```bash
cd ~/serana
export PATH="$HOME/.local/bin:$PATH"

nohup python3 -u <script> <args> > run.log 2>&1 < /dev/null &
disown
#   -u            : unbuffered, so `tail -f run.log` is live
#   < /dev/null   : detaches stdin so the nohup'd process can't block the SSH channel
#   disown        : the shell forgets the job, so logout won't SIGHUP it
```

Monitor from a separate SSH command (don't hold the shell open):

```bash
gcloud compute ssh <VM> --zone=<ZONE> --tunnel-through-iap --command="
  cd ~/serana &&
  pgrep -af '<script>' &&
  tail -c 3000 run.log &&
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
"
```

Config overrides without editing files: `--set dotted.path=value`, e.g.
`--set train.attn_implementation=sdpa` (flash-attn is not installed on a
fresh VM — `TROUBLESHOOTING.md B3`), `--set train.max_steps=60`.

---

## 6. Current task — P8 step 3: eval-set v2 generation

Plan: `artifacts/runs/p8_plan.md` step 3. On-demand VM. ~$2, ~1 h.

```bash
# on the VM, server up (bf16 base + all local adapters)
export PATH="$HOME/.local/bin:$PATH"
nohup python3 -u scripts/serve_up.py > serve.log 2>&1 < /dev/null &
disown
# wait until serve.log shows the server listening and /v1/models lists
# Qwen/Qwen3-8B, serana-sft, serana-dpo, serana-dpo-v3

# generate replies for the 4 configs against the v2 eval set
for c in b sft dpo dpo_v3; do
  python3 scripts/generate_eval_replies.py \
    --config config/experiments/$c.yaml --eval-version v2
done
# -> artifacts/runs/raw_eval_v2_{b,sft,dpo,dpo_v3}.json
```

```bash
# back on the laptop: pull the raw files, stop the VM
gcloud compute scp --recurse \
  <VM>:~/serana/artifacts/runs ./artifacts/ --zone=<ZONE> --tunnel-through-iap
gcloud compute instances stop <VM> --zone=<ZONE>

# score locally with gpt-4o-mini (config/eval.yaml judge_v2), ~$0.25
uv run scripts/score_eval_replies.py --config b sft dpo dpo_v3 --eval-version v2

# gate
uv run scripts/paired_analysis.py --eval-version v2
uv run scripts/make_results_tables.py --eval-version v2
```

**Gate:** open `artifacts/runs/paired_analysis_v2.md`. If the B→SFT
paired PCS diff CI **excludes zero** → proceed to step 4. If it still
brackets zero → stop, write `artifacts/runs/p8_gate_failed.md`.

Also do the step-2 residual: hand-read ~20 v2 items against their
`gpt-4o-mini` scores in `eval_v2_*.json` before trusting the table.

---

## 7. Current task — P8 step 4: 14B pipeline

Plan: `artifacts/runs/p8_plan.md` step 4. Only if step 3's gate passes.
Spot VM for training, on-demand for the eval generation. ~$12-15, ~3 days
with preemptions. **200 GB boot disk.**

```bash
# one line: point the whole pipeline at 14B
#   config/base.yaml  model.base_id: Qwen/Qwen3-8B  ->  Qwen/Qwen3-14B
# then the mirrored train configs (config/train_runs/*_14b.yaml) that
# p8_plan.md step 4 calls for.

# CPT -> SFT -> DPO, each continuing the previous adapter, on Spot:
nohup python3 -u scripts/train.py --config config/train_runs/cpt_14b.yaml \
  > cpt.log 2>&1 < /dev/null & disown
# ... wait, verify, then sft_14b.yaml, then regenerate preference pairs
# from the 14B SFT adapter, then dpo_14b.yaml.

# predicted peak VRAM ~15-16 GB (measure and log the gap if > 20%,
# CLAUDE.md working principle #5). eval-step OOM guard:
#   --set train.per_device_eval_batch_size=1
```

Checkpoints go to `<BUCKET>/artifacts/lora` (`train.checkpoint_uri` in
`config/base.yaml`) so a preemption resumes cleanly. After each stage:
copy the adapter to GCS and pull it to the laptop before stopping the VM.

Then eval on v2 exactly as §6, with the 14B config names, into
`results_quality_v2.md` (the 14B rows) and the hardware table (8B vs 14B
on the same L4).

---

## 8. Teardown checklist (every session)

1. `artifacts/` you care about copied to `<BUCKET>` **and** pulled to the
   laptop.
2. `gcloud compute instances stop <VM> --zone=<ZONE>` — confirm with
   `instances list` that `STATUS` is `TERMINATED`.
3. Any `nohup` job actually finished (not still burning GPU) — check
   `run.log` / `pgrep` before stopping.
4. Console → Billing → Reports: sanity-check the day's spend against the
   estimate in `p8_plan.md`.

Leaving a `g2-standard-8` running overnight by accident is ~$15-22. The
stop step is the one that matters.

---

## 9. GKE serving (P9)

The VM path above still works and is what every P2-P8 number came from.
This is the containerised version of the same server: one image, one
cluster, `deploy/` owns the lifecycle the way `scripts/` owns the VM's.

Why at all, stated plainly: at one replica with no traffic, GKE buys
nothing functionally. What it buys is a pinned, reproducible image, a
GPU that bills only while a pod needs it, and the operational surface
(Workload Identity, probes, init containers) that a VM plus `nohup`
never exercises.

### 9.1 One command up, one down

```bash
deploy/cluster_up.sh                 # cluster + GPU pool + identity + image + manifests
deploy/cluster_up.sh <IMAGE>         # ...reusing an image that already exists
kubectl port-forward svc/serana-vllm 8000:8000    # reach it locally
deploy/cluster_down.sh               # GPU billing stops when this returns
```

`port-forward` is what makes `config/base.yaml`'s
`serving.base_url: http://localhost:8000/v1` correct unchanged, so every
existing script (`scripts/generate_eval_replies.py`,
`throughput_sweep.py`, `parity_check.py`) runs against the cluster with
no edits. The Service is deliberately ClusterIP: nothing about this
model should sit on a public address.

### 9.2 Prerequisites that are not obvious

| | |
|---|---|
| `gke-gcloud-auth-plugin` | **`kubectl` cannot talk to GKE without it** and it is not part of the Homebrew gcloud cask. `gcloud components install gke-gcloud-auth-plugin`, then symlink it onto PATH: `ln -sf /opt/homebrew/share/google-cloud-sdk/bin/gke-gcloud-auth-plugin /opt/homebrew/bin/`. The failure is `executable gke-gcloud-auth-plugin not found`, at the first `kubectl` call, after the cluster has already been created and started billing. |
| Cloud Build service account | Projects created after ~2024 do not grant the Compute Engine default service account the roles a build needs. `deploy/build_push.sh` grants `roles/cloudbuild.builds.builder` itself. Without it the build fails with a **403 naming Cloud Storage**, not IAM — which sends you to the wrong page entirely. |
| L4 quota | 1 in `asia-northeast3` (`NVIDIA_L4_GPUS`), which is exactly one node. Check before creating: `gcloud compute regions describe asia-northeast3 --format="json(quotas)"`. |

### 9.3 Two failures worth expecting

**Workload Identity does not take effect immediately.** The binding is
correct the moment `cluster_up.sh` prints it, but the first pod scheduled
right after still fails with:

```
ERROR: gcloud crashed (MetadataServerException): The request is rejected.
Please check if the metadata server is concealed.
```

That message points at metadata concealment, which is not the problem —
the binding simply has not propagated. Verify rather than guess, with a
throwaway pod using the same service account:

```bash
kubectl run wi-test --restart=Never --image=gcr.io/google.com/cloudsdktool/google-cloud-cli:slim \
  --overrides='{"spec":{"serviceAccountName":"serana"}}' \
  --command -- bash -c 'curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email'
kubectl logs wi-test        # should print serana-serving@<PROJECT>.iam.gserviceaccount.com
```

Once that prints the right account, `kubectl delete pod -l app=serana-vllm`
and the recreated pod works. Measured here: it needed one delete, roughly
two minutes after the binding was created.

**A GPU node reports `Ready` before its GPU is usable.** GKE installs the
NVIDIA driver with a DaemonSet that runs *after* the node joins, so for a
couple of minutes `kubectl get nodes` shows `Ready` while the pod sits in
`Pending`:

```
Warning  FailedScheduling  0/2 nodes are available: 1 Insufficient nvidia.com/gpu, ...
```

Nothing is wrong; the scheduler is telling the truth. Check the node's
allocatable GPUs rather than its status:

```bash
kubectl get node -l cloud.google.com/gke-accelerator=nvidia-l4 \
  -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # empty until ready
kubectl get pods -n kube-system | grep nvidia                    # Init:1/4 while installing
```

It resolves itself and the pod schedules; no action needed beyond waiting.
Worth knowing so the first minutes of a fresh cluster are not spent
debugging a non-problem.

**The startup probe is the thing that makes or breaks the first deploy.**
The 16 GB base model downloads into an emptyDir and then loads onto the
card before `/health` answers. `deployment.yaml` allows 20 minutes
(`periodSeconds: 15 x failureThreshold: 80`). With only a readiness or
liveness probe at default settings, Kubernetes kills the pod long before
the server is ever ready, and the symptom — a crash loop with no error in
the container log — looks nothing like its cause.

### 9.4 Timings measured on the first run

| step | predicted | measured |
|---|---|---|
| image build (Cloud Build, 10 GB+ base) | 8-15 min | **see `artifacts/runs/p9_progress.md`** |
| cluster + GPU node pool | 10-18 min | ~13 min |
| adapters from GCS (init container) | - | seconds, 269 MiB/s |
| pod start to `/health` | 10-15 min | see the progress log |

---

## 10. Quick reference

```bash
gcloud compute instances list
gcloud compute instances start  <VM> --zone=<ZONE>
gcloud compute instances stop   <VM> --zone=<ZONE>
gcloud compute instances reset  <VM> --zone=<ZONE>          # hard reboot, SSH dead (TROUBLESHOOTING.md A4)
gcloud compute ssh  <VM> --zone=<ZONE> --tunnel-through-iap
gcloud compute scp  --recurse <VM>:~/serana/<p> ./<p> --zone=<ZONE> --tunnel-through-iap
gcloud storage cp -r <local> <BUCKET>/<path>
gcloud compute regions describe asia-northeast3 --format="json(quotas)"   # NVIDIA_L4_GPUS
```
