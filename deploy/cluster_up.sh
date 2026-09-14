#!/usr/bin/env bash
# One command up (DESIGN.md §9.3, extended from the VM to GKE): cluster,
# GPU node pool, Workload Identity, image, manifests.
#
#   deploy/cluster_up.sh            # build the image as part of the run
#   deploy/cluster_up.sh <IMAGE>    # deploy an image that already exists
#
# Every step is describe-or-create, so re-running after a failure resumes
# rather than starting over.
#
# Billing starts at the node-pool step. Rough rates (verify before running,
# per §9.2's standing rule):
#   GKE management  ~$0.10/hr -- covered by the free tier for ONE zonal cluster
#   e2-medium pool  ~$0.04/hr
#   g2-standard-8   ~$0.90/hr  <- the real cost, and only while a pod needs it
#   200 GB pd-balanced ~$0.03/hr
# The GPU pool autoscales to zero, so an idle cluster is ~$0.07/hr.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/env.sh

SA_EMAIL="${GSA}@${PROJECT}.iam.gserviceaccount.com"

echo ">> [1/6] enabling the GKE API (idempotent)"
gcloud services enable container.googleapis.com --project "$PROJECT"

echo ">> [2/6] cluster ${CLUSTER} in ${ZONE}"
if gcloud container clusters describe "$CLUSTER" \
     --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "   already exists, skipping"
else
  # --workload-pool is what lets a pod authenticate to GCS as a Google
  # service account with no key file anywhere in the cluster. It cannot be
  # added later without a cluster update, so it goes in at creation.
  gcloud container clusters create "$CLUSTER" \
    --zone "$ZONE" --project "$PROJECT" \
    --workload-pool="${PROJECT}.svc.id.goog" \
    --machine-type=e2-medium --num-nodes=1 \
    --release-channel=regular
fi

echo ">> [3/6] GPU node pool ${NODEPOOL_GPU} (1x L4, autoscales 0..1)"
if gcloud container node-pools describe "$NODEPOOL_GPU" \
     --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "   already exists, skipping"
else
  # gpu-driver-version=default makes GKE install the NVIDIA driver itself.
  # Without it the node comes up, schedules the pod, and CUDA fails at
  # runtime -- a failure that reads like a container bug, not a setup gap.
  # 200 GB disk: the image is >10 GB and the 16 GB base model lands in an
  # emptyDir, which is backed by this same boot disk.
  # On-demand, not Spot: DESIGN.md §9.3 -- preemption is fine for training
  # and fatal for latency numbers, and this node serves the latency work.
  gcloud container node-pools create "$NODEPOOL_GPU" \
    --cluster "$CLUSTER" --zone "$ZONE" --project "$PROJECT" \
    --machine-type=g2-standard-8 \
    --accelerator "type=nvidia-l4,count=1,gpu-driver-version=default" \
    --num-nodes=1 --enable-autoscaling --min-nodes=0 --max-nodes=1 \
    --disk-type=pd-balanced --disk-size=200
fi

echo ">> [4/6] kubectl credentials"
gcloud container clusters get-credentials "$CLUSTER" \
  --zone "$ZONE" --project "$PROJECT"

echo ">> [5/6] Workload Identity: pod -> GCS, no key file"
if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$GSA" \
    --display-name="serana vLLM serving" --project "$PROJECT"
fi
# Read-only on the bucket: the pod pulls adapters and must never write.
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role=roles/storage.objectViewer >/dev/null
kubectl create serviceaccount "$KSA" --dry-run=client -o yaml | kubectl apply -f -
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:${PROJECT}.svc.id.goog[default/${KSA}]" \
  --project "$PROJECT" >/dev/null
kubectl annotate serviceaccount "$KSA" \
  "iam.gke.io/gcp-service-account=${SA_EMAIL}" --overwrite

echo ">> [6/6] image and manifests"
# An image reference may be passed in ($1) to deploy one that is already
# built -- useful because cluster creation and the image build are
# independent and can run at the same time. With no argument this builds.
if [ "$#" -ge 1 ]; then
  IMAGE="$1"
  echo "   using the image passed in, skipping the build"
else
  IMAGE="$(deploy/build_push.sh | tail -1)"
fi
echo "   image: ${IMAGE}"
sed "s|__IMAGE__|${IMAGE}|" deploy/k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f deploy/k8s/service.yaml

# 16 GB of weights download and load before /health answers; the startup
# probe allows 20 minutes and this wait matches it.
echo ">> waiting for rollout (the base model downloads on first start, ~10 min)"
kubectl rollout status deployment/serana-vllm --timeout=25m

cat <<MSG

up. The service is ClusterIP only -- reach it with:

    kubectl port-forward svc/serana-vllm 8000:8000

which makes config/base.yaml's serving.base_url (http://localhost:8000/v1)
correct as-is, so every existing script works unchanged against the cluster.

When finished:  deploy/cluster_down.sh
MSG
