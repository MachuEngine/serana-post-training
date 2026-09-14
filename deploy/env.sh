# Shared values for the deploy scripts. Infrastructure identifiers only --
# model ids, serving knobs and hyperparameters stay in config/*.yaml
# (CLAUDE.md repo conventions); nothing here is a secret.
PROJECT="project-9a113a17-2211-4c9a-ae7"
# Zonal, not regional, on purpose: GKE's free tier covers the management
# fee for one zonal cluster per billing account, and -b is the Seoul zone
# that actually has L4 (RUNBOOK.md §1; -c has none).
ZONE="asia-northeast3-b"
REGION="asia-northeast3"
CLUSTER="serana"
NODEPOOL_GPU="l4-pool"
AR_REPO="serana"
IMAGE_NAME="serana-vllm"
# Google service account the pod impersonates to read adapters from GCS.
GSA="serana-serving"
KSA="serana"
BUCKET="gs://serana-post-training-ann10266"

IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${IMAGE_NAME}"
