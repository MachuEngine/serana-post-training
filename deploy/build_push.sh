#!/usr/bin/env bash
# Builds deploy/Dockerfile with Cloud Build and pushes to Artifact Registry.
#
# Cloud Build rather than a local `docker build`: the vLLM base image is
# >10 GB and linux/amd64, so building it on the M5 means a long pull of an
# image that cannot run there anyway. The local verification that matters
# already happened -- `scripts/serve_up.py --dry-run` proves the argument
# construction the image depends on.
#
# Prints the full image reference (with tag) on the last line, so callers
# can capture it:  IMAGE=$(deploy/build_push.sh | tail -1)
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/env.sh

# Tag by commit, so a running pod is always traceable to a revision.
TAG="$(git rev-parse --short HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
  TAG="${TAG}-dirty"
fi
IMAGE="${IMAGE_BASE}:${TAG}"

# Idempotent; safe to re-run.
echo ">> enabling Cloud Build + Artifact Registry APIs (idempotent)" >&2
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com \
  --project "$PROJECT" >&2

# Projects created after ~2024 no longer grant the Compute Engine default
# service account the roles a build needs (read the uploaded source tarball,
# write logs, push to Artifact Registry). Without this the build dies at
# "could not resolve source" with a 403 that names Cloud Storage rather than
# IAM, which sends you looking in the wrong place. Idempotent.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
echo ">> granting the build service account its standard role (idempotent)" >&2
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder" --condition=None >/dev/null

# One Docker repository in Seoul. describe-or-create keeps this re-runnable.
if ! gcloud artifacts repositories describe "$AR_REPO" \
      --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  echo ">> creating Artifact Registry repo ${AR_REPO} in ${REGION}" >&2
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker --location "$REGION" \
    --description="serana serving images" --project "$PROJECT" >&2
fi

echo ">> building ${IMAGE} on Cloud Build" >&2
echo "   (build context excludes .env, data/ and artifacts/ -- see .dockerignore)" >&2
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions="_IMAGE=${IMAGE}" \
  --project "$PROJECT" --region "$REGION" . >&2

echo "$IMAGE"
