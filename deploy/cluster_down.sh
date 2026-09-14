#!/usr/bin/env bash
# One command down. DESIGN.md §9.3's "the instance is stopped when it is not
# computing", applied to GKE: the GPU node is the expensive part and it lives
# in the cluster, so the cluster goes away between sessions.
#
# Deliberately left behind: the Artifact Registry images (pennies, and
# rebuilding is slow) and the GCS bucket (the artifacts themselves).
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/env.sh

echo ">> deleting cluster ${CLUSTER} in ${ZONE} -- GPU billing stops when this returns"
gcloud container clusters delete "$CLUSTER" \
  --zone "$ZONE" --project "$PROJECT" --quiet

cat <<MSG

down. Still billable, both small:
  images  gcloud artifacts docker images list ${IMAGE_BASE}
  bucket  gcloud storage du -s ${BUCKET}
MSG
