#!/usr/bin/env bash
# Builds and deploys the service layer (src/serve/api.py) plus Prometheus
# and Grafana. Run after deploy/cluster_up.sh -- it assumes the cluster and
# the vLLM Service already exist.
#
#   deploy/monitoring_up.sh              # build the gateway image, then deploy
#   deploy/monitoring_up.sh <IMAGE>      # deploy a gateway image that exists
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/env.sh

GATEWAY_IMAGE_BASE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/serana-gateway"

if [ "$#" -ge 1 ]; then
  GATEWAY_IMAGE="$1"
  echo ">> using the gateway image passed in, skipping the build"
else
  TAG="$(git rev-parse --short HEAD)"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    TAG="${TAG}-dirty"
  fi
  GATEWAY_IMAGE="${GATEWAY_IMAGE_BASE}:${TAG}"
  echo ">> building ${GATEWAY_IMAGE} (small image, ~2 min unlike the 12 GB vLLM one)"
  gcloud builds submit --config deploy/cloudbuild.gateway.yaml \
    --substitutions="_IMAGE=${GATEWAY_IMAGE}" \
    --project "$PROJECT" --region "$REGION" .
fi

echo ">> deploying the gateway"
sed "s|__GATEWAY_IMAGE__|${GATEWAY_IMAGE}|" deploy/k8s/gateway.yaml | kubectl apply -f -

echo ">> dashboard ConfigMap from the checked-in JSON"
# --dry-run|apply rather than `create`, so re-running updates it in place.
kubectl create configmap grafana-dashboards \
  --from-file=deploy/grafana/serana-dashboard.json \
  --dry-run=client -o yaml | kubectl apply -f -

echo ">> deploying Prometheus and Grafana"
kubectl apply -f deploy/k8s/monitoring.yaml

echo ">> waiting for rollouts"
kubectl rollout status deployment/serana-gateway --timeout=5m
kubectl rollout status deployment/prometheus --timeout=5m
kubectl rollout status deployment/grafana --timeout=5m

cat <<MSG

up. Reach them locally:

    kubectl port-forward svc/grafana 3000:3000      # dashboard, no login
    kubectl port-forward svc/prometheus 9090:9090   # raw queries and alert state
    kubectl port-forward svc/serana-gateway 8080:8080

Generate traffic to put something on the dashboard:

    curl -s localhost:8080/chat -H 'content-type: application/json' \\
      -d '{"turn":"너는 누구야?","config":"sft"}'
MSG
