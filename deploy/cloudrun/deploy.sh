#!/usr/bin/env bash
# Deploy the job API ALONE to Cloud Run — no accounts service, no nginx.
#
# The API has NO authentication of its own and gains none. The only thing that
# makes that safe is that nothing can reach it except your proxy, which is what
# the INGRESS default below enforces. Do NOT set INGRESS=all unless your proxy
# is the thing answering that URL (and then auth is on you).
#
# The GPU worker is NOT here: it runs as a Vertex AI Custom Job from the image
# referenced by WORKER_IMAGE_URI, exactly as in the compose deployment.
#
# Usage:
#   cp deploy/cloudrun/env.sh.example deploy/cloudrun/env.sh   # fill it in
#   source deploy/cloudrun/env.sh
#   deploy/cloudrun/deploy.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Identity / destination ----------------------------------------------------
PROJECT="${PROJECT:?set PROJECT to the GCP project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-peptide-api}"
AR_REPO="${AR_REPO:-peptide}"          # Artifact Registry repository name
TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${SERVICE}:${TAG}"

# ---- Required API settings (the API answers 503 while any is missing) ----------
VERTEX_CLOUD_PROJECT="${VERTEX_CLOUD_PROJECT:-$PROJECT}"
VERTEX_LOCATION="${VERTEX_LOCATION:-$REGION}"
VERTEX_ARTIFACTS_DIR="${VERTEX_ARTIFACTS_DIR:?set VERTEX_ARTIFACTS_DIR, e.g. gs://<bucket>/artifacts}"
VERTEX_MODEL_STORE="${VERTEX_MODEL_STORE:?set VERTEX_MODEL_STORE, e.g. gs://<bucket>/model_weights}"
SEED_CANDIDATES_FILE="${SEED_CANDIDATES_FILE:?set SEED_CANDIDATES_FILE, e.g. gs://<bucket>/curated_peptides.fasta}"
WORKER_IMAGE_URI="${WORKER_IMAGE_URI:?set WORKER_IMAGE_URI to the worker image in Artifact Registry}"
WORKER_SERVICE_ACCOUNT="${WORKER_SERVICE_ACCOUNT:?set WORKER_SERVICE_ACCOUNT, e.g. worker@PROJECT.iam.gserviceaccount.com}"

# Required pairing: NVIDIA_L4 only attaches to g2-standard-*.
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_L4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

# ---- The service account the Cloud Run service runs AS (its ADC). --------------
# It must be able to create/cancel Vertex AI Custom Jobs (roles/aiplatform.user)
# and read/write VERTEX_ARTIFACTS_DIR (roles/storage.objectAdmin on that bucket).
# Leave empty to use the project's default compute service account.
API_SERVICE_ACCOUNT="${API_SERVICE_ACCOUNT:-}"

# ---- Ingress / auth ------------------------------------------------------------
#   internal-and-cloud-load-balancing : only your external HTTPS LB (the proxy) can
#                                       reach it. RECOMMENDED default.
#   internal                           : VPC / private Google access only.
#   all                                : public URL — do NOT use with a no-auth API.
INGRESS="${INGRESS:-internal-and-cloud-load-balancing}"

# Whether the Cloud Run service itself requires a Google identity. With a custom
# proxy doing the real auth, "true" is the usual choice — security then comes
# from INGRESS, not from Cloud Run's own IAM check. Set "false" to layer on IAP
# or require your proxy to present a service-account token.
ALLOW_UNAUTH="${ALLOW_UNAUTH:-true}"
if [[ "$ALLOW_UNAUTH" == "true" ]]; then
  AUTH_FLAG=(--allow-unauthenticated)
else
  AUTH_FLAG=(--no-allow-unauthenticated)
fi

# Ensure the Artifact Registry repository exists (idempotent).
if ! gcloud artifacts repositories describe "$AR_REPO" \
    --location "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository $AR_REPO ($REGION)"
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format docker --location "$REGION" --project "$PROJECT"
fi

echo "Building $IMAGE (context: $REPO_ROOT, dockerfile: deploy/api.Dockerfile)"
gcloud builds submit "$REPO_ROOT" \
  --project "$PROJECT" \
  --region "$REGION" \
  --config deploy/cloudbuild.api.yaml \
  --substitutions "_IMAGE_URI=$IMAGE"

echo "Deploying $SERVICE to Cloud Run ($REGION, ingress=$INGRESS)"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --platform managed \
  --image "$IMAGE" \
  --port 8080 \
  --cpu 1 --memory 512Mi \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 5 \
  --ingress "$INGRESS" \
  "${AUTH_FLAG[@]}" \
  ${API_SERVICE_ACCOUNT:+--service-account "$API_SERVICE_ACCOUNT"} \
  --set-env-vars \
    "VERTEX_CLOUD_PROJECT=${VERTEX_CLOUD_PROJECT},"\
"VERTEX_LOCATION=${VERTEX_LOCATION},"\
"VERTEX_ARTIFACTS_DIR=${VERTEX_ARTIFACTS_DIR},"\
"VERTEX_MODEL_STORE=${VERTEX_MODEL_STORE},"\
"SEED_CANDIDATES_FILE=${SEED_CANDIDATES_FILE},"\
"WORKER_IMAGE_URI=${WORKER_IMAGE_URI},"\
"WORKER_SERVICE_ACCOUNT=${WORKER_SERVICE_ACCOUNT},"\
"MACHINE_TYPE=${MACHINE_TYPE},"\
"ACCELERATOR_TYPE=${ACCELERATOR_TYPE},"\
"ACCELERATOR_COUNT=${ACCELERATOR_COUNT}"

echo "Done. Verify with:"
echo "  gcloud run services describe $SERVICE --region $REGION --format='value(status.url)'"
echo "  curl -s https://$SERVICE-<hash>-uc.a.run.app/healthz"
