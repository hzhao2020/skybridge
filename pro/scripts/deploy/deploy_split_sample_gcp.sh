#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGIONS="${REGIONS:-us-east1 us-west1 europe-west1 asia-east1}"
SERVICE_PREFIX="${SERVICE_PREFIX:-skyflow-prototype-split-sample}"
SOURCE_DIR="${SOURCE_DIR:-cloud/split_sample}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is empty. Set PROJECT_ID or run gcloud config set project ..." >&2
  exit 1
fi

for region in ${REGIONS}; do
  service="${SERVICE_PREFIX}-${region}"
  echo "Deploying ${service} to ${region}"
  gcloud run deploy "${service}" \
    --project "${PROJECT_ID}" \
    --region "${region}" \
    --source "${SOURCE_DIR}" \
    --cpu 2 \
    --memory 2Gi \
    --timeout 900 \
    --concurrency 1 \
    --allow-unauthenticated \
    --set-env-vars SAMPLES_PER_SHOT=3
done

