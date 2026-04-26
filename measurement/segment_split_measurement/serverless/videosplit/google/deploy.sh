#!/bin/bash

set -e

export PROJECT_ID=$(gcloud config get-value project)
REGION="us-west1"
REPO_NAME="vqa"
FUNCTION_NAME="split_measurement"
ENTRY_POINT="video_split"
MEMORY="1Gi"
CPU="1"
TIMEOUT="300s"
MAX_INSTANCES="10"

echo "当前项目 ID: $PROJECT_ID"
echo "部署区域: $REGION (memory=${MEMORY}, cpu=${CPU})"

gcloud artifacts repositories describe "$REPO_NAME" \
  --project="$PROJECT_ID" \
  --location="$REGION" > /dev/null 2>&1 || \
gcloud artifacts repositories create "$REPO_NAME" \
  --project="$PROJECT_ID" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Docker repository for Cloud Functions"

echo "部署 Cloud Function 到 $REGION ..."

gcloud functions deploy "$FUNCTION_NAME" \
  --gen2 \
  --region "$REGION" \
  --runtime=python311 \
  --entry-point "$ENTRY_POINT" \
  --trigger-http \
  --allow-unauthenticated \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --timeout "$TIMEOUT" \
  --max-instances "$MAX_INSTANCES" \
  --source . \
  --quiet

echo "部署完成: $FUNCTION_NAME ($REGION)"
