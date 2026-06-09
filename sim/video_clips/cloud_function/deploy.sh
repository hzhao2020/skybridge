#!/usr/bin/env bash
# 部署 split & sample 到 Cloud Run（Gen2 函数本质即 Cloud Run；自定义容器 + apt ffmpeg）。
# 在本目录（cloud_function/）下执行： bash deploy.sh
set -euo pipefail

PROJECT="project-525a8937-7589-4708-842"
REGION="asia-east1"
SERVICE="split-and-sample"

gcloud config set project "$PROJECT"

PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
COMPUTE_SA="${PN}-compute@developer.gserviceaccount.com"
CLOUDBUILD_SA="${PN}@cloudbuild.gserviceaccount.com"

echo "== 0) 建议（可选）：让 ADC quota project 与当前项目一致 =="
echo "  gcloud auth application-default set-quota-project $PROJECT"
echo

echo "== 1) 启用所需 API =="
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  storage.googleapis.com \
  cloudfunctions.googleapis.com

echo "== 2) 为 Cloud Run「从源码部署」预授权 IAM =="
echo "    （run deploy --source 实际用 Compute 默认 SA 跑 Cloud Build，不是 @cloudbuild）"
# Compute SA：读 run-sources 暂存 zip + 推镜像到 Artifact Registry + 写构建日志
for ROLE in \
  roles/storage.objectAdmin \
  roles/artifactregistry.writer \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="$ROLE" \
    --condition=None \
    >/dev/null 2>&1 || gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="$ROLE"
done
# Cloud Build SA（部分项目/旧流程仍可能用到）
for ROLE in \
  roles/storage.admin \
  roles/artifactregistry.writer \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="$ROLE" \
    --condition=None \
    >/dev/null 2>&1 || gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="$ROLE"
done
echo "    已授予 ${COMPUTE_SA} -> storage.objectAdmin, artifactregistry.writer, logging.logWriter"
echo "    已授予 ${CLOUDBUILD_SA} -> storage.admin, artifactregistry.writer, run.admin, iam.serviceAccountUser, logging.logWriter"
echo "    （IAM 传播可能需要 30–60 秒；若仍 403，稍等后重跑本脚本）"
sleep 15

echo "== 3) 部署（Cloud Build 构建 Dockerfile）CPU=2 内存=2Gi 区域=$REGION =="
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --max-instances 3 \
  --allow-unauthenticated \
  --quiet

echo "== 4) 给运行时服务账号授权（GCS 签名 URL 读视频）=="
SA=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(spec.template.spec.serviceAccountName)' || true)
if [ -z "${SA}" ]; then
  SA="${COMPUTE_SA}"
fi
echo "运行时服务账号: $SA"
gcloud iam service-accounts add-iam-policy-binding "$SA" \
  --member="serviceAccount:$SA" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --quiet
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectAdmin" \
  --quiet

echo "== 部署完成，函数 URL： =="
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
echo
echo "把上面的 URL 填到 scripts/config.py 的 SPLIT_SAMPLE_URL，然后运行测量客户端："
echo "  cd ../scripts && python run_split_sample_measurement.py --repeat 3"
echo
echo "测完可删除服务： gcloud run services delete $SERVICE --region $REGION --quiet"
