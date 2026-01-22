#!/bin/bash

# 遇到任何错误立即停止脚本 (防止错误滚雪球)
set -e 

export PROJECT_ID=$(gcloud config get-value project)
# 你的目标 Regions
REGIONS=("us-west1" "europe-west1" "asia-southeast1")
REPO_NAME="experiment-repo"
FUNCTION_NAME="video-splitter"
# 如果使用自定义镜像，--entry-point 其实通常不需要，由 Dockerfile 的 CMD 决定
# 但为了保险起见保留，或者你可以删除这一行
ENTRY_POINT="video_split" 
MEMORY="2Gi"

echo "当前项目 ID: $PROJECT_ID"

for REGION in "${REGIONS[@]}"
do
  echo "========================================================"
  echo "🚀 正在处理区域: $REGION"
  
  # 1. 确保 Artifact Registry 仓库存在
  gcloud artifacts repositories describe $REPO_NAME \
    --project=$PROJECT_ID \
    --location=$REGION > /dev/null 2>&1 || \
  gcloud artifacts repositories create $REPO_NAME \
    --project=$PROJECT_ID \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Cloud Functions"

  # 2. 构建并推送镜像 (这一步你之前已经成功了，但再跑一次确保最新)
  REMOTE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${FUNCTION_NAME}:latest"
  
  echo "[1/2] 构建镜像并推送到 $REGION ..."
  gcloud builds submit --tag $REMOTE_IMAGE .

  # 3. 部署 Cloud Function (使用 Cloud Run 部署预构建镜像)
  echo "[2/2] 部署 Cloud Function 到 $REGION ..."
  
  # ---------------------------------------------------------
  # 关键修改点：
  # Cloud Functions 2nd gen 基于 Cloud Run，可以直接使用 gcloud run deploy 部署预构建镜像
  # 使用 --source 参数指向当前目录，让 Cloud Functions 自动构建会重复构建
  # 因此改用 gcloud run deploy 直接使用已构建的镜像
  # ---------------------------------------------------------
  gcloud run deploy $FUNCTION_NAME \
    --image $REMOTE_IMAGE \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory $MEMORY \
    --timeout 600 \
    --min-instances 0 \
    --max-instances 10 \
    --quiet

  echo "✅ 区域 $REGION 部署完成！"
done

echo "🎉 所有流程结束"