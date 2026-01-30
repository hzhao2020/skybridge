#!/bin/bash

# 遇到任何错误立即停止脚本 (防止错误滚雪球)
set -e 

export PROJECT_ID=$(gcloud config get-value project)
# 你的目标 Regions
# REGIONS=("us-west1" "asia-southeast1")
REGIONS=("asia-southeast1")
REPO_NAME="experiment-repo"
FUNCTION_NAME="video-splitter"
ENTRY_POINT="video_split" 
MEMORY="2Gi"
TIMEOUT="600s"
MAX_INSTANCES="10"

echo "当前项目 ID: $PROJECT_ID"

for REGION in "${REGIONS[@]}"
do
  echo "========================================================"
  echo "🚀 正在处理区域: $REGION"
  
  # 1. 确保 Artifact Registry 仓库存在（Cloud Functions 会自动创建，但提前创建可以指定名称）
  gcloud artifacts repositories describe $REPO_NAME \
    --project=$PROJECT_ID \
    --location=$REGION > /dev/null 2>&1 || \
  gcloud artifacts repositories create $REPO_NAME \
    --project=$PROJECT_ID \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Cloud Functions"

  # 2. 部署 Cloud Function (使用 Cloud Functions Gen2)
  # Cloud Functions Gen2 会自动检测 Dockerfile 并使用 Cloud Build 构建镜像
  echo "部署 Cloud Function 到 $REGION ..."
  
  gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --region $REGION \
    --runtime=python311 \
    --entry-point $ENTRY_POINT \
    --trigger-http \
    --allow-unauthenticated \
    --memory $MEMORY \
    --timeout $TIMEOUT \
    --max-instances $MAX_INSTANCES \
    --source . \
    --quiet

  echo "✅ 区域 $REGION 部署完成！"
done

echo "🎉 所有流程结束"