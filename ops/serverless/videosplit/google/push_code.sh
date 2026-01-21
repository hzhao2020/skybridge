#!/bin/bash

# 确保变量已设置
export PROJECT_ID=$(gcloud config get-value project)
REGIONS=("us-west1" "europe-west1" "asia-southeast1")
REPO_NAME="experiment-repo"
SERVICE_NAME="video-splitter-service"

echo "当前项目 ID: $PROJECT_ID"

# 循环处理每个 Region
for REGION in "${REGIONS[@]}"
do
  echo "========================================================"
  echo "正在处理区域: $REGION"
  
  # 目标镜像地址
  REMOTE_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

  echo "[1/2] 使用 Cloud Build 构建并推送到 $REGION ..."
  # 核心修改：使用 gcloud builds submit 代替本地 docker 命令
  # 这会自动完成：上传代码 -> 云端构建 -> 推送到 Registry
  gcloud builds submit --tag $REMOTE_IMAGE .

  if [ $? -ne 0 ]; then
    echo "❌ 构建失败，停止后续操作"
    exit 1
  fi

  echo "[2/2] 部署 Cloud Run 到 $REGION (从本地 Registry 拉取)..."
  gcloud run deploy $SERVICE_NAME \
    --image $REMOTE_IMAGE \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 900 \
    --max-instances 10
    
  echo "✅ 区域 $REGION 部署完成！"
done

echo "========================================================"
echo "🎉 所有 4 个 Region 部署完毕！"