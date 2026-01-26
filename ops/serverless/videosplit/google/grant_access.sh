#!/bin/bash
# 定义桶列表
buckets=("video_eu" "video_sg" "video_tw" "video_us")

# 获取服务代理账号
SA_EMAIL="service-587417646945@gcp-sa-aiplatform.iam.gserviceaccount.com"

# 循环授权
for bucket in "${buckets[@]}"; do
  gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectViewer" "gs://${bucket}"
  echo "Authorized ${bucket} for ${SA_EMAIL}"
done