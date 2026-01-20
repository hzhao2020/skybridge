# Google Cloud Function Video Split

## 部署说明

### 前置要求

1. 安装 Google Cloud SDK
2. 确保已启用 Cloud Functions API
3. 确保有 GCS bucket 的读写权限

### 部署步骤

1. 进入函数目录：
```bash
cd cloud_functions/google_video_split
```

2. 部署函数：
```bash
gcloud functions deploy video-split-us-west1 \
  --gen2 \
  --runtime python311 \
  --region us-west1 \
  --source . \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 540s \
  --max-instances 10 \
  --set-env-vars GCS_BUCKET=your-bucket-name
```

### 使用不同的区域

```bash
# US West
gcloud functions deploy video-split-us-west1 \
  --gen2 --runtime python311 --region us-west1 ...

# Europe West
gcloud functions deploy video-split-europe-west1 \
  --gen2 --runtime python311 --region europe-west1 ...

# Asia Southeast
gcloud functions deploy video-split-asia-southeast1 \
  --gen2 --runtime python311 --region asia-southeast1 ...
```

### 注意事项

1. **ffmpeg 支持**：
   - Cloud Functions Gen2 支持自定义容器，可以包含 ffmpeg
   - 或者使用 Cloud Run Jobs（更适合长时间运行的任务）

2. **超时限制**：
   - Cloud Functions Gen2 最大超时：540秒（9分钟）
   - 对于长视频，考虑使用 Cloud Run Jobs

3. **内存设置**：
   - 建议至少 2GB 内存用于视频处理

4. **权限**：
   - 确保 Cloud Function 的服务账号有 GCS 读写权限

### 获取函数 URL

部署后，获取函数 URL：
```bash
gcloud functions describe video-split-us-west1 \
  --gen2 --region us-west1 --format="value(serviceConfig.uri)"
```

将此 URL 配置到 `GoogleCloudFunctionSplitImpl` 的 `function_url` 参数中。
