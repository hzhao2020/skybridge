# VideoSplit（Google Cloud Function）

本目录提供 **Google Cloud Function** 版本的视频切割服务。

## HTTP API

- `POST /` 或 `POST /video_split`
- `GET /healthz` (健康检查)
- Body（JSON）：

```json
{
  "video_uri": "gs://bucket/path/to/video.mp4",
  "segments": [{"start": 0.0, "end": 10.0}],
  "output_bucket": "bucket",
  "output_path": "split_segments",
  "output_format": "mp4"
}
```

返回：

```json
{"status":"success","output_uris":["gs://..."],"segment_count":1}
```

## 部署（按 region 部署多个服务）

### 使用部署脚本（推荐）

#### 1. 初始化 Artifact Registry（首次部署前运行一次）

如果使用自定义容器部署（推荐，因为需要 ffmpeg），需要先创建 Artifact Registry 仓库：

```bash
cd ops/serverless/videosplit/google
./init.sh
```

这会为每个 region 创建 Artifact Registry 仓库（如果不存在）。

#### 2. 部署到所有 Region

**方式一：自定义容器部署（推荐，包含 ffmpeg）**

```bash
cd ops/serverless/videosplit/google
./deploy.sh
```

此脚本会：
- 使用 Cloud Build 构建 Docker 镜像（包含 ffmpeg）
- 推送到各 region 的 Artifact Registry
- 部署 Cloud Function 到所有配置的 region

**方式二：标准运行时部署（不包含 ffmpeg，仅用于测试）**

```bash
cd ops/serverless/videosplit/google
./deploy_simple.sh
```

⚠️ **注意**：标准运行时不包含 ffmpeg，无法正常工作。仅用于测试 Cloud Function 框架本身。

### 手动部署（可选）

如果你想手动部署单个 region：

#### 使用自定义容器（推荐）

```bash
# 1. 初始化 Artifact Registry（如果还没运行过）
./init.sh

# 2. 构建并推送镜像
REMOTE_IMAGE="us-west1-docker.pkg.dev/$(gcloud config get-value project)/experiment-repo/video-splitter:latest"
gcloud builds submit --tag $REMOTE_IMAGE .

# 3. 部署函数
gcloud functions deploy video-splitter \
  --gen2 \
  --region us-west1 \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 600s \
  --max-instances 10 \
  --docker-registry CONTAINER_REGISTRY \
  --docker-image $REMOTE_IMAGE
```

#### 使用标准运行时（不推荐，缺少 ffmpeg）

```bash
gcloud functions deploy video-splitter \
  --gen2 \
  --region us-west1 \
  --runtime python311 \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 600s \
  --max-instances 10 \
  --source .
```

**注意**：由于需要 ffmpeg，强烈建议使用自定义容器方式部署（`deploy.sh`）。

部署完成后得到 function URL（例如 `https://...cloudfunctions.net`）。本地调用侧用：

- `split_google_us.execute(..., service_url="<Cloud Function URL>")`

你也可以把 `service_url` 填到 `GoogleCloudFunctionSplitImpl(service_url=...)`（如果你选择自己在外层封装/复用实例）。

