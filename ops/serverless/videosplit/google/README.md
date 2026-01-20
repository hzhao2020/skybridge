# VideoSplit（Google Cloud Run）

本目录提供 **Google Cloud Run** 版本的视频切割服务。

## HTTP API

- `POST /` 或 `POST /video_split`
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

在每个 region 分别部署一个 Cloud Run service（服务名建议包含 region，便于管理），例如：

```bash
cd ops/serverless/videosplit/google

gcloud run deploy videosplit-us-west1 \
  --region us-west1 \
  --source . \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 600
```

部署完成后得到 service URL（例如 `https://...run.app`）。本地调用侧用：

- `split_google_us.execute(..., service_url="<Cloud Run URL>")`

你也可以把 `service_url` 填到 `GoogleCloudRunSplitImpl(service_url=...)`（如果你选择自己在外层封装/复用实例）。

