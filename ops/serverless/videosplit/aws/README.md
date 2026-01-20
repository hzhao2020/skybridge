# VideoSplit（AWS Lambda）

本目录提供 **AWS Lambda（容器镜像）** 版本的视频切割函数。

## 事件格式

```json
{
  "video_uri": "s3://bucket/path/to/video.mp4",
  "segments": [{"start": 0.0, "end": 10.0}],
  "output_bucket": "output-bucket",
  "output_path": "split_segments",
  "output_format": "mp4"
}
```

返回（函数成功时，直接返回业务结果 dict）：

```json
{"status":"success","output_uris":["s3://..."],"segment_count":1}
```

## 部署（按 region 部署多个函数）

建议每个 region 部署一个 Lambda 函数（函数名建议与 region 绑定，便于本地侧选择调用）。

本地 `AWSLambdaSplitImpl` 默认函数名规则是：

- `video-split-<region_with_underscores>`  
  例如：`us-west-2` -> `video-split-us_west_2`

你也可以在调用 `execute(..., function_name="...")` 时覆盖函数名。

