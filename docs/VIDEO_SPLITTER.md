# 视频分割 (Video Splitter) 功能说明

## 概述

`VideoSplitter` 是一个新的 operation 类，用于将视频物理切割成多个片段文件。与 `VideoSegmenter`（用于检测视频中的场景切换）不同，`VideoSplitter` 会实际创建多个视频文件。

## 架构

### 基类

在 `ops/base.py` 中定义了 `VideoSplitter` 基类：

```python
class VideoSplitter(Operation):
    @abstractmethod
    def execute(self, video_uri: str, segments: list, **kwargs) -> Dict[str, Any]:
        """
        执行视频分割
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            segments: 片段列表，每个片段包含 start_time 和 end_time
            **kwargs: 其他参数，如 target_path, output_format 等
        """
        pass
```

### 实现类

1. **GoogleCloudFunctionSplitImpl** (`ops/impl/google_ops.py`)
   - 使用 Google Cloud Functions 进行视频分割
   - 支持多个区域：us-west1, europe-west1, asia-southeast1, asia-east1
   - 通过 HTTP POST 调用 Cloud Function

2. **AWSLambdaSplitImpl** (`ops/impl/amazon_ops.py`)
   - 使用 AWS Lambda 进行视频分割
   - 支持多个区域：us-west-2, eu-central-1, ap-southeast-1
   - 通过 boto3 Lambda 客户端调用

## 注册的 Operation IDs

### Google Cloud Functions
- `split_google_us` - us-west1 区域
- `split_google_eu` - europe-west1 区域
- `split_google_sg` - asia-southeast1 区域
- `split_google_tw` - asia-east1 区域

### AWS Lambda
- `split_aws_us` - us-west-2 区域
- `split_aws_eu` - eu-central-1 区域
- `split_aws_sg` - ap-southeast-1 区域

## 使用方法

### 基本用法

```python
from ops.registry import get_operation

# 获取操作实例
split_op = get_operation("split_google_us")

# 定义要切割的片段
segments = [
    {"start": 0.0, "end": 10.0},
    {"start": 10.0, "end": 20.0},
]

# 执行分割
result = split_op.execute(
    video_uri="gs://bucket/video.mp4",
    segments=segments,
    target_path="output/segments",
    output_format="mp4"
)

# 查看结果
print(f"输出片段数量: {result['output_count']}")
for uri in result['output_uris']:
    print(f"  - {uri}")
```

### 自定义函数 URL

```python
from ops.impl.google_ops import GoogleCloudFunctionSplitImpl

split_op = GoogleCloudFunctionSplitImpl(
    provider="google",
    region="us-west1",
    storage_bucket="your-bucket",
    function_url="https://us-west1-PROJECT.cloudfunctions.net/custom-function"
)
```

## 部署 Cloud Functions / Lambda

### Google Cloud Functions

1. 参考 `cloud_functions/google_video_split/README.md`
2. 部署命令：
```bash
gcloud functions deploy video-split-us-west1 \
  --gen2 \
  --runtime python311 \
  --region us-west1 \
  --source cloud_functions/google_video_split \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 540s
```

### AWS Lambda

1. 参考 `lambda_functions/aws_video_split/README.md`
2. 推荐使用容器镜像部署（支持 ffmpeg）
3. 设置超时：600秒，内存：2048 MB

## 参数说明

### execute 方法参数

- `video_uri` (str): 视频 URI，支持：
  - 本地路径：`/path/to/video.mp4`
  - GCS URI：`gs://bucket/path/to/video.mp4`
  - S3 URI：`s3://bucket/path/to/video.mp4`

- `segments` (list): 片段列表，每个片段包含：
  - `start` (float): 开始时间（秒）
  - `end` (float): 结束时间（秒）

- `target_path` (str, optional): 输出路径前缀
- `output_format` (str, optional): 输出格式，默认 "mp4"
- `function_url` / `function_name` (str, optional): 自定义函数 URL/名称

### 返回值

```python
{
    "provider": "google_cloud_functions" | "aws_lambda",
    "region": "us-west1" | "us-west-2" | ...,
    "input_video": "gs://bucket/video.mp4",
    "segments": [...],
    "output_uris": [
        "gs://bucket/output/segment_1.mp4",
        "gs://bucket/output/segment_2.mp4",
        ...
    ],
    "output_count": 2
}
```

## 注意事项

1. **需要先部署 Cloud Function / Lambda**：
   - 代码提供了部署模板，但需要手动部署
   - 确保函数有权限访问云存储

2. **ffmpeg 支持**：
   - Cloud Functions: 需要使用 Gen2（支持自定义容器）或 Cloud Run Jobs
   - Lambda: 推荐使用容器镜像部署

3. **超时限制**：
   - Cloud Functions: 最大 540秒（9分钟）
   - Lambda: 最大 900秒（15分钟）

4. **成本考虑**：
   - 视频处理是计算密集型任务
   - 建议根据视频大小和数量选择合适的资源配置

## 与 VideoSegmenter 的区别

| 特性 | VideoSegmenter | VideoSplitter |
|------|---------------|---------------|
| 功能 | 检测场景切换 | 物理切割视频 |
| 输出 | 时间段列表（JSON） | 多个视频文件 |
| 用途 | 分析视频结构 | 创建视频片段 |
| 实现 | Video Intelligence API | Cloud Functions / Lambda |

## 示例

完整示例请参考 `examples/video_split_example.py`
