"""
复制本文件为 config.py，填写 GCP（Google Video Intelligence）配置。

镜头检测使用 Video Intelligence API 的 SHOT_CHANGE_DETECTION 功能：
  annotate_video(input_uri=gs://...) -> operation -> operation.result()

鉴权使用 Application Default Credentials（ADC），二选一：
  1) gcloud auth application-default login          # 本地开发推荐
  2) export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

测量语义（三类时延分列）：
  - upload_sec            ：本地视频上传至 GCS 的墙钟时间，单独记录，不计入执行耗时。
  - execution_latency_sec ：客户端观测的执行时延（从 annotate_video 提交完成到
                            operation.result() 返回的墙钟时间）。
  - server_execution_sec  ：服务端 update_time - start_time（来自 operation.metadata
                            的 annotation_progress），最纯净的服务端处理耗时。

注意：
  - GVI 输入必须是 GCS（gs://），不能用阿里云 OSS。请先创建一个 GCS bucket。
  - GVI 支持 mkv 容器及 ffmpeg 可解码的编码（含 AV1，绝大多数情况可直接用原片）；
    仅当极少数环境无法解码时，才需转成 H.264 mp4。
"""

# GCP 配置
# GCP_CONFIG = {
#     "project": "your-gcp-project-id",      # 可留空，走 ADC 默认项目
#     "gcs_bucket": "your-gcs-bucket",        # 必填：存放待解析视频的 GCS bucket
#     "location_id": "us-west1",              # GVI 处理区域：us-west1 / asia-east1
#     "upload_prefix": "gvi_shot_measure",    # bucket 内的目录前缀
# }

# 访问 Google API 需要的代理（VPN）。设置后脚本会自动把
# http_proxy/https_proxy/grpc_proxy 指向它（gRPC 也走代理）。不需要则留空/删除。
# PROXY = "http://127.0.0.1:6454"

# 单任务最长等待秒数（operation.result 超时）
# JOB_TIMEOUT_SEC = 1800
