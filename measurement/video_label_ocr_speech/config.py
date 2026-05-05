"""
本目录测量脚本可选配置。使用 vi_*_google_us 时只需覆盖 gcp_us 桶名。
若改用 vi_*_google_tw，请为 gcp_tw 增加桶映射或保持默认 video_tw。
"""

import os

# 与应用默认凭证（ADC）配合，供 Storage / Video Intelligence 客户端解析项目
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "project-81ef0a83-73fc-42d4-bd4")

GCS_BUCKETS = {
    "gcp_us": "sky_bucket_us_west1",
}
