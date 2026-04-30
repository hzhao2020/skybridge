"""
复制本文件为 config.py，仅在需要覆盖默认值或代理时填写。
打包上传到 GCP VM 后：若使用实例服务账号 ADC，多数情况下可不创建 config.py。
"""

# 可选：覆盖 registry 中的默认 GCS 桶名（默认美西 video_us、台湾 video_tw）
# GCS_BUCKETS = {
#     "gcp_us": "your-project-us-bucket",
#     "gcp_tw": "your-project-tw-bucket",
# }

# 可选：其它脚本读取的项目号
# GCP_PROJECT_NUMBER = "123456789012"

# --- OCR（画面文字）---
# Registry 物理 ID = Google Video Intelligence Feature.TEXT_DETECTION（单独一次 annotate_video）
#   VI_OCR_GOOGLE_US = "vi_ocr_google_us"   （us-west1，桶 gcp_us）
#   VI_OCR_GOOGLE_TW = "vi_ocr_google_tw"   （asia-east1，桶 gcp_tw）
# 默认测量 pid：DEFAULT_OCR_PID → vi_ocr_google_us
# Python 常量：ops/registry.py（常量 VI_FEATURE_TEXT_DETECTION）或包根目录 pids.py

# --- Speech Transcription（语音转写）---
# Registry 物理 ID = Feature.SPEECH_TRANSCRIPTION（带 SpeechTranscriptionConfig）
#   VI_SPEECH_GOOGLE_US = "vi_speech_google_us"
#   VI_SPEECH_GOOGLE_TW = "vi_speech_google_tw"
# 默认测量 pid：DEFAULT_SPEECH_PID → vi_speech_google_us
# 默认识别语言（BCP-47）；取消下行注释后 run_measurement / batch 优先使用：
# DEFAULT_SPEECH_LANGUAGE_CODE = "zh-CN"

# 默认测量使用的三类 pid（改用台湾区域时可改为 *_google_tw）：
# MEASUREMENT_LABEL_PID = "vi_label_google_us"
# MEASUREMENT_OCR_PID = "vi_ocr_google_us"
# MEASUREMENT_SPEECH_PID = "vi_speech_google_us"

# 若 VM 访问 Google API 需 HTTP 代理（按需取消注释）
# import os
# os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
