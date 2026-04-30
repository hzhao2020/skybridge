"""
测量包：Google Video Intelligence — Label Detection / Text(OCR) / Speech Transcription。
每项能力单独 annotate_video，便于分项计时。

所有 pid 在导入本模块时即注册（不依赖当时能否 import videointelligence），避免 DEFAULT_* 指向未注册的物理 ID。
实际调用 API 时若未安装 google-cloud-videointelligence，execute 会抛出明确 ImportError。
"""

from typing import Dict, List, Optional

try:
    import config
except ImportError:
    config = None

from ops.base import Operation
from ops.impl.google_ops import GoogleVideoIntelligenceFeatureImpl

# --- Video Intelligence API：Feature 枚举名（google.cloud.videointelligence.Feature.*）---
VI_FEATURE_LABEL_DETECTION = "LABEL_DETECTION"
VI_FEATURE_TEXT_DETECTION = "TEXT_DETECTION"  # OCR：画面中出现的文字
VI_FEATURE_SPEECH_TRANSCRIPTION = "SPEECH_TRANSCRIPTION"  # 语音转写（需 SpeechTranscriptionConfig）

# --- 物理 ID：Label（镜头级标签 / 实体标签检测）---
VI_LABEL_GOOGLE_US = "vi_label_google_us"
VI_LABEL_GOOGLE_TW = "vi_label_google_tw"

# --- 物理 ID：OCR（同上 VI_FEATURE_TEXT_DETECTION，单独 annotate_video）---
VI_OCR_GOOGLE_US = "vi_ocr_google_us"
VI_OCR_GOOGLE_TW = "vi_ocr_google_tw"

# --- 物理 ID：Speech Transcription（语音转文字）---
VI_SPEECH_GOOGLE_US = "vi_speech_google_us"
VI_SPEECH_GOOGLE_TW = "vi_speech_google_tw"

DEFAULT_LABEL_PID = VI_LABEL_GOOGLE_US
DEFAULT_OCR_PID = VI_OCR_GOOGLE_US
DEFAULT_SPEECH_PID = VI_SPEECH_GOOGLE_US

# Speech Transcription 默认 BCP-47 语言码（可在命令行 --speech-language 或 config.py 覆盖）
DEFAULT_SPEECH_LANGUAGE_CODE = "en-US"

REGISTRY: Dict[str, Operation] = {}


def register(pid: str, instance: Operation) -> None:
    REGISTRY[pid] = instance


def get_operation(pid: str) -> Operation:
    if pid not in REGISTRY:
        raise ValueError(f"Physical ID '{pid}' not found. 已知 pid 见 list_supported_operations()。")
    return REGISTRY[pid]


BUCKETS = {
    "gcp_us": "video_us",
    "gcp_tw": "video_tw",
}


def _bucket_for_key(bucket_key: str) -> str:
    name = BUCKETS.get(bucket_key)
    if config and hasattr(config, "GCS_BUCKETS") and isinstance(config.GCS_BUCKETS, dict):
        override = config.GCS_BUCKETS.get(bucket_key)
        if override:
            return str(override)
    return name or ""


# feature_name 为 Video Intelligence API 的 Feature 枚举名（字符串），在 execute 时再解析为枚举。
VIDEO_VI_FEATURE_CATALOG: List[Dict[str, object]] = [
    {
        "pid": VI_LABEL_GOOGLE_US,
        "feature_name": VI_FEATURE_LABEL_DETECTION,
        "operation_name_for_path": "vi_label_detection",
        "provider": "google",
        "region": "us-west1",
        "bucket_key": "gcp_us",
    },
    {
        "pid": VI_OCR_GOOGLE_US,
        "feature_name": VI_FEATURE_TEXT_DETECTION,
        "operation_name_for_path": "vi_text_detection",
        "provider": "google",
        "region": "us-west1",
        "bucket_key": "gcp_us",
    },
    {
        "pid": VI_SPEECH_GOOGLE_US,
        "feature_name": VI_FEATURE_SPEECH_TRANSCRIPTION,
        "operation_name_for_path": "vi_speech_transcription",
        "provider": "google",
        "region": "us-west1",
        "bucket_key": "gcp_us",
    },
    {
        "pid": VI_LABEL_GOOGLE_TW,
        "feature_name": VI_FEATURE_LABEL_DETECTION,
        "operation_name_for_path": "vi_label_detection",
        "provider": "google",
        "region": "asia-east1",
        "bucket_key": "gcp_tw",
    },
    {
        "pid": VI_OCR_GOOGLE_TW,
        "feature_name": VI_FEATURE_TEXT_DETECTION,
        "operation_name_for_path": "vi_text_detection",
        "provider": "google",
        "region": "asia-east1",
        "bucket_key": "gcp_tw",
    },
    {
        "pid": VI_SPEECH_GOOGLE_TW,
        "feature_name": VI_FEATURE_SPEECH_TRANSCRIPTION,
        "operation_name_for_path": "vi_speech_transcription",
        "provider": "google",
        "region": "asia-east1",
        "bucket_key": "gcp_tw",
    },
]


for item in VIDEO_VI_FEATURE_CATALOG:
    bucket_name = _bucket_for_key(str(item["bucket_key"]))
    register(
        str(item["pid"]),
        GoogleVideoIntelligenceFeatureImpl(
            str(item["provider"]),
            str(item["region"]),
            bucket_name,
            str(item["feature_name"]),
            operation_name_for_path=str(item["operation_name_for_path"]),
        ),
    )


def list_supported_operations() -> str:
    lines = ["Video Intelligence label / OCR / speech (measurement bundle):"]
    for item in VIDEO_VI_FEATURE_CATALOG:
        lines.append(f"  - {item['pid']}")
    return "\n".join(lines)


def get_operation_info(pid: str, include_class: bool = False) -> Optional[Dict]:
    for item in VIDEO_VI_FEATURE_CATALOG:
        if item.get("pid") == pid:
            if include_class:
                return {k: v for k, v in item.items()}
            return {
                "pid": item.get("pid"),
                "provider": item.get("provider"),
                "region": item.get("region"),
                "bucket_key": item.get("bucket_key"),
                "feature_name": item.get("feature_name"),
                "class": GoogleVideoIntelligenceFeatureImpl.__name__,
            }
    return None
