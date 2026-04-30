"""
本测量包使用的 registry 物理 ID 与 Video Intelligence Feature 名。

单一引用方式：
  from ops.registry import VI_OCR_GOOGLE_US, VI_FEATURE_TEXT_DETECTION, ...

若希望从包根目录导入（已将项目根加入 sys.path 时）：
  from pids import VI_OCR_GOOGLE_US
"""

from ops.registry import (  # noqa: F401
    DEFAULT_LABEL_PID,
    DEFAULT_OCR_PID,
    DEFAULT_SPEECH_PID,
    DEFAULT_SPEECH_LANGUAGE_CODE,
    VI_FEATURE_LABEL_DETECTION,
    VI_FEATURE_TEXT_DETECTION,
    VI_FEATURE_SPEECH_TRANSCRIPTION,
    VI_LABEL_GOOGLE_US,
    VI_LABEL_GOOGLE_TW,
    VI_OCR_GOOGLE_US,
    VI_OCR_GOOGLE_TW,
    VI_SPEECH_GOOGLE_US,
    VI_SPEECH_GOOGLE_TW,
)
