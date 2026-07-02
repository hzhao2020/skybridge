"""Measurement-backed execution latency models for SkyFlow simulation."""

from __future__ import annotations

from . import (
    database,
    label_detection,
    ocr,
    shot_detection,
    speech_transcription,
    video_split,
)

MEASURED_OPS = {
    "Sample": video_split.build_params,
    "Shot Detection": shot_detection.build_params,
    "Split & Sample": video_split.build_params,
    "Temporal Grounding": database.build_params,
    "Speech Transcription": speech_transcription.build_params,
    "OCR": ocr.build_params,
    "Label Detection": label_detection.build_params,
    "Database": database.build_params,
}

__all__ = ["MEASURED_OPS"]
