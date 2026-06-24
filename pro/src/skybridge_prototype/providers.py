from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Answer, Caption, FramePayload, ProviderRef, Shot


class ShotDetectionProvider(Protocol):
    ref: ProviderRef

    def detect_shots(self, video_path: Path) -> list[Shot]:
        """Run shot detection in a cloud service and return boundaries."""


class SplitSampleProvider(Protocol):
    ref: ProviderRef

    def split_and_sample(self, video_path: Path, shots: list[Shot]) -> list[FramePayload]:
        """Run a serverless splitter/sampler and return sampled frame payloads."""


class CaptionProvider(Protocol):
    ref: ProviderRef

    def caption_frame(self, frame_path: Path, timestamp_ms: int) -> Caption:
        """Run a vision LLM caption call for one sampled frame."""


class QAProvider(Protocol):
    ref: ProviderRef

    def answer_question(self, question: str, captions: list[Caption]) -> Answer:
        """Run final QA over captions and return the user-facing answer."""
