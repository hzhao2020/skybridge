from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Any

from ..models import Answer, Caption, FramePayload, ProviderRef, Shot, to_jsonable


class HttpJsonShotDetector:
    """Generic cloud shot detector bridge.

    Contract:
      request:  {"video": {"name": str, "data_b64": str}}
      response: {"shots": [{"shot_id": str, "start_ms": int, "end_ms": int, "confidence": float}]}
    """

    def __init__(self, ref: ProviderRef, timeout_seconds: int = 120):
        self.ref = ref
        self.timeout_seconds = timeout_seconds

    def detect_shots(self, video_path: Path) -> list[Shot]:
        response = _post_json(
            self.ref,
            {"video": _file_payload(video_path)},
            self.timeout_seconds,
        )
        return [Shot(**item) for item in response["shots"]]


class HttpJsonSplitSampler:
    """Generic serverless split/sample bridge.

    Contract:
      request:  {"video": {...}, "shots": [...], "samples_per_shot": int}
      response: {"frames": [{"frame_id": str, "shot_id": str, "timestamp_ms": int,
                             "media_type": str, "data_b64": str, "metadata": dict}]}
    """

    def __init__(self, ref: ProviderRef, timeout_seconds: int = 120, samples_per_shot: int = 3):
        self.ref = ref
        self.timeout_seconds = timeout_seconds
        self.samples_per_shot = samples_per_shot

    def split_and_sample(self, video_path: Path, shots: list[Shot]) -> list[FramePayload]:
        response = _post_json(
            self.ref,
            {
                "video": _file_payload(video_path),
                "shots": to_jsonable(shots),
                "samples_per_shot": self.samples_per_shot,
            },
            self.timeout_seconds,
        )
        frames: list[FramePayload] = []
        for item in response["frames"]:
            frames.append(
                FramePayload(
                    frame_id=item["frame_id"],
                    shot_id=item["shot_id"],
                    timestamp_ms=int(item["timestamp_ms"]),
                    media_type=item["media_type"],
                    data=base64.b64decode(item["data_b64"]),
                    metadata=item.get("metadata", {}),
                )
            )
        return frames


class HttpJsonCaptioner:
    """Generic GCP/Anthropic-compatible caption bridge behind an HTTP endpoint."""

    def __init__(self, ref: ProviderRef, timeout_seconds: int = 120):
        self.ref = ref
        self.timeout_seconds = timeout_seconds

    def caption_frame(self, frame_path: Path, timestamp_ms: int) -> Caption:
        response = _post_json(
            self.ref,
            {"frame": _file_payload(frame_path), "timestamp_ms": timestamp_ms},
            self.timeout_seconds,
        )
        return Caption(
            frame_id=response.get("frame_id", frame_path.stem),
            timestamp_ms=int(response.get("timestamp_ms", timestamp_ms)),
            text=response["text"],
            provider=self.ref.provider,
        )


class HttpJsonQA:
    """Generic GCP/Anthropic-compatible QA bridge behind an HTTP endpoint."""

    def __init__(self, ref: ProviderRef, timeout_seconds: int = 120):
        self.ref = ref
        self.timeout_seconds = timeout_seconds

    def answer_question(self, question: str, captions: list[Caption]) -> Answer:
        response = _post_json(
            self.ref,
            {"question": question, "captions": to_jsonable(captions)},
            self.timeout_seconds,
        )
        return Answer(
            text=response["text"],
            provider=self.ref.provider,
            evidence_frame_ids=list(response.get("evidence_frame_ids", [])),
        )


def _post_json(ref: ProviderRef, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    if not ref.endpoint:
        raise ValueError(f"HTTP provider {ref.provider}/{ref.role} requires an endpoint")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ref.endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _file_payload(path: Path) -> dict[str, str]:
    return {
        "name": path.name,
        "data_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
