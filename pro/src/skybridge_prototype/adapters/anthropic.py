from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from ..models import Answer, Caption, ProviderRef, to_jsonable


class AnthropicMessagesProvider:
    """Minimal Anthropic Messages API adapter for caption and QA nodes."""

    def __init__(
        self,
        ref: ProviderRef,
        model: str | None = None,
        model_env: str = "ANTHROPIC_MODEL",
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_seconds: int = 120,
    ):
        self.ref = ref
        self.model = model
        self.model_env = model_env
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.endpoint = ref.endpoint or "https://api.anthropic.com/v1/messages"

    def caption_frame(self, frame_path: Path, timestamp_ms: int) -> Caption:
        media_type = _media_type(frame_path)
        if media_type.startswith("image/"):
            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
                    },
                },
                {
                    "type": "text",
                    "text": f"Caption this sampled video frame at {timestamp_ms} ms in one concise sentence.",
                },
            ]
        else:
            content = [
                {
                    "type": "text",
                    "text": (
                        f"This prototype frame artifact is text, not an image. "
                        f"Summarize it as a video-frame caption.\n\n{frame_path.read_text(errors='ignore')}"
                    ),
                }
            ]
        response = self._messages(content)
        return Caption(
            frame_id=frame_path.stem,
            timestamp_ms=timestamp_ms,
            text=_extract_text(response),
            provider=self.ref.provider,
        )

    def answer_question(self, question: str, captions: list[Caption]) -> Answer:
        response = self._messages(
            [
                {
                    "type": "text",
                    "text": (
                        "Answer the user's question using only these frame captions. "
                        "Mention the most relevant frame ids when useful.\n\n"
                        f"Question: {question}\n"
                        f"Captions: {json.dumps(to_jsonable(captions), ensure_ascii=False)}"
                    ),
                }
            ]
        )
        return Answer(
            text=_extract_text(response),
            provider=self.ref.provider,
            evidence_frame_ids=[caption.frame_id for caption in captions[:3]],
        )

    def _messages(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env} for Anthropic API provider")
        model = self.model or os.environ.get(self.model_env)
        if not model:
            raise RuntimeError(f"Missing Anthropic model. Set config model or {self.model_env}")
        body = json.dumps(
            {
                "model": model,
                "max_tokens": 700,
                "messages": [{"role": "user", "content": content}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _extract_text(response: dict[str, Any]) -> str:
    chunks = response.get("content", [])
    text_parts = [chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text"]
    return "\n".join(part for part in text_parts if part).strip()


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    return "text/plain"
