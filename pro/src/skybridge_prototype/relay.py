from __future__ import annotations

import json
import shutil
from pathlib import Path

from .models import Artifact, FramePayload, ProviderRef, TransferEvent, to_jsonable


class UserRelay:
    """Local artifact relay.

    Every cloud node receives inputs from this relay and returns outputs to it.
    The broker never passes a provider-owned object directly to another provider.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.artifacts_dir = run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.transfers: list[TransferEvent] = []

    def ingest_video(self, video_path: Path) -> Artifact:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        suffix = video_path.suffix or ".bin"
        target = self.artifacts_dir / f"input_video{suffix}"
        shutil.copy2(video_path, target)
        return Artifact(
            artifact_id="input_video",
            kind="video",
            path=target,
            media_type=_guess_media_type(target),
            metadata={"source_path": str(video_path)},
        )

    def save_json_artifact(self, artifact_id: str, kind: str, payload: object) -> Artifact:
        target = self.artifacts_dir / f"{artifact_id}.json"
        target.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
        return Artifact(artifact_id=artifact_id, kind=kind, path=target, media_type="application/json")

    def save_frame(self, payload: FramePayload) -> Artifact:
        suffix = _suffix_for_media_type(payload.media_type)
        target = self.artifacts_dir / f"{payload.frame_id}{suffix}"
        target.write_bytes(payload.data)
        return Artifact(
            artifact_id=payload.frame_id,
            kind="frame",
            path=target,
            media_type=payload.media_type,
            metadata={
                "shot_id": payload.shot_id,
                "timestamp_ms": payload.timestamp_ms,
                **payload.metadata,
            },
        )

    def record_transfer(
        self,
        provider_ref: ProviderRef,
        node: str,
        direction: str,
        artifact: Artifact,
        bytes_count: int | None = None,
    ) -> None:
        size = bytes_count if bytes_count is not None else _file_size(artifact.path)
        self.transfers.append(
            TransferEvent(
                node=node,
                provider=provider_ref.provider,
                region=provider_ref.region,
                direction=direction,
                artifact_id=artifact.artifact_id,
                artifact_kind=artifact.kind,
                bytes_count=size,
            )
        )


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix in {".mov", ".qt"}:
        return "video/quicktime"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".json":
        return "application/json"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def _suffix_for_media_type(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "text/plain": ".txt",
        "application/json": ".json",
    }.get(media_type, ".bin")
