from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ProviderRef:
    role: str
    provider: str
    region: str = "local"
    endpoint: str | None = None


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    path: Path
    media_type: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TransferEvent:
    node: str
    provider: str
    region: str
    direction: str
    artifact_id: str
    artifact_kind: str
    bytes_count: int


@dataclass(frozen=True)
class Shot:
    shot_id: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass(frozen=True)
class FramePayload:
    frame_id: str
    shot_id: str
    timestamp_ms: int
    media_type: str
    data: bytes
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Caption:
    frame_id: str
    timestamp_ms: int
    text: str
    provider: str


@dataclass(frozen=True)
class Answer:
    text: str
    provider: str
    evidence_frame_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowRequest:
    video_path: Path
    question: str
    selected_providers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    answer: Answer
    shots: list[Shot]
    frames: list[Artifact]
    captions: list[Caption]
    transfers: list[TransferEvent]
    run_dir: Path

    def to_dict(self) -> JsonDict:
        return to_jsonable(self)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
