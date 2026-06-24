from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import Answer, Artifact, Caption, WorkflowRequest, WorkflowResult, to_jsonable
from .providers import CaptionProvider, QAProvider, ShotDetectionProvider, SplitSampleProvider
from .relay import UserRelay


class UserSideBroker:
    """Local scheduler that bridges all cross-cloud workflow edges via the user side."""

    def __init__(
        self,
        shot_detector: ShotDetectionProvider,
        split_sampler: SplitSampleProvider,
        captioner: CaptionProvider,
        qa: QAProvider,
        runs_dir: Path = Path("runs"),
    ):
        self.shot_detector = shot_detector
        self.split_sampler = split_sampler
        self.captioner = captioner
        self.qa = qa
        self.runs_dir = runs_dir

    def run(self, request: WorkflowRequest) -> WorkflowResult:
        run_id = _new_run_id()
        relay = UserRelay(self.runs_dir / run_id)
        video = relay.ingest_video(Path(request.video_path))

        relay.record_transfer(self.shot_detector.ref, "shot_detection", "user_to_provider", video)
        shots = self.shot_detector.detect_shots(video.path)
        shots_artifact = relay.save_json_artifact("shots", "shot_list", shots)
        relay.record_transfer(self.shot_detector.ref, "shot_detection", "provider_to_user", shots_artifact)

        relay.record_transfer(self.split_sampler.ref, "split_sample", "user_to_provider", video)
        relay.record_transfer(self.split_sampler.ref, "split_sample", "user_to_provider", shots_artifact)
        frame_payloads = self.split_sampler.split_and_sample(video.path, shots)
        frames: list[Artifact] = []
        for payload in frame_payloads:
            frame = relay.save_frame(payload)
            frames.append(frame)
            relay.record_transfer(self.split_sampler.ref, "split_sample", "provider_to_user", frame)

        captions: list[Caption] = []
        for frame in frames:
            relay.record_transfer(self.captioner.ref, "frame_caption", "user_to_provider", frame)
            timestamp_ms = int(frame.metadata.get("timestamp_ms", 0))
            caption = self.captioner.caption_frame(frame.path, timestamp_ms)
            captions.append(caption)
        captions_artifact = relay.save_json_artifact("captions", "caption_list", captions)
        relay.record_transfer(self.captioner.ref, "frame_caption", "provider_to_user", captions_artifact)

        relay.record_transfer(self.qa.ref, "qa", "user_to_provider", captions_artifact)
        answer = self.qa.answer_question(request.question, captions)
        answer_artifact = relay.save_json_artifact("answer", "answer", answer)
        relay.record_transfer(self.qa.ref, "qa", "provider_to_user", answer_artifact)

        result = WorkflowResult(
            run_id=run_id,
            answer=answer,
            shots=shots,
            frames=frames,
            captions=captions,
            transfers=relay.transfers,
            run_dir=relay.run_dir,
        )
        _write_result(relay.run_dir, result)
        return result


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _write_result(run_dir: Path, result: WorkflowResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(to_jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
