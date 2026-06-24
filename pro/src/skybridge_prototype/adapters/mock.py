from __future__ import annotations

from pathlib import Path

from ..models import Answer, Caption, FramePayload, ProviderRef, Shot


class MockShotDetector:
    def __init__(self, ref: ProviderRef, shots_per_video: int = 3):
        self.ref = ref
        self.shots_per_video = max(1, shots_per_video)

    def detect_shots(self, video_path: Path) -> list[Shot]:
        seed = video_path.stat().st_size if video_path.exists() else 1
        total_ms = 18_000 + (seed % 7) * 2_000
        width = total_ms // self.shots_per_video
        shots: list[Shot] = []
        for index in range(self.shots_per_video):
            start = index * width
            end = total_ms if index == self.shots_per_video - 1 else (index + 1) * width
            shots.append(
                Shot(
                    shot_id=f"shot-{index + 1:03d}",
                    start_ms=start,
                    end_ms=end,
                    confidence=0.91 - index * 0.02,
                )
            )
        return shots


class MockSplitSampler:
    def __init__(self, ref: ProviderRef, samples_per_shot: int = 3):
        self.ref = ref
        self.samples_per_shot = max(1, samples_per_shot)

    def split_and_sample(self, video_path: Path, shots: list[Shot]) -> list[FramePayload]:
        payloads: list[FramePayload] = []
        video_name = video_path.name
        for shot_index, shot in enumerate(shots, start=1):
            duration = max(1, shot.end_ms - shot.start_ms)
            for sample_index in range(self.samples_per_shot):
                timestamp = shot.start_ms + duration * (sample_index + 1) // (self.samples_per_shot + 1)
                frame_id = f"frame-{shot_index:03d}-{sample_index + 1:02d}"
                content = (
                    f"mock frame {frame_id}\n"
                    f"video={video_name}\n"
                    f"shot={shot.shot_id}\n"
                    f"timestamp_ms={timestamp}\n"
                    f"samples_per_shot={self.samples_per_shot}\n"
                )
                payloads.append(
                    FramePayload(
                        frame_id=frame_id,
                        shot_id=shot.shot_id,
                        timestamp_ms=timestamp,
                        media_type="text/plain",
                        data=content.encode("utf-8"),
                        metadata={"mock": True},
                    )
                )
        return payloads


class MockCaptioner:
    def __init__(self, ref: ProviderRef):
        self.ref = ref

    def caption_frame(self, frame_path: Path, timestamp_ms: int) -> Caption:
        frame_text = frame_path.read_text(encoding="utf-8", errors="ignore")
        shot_line = next((line for line in frame_text.splitlines() if line.startswith("shot=")), "shot=unknown")
        shot_id = shot_line.split("=", 1)[1]
        return Caption(
            frame_id=frame_path.stem,
            timestamp_ms=timestamp_ms,
            text=f"At {timestamp_ms} ms, sampled {shot_id}; visible content is represented by {frame_path.name}.",
            provider=self.ref.provider,
        )


class MockQA:
    def __init__(self, ref: ProviderRef):
        self.ref = ref

    def answer_question(self, question: str, captions: list[Caption]) -> Answer:
        evidence = [caption.frame_id for caption in captions[:3]]
        joined = " ".join(caption.text for caption in captions[:3])
        text = (
            f"Question: {question}\n"
            f"Prototype answer: the workflow inspected {len(captions)} sampled frames. "
            f"Key observations: {joined}"
        )
        return Answer(text=text, provider=self.ref.provider, evidence_frame_ids=evidence)
