from pathlib import Path

from skybridge_prototype.broker import UserSideBroker
from skybridge_prototype.models import ProviderRef, WorkflowRequest
from skybridge_prototype.adapters.mock import MockCaptioner, MockQA, MockShotDetector, MockSplitSampler


def test_mock_workflow_relays_every_cross_cloud_edge(tmp_path: Path) -> None:
    video = tmp_path / "video.txt"
    video.write_text("demo video", encoding="utf-8")

    broker = UserSideBroker(
        shot_detector=MockShotDetector(ProviderRef("shot_detection", "aws", "us-west-2"), shots_per_video=2),
        split_sampler=MockSplitSampler(ProviderRef("split_sample", "gcp", "us-west1"), samples_per_shot=3),
        captioner=MockCaptioner(ProviderRef("caption", "gcp", "us-west1")),
        qa=MockQA(ProviderRef("qa", "anthropic", "api")),
        runs_dir=tmp_path / "runs",
    )

    result = broker.run(WorkflowRequest(video_path=video, question="What happens?"))

    assert len(result.shots) == 2
    assert len(result.frames) == 6
    assert len(result.captions) == 6
    assert result.answer.provider == "anthropic"
    assert (result.run_dir / "result.json").exists()
    assert all(event.direction in {"user_to_provider", "provider_to_user"} for event in result.transfers)
    assert {"aws", "gcp", "anthropic"}.issubset({event.provider for event in result.transfers})
