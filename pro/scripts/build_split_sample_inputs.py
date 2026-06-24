#!/usr/bin/env python3
"""Build normalized split/sample inputs from saved shot-detection results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SHOT_ROOT = Path("experiments/shot_detection_profile/final_50_seed20260622")
DEFAULT_OUTPUT_DIR = Path("experiments/split_sample_inputs/final_50_seed20260622")


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    filename: str
    path: Path
    size_bytes: int
    duration_sec: float
    fps: float


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_f = float(denominator)
        return float(numerator) / denominator_f if denominator_f else 0.0
    return float(value)


def ffprobe_video(path: Path) -> tuple[float, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    duration = float(data.get("format", {}).get("duration") or 0.0)
    stream = (data.get("streams") or [{}])[0]
    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
    return duration, fps


def load_manifest(path: Path) -> list[VideoInfo]:
    manifest = load_json(path)
    items = manifest["items"] if isinstance(manifest, dict) else manifest
    videos: list[VideoInfo] = []
    for item in items:
        video_path = Path(item["path"])
        duration_sec = float(item.get("duration_sec") or 0.0)
        fps = 0.0
        if duration_sec <= 0 or fps <= 0:
            probe_duration, probe_fps = ffprobe_video(video_path)
            duration_sec = duration_sec or probe_duration
            fps = probe_fps
        videos.append(
            VideoInfo(
                video_id=item["video_id"],
                filename=item.get("filename") or video_path.name,
                path=video_path,
                size_bytes=int(item.get("size_bytes") or video_path.stat().st_size),
                duration_sec=duration_sec,
                fps=fps,
            )
        )
    return videos


def seconds_string_to_ms(value: str | int | float | None) -> int:
    if value in ("", None):
        return 0
    if isinstance(value, (int, float)):
        return int(round(float(value) * 1000))
    text = str(value).strip()
    if text.endswith("s"):
        text = text[:-1]
    return int(round(float(text) * 1000))


def normalize_shots(
    result: dict[str, Any],
    video: VideoInfo,
    provider_key: str,
    min_shot_duration_ms: int,
) -> list[dict[str, Any]]:
    provider = result.get("provider") or provider_key.split("_", 1)[0]
    if provider == "gcp":
        return normalize_gcp_shots(result, min_shot_duration_ms)
    if provider == "aws":
        return normalize_aws_shots(result, min_shot_duration_ms)
    if provider == "aliyun":
        return normalize_aliyun_shots(result, video, min_shot_duration_ms)
    raise ValueError(f"Unsupported shot provider for {video.video_id}: {provider!r}")


def normalize_gcp_shots(result: dict[str, Any], min_shot_duration_ms: int) -> list[dict[str, Any]]:
    response = result.get("operation", {}).get("response", {})
    annotations: list[dict[str, Any]] = []
    for item in response.get("annotationResults", []):
        annotations.extend(item.get("shotAnnotations", []))
    shots = []
    for index, shot in enumerate(annotations, start=1):
        shots.append(
            {
                "shot_id": f"shot-{index:03d}",
                "start_ms": seconds_string_to_ms(shot.get("startTimeOffset")),
                "end_ms": seconds_string_to_ms(shot.get("endTimeOffset")),
                "confidence": None,
                "source": "gcp_shotAnnotations",
            }
        )
    return clean_shots(shots, min_shot_duration_ms=min_shot_duration_ms)


def normalize_aws_shots(result: dict[str, Any], min_shot_duration_ms: int) -> list[dict[str, Any]]:
    segments = [segment for segment in result.get("segments", []) if segment.get("Type") == "SHOT"]
    shots = []
    for index, segment in enumerate(segments, start=1):
        shots.append(
            {
                "shot_id": f"shot-{index:03d}",
                "start_ms": int(segment["StartTimestampMillis"]),
                "end_ms": int(segment["EndTimestampMillis"]),
                "confidence": segment.get("ShotSegment", {}).get("Confidence"),
                "source": "aws_segments",
            }
        )
    return clean_shots(shots, min_shot_duration_ms=min_shot_duration_ms)


def normalize_aliyun_shots(result: dict[str, Any], video: VideoInfo, min_shot_duration_ms: int) -> list[dict[str, Any]]:
    raw = result.get("result", {}).get("ShotFrameIds")
    if raw is None:
        raise ValueError(f"Aliyun result has no ShotFrameIds for {video.video_id}")
    frame_ids = json.loads(raw) if isinstance(raw, str) else list(raw)
    frame_ids = sorted({int(frame) for frame in frame_ids if int(frame) >= 0})
    if not frame_ids or frame_ids[0] != 0:
        frame_ids.insert(0, 0)
    fps = video.fps or (frame_ids[-1] / video.duration_sec if video.duration_sec > 0 else 0.0)
    if fps <= 0:
        raise ValueError(f"Cannot infer FPS for {video.video_id}")
    duration_ms = int(round(video.duration_sec * 1000))
    shots = []
    for index, start_frame in enumerate(frame_ids):
        start_ms = int(round(start_frame * 1000 / fps))
        if index + 1 < len(frame_ids):
            end_ms = int(round(frame_ids[index + 1] * 1000 / fps))
        else:
            end_ms = duration_ms
        shots.append(
            {
                "shot_id": f"shot-{index + 1:03d}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": None,
                "source": "aliyun_ShotFrameIds",
                "start_frame": start_frame,
            }
        )
    return clean_shots(shots, duration_ms=duration_ms, min_shot_duration_ms=min_shot_duration_ms)


def clean_shots(
    shots: list[dict[str, Any]],
    duration_ms: int | None = None,
    min_shot_duration_ms: int = 1,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for shot in shots:
        start_ms = max(0, int(shot["start_ms"]))
        end_ms = max(start_ms + 1, int(shot["end_ms"]))
        if duration_ms is not None:
            start_ms = min(start_ms, duration_ms)
            end_ms = min(max(start_ms + 1, end_ms), duration_ms)
        if duration_ms is not None and start_ms >= duration_ms:
            continue
        if end_ms - start_ms < min_shot_duration_ms:
            continue
        normalized = dict(shot)
        normalized["start_ms"] = start_ms
        normalized["end_ms"] = end_ms
        cleaned.append(normalized)
    return cleaned


def build_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shot_root = Path(args.shot_results_root)
    index = load_json(shot_root / "index.json")
    manifest_path = Path(args.manifest or index["manifest"])
    videos = load_manifest(manifest_path)
    if args.max_videos:
        videos = videos[: args.max_videos]

    provider_key = args.shot_provider
    provider_info = index["providers"][provider_key]
    result_dir = Path(provider_info["result_dir"])
    usable_by_video = {record["video_id"]: bool(record["usable_for_split_sample"]) for record in provider_info["records"]}

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for video in videos:
        result_path = result_dir / f"{video.video_id}.json"
        if not usable_by_video.get(video.video_id, False):
            skipped.append({"video_id": video.video_id, "reason": "shot_result_not_usable", "path": str(result_path)})
            if not args.allow_missing:
                raise ValueError(f"Shot result is not usable for {video.video_id}: {result_path}")
            continue
        result = load_json(result_path)
        raw_shots = normalize_shots(result, video, provider_key, min_shot_duration_ms=1)
        shots = normalize_shots(result, video, provider_key, min_shot_duration_ms=args.min_shot_duration_ms)
        if not shots:
            skipped.append({"video_id": video.video_id, "reason": "no_normalized_shots", "path": str(result_path)})
            if not args.allow_missing:
                raise ValueError(f"No normalized shots for {video.video_id}: {result_path}")
            continue
        items.append(
            {
                "video_id": video.video_id,
                "filename": video.filename,
                "video_path": str(video.path),
                "size_bytes": video.size_bytes,
                "duration_sec": video.duration_sec,
                "fps": video.fps,
                "shot_provider": provider_key,
                "shot_result_path": str(result_path),
                "shot_count": len(shots),
                "raw_shot_count": len(raw_shots),
                "dropped_short_shot_count": len(raw_shots) - len(shots),
                "min_shot_duration_ms": args.min_shot_duration_ms,
                "samples_per_shot": args.samples_per_shot,
                "expected_frame_count": len(shots) * args.samples_per_shot,
                "shots": shots,
            }
        )
    return items, skipped


def write_outputs(args: argparse.Namespace, items: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "shot_results_root": str(Path(args.shot_results_root)),
        "shot_provider": args.shot_provider,
        "samples_per_shot": args.samples_per_shot,
        "video_count": len(items),
        "skipped_count": len(skipped),
        "items": items,
    }
    write_json(output_dir / "split_sample_inputs.json", manifest)
    with (output_dir / "split_sample_inputs.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "split_sample_inputs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "filename",
                "video_path",
                "size_bytes",
                "duration_sec",
                "fps",
                "shot_provider",
                "shot_count",
                "raw_shot_count",
                "dropped_short_shot_count",
                "min_shot_duration_ms",
                "samples_per_shot",
                "expected_frame_count",
                "shot_result_path",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    shot_counts = [int(item["shot_count"]) for item in items]
    summary = {
        "video_count": len(items),
        "skipped": skipped,
        "shot_count_min": min(shot_counts) if shot_counts else 0,
        "shot_count_max": max(shot_counts) if shot_counts else 0,
        "shot_count_mean": sum(shot_counts) / len(shot_counts) if shot_counts else 0,
        "frame_count_total": sum(int(item["expected_frame_count"]) for item in items),
        "dropped_short_shot_total": sum(int(item.get("dropped_short_shot_count") or 0) for item in items),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot-results-root", default=str(DEFAULT_SHOT_ROOT))
    parser.add_argument("--shot-provider", default="aliyun_cn_shanghai")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--samples-per-shot", type=int, default=3)
    parser.add_argument("--min-shot-duration-ms", type=int, default=500)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_shot < 1:
        raise SystemExit("--samples-per-shot must be >= 1")
    items, skipped = build_items(args)
    write_outputs(args, items, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
