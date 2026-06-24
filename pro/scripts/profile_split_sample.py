#!/usr/bin/env python3
"""Profile cloud split/sample endpoints with normalized shot inputs."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "node_id",
    "provider",
    "region",
    "video_id",
    "filename",
    "video_bytes",
    "duration_sec",
    "shot_count",
    "samples_per_shot",
    "expected_frame_count",
    "submit_start_ts",
    "response_headers_ts",
    "complete_ts",
    "wall_latency_ms",
    "time_to_headers_ms",
    "response_read_ms",
    "status",
    "http_status",
    "frame_count",
    "frame_image_bytes",
    "request_json_bytes",
    "request_video_b64_bytes",
    "request_shots_json_bytes",
    "response_json_bytes",
    "upload_mb_per_sec_observed",
    "download_mb_per_sec_observed",
    "result_path",
    "frames_dir",
    "error_type",
    "error_message",
]


@dataclass(frozen=True)
class Node:
    node_id: str
    provider: str
    region: str
    endpoint: str
    samples_per_shot: int


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def elapsed_ms(start: float, end: float) -> int:
    return int(round((end - start) * 1000))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


class Recorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.csv_path = run_dir / "split_sample_profile.csv"
        self.jsonl_path = run_dir / "split_sample_profile_raw.jsonl"
        self.lock = threading.Lock()
        run_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def write_row(self, row: dict[str, Any], raw: dict[str, Any]) -> None:
        with self.lock:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"row": row, "raw": raw}, ensure_ascii=False) + "\n")

    def result_paths(self, node: Node, item: dict[str, Any]) -> tuple[Path, Path, str, str]:
        result_dir = self.run_dir / "results" / node.node_id / item["video_id"]
        frames_dir = result_dir / "frames"
        result_path = result_dir / "response.json"
        return (
            result_path,
            frames_dir,
            str(result_path.relative_to(self.run_dir)),
            str(frames_dir.relative_to(self.run_dir)),
        )


def load_inputs(path: Path, max_videos: int = 0) -> list[dict[str, Any]]:
    data = load_json(path)
    items = data["items"] if isinstance(data, dict) else data
    if max_videos:
        items = items[:max_videos]
    return items


def load_nodes(config_path: Path, requested_nodes: str = "") -> list[Node]:
    config = load_json(config_path)
    providers = config["providers"]["split_sample"]
    requested = {node.strip() for node in requested_nodes.split(",") if node.strip()}
    nodes: list[Node] = []
    for node_id, data in providers.items():
        if requested and node_id not in requested:
            continue
        if data.get("type") != "http_json":
            continue
        nodes.append(
            Node(
                node_id=node_id,
                provider=data["provider"],
                region=data["region"],
                endpoint=data["endpoint"],
                samples_per_shot=int(data.get("samples_per_shot") or 3),
            )
        )
    missing = sorted(requested - {node.node_id for node in nodes})
    if missing:
        raise SystemExit(f"Unknown split_sample node ids: {', '.join(missing)}")
    return nodes


def completed_pairs(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["node_id"], row["video_id"]))
    return done


def make_payload(item: dict[str, Any], samples_per_shot: int) -> tuple[dict[str, Any], dict[str, int]]:
    video_path = Path(item["video_path"])
    video_bytes = video_path.read_bytes()
    video_b64 = base64.b64encode(video_bytes).decode("ascii")
    shots = item["shots"]
    payload = {
        "video": {"name": item["filename"], "data_b64": video_b64},
        "shots": shots,
        "samples_per_shot": samples_per_shot,
    }
    metrics = {
        "video_bytes": len(video_bytes),
        "request_video_b64_bytes": len(video_b64.encode("ascii")),
        "request_shots_json_bytes": len(json.dumps(shots, separators=(",", ":")).encode("utf-8")),
    }
    return payload, metrics


def post_json(endpoint: str, payload: dict[str, Any], timeout_sec: int) -> tuple[int, bytes, int, int, int]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            headers_at = time.perf_counter()
            response_body = response.read()
            end = time.perf_counter()
            return response.status, response_body, elapsed_ms(start, headers_at), elapsed_ms(headers_at, end), len(body)
    except urllib.error.HTTPError as exc:
        headers_at = time.perf_counter()
        response_body = exc.read()
        end = time.perf_counter()
        return exc.code, response_body, elapsed_ms(start, headers_at), elapsed_ms(headers_at, end), len(body)


def strip_and_save_response(
    response: dict[str, Any],
    result_path: Path,
    frames_dir: Path,
    save_frames: bool,
    keep_frame_b64: bool,
) -> tuple[int, int]:
    frames = response.get("frames") or []
    frame_image_bytes = 0
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    stored_frames = []
    for frame in frames:
        stored = dict(frame)
        data_b64 = frame.get("data_b64") or ""
        if data_b64:
            frame_bytes = base64.b64decode(data_b64)
            frame_image_bytes += len(frame_bytes)
            if save_frames:
                frame_path = frames_dir / f"{frame['frame_id']}.jpg"
                frame_path.write_bytes(frame_bytes)
                stored["local_path"] = str(frame_path.relative_to(result_path.parent.parent.parent.parent))
        if not keep_frame_b64:
            stored.pop("data_b64", None)
        stored_frames.append(stored)
    stored_response = dict(response)
    stored_response["frames"] = stored_frames
    write_json(result_path, stored_response)
    return len(frames), frame_image_bytes


def profile_one(
    node: Node,
    item: dict[str, Any],
    recorder: Recorder,
    timeout_sec: int,
    save_frames: bool,
    keep_frame_b64: bool,
    samples_per_shot_override: int,
) -> None:
    samples_per_shot = samples_per_shot_override or int(item.get("samples_per_shot") or node.samples_per_shot)
    result_path, frames_dir, result_rel, frames_rel = recorder.result_paths(node, item)
    row: dict[str, Any] = {
        "node_id": node.node_id,
        "provider": node.provider,
        "region": node.region,
        "video_id": item["video_id"],
        "filename": item["filename"],
        "duration_sec": item.get("duration_sec", ""),
        "shot_count": item.get("shot_count", len(item.get("shots", []))),
        "samples_per_shot": samples_per_shot,
        "expected_frame_count": int(item.get("shot_count", len(item.get("shots", [])))) * samples_per_shot,
        "result_path": result_rel,
        "frames_dir": frames_rel if save_frames else "",
    }
    raw: dict[str, Any] = {"endpoint": node.endpoint}
    try:
        payload, payload_metrics = make_payload(item, samples_per_shot)
        row.update(payload_metrics)
        row["submit_start_ts"] = iso_now()
        wall_start = time.perf_counter()
        http_status, response_body, time_to_headers_ms, response_read_ms, request_json_bytes = post_json(
            node.endpoint, payload, timeout_sec
        )
        wall_end = time.perf_counter()
        row["response_headers_ts"] = iso_now()
        row["complete_ts"] = iso_now()
        row["wall_latency_ms"] = elapsed_ms(wall_start, wall_end)
        row["time_to_headers_ms"] = time_to_headers_ms
        row["response_read_ms"] = response_read_ms
        row["request_json_bytes"] = request_json_bytes
        row["response_json_bytes"] = len(response_body)
        row["http_status"] = http_status
        response_text = response_body.decode("utf-8")
        response = json.loads(response_text)
        raw["response_preview"] = response_text[:4000]
        if http_status >= 400 or "error" in response:
            row["status"] = "error"
            row["error_type"] = f"HTTP_{http_status}"
            row["error_message"] = response.get("error", response_text[:1000])
            write_json(result_path, response)
        else:
            frame_count, frame_image_bytes = strip_and_save_response(
                response, result_path, frames_dir, save_frames=save_frames, keep_frame_b64=keep_frame_b64
            )
            row["status"] = "success"
            row["frame_count"] = frame_count
            row["frame_image_bytes"] = frame_image_bytes
        wall_sec = max(0.001, float(row["wall_latency_ms"]) / 1000.0)
        row["upload_mb_per_sec_observed"] = (row["request_json_bytes"] / 1024 / 1024) / wall_sec
        row["download_mb_per_sec_observed"] = (row["response_json_bytes"] / 1024 / 1024) / wall_sec
    except Exception as exc:  # noqa: BLE001 - keep the run moving across videos.
        row["complete_ts"] = iso_now()
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)[:1000]
        raw["traceback"] = traceback.format_exc()
    recorder.write_row(row, raw)
    print(
        f"[{node.node_id}] {item['video_id']} status={row.get('status')} "
        f"latency_ms={row.get('wall_latency_ms', '')} frames={row.get('frame_count', '')}",
        flush=True,
    )


def profile_node(
    node: Node,
    items: list[dict[str, Any]],
    recorder: Recorder,
    done: set[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    print(f"[{iso_now()}] start {node.node_id} ({len(items)} videos)", flush=True)
    for item in items:
        if not args.rerun_existing and (node.node_id, item["video_id"]) in done:
            print(f"[{node.node_id}] skip {item['video_id']}", flush=True)
            continue
        profile_one(
            node,
            item,
            recorder,
            timeout_sec=args.timeout_sec,
            save_frames=not args.no_save_frames,
            keep_frame_b64=args.keep_frame_b64,
            samples_per_shot_override=args.samples_per_shot,
        )
        if args.per_video_sleep_sec > 0:
            time.sleep(args.per_video_sleep_sec)
    print(f"[{iso_now()}] done {node.node_id}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        default="experiments/split_sample_inputs/final_50_seed20260622/split_sample_inputs.json",
    )
    parser.add_argument("--config", default="configs/prototype.hybrid.split_sample.json")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--samples-per-shot", type=int, default=0, help="Override manifest/node samples_per_shot.")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--node-concurrency", type=int, default=3)
    parser.add_argument("--per-video-sleep-sec", type=float, default=0.0)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--keep-frame-b64", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"experiments/split_sample_profile/run_{run_id}")
    recorder = Recorder(run_dir)
    items = load_inputs(Path(args.inputs), args.max_videos)
    nodes = load_nodes(Path(args.config), args.nodes)
    metadata = {
        "run_id": run_id,
        "inputs": args.inputs,
        "config": args.config,
        "run_dir": str(run_dir),
        "nodes": [node.__dict__ for node in nodes],
        "videos": len(items),
        "started_at": iso_now(),
        "save_frames": not args.no_save_frames,
        "keep_frame_b64": args.keep_frame_b64,
    }
    write_json(run_dir / "metadata.json", metadata)
    done = completed_pairs(recorder.csv_path)
    with ThreadPoolExecutor(max_workers=args.node_concurrency) as executor:
        futures = [executor.submit(profile_node, node, items, recorder, done, args) for node in nodes]
        for future in as_completed(futures):
            future.result()
    metadata["finished_at"] = iso_now()
    write_json(run_dir / "metadata.json", metadata)
    print(f"[{iso_now()}] all done: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
