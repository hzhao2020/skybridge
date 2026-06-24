#!/usr/bin/env python3
"""Measure size-aware video upload/download performance for all cloud buckets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRANSFER_FIELDS = [
    "timestamp_utc",
    "run_id",
    "node_id",
    "provider",
    "region",
    "bucket",
    "direction",
    "repeat",
    "sample_index",
    "source_rank",
    "video_id",
    "filename",
    "size_bytes",
    "size_mib",
    "elapsed_sec",
    "throughput_mib_per_sec",
    "throughput_mbps",
    "status",
    "verified_size_bytes",
    "object_uri",
    "error_type",
    "error_message",
]

SUMMARY_FIELDS = [
    "node_id",
    "provider",
    "region",
    "direction",
    "sample_count",
    "success_count",
    "error_count",
    "total_mib",
    "total_elapsed_sec",
    "aggregate_mib_per_sec",
    "throughput_mib_per_sec_mean",
    "throughput_mib_per_sec_p50",
    "throughput_mib_per_sec_p10",
    "elapsed_sec_mean",
    "elapsed_sec_p50",
    "elapsed_sec_p90",
    "model_fixed_overhead_sec",
    "model_mib_per_sec",
    "model_r_squared",
]


@dataclass(frozen=True)
class Node:
    node_id: str
    provider: str
    region: str
    bucket: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_nodes(path: Path, requested_nodes: str) -> list[Node]:
    buckets = load_json(path)["buckets"]
    requested = {value.strip() for value in requested_nodes.split(",") if value.strip()}
    nodes = []
    for provider, regions in buckets.items():
        for region, bucket in regions.items():
            node_id = f"{provider}_{region.replace('-', '_')}"
            if requested and node_id not in requested:
                continue
            nodes.append(Node(node_id=node_id, provider=provider, region=region, bucket=bucket))
    missing = requested - {node.node_id for node in nodes}
    if missing:
        raise SystemExit(f"Unknown node ids: {', '.join(sorted(missing))}")
    return nodes


def select_size_stratified(items: list[dict[str, Any]], sample_count: int) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: int(item["size_bytes"]))
    for rank, item in enumerate(ordered):
        item["_source_rank"] = rank
    if sample_count <= 0 or sample_count >= len(ordered):
        return ordered
    if sample_count == 1:
        return [ordered[len(ordered) // 2]]
    ranks = [round(index * (len(ordered) - 1) / (sample_count - 1)) for index in range(sample_count)]
    return [ordered[rank] for rank in dict.fromkeys(ranks)]


def run_command(command: list[str], timeout_sec: int) -> tuple[float, str, str]:
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec)
    elapsed = time.perf_counter() - started
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(f"command exited {process.returncode}: {detail[-2000:]}")
    return elapsed, process.stdout, process.stderr


def object_uri(node: Node, key: str) -> str:
    schemes = {"gcp": "gs", "aws": "s3", "aliyun": "oss"}
    return f"{schemes[node.provider]}://{node.bucket}/{key}"


def aliyun_common(node: Node, timeout_sec: int) -> list[str]:
    return [
        "--region",
        node.region,
        "--endpoint",
        f"oss-{node.region}.aliyuncs.com",
        "--connect-timeout",
        "30",
        "--read-timeout",
        str(timeout_sec),
        "--retry-times",
        "20",
    ]


def upload_command(node: Node, local_path: Path, uri: str, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return ["gcloud", "storage", "cp", "--quiet", str(local_path), uri]
    if node.provider == "aws":
        return [
            "aws",
            "s3",
            "cp",
            str(local_path),
            uri,
            "--region",
            node.region,
            "--only-show-errors",
            "--no-progress",
        ]
    return ["aliyun", "oss", "cp", str(local_path), uri, *aliyun_common(node, timeout_sec), "--force"]


def download_command(node: Node, uri: str, local_path: Path, timeout_sec: int) -> list[str]:
    if node.provider == "gcp":
        return ["gcloud", "storage", "cp", "--quiet", uri, str(local_path)]
    if node.provider == "aws":
        return [
            "aws",
            "s3",
            "cp",
            uri,
            str(local_path),
            "--region",
            node.region,
            "--only-show-errors",
            "--no-progress",
        ]
    return ["aliyun", "oss", "cp", uri, str(local_path), *aliyun_common(node, timeout_sec), "--force"]


def remote_size(node: Node, uri: str, timeout_sec: int) -> int:
    if node.provider == "gcp":
        _, stdout, _ = run_command(
            ["gcloud", "storage", "objects", "describe", uri, "--format=value(size)"], timeout_sec
        )
        return int(stdout.strip())
    if node.provider == "aws":
        key = uri.split("/", 3)[3]
        _, stdout, _ = run_command(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                node.bucket,
                "--key",
                key,
                "--region",
                node.region,
                "--query",
                "ContentLength",
                "--output",
                "text",
            ],
            timeout_sec,
        )
        return int(stdout.strip())
    _, stdout, _ = run_command(["aliyun", "oss", "stat", uri, *aliyun_common(node, timeout_sec)], timeout_sec)
    for line in stdout.splitlines():
        if line.strip().lower().startswith("content-length"):
            return int(line.split(":", 1)[1].strip())
        if line.strip().lower().startswith("size"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not parse OSS object size: {stdout[-1000:]}")


def delete_object(node: Node, uri: str, timeout_sec: int) -> None:
    if node.provider == "gcp":
        command = ["gcloud", "storage", "rm", "--quiet", uri]
    elif node.provider == "aws":
        command = ["aws", "s3", "rm", uri, "--region", node.region, "--only-show-errors"]
    else:
        command = ["aliyun", "oss", "rm", uri, *aliyun_common(node, timeout_sec), "--force"]
    run_command(command, timeout_sec)


class Recorder:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "video_transfer_measurements.csv"
        self.lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=TRANSFER_FIELDS).writeheader()

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=TRANSFER_FIELDS, extrasaction="ignore")
                writer.writerow({field: row.get(field, "") for field in TRANSFER_FIELDS})


def measure_transfer(
    node: Node,
    item: dict[str, Any],
    direction: str,
    repeat: int,
    sample_index: int,
    run_id: str,
    temp_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    size_bytes = int(item["size_bytes"])
    source_path = Path(item["path"])
    original_key = item["object_key"]
    temporary_key = f"network_measurement/{run_id}/{node.node_id}/{repeat}/{item['filename']}"
    uri = object_uri(node, temporary_key if direction == "upload" else original_key)
    row = {
        "timestamp_utc": iso_now(),
        "run_id": run_id,
        "node_id": node.node_id,
        "provider": node.provider,
        "region": node.region,
        "bucket": node.bucket,
        "direction": direction,
        "repeat": repeat,
        "sample_index": sample_index,
        "source_rank": item["_source_rank"],
        "video_id": item["video_id"],
        "filename": item["filename"],
        "size_bytes": size_bytes,
        "size_mib": size_bytes / 1024 / 1024,
        "object_uri": uri,
    }
    download_path = temp_dir / f"{node.node_id}-{repeat}-{item['filename']}"
    uploaded = False
    try:
        if not source_path.is_file() or source_path.stat().st_size != size_bytes:
            raise ValueError(f"Local source missing or size mismatch: {source_path}")
        if direction == "upload":
            elapsed, _, _ = run_command(upload_command(node, source_path, uri, timeout_sec), timeout_sec)
            uploaded = True
            verified_size = remote_size(node, uri, timeout_sec)
        else:
            elapsed, _, _ = run_command(download_command(node, uri, download_path, timeout_sec), timeout_sec)
            verified_size = download_path.stat().st_size
        if verified_size != size_bytes:
            raise ValueError(f"Transferred size mismatch: expected={size_bytes}, actual={verified_size}")
        elapsed = max(elapsed, 0.000001)
        row.update(
            {
                "elapsed_sec": elapsed,
                "throughput_mib_per_sec": row["size_mib"] / elapsed,
                "throughput_mbps": row["size_mib"] * 8 / elapsed,
                "status": "success",
                "verified_size_bytes": verified_size,
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain errors and keep the run moving.
        row.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
            }
        )
    finally:
        download_path.unlink(missing_ok=True)
        if uploaded:
            try:
                delete_object(node, uri, timeout_sec)
            except Exception as exc:  # noqa: BLE001
                cleanup_message = f"cleanup failed: {type(exc).__name__}: {exc}"
                row["error_message"] = (
                    f"{row.get('error_message', '')}; {cleanup_message}".strip("; ")
                )[:2000]
    return row


def measure_node(
    node: Node,
    items: list[dict[str, Any]],
    recorder: Recorder,
    args: argparse.Namespace,
    run_id: str,
    temp_dir: Path,
) -> None:
    print(f"[{iso_now()}] start {node.node_id}: {len(items)} videos x {args.repeats} repeats", flush=True)
    for repeat in range(1, args.repeats + 1):
        for sample_index, item in enumerate(items, start=1):
            for direction in ("upload", "download"):
                row = measure_transfer(
                    node,
                    item,
                    direction,
                    repeat,
                    sample_index,
                    run_id,
                    temp_dir,
                    args.timeout_sec,
                )
                recorder.write(row)
                print(
                    f"[{node.node_id}] {direction} {sample_index}/{len(items)} "
                    f"{row['size_mib']:.2f} MiB status={row['status']} "
                    f"elapsed={row.get('elapsed_sec', math.nan):.3f}s "
                    f"rate={row.get('throughput_mib_per_sec', math.nan):.2f} MiB/s",
                    flush=True,
                )
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
    print(f"[{iso_now()}] done {node.node_id}", flush=True)


def fit_size_latency(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    if len(rows) < 2:
        return math.nan, math.nan, math.nan
    x = [float(row["size_mib"]) for row in rows]
    y = [float(row["elapsed_sec"]) for row in rows]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        return math.nan, math.nan, math.nan
    slope = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * value for value in x]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y, predictions))
    total = sum((actual - y_mean) ** 2 for actual in y)
    r_squared = 1 - residual / total if total > 0 else math.nan
    bandwidth = 1 / slope if slope > 0 else math.nan
    return max(0.0, intercept), bandwidth, r_squared


def summarize(csv_path: Path) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["node_id"], row["direction"]), []).append(row)
    summaries = []
    for (node_id, direction), group in sorted(groups.items()):
        successes = [row for row in group if row["status"] == "success"]
        throughputs = [float(row["throughput_mib_per_sec"]) for row in successes]
        elapsed = [float(row["elapsed_sec"]) for row in successes]
        total_mib = sum(float(row["size_mib"]) for row in successes)
        total_elapsed = sum(elapsed)
        overhead, bandwidth, r_squared = fit_size_latency(successes)
        first = group[0]
        summaries.append(
            {
                "node_id": node_id,
                "provider": first["provider"],
                "region": first["region"],
                "direction": direction,
                "sample_count": len(group),
                "success_count": len(successes),
                "error_count": len(group) - len(successes),
                "total_mib": total_mib,
                "total_elapsed_sec": total_elapsed,
                "aggregate_mib_per_sec": total_mib / total_elapsed if total_elapsed else math.nan,
                "throughput_mib_per_sec_mean": statistics.mean(throughputs) if throughputs else math.nan,
                "throughput_mib_per_sec_p50": percentile(throughputs, 50),
                "throughput_mib_per_sec_p10": percentile(throughputs, 10),
                "elapsed_sec_mean": statistics.mean(elapsed) if elapsed else math.nan,
                "elapsed_sec_p50": percentile(elapsed, 50),
                "elapsed_sec_p90": percentile(elapsed, 90),
                "model_fixed_overhead_sec": overhead,
                "model_mib_per_sec": bandwidth,
                "model_r_squared": r_squared,
            }
        )
    return summaries


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="experiments/activitynet_splits/upload_manifests/profile_videos_upload_manifest_seed20260622.json",
    )
    parser.add_argument("--buckets", default="configs/buckets.json")
    parser.add_argument("--nodes", default="", help="Comma-separated node ids; default is all 12.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Evenly spaced files across the size distribution; 0 uses all videos.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--node-concurrency", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--run-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"experiments/video_transfer_network/run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    items = load_json(Path(args.manifest))["items"]
    selected = select_size_stratified(items, args.sample_count)
    nodes = load_nodes(Path(args.buckets), args.nodes)
    metadata = {
        "run_id": run_id,
        "created_at": iso_now(),
        "manifest": args.manifest,
        "buckets": args.buckets,
        "nodes": [node.__dict__ for node in nodes],
        "video_population_count": len(items),
        "sample_count": len(selected),
        "repeats": args.repeats,
        "node_concurrency": args.node_concurrency,
        "selection": [
            {
                "video_id": item["video_id"],
                "filename": item["filename"],
                "source_rank": item["_source_rank"],
                "size_bytes": item["size_bytes"],
                "size_mib": item["size_bytes"] / 1024 / 1024,
            }
            for item in selected
        ],
        "method": (
            "Wall-clock CLI upload/download of real videos. Uploads use unique temporary object keys and are "
            "deleted after verification. Downloads use existing profile objects and are size-verified locally. "
            "Nodes run sequentially by default to avoid client bandwidth contention."
        ),
    }
    write_json(run_dir / "metadata.json", metadata)
    recorder = Recorder(run_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"skyflow-network-{run_id}-"))
    try:
        with ThreadPoolExecutor(max_workers=args.node_concurrency) as executor:
            futures = [
                executor.submit(measure_node, node, selected, recorder, args, run_id, temp_dir) for node in nodes
            ]
            for future in as_completed(futures):
                future.result()
        summaries = summarize(recorder.path)
        write_summary(run_dir / "video_transfer_summary.csv", summaries)
        metadata["finished_at"] = iso_now()
        metadata["summary"] = summaries
        write_json(run_dir / "metadata.json", metadata)
        print(f"[{iso_now()}] all done: {run_dir}", flush=True)
        return 0
    except Exception:  # noqa: BLE001
        write_json(run_dir / "error.json", {"timestamp": iso_now(), "traceback": traceback.format_exc()})
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
