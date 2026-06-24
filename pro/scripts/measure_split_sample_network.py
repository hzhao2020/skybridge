#!/usr/bin/env python3
"""Measure split/sample endpoint network probes and summarize workload transfer metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBE_FIELDS = [
    "timestamp_utc",
    "node_id",
    "provider",
    "region",
    "endpoint",
    "probe_index",
    "status",
    "http_status",
    "latency_ms",
    "response_bytes",
    "error_type",
    "error_message",
]

SUMMARY_FIELDS = [
    "node_id",
    "provider",
    "region",
    "sample_count",
    "success_count",
    "error_count",
    "request_json_mb_mean",
    "response_json_mb_mean",
    "wall_latency_ms_mean",
    "wall_latency_ms_p50",
    "wall_latency_ms_p90",
    "observed_upload_mb_per_sec_mean",
    "observed_download_mb_per_sec_mean",
    "total_request_json_mb",
    "total_response_json_mb",
]


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


def load_nodes(config_path: Path, requested_nodes: str = "") -> list[dict[str, Any]]:
    config = load_json(config_path)
    providers = config["providers"]["split_sample"]
    requested = {node.strip() for node in requested_nodes.split(",") if node.strip()}
    nodes = []
    for node_id, data in providers.items():
        if requested and node_id not in requested:
            continue
        if data.get("type") != "http_json":
            continue
        nodes.append(
            {
                "node_id": node_id,
                "provider": data["provider"],
                "region": data["region"],
                "endpoint": data["endpoint"],
            }
        )
    missing = sorted(requested - {node["node_id"] for node in nodes})
    if missing:
        raise SystemExit(f"Unknown split_sample node ids: {', '.join(missing)}")
    return nodes


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def probe_endpoint(node: dict[str, Any], probe_index: int, timeout_sec: int) -> dict[str, Any]:
    row = {
        "timestamp_utc": iso_now(),
        "node_id": node["node_id"],
        "provider": node["provider"],
        "region": node["region"],
        "endpoint": node["endpoint"],
        "probe_index": probe_index,
    }
    start = time.perf_counter()
    request = urllib.request.Request(node["endpoint"], method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read()
            end = time.perf_counter()
            row.update(
                {
                    "status": "success",
                    "http_status": response.status,
                    "latency_ms": int(round((end - start) * 1000)),
                    "response_bytes": len(body),
                }
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        end = time.perf_counter()
        row.update(
            {
                "status": "error",
                "http_status": exc.code,
                "latency_ms": int(round((end - start) * 1000)),
                "response_bytes": len(body),
                "error_type": "HTTPError",
                "error_message": body.decode("utf-8", errors="replace")[:1000],
            }
        )
    except Exception as exc:  # noqa: BLE001 - record failed probes.
        end = time.perf_counter()
        row.update(
            {
                "status": "error",
                "latency_ms": int(round((end - start) * 1000)),
                "response_bytes": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            }
        )
    return row


def write_probe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROBE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_probes(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    if args.probes <= 0:
        return []
    nodes = load_nodes(Path(args.config), args.nodes)
    rows: list[dict[str, Any]] = []
    for probe_index in range(1, args.probes + 1):
        for node in nodes:
            row = probe_endpoint(node, probe_index, args.timeout_sec)
            rows.append(row)
            print(
                f"[probe] {node['node_id']} #{probe_index} status={row.get('status')} "
                f"latency_ms={row.get('latency_ms')} http={row.get('http_status', '')}",
                flush=True,
            )
            if args.probe_sleep_sec > 0:
                time.sleep(args.probe_sleep_sec)
    write_probe_csv(run_dir / "endpoint_network_probe.csv", rows)
    return rows


def as_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    return float(value)


def summarize_profile(profile_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not profile_csv:
        return [], {}
    rows = list(csv.DictReader(profile_csv.open(encoding="utf-8")))
    by_node: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_node.setdefault(row["node_id"], []).append(row)
    summaries = []
    for node_id, node_rows in sorted(by_node.items()):
        success_rows = [row for row in node_rows if row.get("status") == "success"]
        latencies = [as_float(row.get("wall_latency_ms")) for row in success_rows]
        latencies = [value for value in latencies if not math.isnan(value)]
        request_mb = [as_float(row.get("request_json_bytes")) / 1024 / 1024 for row in success_rows]
        response_mb = [as_float(row.get("response_json_bytes")) / 1024 / 1024 for row in success_rows]
        upload = [as_float(row.get("upload_mb_per_sec_observed")) for row in success_rows]
        download = [as_float(row.get("download_mb_per_sec_observed")) for row in success_rows]
        request_mb = [value for value in request_mb if not math.isnan(value)]
        response_mb = [value for value in response_mb if not math.isnan(value)]
        upload = [value for value in upload if not math.isnan(value)]
        download = [value for value in download if not math.isnan(value)]
        first = node_rows[0]
        summaries.append(
            {
                "node_id": node_id,
                "provider": first.get("provider", ""),
                "region": first.get("region", ""),
                "sample_count": len(node_rows),
                "success_count": len(success_rows),
                "error_count": len(node_rows) - len(success_rows),
                "request_json_mb_mean": statistics.mean(request_mb) if request_mb else math.nan,
                "response_json_mb_mean": statistics.mean(response_mb) if response_mb else math.nan,
                "wall_latency_ms_mean": statistics.mean(latencies) if latencies else math.nan,
                "wall_latency_ms_p50": percentile(latencies, 50),
                "wall_latency_ms_p90": percentile(latencies, 90),
                "observed_upload_mb_per_sec_mean": statistics.mean(upload) if upload else math.nan,
                "observed_download_mb_per_sec_mean": statistics.mean(download) if download else math.nan,
                "total_request_json_mb": sum(request_mb),
                "total_response_json_mb": sum(response_mb),
            }
        )
    metadata = {
        "profile_csv": str(profile_csv),
        "rows": len(rows),
        "nodes": len(summaries),
        "note": "Throughput is observed over the whole HTTP request, so it includes server execution time as well as network transfer.",
    }
    return summaries, metadata


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prototype.hybrid.split_sample.json")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--probe-sleep-sec", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--profile-csv", default="", help="Optional split_sample_profile.csv to summarize.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir or f"experiments/split_sample_network/run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        probe_rows = run_probes(args, run_dir)
        profile_summaries: list[dict[str, Any]] = []
        profile_metadata: dict[str, Any] = {}
        if args.profile_csv:
            profile_summaries, profile_metadata = summarize_profile(Path(args.profile_csv))
            write_summary_csv(run_dir / "split_sample_network_summary.csv", profile_summaries)
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "created_at": iso_now(),
            "probe_count": len(probe_rows),
            "probe_success_count": sum(1 for row in probe_rows if row.get("status") == "success"),
            "profile_summary": profile_metadata,
            "profile_nodes": profile_summaries,
        }
        write_json(run_dir / "network_measurement_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception:  # noqa: BLE001 - leave a useful failure artifact.
        write_json(run_dir / "network_measurement_error.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
