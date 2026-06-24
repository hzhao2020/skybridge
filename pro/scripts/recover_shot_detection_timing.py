#!/usr/bin/env python3
"""Recover timing signals from an existing shot-detection profile run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


EXTRA_FIELDS = [
    "recovered_effective_latency_ms",
    "recovered_effective_latency_source",
    "recovered_latency_uncertainty_ms",
    "recovered_observed_lower_bound_ms",
    "recovered_observed_upper_bound_ms",
    "recovered_provider_start_ts",
    "recovered_provider_end_ts",
    "recovered_provider_elapsed_ms",
    "recovery_note",
]


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_ms(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() * 1000)


def gcp_provider_timing(result: dict[str, Any]) -> dict[str, Any]:
    progress = result.get("operation", {}).get("metadata", {}).get("annotationProgress", [])
    starts: list[datetime] = []
    ends: list[datetime] = []
    for item in progress:
        start = parse_iso_datetime(item.get("startTime"))
        end = parse_iso_datetime(item.get("updateTime"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return {}
    start = min(starts)
    end = max(ends)
    return {
        "recovered_provider_start_ts": start.isoformat(timespec="milliseconds"),
        "recovered_provider_end_ts": end.isoformat(timespec="milliseconds"),
        "recovered_provider_elapsed_ms": elapsed_ms(start, end),
    }


def observed_bounds(row: dict[str, str]) -> dict[str, Any]:
    upper = int(row["job_elapsed_ms"]) if row.get("job_elapsed_ms") else ""
    interval = int(float(row.get("poll_interval_sec") or 0) * 1000)
    poll_count = int(row.get("poll_count") or 0)
    if upper == "":
        return {"recovered_observed_lower_bound_ms": "", "recovered_observed_upper_bound_ms": ""}
    lower = max(0, (poll_count - 1) * interval)
    return {
        "recovered_observed_lower_bound_ms": lower,
        "recovered_observed_upper_bound_ms": upper,
    }


def set_effective_latency(row: dict[str, Any]) -> None:
    provider_elapsed = row.get("recovered_provider_elapsed_ms")
    upper = row.get("recovered_observed_upper_bound_ms")
    lower = row.get("recovered_observed_lower_bound_ms")
    if provider_elapsed != "":
        row["recovered_effective_latency_ms"] = provider_elapsed
        row["recovered_effective_latency_source"] = "gcp_operation_metadata"
        row["recovered_latency_uncertainty_ms"] = 0
        return
    row["recovered_effective_latency_ms"] = upper
    row["recovered_effective_latency_source"] = "observed_poll_upper_bound" if upper != "" else ""
    row["recovered_latency_uncertainty_ms"] = upper - lower if upper != "" and lower != "" else ""


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    k = (len(values) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def recover_run(run_dir: Path, output_path: Path) -> None:
    csv_path = run_dir / "shot_detection_profile.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    fieldnames = list(rows[0].keys()) if rows else []
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    recovered_rows: list[dict[str, Any]] = []
    effective_elapsed_by_node: dict[str, list[float]] = {}
    for row in rows:
        recovered = dict(row)
        recovered.update({field: "" for field in EXTRA_FIELDS})
        recovered.update(observed_bounds(row))

        result_path = row.get("result_path")
        result_file = run_dir / result_path if result_path else None
        if row.get("provider") == "gcp" and result_file and result_file.exists():
            timing = gcp_provider_timing(json.loads(result_file.read_text(encoding="utf-8")))
            recovered.update(timing)
            if timing.get("recovered_provider_elapsed_ms") != "":
                recovered["recovery_note"] = "gcp_provider_metadata"
        if not recovered.get("recovery_note"):
            recovered["recovery_note"] = "observed_poll_bounds_only"
        set_effective_latency(recovered)
        if recovered.get("recovered_effective_latency_ms") != "":
            effective_elapsed_by_node.setdefault(row["node_id"], []).append(float(recovered["recovered_effective_latency_ms"]))
        recovered_rows.append(recovered)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recovered_rows)

    print(f"wrote {output_path}")
    for node, values in sorted(effective_elapsed_by_node.items()):
        seconds = [value / 1000 for value in values]
        print(
            f"{node}: n={len(seconds)} "
            f"p50={percentile(seconds, 50):.2f}s "
            f"p95={percentile(seconds, 95):.2f}s "
            f"min={min(seconds):.2f}s max={max(seconds):.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = args.output or (run_dir / "shot_detection_timing_recovered.csv")
    recover_run(run_dir, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
