"""Build database query simulation bounds from database measurement."""

from __future__ import annotations

import csv

from src.measurement._common import DATABASE_CSV, scale_bounds


def build_params(node_number: int) -> tuple[float, float]:
    latencies_sec: list[float] = []
    with DATABASE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("Latency_Avg_ms") or "").strip()
            if not raw:
                continue
            latencies_sec.append(float(raw) / 1000.0)
    if not latencies_sec:
        raise RuntimeError(f"No Latency_Avg_ms rows in {DATABASE_CSV}")
    return scale_bounds(min(latencies_sec), max(latencies_sec), node_number)
