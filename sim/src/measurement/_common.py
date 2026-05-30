"""Shared helpers for building simulation latency parameters from measurement CSVs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Callable, Iterable, Sequence

from src.config import DATA_DIR

MEASUREMENT_DATA_DIR = DATA_DIR.parent / "measurement"
SCALE_HALF_WIDTH = 0.3
DEFAULT_RANDOM_SEED = 42

SEGMENT_SPLIT_CSV = MEASUREMENT_DATA_DIR / "segment_split_execution_time.csv"
SPEECH_CSV = MEASUREMENT_DATA_DIR / "speech_transcription_execution_time.csv"
OCR_CSV = MEASUREMENT_DATA_DIR / "ocr_execution_time.csv"
LABEL_CSV = MEASUREMENT_DATA_DIR / "label_detection_execution_time.csv"
DATABASE_CSV = MEASUREMENT_DATA_DIR / "alloydb_latency.csv"


def least_squares_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points for linear fit")
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate x values for linear fit")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def node_scale_factor(
    node_number: int,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    half_width: float = SCALE_HALF_WIDTH,
) -> float:
    """Deterministic scale k in [1 - half_width, 1 + half_width] from seed + node index."""
    rng = random.Random(f"{int(seed)}:{int(node_number)}")
    return rng.uniform(1.0 - half_width, 1.0 + half_width)


def scale_linear_intercepts(
    a_lower: float,
    b_lower: float,
    a_upper: float,
    b_upper: float,
    node_number: int,
) -> tuple[float, float, float, float]:
    k = node_scale_factor(node_number)
    return a_lower, b_lower * k, a_upper, b_upper * k


def scale_bounds(
    lower: float,
    upper: float,
    node_number: int,
) -> tuple[float, float]:
    k = node_scale_factor(node_number)
    lo, hi = lower * k, upper * k
    return (lo, hi) if lo <= hi else (hi, lo)


def _is_success(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def duration_min_max_series(
    csv_path: Path,
    *,
    duration_col: str,
    value_col: str,
    success_col: str | None = "success",
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    by_duration: dict[float, list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if success_col is not None and not _is_success(row.get(success_col)):
                continue
            if row_filter is not None and not row_filter(row):
                continue
            raw = (row.get(value_col) or "").strip()
            if not raw:
                continue
            duration = float(row[duration_col])
            by_duration.setdefault(duration, []).append(float(raw))
    if not by_duration:
        raise RuntimeError(f"No usable rows in {csv_path}")
    durations = sorted(by_duration)
    mins = [min(by_duration[d]) for d in durations]
    maxs = [max(by_duration[d]) for d in durations]
    return durations, mins, maxs


def fit_linear_uniform_params(
    csv_path: Path,
    *,
    duration_col: str,
    value_col: str,
    success_col: str | None = "success",
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[float, float, float, float]:
    xs, mins, maxs = duration_min_max_series(
        csv_path,
        duration_col=duration_col,
        value_col=value_col,
        success_col=success_col,
        row_filter=row_filter,
    )
    a_lower, b_lower = least_squares_line(xs, mins)
    a_upper, b_upper = least_squares_line(xs, maxs)
    return a_lower, b_lower, a_upper, b_upper


def execution_time_bounds(
    csv_path: Path,
    *,
    value_col: str = "execution_time_sec",
    success_col: str | None = "success",
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> tuple[float, float]:
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if success_col is not None and not _is_success(row.get(success_col)):
                continue
            if row_filter is not None and not row_filter(row):
                continue
            raw = (row.get(value_col) or "").strip()
            if not raw:
                continue
            values.append(float(raw))
    if not values:
        raise RuntimeError(f"No usable rows in {csv_path}")
    return min(values), max(values)


def add_node_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "node_number",
        type=int,
        help="Node index used to derive a deterministic ±30%% scale factor",
    )


def format_linear_tuple(params: Iterable[float]) -> str:
    a_lower, b_lower, a_upper, b_upper = params
    return (
        f"({a_lower:.6f}, {b_lower:.6f}, {a_upper:.6f}, {b_upper:.6f})"
    )


def format_bounds_tuple(bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    return f"({lo:.6f}, {hi:.6f})"
