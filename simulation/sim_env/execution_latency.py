"""
execution_latency.py
Execution-time models for segment/split (empirical from measurements) and LLMs (TTFT + tokens / throughput).

Segment & split: driven by video clip duration in seconds (the sweep axis in
segment_split_execution_time.csv). Samples are drawn from per-bucket empirical
distributions; between measured durations, two bucket samples are blended
linearly in duration space.

LLM: total decode latency = TTFT + output_tokens / (output tokens per second),
using SkyXXX Simulation.md benchmark table (model names match config.py).
"""

from __future__ import annotations

import csv
import random
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM: TTFT (seconds) and output throughput (tokens / second)
# ---------------------------------------------------------------------------
LLM_LATENCY_SPECS: dict[str, tuple[float, float]] = {
    "Gemini 2.5 Pro": (2.64, 72.0),
    "Gemini 2.5 Flash": (0.90, 64.0),
    "Claude 3.5 Sonnet": (0.99, 30.5),
    "Claude 3.5 Haiku": (0.68, 32.50),
    "Qwen3-VL-Plus": (0.69, 39.44),
    "Qwen3-VL-Flash": (0.76, 45.90),
    "Qwen3.5-Plus": (2.32, 94.0),
    "Qwen3.5-Flash": (0.50, 84.0),
}


def llm_decode_duration_sec(model: str, output_tokens: float) -> float:
    """
    End-to-end generation time for the decoding phase:
    TTFT + (output tokens) / (output tokens per second).

    Caption vs query only changes how you model `output_tokens`; latency params
    are per model name in the benchmark table.
    """
    try:
        ttft_s, tps = LLM_LATENCY_SPECS[model]
    except KeyError as e:
        raise KeyError(f"Unknown LLM model for latency: {model!r}") from e
    if tps <= 0:
        raise ValueError(f"Non-positive throughput for model {model!r}")
    return ttft_s + float(output_tokens) / tps


def get_llm_ttft_and_throughput(model: str) -> tuple[float, float]:
    """Return (ttft_seconds, output_tokens_per_second)."""
    return LLM_LATENCY_SPECS[model]


# ---------------------------------------------------------------------------
# Segment / split empirical model
# ---------------------------------------------------------------------------
_MEASUREMENT_CSV = (
    Path(__file__).resolve().parent
    / "measurement_data"
    / "segment_split_execution_time.csv"
)


@dataclass(frozen=True)
class SegmentSplitSample:
    segment_execute_sec: float
    split_execute_sec: float
    duration_sec: float


class SegmentSplitLatencyModel:
    """
    Empirical distributions of `node_segment_execute_sec` and
    `node_split_execute_http_observed_sec` conditioned on `duration_sec`.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path or _MEASUREMENT_CSV
        self._by_duration: dict[float, list[tuple[float, float]]] | None = None
        self._sorted_durations: list[float] | None = None

    def _ensure_loaded(self) -> None:
        if self._by_duration is not None:
            return
        by_d: dict[float, list[tuple[float, float]]] = {}
        with self._csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("success") != "True":
                    continue
                d = float(row["duration_sec"])
                seg = float(row["node_segment_execute_sec"])
                spl = float(row["node_split_execute_http_observed_sec"])
                by_d.setdefault(d, []).append((seg, spl))
        if not by_d:
            raise RuntimeError(f"No successful rows in {self._csv_path}")
        self._by_duration = by_d
        self._sorted_durations = sorted(by_d.keys())

    @property
    def measured_durations_sec(self) -> list[float]:
        self._ensure_loaded()
        assert self._sorted_durations is not None
        return list(self._sorted_durations)

    def sample(
        self,
        video_duration_sec: float,
        rng: random.Random | None = None,
    ) -> SegmentSplitSample:
        """
        Draw one joint sample of segment and split execution times for a clip
        of length `video_duration_sec` (seconds).

        Uses empirical resampling within the bracketing measurement buckets and
        linearly blends the two draws when duration falls strictly between
        adjacent measured lengths.
        """
        self._ensure_loaded()
        assert self._by_duration is not None and self._sorted_durations is not None
        r = rng or random.Random()
        d_req = float(video_duration_sec)
        if d_req <= 0:
            raise ValueError("video_duration_sec must be positive")

        anchors = self._sorted_durations
        if d_req <= anchors[0]:
            seg, spl = self._sample_bucket(anchors[0], r)
            return SegmentSplitSample(seg, spl, anchors[0])
        if d_req >= anchors[-1]:
            seg, spl = self._sample_bucket(anchors[-1], r)
            return SegmentSplitSample(seg, spl, anchors[-1])

        i = bisect_left(anchors, d_req)
        if i < len(anchors) and anchors[i] == d_req:
            seg, spl = self._sample_bucket(d_req, r)
            return SegmentSplitSample(seg, spl, d_req)

        d_lo, d_hi = anchors[i - 1], anchors[i]
        w = (d_req - d_lo) / (d_hi - d_lo)
        s_lo = self._sample_bucket(d_lo, r)
        s_hi = self._sample_bucket(d_hi, r)
        seg = (1.0 - w) * s_lo[0] + w * s_hi[0]
        spl = (1.0 - w) * s_lo[1] + w * s_hi[1]
        return SegmentSplitSample(seg, spl, d_req)

    def _sample_bucket(self, duration_sec: float, rng: random.Random) -> tuple[float, float]:
        assert self._by_duration is not None
        pairs = self._by_duration[duration_sec]
        return rng.choice(pairs)


_DEFAULT_SEGMENT_SPLIT_MODEL = SegmentSplitLatencyModel()


def sample_segment_split(
    video_duration_sec: float,
    rng: random.Random | None = None,
) -> SegmentSplitSample:
    """Convenience wrapper using the bundled measurement CSV."""
    return _DEFAULT_SEGMENT_SPLIT_MODEL.sample(video_duration_sec, rng=rng)


def sample_segment_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
) -> float:
    return sample_segment_split(video_duration_sec, rng=rng).segment_execute_sec


def sample_split_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
) -> float:
    return sample_segment_split(video_duration_sec, rng=rng).split_execute_sec


def segment_split_bucket_sizes() -> list[float]:
    """Measured clip durations present in the CSV (seconds)."""
    return _DEFAULT_SEGMENT_SPLIT_MODEL.measured_durations_sec
