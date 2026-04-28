"""
execution_latency.py
Execution-time models for segment/split (min/max envelope from measurements) and LLMs
(TTFT + tokens / throughput).

Segment & split: For each measured clip duration in segment_split_execution_time.csv,
we take min/max of segment and of split times across trials. Piecewise-linear (linear
interpolation) curves connect these points in duration space; outside the measured
range we linearly extrapolate from the nearest end segment. For any video duration,
segment and split times are sampled uniformly and independently between their
respective min and max bounds at that duration.

LLM: total decode latency = TTFT + output_tokens / (output tokens per second),
using SkyXXX Simulation.md benchmark table (model names match config.py).
"""

from __future__ import annotations

import csv
import random
from bisect import bisect_right
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
# Segment / split: min–max envelope + uniform sampling
# ---------------------------------------------------------------------------
_MEASUREMENT_CSV = (
    Path(__file__).resolve().parent
    / "measurement_data"
    / "segment_split_execution_time.csv"
)


@dataclass(frozen=True)
class SegmentSplitBounds:
    """Interpolated/extrapolated min–max bounds (seconds) at a clip duration."""

    duration_sec: float
    segment_min_sec: float
    segment_max_sec: float
    split_min_sec: float
    split_max_sec: float


@dataclass(frozen=True)
class SegmentSplitSample:
    segment_execute_sec: float
    split_execute_sec: float
    duration_sec: float


def _piecewise_linear_eval(
    anchors: list[float],
    values: list[float],
    d_req: float,
) -> float:
    """Linear interpolation between anchors; linear extrapolation beyond ends."""
    if len(anchors) != len(values):
        raise ValueError("anchors and values length mismatch")
    if len(anchors) == 1:
        return values[0]
    if d_req <= anchors[0]:
        slope = (values[1] - values[0]) / (anchors[1] - anchors[0])
        return values[0] + slope * (d_req - anchors[0])
    if d_req >= anchors[-1]:
        slope = (values[-1] - values[-2]) / (anchors[-1] - anchors[-2])
        return values[-1] + slope * (d_req - anchors[-1])
    i = bisect_right(anchors, d_req) - 1
    d0, d1 = anchors[i], anchors[i + 1]
    t = (d_req - d0) / (d1 - d0)
    return (1.0 - t) * values[i] + t * values[i + 1]


def _enforce_min_le_max(lo: float, hi: float) -> tuple[float, float]:
    if lo <= hi:
        return lo, hi
    return hi, lo


class SegmentSplitLatencyModel:
    """
    Per measured duration: min/max of segment and of split execution times.
    Bounds vs. duration are piecewise-linear (linear fit between measurement points);
    sample uniformly in [min, max] for segment and for split independently.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path or _MEASUREMENT_CSV
        self._anchors: list[float] | None = None
        self._seg_min: list[float] | None = None
        self._seg_max: list[float] | None = None
        self._spl_min: list[float] | None = None
        self._spl_max: list[float] | None = None

    def _ensure_loaded(self) -> None:
        if self._anchors is not None:
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
        durations = sorted(by_d.keys())
        seg_min, seg_max, spl_min, spl_max = [], [], [], []
        for d in durations:
            segs = [p[0] for p in by_d[d]]
            spls = [p[1] for p in by_d[d]]
            sm_s, sx_s = min(segs), max(segs)
            sm_p, sx_p = min(spls), max(spls)
            sm_s, sx_s = _enforce_min_le_max(sm_s, sx_s)
            sm_p, sx_p = _enforce_min_le_max(sm_p, sx_p)
            seg_min.append(sm_s)
            seg_max.append(sx_s)
            spl_min.append(sm_p)
            spl_max.append(sx_p)
        self._anchors = durations
        self._seg_min = seg_min
        self._seg_max = seg_max
        self._spl_min = spl_min
        self._spl_max = spl_max

    @property
    def measured_durations_sec(self) -> list[float]:
        self._ensure_loaded()
        assert self._anchors is not None
        return list(self._anchors)

    def bounds_at(self, video_duration_sec: float) -> SegmentSplitBounds:
        """
        Min/max envelope for segment and split at `video_duration_sec` (seconds),
        from piecewise-linear curves through per-bucket minima/maxima.
        """
        self._ensure_loaded()
        assert (
            self._anchors is not None
            and self._seg_min is not None
            and self._seg_max is not None
            and self._spl_min is not None
            and self._spl_max is not None
        )
        d_req = float(video_duration_sec)
        if d_req <= 0:
            raise ValueError("video_duration_sec must be positive")
        sm = _piecewise_linear_eval(self._anchors, self._seg_min, d_req)
        sx = _piecewise_linear_eval(self._anchors, self._seg_max, d_req)
        pm = _piecewise_linear_eval(self._anchors, self._spl_min, d_req)
        px = _piecewise_linear_eval(self._anchors, self._spl_max, d_req)
        sm, sx = _enforce_min_le_max(sm, sx)
        pm, px = _enforce_min_le_max(pm, px)
        return SegmentSplitBounds(d_req, sm, sx, pm, px)

    def sample(
        self,
        video_duration_sec: float,
        rng: random.Random | None = None,
    ) -> SegmentSplitSample:
        """
        Independent uniform draws: segment ~ U(segment_min, segment_max),
        split ~ U(split_min, split_max) at the interpolated bounds for this duration.
        """
        r = rng or random.Random()
        b = self.bounds_at(video_duration_sec)
        seg = r.uniform(b.segment_min_sec, b.segment_max_sec)
        spl = r.uniform(b.split_min_sec, b.split_max_sec)
        return SegmentSplitSample(seg, spl, b.duration_sec)


_DEFAULT_SEGMENT_SPLIT_MODEL = SegmentSplitLatencyModel()


def sample_segment_split(
    video_duration_sec: float,
    rng: random.Random | None = None,
) -> SegmentSplitSample:
    """Convenience wrapper using the bundled measurement CSV."""
    return _DEFAULT_SEGMENT_SPLIT_MODEL.sample(video_duration_sec, rng=rng)


def segment_split_bounds_at(video_duration_sec: float) -> SegmentSplitBounds:
    """Interpolated min/max bounds for segment and split at the given clip length."""
    return _DEFAULT_SEGMENT_SPLIT_MODEL.bounds_at(video_duration_sec)


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
    """Clip durations (seconds) that appear in the measurement CSV."""
    return _DEFAULT_SEGMENT_SPLIT_MODEL.measured_durations_sec
