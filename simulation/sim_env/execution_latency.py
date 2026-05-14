"""
execution_latency.py
Execution-time models for measured video/split steps, Video Intelligence (label / OCR /
speech), and LLMs (TTFT + tokens / throughput).

Shot detection & video split: ``segment_split_execution_time.csv`` bundles two columns for measurement
convenience only; **shot_detection and video_split are independent** in the simulator. Interpolated
min/max per operation; sample only via ``sample_shot_detection_execute_sec`` /
``sample_video_split_execute_sec`` (never one draw for both). ``shot_detection_video_split_bounds_at`` returns
both envelopes for inspection only.

Label / OCR / speech: ``label_ocr_speech_execution_time.csv`` — **same idea**: one file,
**three independent** operations; sample only via ``sample_label_detection_execute_sec``,
``sample_ocr_execute_sec``, ``sample_speech_transcription_execute_sec``. Inspect-only combined
bounds: ``label_ocr_speech_bounds_at``.

Database retrieval (query latency): ``alloydb_latency.csv`` — average latency samples in ms;
bounds are the file min/max converted to seconds. Sample via ``sample_database_query_execute_sec``
(uniform inside scaled bounds, same per-node ``k`` as other execution ops).

With a ``PhysicalNode``, scaled latency uses ``k ~ Uniform([1±0.3])`` per endpoint and
latency operation.

**Recommended (reproducible):** pass ``execution_scale_seed: int``. Then ``k`` is derived
deterministically from ``(seed, execution_scale_scope, provider, region, latency_op)`` via
a stable hashed sub-RNG—no dependence on outer ``rng`` call order.

**Alternative:** omit ``execution_scale_seed`` (``None``). Then ``k`` is drawn once per
cached key ``(execution_scale_scope, provider, region, latency_op)`` using the caller's
``rng`` on first access, then reused (still fixed per endpoint/op within a scope).

LLM jitter is unchanged: each ``llm_decode_duration_sec(..., rng=…)`` draws **fresh**
independent TTFT / throughput factors.

LLM decode latency uses the benchmark table as **nominals** (model names match
``config.py``). When a ``random.Random`` is passed into ``llm_decode_duration_sec`` / helpers,
TTFT and throughput each get **independent** multiplicative jitter
``Uniform(1 - w, 1 + w)`` with ``w = LATENCY_RANDOM_HALF_WIDTH`` (default 30%), every call.
"""

from __future__ import annotations

import csv
import hashlib
import random
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sim_env.utility import PhysicalNode

# ---------------------------------------------------------------------------
# LLM: nominal TTFT (seconds) + output throughput (tokens / second); optional ±jitter.
# ---------------------------------------------------------------------------
LATENCY_RANDOM_HALF_WIDTH = 0.3


LLM_LATENCY_SPECS: dict[str, tuple[float, float]] = {
    # Nominals from SkyAPI Evaluation.md (TTFT s, output tokens/s); jitter ±30% elsewhere.
    "Gemini 2.5 Pro": (2.64, 86.0),
    "Gemini 2.5 Flash": (0.67, 77.0),
    "Amazon Nova Pro": (0.43, 34.50),
    "Amazon Nova Lite": (0.46, 51.50),
    "Qwen3-VL-Plus": (1.29, 54.0),
    "Qwen3-VL-Flash": (0.54, 87.0),
}


def llm_nominal_ttft_and_tps(model: str) -> tuple[float, float]:
    """Baseline table (no jitter): ``(ttft_sec, tokens_per_second)``."""
    try:
        ttft_s, tps = LLM_LATENCY_SPECS[model]
    except KeyError as e:
        raise KeyError(f"Unknown LLM model for latency: {model!r}") from e
    return ttft_s, tps


def llm_ttft_and_tps_jittered(model: str, rng: random.Random) -> tuple[float, float]:
    """
    Nominal TTFT and throughput multiplied by **independent**
    ``Uniform(1 - LATENCY_RANDOM_HALF_WIDTH, 1 + LATENCY_RANDOM_HALF_WIDTH)`` draws (two RNG calls).
    """
    ttft_b, tps_b = llm_nominal_ttft_and_tps(model)
    hw = LATENCY_RANDOM_HALF_WIDTH
    m_t = rng.uniform(1.0 - hw, 1.0 + hw)
    m_p = rng.uniform(1.0 - hw, 1.0 + hw)
    ttft = ttft_b * m_t
    tps = tps_b * m_p
    if tps <= 0:
        raise ValueError(f"Jitter yielded non-positive throughput for model {model!r}")
    return ttft, tps


def llm_decode_duration_sec(
    model: str,
    output_tokens: float,
    *,
    rng: random.Random | None = None,
) -> float:
    """
    End-to-end generation time for the decoding phase:
    TTFT + (output tokens) / (output tokens per second).

    ``rng``: if provided, nominal TTFT and throughput are jittered independently each call.
    ``rng=None`` uses table values exactly (backward compatible).

    Caption vs query differs only via ``output_tokens``.
    """
    if rng is None:
        ttft_s, tps = llm_nominal_ttft_and_tps(model)
    else:
        ttft_s, tps = llm_ttft_and_tps_jittered(model, rng)
    return ttft_s + float(output_tokens) / tps


def get_llm_ttft_and_throughput(
    model: str,
    *,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Nominal TTFT/tps from table, or one jittered draw pair if ``rng`` is provided."""
    if rng is None:
        return llm_nominal_ttft_and_tps(model)
    return llm_ttft_and_tps_jittered(model, rng)


# ---------------------------------------------------------------------------
# Segment / split: min–max envelope + uniform sampling
# ---------------------------------------------------------------------------
_MEASUREMENT_CSV = (
    Path(__file__).resolve().parent
    / "measurement_data"
    / "segment_split_execution_time.csv"
)
_LABEL_OCR_SPEECH_MEASUREMENT_CSV = (
    Path(__file__).resolve().parent
    / "measurement_data"
    / "label_ocr_speech_execution_time.csv"
)
_DATABASE_QUERY_LATENCY_CSV = (
    Path(__file__).resolve().parent
    / "measurement_data"
    / "alloydb_latency.csv"
)


@dataclass(frozen=True)
class ShotDetectionVideoSplitBounds:
    """Interpolated/extrapolated min–max bounds (seconds) at a clip duration."""

    duration_sec: float
    shot_detection_min_sec: float
    shot_detection_max_sec: float
    video_split_min_sec: float
    video_split_max_sec: float


@dataclass(frozen=True)
class LabelOcrSpeechBounds:
    duration_sec: float
    label_detection_min_sec: float
    label_detection_max_sec: float
    ocr_min_sec: float
    ocr_max_sec: float
    speech_transcription_min_sec: float
    speech_transcription_max_sec: float


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


# --- Execution envelope scale k ~ Uniform(1±LATENCY_RANDOM_HALF_WIDTH) -------------
NODE_EXECUTION_SCALE_HALF_WIDTH = LATENCY_RANDOM_HALF_WIDTH

_NODE_EXECUTION_SCALE_CACHE: dict[tuple[str, str, str, str], float] = {}

OperationShotDetectionVideoSplit = Literal["shot_detection", "video_split"]
OperationLabelOcrSpeech = Literal["label_detection", "ocr", "speech_transcription"]

LatencyOpKind = Literal[
    "shot_detection",
    "video_split",
    "label_detection",
    "ocr",
    "speech_transcription",
    "database_query",
]


def clear_node_execution_scale_cache() -> None:
    """Drop all cached execution-scale factors ``k`` (e.g. between batch experiments)."""

    _NODE_EXECUTION_SCALE_CACHE.clear()


def _node_execution_scale_key(
    *,
    execution_scale_scope: str,
    provider: str,
    region: str,
    latency_op: LatencyOpKind,
) -> tuple[str, str, str, str]:
    return (execution_scale_scope, provider, region, latency_op)


def _deterministic_rng_execution_scale(
    execution_scale_seed: int,
    execution_scale_scope: str,
    provider: str,
    region: str,
    latency_op: LatencyOpKind,
) -> random.Random:
    """
    Stable ``random.Random`` from integer seed + scope + endpoint + operation (cross-process).

    Mirrors the salted SHA256 → 64-bit pattern used elsewhere in this repo for ``det_rng``.
    """

    buf = hashlib.sha256(str(int(execution_scale_seed)).encode("utf-8"))
    for p in (
        "sim_env.node_execution_scale_k_v1",
        execution_scale_scope,
        provider,
        region,
        latency_op,
    ):
        buf.update(b"|")
        buf.update(str(p).encode("utf-8"))
        buf.update(b"\x1e")
    return random.Random(int.from_bytes(buf.digest()[:8], "big"))


def _get_fixed_node_execution_scale(
    *,
    execution_scale_scope: str,
    provider: str,
    region: str,
    latency_op: LatencyOpKind,
    rng: random.Random,
) -> float:
    """
    Cached ``k`` for this scope + endpoint + operation. On cache miss draws once via ``rng``.
    On hit, ``rng`` is not consumed.
    """

    key = _node_execution_scale_key(
        execution_scale_scope=execution_scale_scope,
        provider=provider,
        region=region,
        latency_op=latency_op,
    )
    k = _NODE_EXECUTION_SCALE_CACHE.get(key)
    if k is not None:
        return k
    k = sample_execution_scale(rng)
    _NODE_EXECUTION_SCALE_CACHE[key] = k
    return k


def sample_execution_scale(rng: random.Random) -> float:
    """
    One draw ``k ~ Uniform(1 - half_width, 1 + half_width)``.

    With ``execution_scale_seed`` set on ``sample_*_execute_sec``, ``k`` comes from
    ``node_execution_scale_k`` (deterministic). Otherwise ``_get_fixed_node_execution_scale``
    draws ``k`` once per cache key via ``rng``.
    """
    hw = NODE_EXECUTION_SCALE_HALF_WIDTH
    return rng.uniform(1.0 - hw, 1.0 + hw)


def node_execution_scale_k(
    execution_scale_seed: int,
    execution_scale_scope: str,
    provider: str,
    region: str,
    latency_op: LatencyOpKind,
) -> float:
    """
    Reproducible scale ``k ∈ [1-w, 1+w]`` for one physical endpoint and latency operation.

    Same arguments ⇒ same ``k`` (deterministic given ``NODE_EXECUTION_SCALE_HALF_WIDTH``).
    """

    r = _deterministic_rng_execution_scale(
        execution_scale_seed,
        execution_scale_scope,
        provider,
        region,
        latency_op,
    )
    return sample_execution_scale(r)


class ShotDetectionVideoSplitLatencyModel:
    """
    Per measured duration: min/max of shot_detection and of video_split execution times.
    Bounds vs. duration are piecewise-linear; draw each operation separately via
    ``sample_shot_detection_execute_sec`` / ``sample_video_split_execute_sec``.
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
            pairs = by_d[d]
            segs = [p[0] for p in pairs]
            spls = [p[1] for p in pairs]
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

    def node_field_bounds_at(
        self,
        video_duration_sec: float,
        operation: OperationShotDetectionVideoSplit,
        provider: str,
        region: str,
    ) -> tuple[float, float]:
        """Aggregate shot_detection/video_split (min, max) at ``video_duration_sec`` before ``k``."""

        _ = provider, region
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
        if operation == "shot_detection":
            lo = _piecewise_linear_eval(self._anchors, self._seg_min, d_req)
            hi = _piecewise_linear_eval(self._anchors, self._seg_max, d_req)
        else:
            lo = _piecewise_linear_eval(self._anchors, self._spl_min, d_req)
            hi = _piecewise_linear_eval(self._anchors, self._spl_max, d_req)
        return _enforce_min_le_max(lo, hi)

    @property
    def measured_durations_sec(self) -> list[float]:
        self._ensure_loaded()
        assert self._anchors is not None
        return list(self._anchors)

    def bounds_at(self, video_duration_sec: float) -> ShotDetectionVideoSplitBounds:
        """
        Min/max envelope for shot_detection and video_split at `video_duration_sec` (seconds),
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
        return ShotDetectionVideoSplitBounds(d_req, sm, sx, pm, px)


_DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL = ShotDetectionVideoSplitLatencyModel()


def shot_detection_video_split_bounds_at(
    video_duration_sec: float,
) -> ShotDetectionVideoSplitBounds:
    """Interpolated min/max bounds for shot_detection and video_split at the given clip length."""
    return _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.bounds_at(video_duration_sec)


def node_shot_detection_execute_bounds_at(
    video_duration_sec: float,
    provider: str,
    region: str,
) -> tuple[float, float]:
    """Aggregate shot_detection (min, max) in seconds at this clip length; scaled by ``k`` when sampling."""
    return _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.node_field_bounds_at(
        video_duration_sec, "shot_detection", provider, region
    )


def node_video_split_execute_bounds_at(
    video_duration_sec: float,
    provider: str,
    region: str,
) -> tuple[float, float]:
    """Aggregate video_split (min, max) in seconds at this clip length; scaled by ``k`` when sampling."""
    return _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.node_field_bounds_at(
        video_duration_sec, "video_split", provider, region
    )


def sample_shot_detection_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    r = rng or random.Random()
    d_req = float(video_duration_sec)
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    if node is None:
        lo, hi = _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.node_field_bounds_at(
            d_req, "shot_detection", "", ""
        )
        k = 1.0
    else:
        lo, hi = node_shot_detection_execute_bounds_at(d_req, node.provider, node.region)
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "shot_detection",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="shot_detection",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)


def sample_video_split_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    r = rng or random.Random()
    d_req = float(video_duration_sec)
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    if node is None:
        lo, hi = _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.node_field_bounds_at(
            d_req, "video_split", "", ""
        )
        k = 1.0
    else:
        lo, hi = node_video_split_execute_bounds_at(d_req, node.provider, node.region)
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "video_split",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="video_split",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)


def shot_detection_video_split_bucket_sizes() -> list[float]:
    """Clip durations (seconds) that appear in the measurement CSV."""
    return _DEFAULT_SHOT_DETECTION_VIDEO_SPLIT_MODEL.measured_durations_sec


# ---------------------------------------------------------------------------
# Label detection / OCR / speech transcription (VI prep + operation wait)
# ---------------------------------------------------------------------------


def _label_prep_plus_vi_sec(row: dict[str, str]) -> float:
    return float(row["node_label_prep_sec"]) + float(row["node_label_vi_operation_wait_sec"])


def _ocr_prep_plus_vi_sec(row: dict[str, str]) -> float:
    return float(row["node_ocr_prep_sec"]) + float(row["node_ocr_vi_operation_wait_sec"])


def _speech_prep_plus_vi_sec(row: dict[str, str]) -> float:
    return float(row["node_speech_prep_sec"]) + float(row["node_speech_vi_operation_wait_sec"])


class LabelOcrSpeechLatencyModel:
    """Min/max envelopes from one CSV (measurement convenience only).

    Label / OCR / speech are **independent**: use ``sample_label_detection_execute_sec``,
    ``sample_ocr_execute_sec``, ``sample_speech_transcription_execute_sec`` separately.
    ``bounds_at`` / ``label_ocr_speech_bounds_at`` expose all six edges for inspection only.
    """

    def __init__(self, csv_path: Path | None = None) -> None:
        self._csv_path = csv_path or _LABEL_OCR_SPEECH_MEASUREMENT_CSV
        self._anchors: list[float] | None = None
        self._lab_min: list[float] | None = None
        self._lab_max: list[float] | None = None
        self._ocr_min: list[float] | None = None
        self._ocr_max: list[float] | None = None
        self._sp_min: list[float] | None = None
        self._sp_max: list[float] | None = None

    def _ensure_loaded(self) -> None:
        if self._anchors is not None:
            return
        by_d: dict[float, list[tuple[float, float, float]]] = {}
        with self._csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("success") != "True":
                    continue
                d = float(row["duration_sec"])
                lab = _label_prep_plus_vi_sec(row)
                ocr = _ocr_prep_plus_vi_sec(row)
                sp = _speech_prep_plus_vi_sec(row)
                by_d.setdefault(d, []).append((lab, ocr, sp))
        if not by_d:
            raise RuntimeError(f"No successful rows in {self._csv_path}")
        durations = sorted(by_d.keys())
        lab_min, lab_max, ocr_min, ocr_max, sp_min, sp_max = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for d in durations:
            triples = by_d[d]
            labs = [t[0] for t in triples]
            ocrs = [t[1] for t in triples]
            sps = [t[2] for t in triples]
            lm, lx = _enforce_min_le_max(min(labs), max(labs))
            om, ox = _enforce_min_le_max(min(ocrs), max(ocrs))
            sm, sx = _enforce_min_le_max(min(sps), max(sps))
            lab_min.append(lm)
            lab_max.append(lx)
            ocr_min.append(om)
            ocr_max.append(ox)
            sp_min.append(sm)
            sp_max.append(sx)
        self._anchors = durations
        self._lab_min = lab_min
        self._lab_max = lab_max
        self._ocr_min = ocr_min
        self._ocr_max = ocr_max
        self._sp_min = sp_min
        self._sp_max = sp_max

    def node_field_bounds_at(
        self,
        video_duration_sec: float,
        operation: OperationLabelOcrSpeech,
        provider: str,
        region: str,
    ) -> tuple[float, float]:
        """Aggregate (min, max) at ``video_duration_sec`` for one modality; ``provider``/``region`` unused."""

        _ = provider, region
        self._ensure_loaded()
        assert (
            self._anchors is not None
            and self._lab_min is not None
            and self._lab_max is not None
            and self._ocr_min is not None
            and self._ocr_max is not None
            and self._sp_min is not None
            and self._sp_max is not None
        )
        d_req = float(video_duration_sec)
        if d_req <= 0:
            raise ValueError("video_duration_sec must be positive")
        if operation == "label_detection":
            lo = _piecewise_linear_eval(self._anchors, self._lab_min, d_req)
            hi = _piecewise_linear_eval(self._anchors, self._lab_max, d_req)
        elif operation == "ocr":
            lo = _piecewise_linear_eval(self._anchors, self._ocr_min, d_req)
            hi = _piecewise_linear_eval(self._anchors, self._ocr_max, d_req)
        else:
            lo = _piecewise_linear_eval(self._anchors, self._sp_min, d_req)
            hi = _piecewise_linear_eval(self._anchors, self._sp_max, d_req)
        return _enforce_min_le_max(lo, hi)

    @property
    def measured_durations_sec(self) -> list[float]:
        self._ensure_loaded()
        assert self._anchors is not None
        return list(self._anchors)

    def bounds_at(self, video_duration_sec: float) -> LabelOcrSpeechBounds:
        self._ensure_loaded()
        assert (
            self._anchors is not None
            and self._lab_min is not None
            and self._lab_max is not None
            and self._ocr_min is not None
            and self._ocr_max is not None
            and self._sp_min is not None
            and self._sp_max is not None
        )
        d_req = float(video_duration_sec)
        if d_req <= 0:
            raise ValueError("video_duration_sec must be positive")
        lm = _piecewise_linear_eval(self._anchors, self._lab_min, d_req)
        lx = _piecewise_linear_eval(self._anchors, self._lab_max, d_req)
        om = _piecewise_linear_eval(self._anchors, self._ocr_min, d_req)
        ox = _piecewise_linear_eval(self._anchors, self._ocr_max, d_req)
        sm = _piecewise_linear_eval(self._anchors, self._sp_min, d_req)
        sx = _piecewise_linear_eval(self._anchors, self._sp_max, d_req)
        lm, lx = _enforce_min_le_max(lm, lx)
        om, ox = _enforce_min_le_max(om, ox)
        sm, sx = _enforce_min_le_max(sm, sx)
        return LabelOcrSpeechBounds(d_req, lm, lx, om, ox, sm, sx)


_DEFAULT_LABEL_OCR_SPEECH_MODEL = LabelOcrSpeechLatencyModel()


def label_ocr_speech_bounds_at(video_duration_sec: float) -> LabelOcrSpeechBounds:
    return _DEFAULT_LABEL_OCR_SPEECH_MODEL.bounds_at(video_duration_sec)


def node_label_detection_execute_bounds_at(
    video_duration_sec: float,
    provider: str,
    region: str,
) -> tuple[float, float]:
    return _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
        video_duration_sec, "label_detection", provider, region
    )


def node_ocr_execute_bounds_at(
    video_duration_sec: float,
    provider: str,
    region: str,
) -> tuple[float, float]:
    return _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
        video_duration_sec, "ocr", provider, region
    )


def node_speech_transcription_execute_bounds_at(
    video_duration_sec: float,
    provider: str,
    region: str,
) -> tuple[float, float]:
    return _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
        video_duration_sec, "speech_transcription", provider, region
    )


def sample_label_detection_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    r = rng or random.Random()
    d_req = float(video_duration_sec)
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    if node is None:
        lo, hi = _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
            d_req, "label_detection", "", ""
        )
        k = 1.0
    else:
        lo, hi = node_label_detection_execute_bounds_at(d_req, node.provider, node.region)
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "label_detection",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="label_detection",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)


def sample_ocr_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    r = rng or random.Random()
    d_req = float(video_duration_sec)
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    if node is None:
        lo, hi = _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
            d_req, "ocr", "", ""
        )
        k = 1.0
    else:
        lo, hi = node_ocr_execute_bounds_at(d_req, node.provider, node.region)
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "ocr",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="ocr",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)


def sample_speech_transcription_execute_sec(
    video_duration_sec: float,
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    r = rng or random.Random()
    d_req = float(video_duration_sec)
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    if node is None:
        lo, hi = _DEFAULT_LABEL_OCR_SPEECH_MODEL.node_field_bounds_at(
            d_req, "speech_transcription", "", ""
        )
        k = 1.0
    else:
        lo, hi = node_speech_transcription_execute_bounds_at(
            d_req, node.provider, node.region
        )
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "speech_transcription",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="speech_transcription",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)


def label_ocr_speech_bucket_sizes() -> list[float]:
    """Clip durations (seconds) present in ``label_ocr_speech_execution_time.csv``."""
    return _DEFAULT_LABEL_OCR_SPEECH_MODEL.measured_durations_sec


# ---------------------------------------------------------------------------
# Database retrieval latency (measured AlloyDB avg latencies, ms → seconds)
# ---------------------------------------------------------------------------

_database_query_latency_bounds_sec: tuple[float, float] | None = None


def _ensure_database_query_bounds_loaded() -> tuple[float, float]:
    global _database_query_latency_bounds_sec
    if _database_query_latency_bounds_sec is not None:
        return _database_query_latency_bounds_sec
    secs: list[float] = []
    with _DATABASE_QUERY_LATENCY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = (row.get("Latency_Avg_ms") or "").strip()
            if not raw:
                continue
            secs.append(float(raw) / 1000.0)
    if not secs:
        raise RuntimeError(f"No Latency_Avg_ms rows in {_DATABASE_QUERY_LATENCY_CSV}")
    lo, hi = _enforce_min_le_max(min(secs), max(secs))
    _database_query_latency_bounds_sec = (lo, hi)
    return _database_query_latency_bounds_sec


def database_query_latency_bounds_sec() -> tuple[float, float]:
    """Min/max query latency (seconds) from ``alloydb_latency.csv`` (nominal, before node ``k``)."""
    return _ensure_database_query_bounds_loaded()


def node_database_query_execute_bounds_at(
    provider: str,
    region: str,
) -> tuple[float, float]:
    """Aggregate (min, max) seconds at the database node; ``provider``/``region`` do not shift the measurement envelope."""
    _ = provider, region
    return _ensure_database_query_bounds_loaded()


def sample_database_query_execute_sec(
    rng: random.Random | None = None,
    *,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    """
    One database retrieval latency draw: uniform between scaled min/max from measurement CSV.

    With a ``PhysicalNode``, latency is scaled by the same endpoint ``k`` as shot_detection/VI ops.
    """
    r = rng or random.Random()
    scope = execution_scale_scope if execution_scale_scope is not None else "_global"
    lo, hi = _ensure_database_query_bounds_loaded()
    if node is None:
        k = 1.0
    else:
        if execution_scale_seed is not None:
            k = node_execution_scale_k(
                execution_scale_seed,
                scope,
                node.provider,
                node.region,
                "database_query",
            )
        else:
            k = _get_fixed_node_execution_scale(
                execution_scale_scope=scope,
                provider=node.provider,
                region=node.region,
                latency_op="database_query",
                rng=r,
            )
    lo, hi = lo * k, hi * k
    lo, hi = _enforce_min_le_max(lo, hi)
    return r.uniform(lo, hi)
