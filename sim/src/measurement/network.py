"""Measurement-backed RTT and bandwidth sampling for synthetic network links."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.measurement._common import MEASUREMENT_DATA_DIR

LOCAL_PROVIDER = "Local"
LOCAL_REGION = "cn-shanghai"
MEASUREMENT_NETWORK_DIR = MEASUREMENT_DATA_DIR / "network"


class LinkCategory(str, Enum):
    NONE = "none"
    INTER_REGION = "inter_region"
    CROSS_PROVIDER_SAME_REGION = "cross_provider_same_region"
    CROSS_PROVIDER_CROSS_REGION = "cross_provider_cross_region"


@dataclass(frozen=True)
class NetworkTraceSample:
    rtt_sec: float
    bandwidth_mb_per_sec: float
    category: LinkCategory


_CROSS_PROVIDER_SAME_REGION_GROUPS: frozenset[frozenset[tuple[str, str]]] = frozenset(
    {
        frozenset({("GCP", "us-west1"), ("AWS", "us-west-2")}),
        frozenset({("AWS", "ap-southeast-1"), ("Aliyun", "ap-southeast-1")}),
        frozenset({(LOCAL_PROVIDER, LOCAL_REGION), ("Aliyun", "cn-shanghai")}),
    }
)


def canonical_provider(provider: str) -> str:
    low = provider.strip().lower()
    if low == "gcp":
        return "GCP"
    if low == "aws":
        return "AWS"
    if low in {"aliyun", "alibaba", "alibabacloud"}:
        return "Aliyun"
    if low in {"azure", "microsoft"}:
        return "Azure"
    if low in {"client", "local"}:
        return LOCAL_PROVIDER
    return provider.strip()


def canonical_region(provider: str, region: str) -> str:
    if canonical_provider(provider) == LOCAL_PROVIDER:
        return LOCAL_REGION
    return region.strip()


def classify_link(src: tuple[str, str], dst: tuple[str, str]) -> LinkCategory:
    src_norm = (canonical_provider(src[0]), canonical_region(src[0], src[1]))
    dst_norm = (canonical_provider(dst[0]), canonical_region(dst[0], dst[1]))
    if src_norm == dst_norm:
        return LinkCategory.NONE
    if src_norm[0] == dst_norm[0]:
        return LinkCategory.INTER_REGION
    if frozenset({src_norm, dst_norm}) in _CROSS_PROVIDER_SAME_REGION_GROUPS:
        return LinkCategory.CROSS_PROVIDER_SAME_REGION
    return LinkCategory.CROSS_PROVIDER_CROSS_REGION


def sample_link_by_index(
    src: tuple[str, str],
    dst: tuple[str, str],
    sample_index: int,
) -> NetworkTraceSample:
    """Return a deterministic measured sample for src -> dst at a shared category index."""
    category = classify_link(src, dst)
    if category is LinkCategory.NONE:
        return NetworkTraceSample(
            rtt_sec=0.0,
            bandwidth_mb_per_sec=1_000_000_000.0,
            category=category,
        )
    series = _load_category_series()[category]
    rtt_ms, bw_out_mbits, bw_in_mbits = series[int(sample_index) % len(series)]
    bw_mbits = min(bw_out_mbits, bw_in_mbits)
    return NetworkTraceSample(
        rtt_sec=rtt_ms / 1000.0,
        bandwidth_mb_per_sec=bw_mbits / 8.0,
        category=category,
    )


def category_sample_counts() -> dict[LinkCategory, int]:
    return {cat: len(series) for cat, series in _load_category_series().items()}


_CATEGORY_SERIES: dict[LinkCategory, list[tuple[float, float, float]]] | None = None


def _load_category_series() -> dict[LinkCategory, list[tuple[float, float, float]]]:
    global _CATEGORY_SERIES
    if _CATEGORY_SERIES is None:
        _CATEGORY_SERIES = {
            LinkCategory.INTER_REGION: _load_trace_dir(
                MEASUREMENT_NETWORK_DIR / LinkCategory.INTER_REGION.value
            ),
            LinkCategory.CROSS_PROVIDER_SAME_REGION: _load_trace_dir(
                MEASUREMENT_NETWORK_DIR / LinkCategory.CROSS_PROVIDER_SAME_REGION.value
            ),
            LinkCategory.CROSS_PROVIDER_CROSS_REGION: _load_trace_dir(
                MEASUREMENT_NETWORK_DIR / LinkCategory.CROSS_PROVIDER_CROSS_REGION.value
            ),
        }
    return _CATEGORY_SERIES


def _load_trace_dir(dirpath: Path) -> list[tuple[float, float, float]]:
    rtt_path, bw_path = _pick_latest_pair(dirpath)
    rtt_by_ts: dict[str, float] = {}
    with rtt_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _ok(row.get("ping_ok")):
                continue
            rtt_by_ts[row["timestamp_utc"]] = float(row["rtt_avg_ms"])

    samples: list[tuple[float, float, float]] = []
    with bw_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _ok(row.get("iperf_ok")):
                continue
            rtt = rtt_by_ts.get(row["timestamp_utc"])
            if rtt is None:
                continue
            out_mbits = float(row["bw_out_mbits_per_sec"])
            in_mbits = float(row["bw_in_mbits_per_sec"])
            if math.isfinite(out_mbits) and math.isfinite(in_mbits):
                samples.append((rtt, out_mbits, in_mbits))

    if not samples:
        raise RuntimeError(f"No aligned RTT/bandwidth samples under {dirpath}")
    return samples


def _pick_latest_pair(dirpath: Path) -> tuple[Path, Path]:
    rtt_files = sorted(dirpath.glob("rtt_*.csv"))
    bw_files = sorted(dirpath.glob("bandwidth_*.csv"))
    if not rtt_files or not bw_files:
        raise FileNotFoundError(f"Expected rtt_*.csv and bandwidth_*.csv under {dirpath}")
    return rtt_files[-1], bw_files[-1]


def _ok(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}
