"""
network.py
RTT and bandwidth sampling for workflow simulation.

Measurement traces live under measurement_data/rtt_bandwidth/ in three folders:
  - inter_region: same cloud provider, different regions (any of the 4 regions per provider)
  - cross_provider_same_region: only these colocation pairs:
      - Oregon: GCP us-west1 <-> AWS us-west-2
      - Singapore: AWS ap-southeast-1 <-> Aliyun ap-southeast-1
      - Shanghai: Local (cn-shanghai) <-> Aliyun cn-shanghai
  - cross_provider_cross_region: all other cross-provider pairs

Same provider + same region: no network transfer (RTT 0, unlimited bandwidth).

Local is modeled as provider "Local" with region "cn-shanghai" (Shanghai).

Each directed link (src -> dst) keeps its own cursor into the appropriate trace;
samples are taken sequentially and independently per link.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator

# Logical endpoint for on-prem / edge; aligns with Aliyun Shanghai placement.
LOCAL_PROVIDER = "Local"
LOCAL_REGION = "cn-shanghai"

_MEASUREMENT_ROOT = Path(__file__).resolve().parent / "measurement_data" / "rtt_bandwidth"

# Unordered endpoint pairs treated as cross-provider "same region" (colocation).
_CROSS_PROVIDER_SAME_REGION_GROUPS: frozenset[frozenset[tuple[str, str]]] = frozenset(
    {
        frozenset({("GCP", "us-west1"), ("AWS", "us-west-2")}),
        frozenset({("AWS", "ap-southeast-1"), ("Aliyun", "ap-southeast-1")}),
        frozenset({(LOCAL_PROVIDER, LOCAL_REGION), ("Aliyun", "cn-shanghai")}),
    }
)


class LinkCategory(str, Enum):
    NONE = "none"  # same provider + same region — no transfer
    INTER_REGION = "inter_region"
    CROSS_PROVIDER_SAME_GEO = "cross_provider_same_region"
    CROSS_PROVIDER_CROSS_GEO = "cross_provider_cross_region"


@dataclass(frozen=True)
class NetworkSample:
    """One draw of RTT (ms) and iperf bandwidth (Mbits/s) for a directed link."""

    rtt_ms: float
    bandwidth_out_mbits_per_sec: float
    bandwidth_in_mbits_per_sec: float
    category: LinkCategory

    @property
    def bandwidth_effective_mbits_per_sec(self) -> float:
        """Conservative single number for capacity planning (bottleneck of the two directions)."""
        return (self.bandwidth_out_mbits_per_sec + self.bandwidth_in_mbits_per_sec) / 2.0


def canonical_provider(name: str) -> str:
    n = name.strip()
    low = n.lower()
    if low == "gcp":
        return "GCP"
    if low == "aws":
        return "AWS"
    if low in ("aliyun", "alibaba", "alibabacloud"):
        return "Aliyun"
    if low == "local":
        return LOCAL_PROVIDER
    return n


def _normalize_endpoint(provider: str, region: str) -> tuple[str, str]:
    p = canonical_provider(provider)
    if p == LOCAL_PROVIDER and region != LOCAL_REGION:
        raise ValueError(
            f"Local node must use region {LOCAL_REGION!r}, got {region!r}"
        )
    return p, region


def _is_cross_provider_same_region_pair(
    a: tuple[str, str], b: tuple[str, str]
) -> bool:
    if a == b:
        return False
    return frozenset({a, b}) in _CROSS_PROVIDER_SAME_REGION_GROUPS


def classify_link(
    src: tuple[str, str],
    dst: tuple[str, str],
) -> LinkCategory:
    """
    Map an ordered pair of (provider, region) endpoints to a measurement bucket.

    - Same provider and same region -> NONE (no transfer).
    - Same provider, different region -> INTER_REGION (any inter-region within GCP/AWS/Aliyun).
    - Different providers, and the unordered pair is Oregon / Singapore / Shanghai
      colocation (see module doc) -> CROSS_PROVIDER_SAME_GEO.
    - Any other different-provider pair -> CROSS_PROVIDER_CROSS_GEO.
    """
    a = _normalize_endpoint(src[0], src[1])
    b = _normalize_endpoint(dst[0], dst[1])
    if a == b:
        return LinkCategory.NONE
    if a[0] == b[0]:
        return LinkCategory.INTER_REGION
    if _is_cross_provider_same_region_pair(a, b):
        return LinkCategory.CROSS_PROVIDER_SAME_GEO
    return LinkCategory.CROSS_PROVIDER_CROSS_GEO


def _iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _pick_latest_pair(dirpath: Path) -> tuple[Path, Path]:
    rtt_files = sorted(dirpath.glob("rtt_*.csv"))
    bw_files = sorted(dirpath.glob("bandwidth_*.csv"))
    if not rtt_files or not bw_files:
        raise FileNotFoundError(
            f"Expected rtt_*.csv and bandwidth_*.csv under {dirpath}"
        )
    return rtt_files[-1], bw_files[-1]


def _merge_inter_region_series(dirpath: Path) -> list[tuple[float, float, float]]:
    rtt_path, bw_path = _pick_latest_pair(dirpath)
    rtt_by_ts: dict[str, dict[str, str]] = {}
    for row in _iter_csv_rows(rtt_path):
        ts = row.get("timestamp_utc", "")
        if row.get("ping_ok", "1") != "1":
            continue
        rtt_by_ts[ts] = row
    out: list[tuple[float, float, float]] = []
    for row in _iter_csv_rows(bw_path):
        if row.get("iperf_ok", "1") != "1":
            continue
        ts = row.get("timestamp_utc", "")
        pr = rtt_by_ts.get(ts)
        if not pr:
            continue
        out.append(
            (
                float(pr["rtt_avg_ms"]),
                float(row["bw_out_mbits_per_sec"]),
                float(row["bw_in_mbits_per_sec"]),
            )
        )
    return out


def _merge_cross_provider_series(dirpath: Path) -> list[tuple[float, float, float]]:
    rtt_path, bw_path = _pick_latest_pair(dirpath)
    rtt_by_ts: dict[str, dict[str, str]] = {}
    for row in _iter_csv_rows(rtt_path):
        if row.get("ping_ok", "1") != "1":
            continue
        rtt_by_ts[row["timestamp_utc"]] = row
    out: list[tuple[float, float, float]] = []
    for row in _iter_csv_rows(bw_path):
        if row.get("iperf_ok", "1") != "1":
            continue
        ts = row.get("timestamp_utc", "")
        pr = rtt_by_ts.get(ts)
        if not pr:
            continue
        out.append(
            (
                float(pr["rtt_avg_ms"]),
                float(row["bw_out_mbits_per_sec"]),
                float(row["bw_in_mbits_per_sec"]),
            )
        )
    return out


def _load_category_series() -> dict[LinkCategory, list[tuple[float, float, float]]]:
    inter = _merge_inter_region_series(_MEASUREMENT_ROOT / "inter_region")
    same = _merge_cross_provider_series(
        _MEASUREMENT_ROOT / "cross_provider_same_region"
    )
    xgeo = _merge_cross_provider_series(
        _MEASUREMENT_ROOT / "cross_provider_cross_region"
    )
    if not inter or not same or not xgeo:
        raise RuntimeError(
            "Failed to build non-empty measurement series for all link categories; "
            f"inter={len(inter)} same={len(same)} cross={len(xgeo)}"
        )
    return {
        LinkCategory.INTER_REGION: inter,
        LinkCategory.CROSS_PROVIDER_SAME_GEO: same,
        LinkCategory.CROSS_PROVIDER_CROSS_GEO: xgeo,
    }


_SERIES: dict[LinkCategory, list[tuple[float, float, float]]] | None = None
_LINK_INDICES: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}


def _get_series() -> dict[LinkCategory, list[tuple[float, float, float]]]:
    global _SERIES
    if _SERIES is None:
        _SERIES = _load_category_series()
    return _SERIES


def reset_link_counters(links: Iterable[tuple[tuple[str, str], tuple[str, str]]] | None = None) -> None:
    """Clear per-link sampling cursors (all links if `links` is None)."""
    global _LINK_INDICES
    if links is None:
        _LINK_INDICES = {}
        return
    for key in links:
        _LINK_INDICES.pop(key, None)


def reset_measurement_cache() -> None:
    """Reload CSVs from disk on next sample (for tests / notebook reload)."""
    global _SERIES
    _SERIES = None


def sample_link(
    src: tuple[str, str],
    dst: tuple[str, str],
) -> NetworkSample:
    """
    Return the next RTT / bandwidth sample for directed link src -> dst.

    Cursors are independent per (src, dst) pair and advance by one on each call.
    """
    sp, sr = canonical_provider(src[0]), src[1]
    dp, dr = canonical_provider(dst[0]), dst[1]
    canon_src = (sp, sr)
    canon_dst = (dp, dr)
    cat = classify_link(canon_src, canon_dst)
    if cat is LinkCategory.NONE:
        return NetworkSample(
            rtt_ms=0.0,
            bandwidth_out_mbits_per_sec=math.inf,
            bandwidth_in_mbits_per_sec=math.inf,
            category=cat,
        )
    series_map = _get_series()
    series = series_map[cat]
    key = (canon_src, canon_dst)
    idx = _LINK_INDICES.get(key, 0)
    _LINK_INDICES[key] = idx + 1
    rtt, bw_out, bw_in = series[idx % len(series)]
    return NetworkSample(
        rtt_ms=rtt,
        bandwidth_out_mbits_per_sec=bw_out,
        bandwidth_in_mbits_per_sec=bw_in,
        category=cat,
    )


def link_cursor_position(src: tuple[str, str], dst: tuple[str, str]) -> int:
    """Number of samples already drawn for this directed link (for debugging)."""
    sp, sr = canonical_provider(src[0]), src[1]
    dp, dr = canonical_provider(dst[0]), dst[1]
    return _LINK_INDICES.get(((sp, sr), (dp, dr)), 0)


if __name__ == "__main__":
    src = ("GCP", "us-east1")
    src2 = ("AWS", "us-west-2")
    dst1 = ("Aliyun", "cn-beijing")
    dst2 = ("AWS", "us-west-2")
    dst3 = ("GCP", "us-west1")


    s1 = sample_link(src, dst1)
    s2 = sample_link(src, dst2)
    s3 = sample_link(src2, dst3)

    print(s1)
    print(s2)
    print(s3)