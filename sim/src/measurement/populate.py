"""Apply measurement-backed latencies to synthetic endpoint CSVs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.measurement import MEASURED_OPS

def _default_measurement_seed() -> int:
    from src.config import load_default_config

    return int(load_default_config().get("random_seed", 42))


DEFAULT_SEED = _default_measurement_seed()
QUALITY_LATENCY_FACTOR = {"Q1": 1.2, "Q2": 1.0, "Q3": 0.8}


def node_number(seed: int, provider: str, region: str, logical_op: str) -> int:
    """Deterministic node index from seed + endpoint identity."""
    buf = hashlib.sha256(str(int(seed)).encode("utf-8"))
    for part in (provider, region, logical_op):
        buf.update(b"|")
        buf.update(part.encode("utf-8"))
    return int.from_bytes(buf.digest()[:8], "big")


def reference_duration_per_mb(queries_df: pd.DataFrame) -> float:
    ratios = queries_df["video_duration_sec"] / queries_df["video_size_mb"]
    return float(ratios.median())


def linear_to_endpoint(
    a_lower: float,
    b_lower: float,
    a_upper: float,
    b_upper: float,
    duration_per_mb: float,
) -> tuple[float, float]:
    mid_a = (a_lower + a_upper) / 2.0
    mid_b = (b_lower + b_upper) / 2.0
    return mid_b, mid_a * duration_per_mb


def bounds_to_endpoint(lower: float, upper: float) -> tuple[float, float]:
    return (lower + upper) / 2.0, 0.0


def latency_from_measurement(
    logical_op: str,
    provider: str,
    region: str,
    quality_level: str,
    *,
    seed: int,
    duration_per_mb: float,
) -> tuple[float, float]:
    build_params = MEASURED_OPS[logical_op]
    node = node_number(seed, provider, region, logical_op)
    params = build_params(node)
    if len(params) == 4:
        base, per_mb = linear_to_endpoint(*params, duration_per_mb)
    else:
        base, per_mb = bounds_to_endpoint(*params)
    factor = QUALITY_LATENCY_FACTOR[quality_level]
    return base * factor, per_mb * factor


def apply_measurement_latencies(
    endpoints_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    duration_per_mb = reference_duration_per_mb(queries_df)
    updated = endpoints_df.copy()
    for idx, row in updated.iterrows():
        if row.get("is_virtual"):
            continue
        op = row["logical_operation"]
        if op not in MEASURED_OPS:
            continue
        base, per_mb = latency_from_measurement(
            op,
            row["provider"],
            row["region"],
            row["quality_level"],
            seed=seed,
            duration_per_mb=duration_per_mb,
        )
        updated.at[idx, "base_latency_sec"] = base
        updated.at[idx, "latency_per_mb"] = per_mb
    return updated


def populate(
    output_dir: Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    regenerate_all: bool = False,
) -> pd.DataFrame:
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    if regenerate_all:
        from src.data_generator import generate_all

        generate_all(out, seed=seed)

    endpoints_path = out / "endpoints.csv"
    queries_path = out / "queries.csv"
    if not endpoints_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"Missing endpoints.csv or queries.csv under {out}. "
            "Run with regenerate_all=True or generate synthetic data first."
        )

    endpoints_df = pd.read_csv(endpoints_path)
    queries_df = pd.read_csv(queries_path)
    endpoints_df = apply_measurement_latencies(endpoints_df, queries_df, seed=seed)
    endpoints_df.to_csv(endpoints_path, index=False)
    return endpoints_df
