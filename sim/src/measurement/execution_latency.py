"""Endpoint-specific execution latency sampling from measurement bounds."""

from __future__ import annotations

import random
from functools import lru_cache

from src.config import load_default_config
from src.measurement import MEASURED_OPS
from src.measurement.populate import QUALITY_LATENCY_FACTOR, node_number
from src.schemas import Endpoint, Query, Scenario


@lru_cache(maxsize=1)
def _default_seed() -> int:
    return int(load_default_config().get("random_seed", 42))


@lru_cache(maxsize=None)
def _measurement_params(
    seed: int,
    provider: str,
    region: str,
    op: str,
) -> tuple[float, ...]:
    node = node_number(seed, provider, region, op)
    return tuple(float(x) for x in MEASURED_OPS[op](node))


@lru_cache(maxsize=None)
def _sampled_execution_latency_cached(
    seed: int,
    query_id: str,
    scenario_id: str,
    endpoint_id: str,
    video_duration_sec: float,
    provider: str,
    region: str,
    op: str,
    quality_level: str,
) -> float:
    params = _measurement_params(seed, provider, region, op)
    if len(params) == 4:
        a_lower, b_lower, a_upper, b_upper = params
        lower = a_lower * video_duration_sec + b_lower
        upper = a_upper * video_duration_sec + b_upper
    else:
        lower, upper = params

    lo, hi = sorted((max(0.0, float(lower)), max(0.0, float(upper))))
    factor = QUALITY_LATENCY_FACTOR[quality_level]
    rng = random.Random(f"{seed}|{query_id}|{scenario_id}|{endpoint_id}")
    return rng.uniform(lo, hi) * factor


def sampled_execution_latency(
    endpoint: Endpoint,
    query: Query,
    scenario: Scenario,
) -> float | None:
    """Sample T_k^exe from measurement-derived ranges for non-LLM endpoints.

    The sample is deterministic for a fixed random seed, query, scenario, and
    endpoint so every solver/evaluator sees the same stochastic realization.
    """
    op = endpoint.logical_operation
    if op not in MEASURED_OPS:
        return None

    seed = _default_seed()
    return _sampled_execution_latency_cached(
        seed,
        query.query_id,
        scenario.scenario_id,
        endpoint.endpoint_id,
        query.video_duration_sec,
        endpoint.provider,
        endpoint.region,
        op,
        endpoint.quality_level,
    )
