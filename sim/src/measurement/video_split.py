"""Build video_split simulation params from segment_split measurement."""

from __future__ import annotations

from src.measurement._common import (
    SEGMENT_SPLIT_CSV,
    fit_linear_uniform_params,
    scale_linear_intercepts,
)


def build_params(node_number: int) -> tuple[float, float, float, float]:
    a_lower, b_lower, a_upper, b_upper = fit_linear_uniform_params(
        SEGMENT_SPLIT_CSV,
        duration_col="duration_sec",
        value_col="node_split_execute_http_observed_sec",
    )
    return scale_linear_intercepts(a_lower, b_lower, a_upper, b_upper, node_number)
