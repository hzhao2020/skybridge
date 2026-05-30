"""Build label_detection simulation bounds from label_vision measurement."""

from __future__ import annotations

from src.measurement._common import LABEL_CSV, execution_time_bounds, scale_bounds


def build_params(node_number: int) -> tuple[float, float]:
    lower, upper = execution_time_bounds(LABEL_CSV)
    return scale_bounds(lower, upper, node_number)
