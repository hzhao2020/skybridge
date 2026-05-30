"""Build OCR simulation bounds from ocr_vision measurement."""

from __future__ import annotations

from src.measurement._common import OCR_CSV, execution_time_bounds, scale_bounds


def build_params(node_number: int) -> tuple[float, float]:
    lower, upper = execution_time_bounds(OCR_CSV)
    return scale_bounds(lower, upper, node_number)
