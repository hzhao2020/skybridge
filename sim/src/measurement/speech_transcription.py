"""Build speech_transcription simulation params from speech_audio measurement."""

from __future__ import annotations

from src.measurement._common import (
    SPEECH_CSV,
    fit_linear_uniform_params,
    scale_linear_intercepts,
)


def build_params(node_number: int) -> tuple[float, float, float, float]:
    a_lower, b_lower, a_upper, b_upper = fit_linear_uniform_params(
        SPEECH_CSV,
        duration_col="audio_duration_sec",
        value_col="execution_time_sec",
    )
    return scale_linear_intercepts(a_lower, b_lower, a_upper, b_upper, node_number)
