"""
llm.py
Helpers for mapping video workloads to LLM token counts used in billing / simulation.

Caption (vision) input:
  InputTokens = duration_sec × sample_fps × tokens_per_frame
  (default 1 FPS, 256 tokens/frame — ViT-L/14 style.)

Caption (text) output length is modeled as **log-t**: let Y ~ StudentT(ν, loc=μ, scale=σ)
on the **log scale** (ln tokens), then output tokens = exp(Y) (floored at 1). With clip
length T (seconds)::

  μ(T) = a·ln(T) + b
  σ(T) = max(ε, c·ln(T) + d)
  ν(T) = max(2.1, ν_base − k·ln(T))

Query output uses the same log-t shape with **fixed** μ=5, σ=0.6, ν=10.

Requires **numpy** for Student-t draws (`numpy.random.Generator.standard_t`).
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

# ---------------------------------------------------------------------------
# Vision input (caption)
# ---------------------------------------------------------------------------
DEFAULT_CAPTION_SAMPLE_FPS = 1.0
DEFAULT_TOKENS_PER_FRAME = 256

# ---------------------------------------------------------------------------
# Caption output: log-t parameters vs. video length T (seconds)
# Tune a,b,c,d,ν_base,k to your data; defaults are placeholders.
# ---------------------------------------------------------------------------
CAPTION_LOGT_MU_A = 0.35
CAPTION_LOGT_MU_B = 2.8
CAPTION_LOGT_SIGMA_C = 0.08
CAPTION_LOGT_SIGMA_D = 0.45
CAPTION_LOGT_NU_BASE = 12.0
CAPTION_LOGT_NU_K = 0.5

MIN_LN_DURATION_SEC = 1.0
MIN_SIGMA_SCALE = 1e-6
MIN_NU = 2.1

# ---------------------------------------------------------------------------
# Query output: fixed log-t (μ, σ, ν on ln-token scale)
# ---------------------------------------------------------------------------
QUERY_OUTPUT_LOG_MU = 5.0
QUERY_OUTPUT_LOG_SIGMA = 0.6
QUERY_OUTPUT_LOG_NU = 10.0

Rng: TypeAlias = int | np.random.Generator | None


def _as_numpy_rng(rng: Rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def caption_visual_input_tokens(
    duration_sec: float,
    *,
    sample_fps: float = DEFAULT_CAPTION_SAMPLE_FPS,
    tokens_per_frame: int = DEFAULT_TOKENS_PER_FRAME,
) -> float:
    """
    Estimated **vision encoder input tokens** for a caption call from clip length.

    Formula: ``duration_sec × sample_fps × tokens_per_frame``.
    """
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if tokens_per_frame < 0:
        raise ValueError("tokens_per_frame must be non-negative")
    return float(duration_sec) * float(sample_fps) * float(tokens_per_frame)


def caption_output_log_t_params(duration_sec: float) -> tuple[float, float, float]:
    """
    Return (μ, σ, ν) for caption output on **ln(token)** scale, with T = max(duration_sec, 1).
    """
    t = max(float(duration_sec), MIN_LN_DURATION_SEC)
    ln_t = math.log(t)
    mu = CAPTION_LOGT_MU_A * ln_t + CAPTION_LOGT_MU_B
    sigma = CAPTION_LOGT_SIGMA_C * ln_t + CAPTION_LOGT_SIGMA_D
    sigma = max(MIN_SIGMA_SCALE, sigma)
    nu = CAPTION_LOGT_NU_BASE - CAPTION_LOGT_NU_K * ln_t
    nu = max(MIN_NU, nu)
    return mu, sigma, nu


def sample_caption_output_tokens(
    duration_sec: float,
    *,
    rng: Rng = None,
) -> float:
    """
    One random **caption** output token count ~ log-t: exp(Y), Y ~ StudentT(ν, μ, σ),
    with μ, σ, ν depending on clip length as in ``caption_output_log_t_params``.
    """
    mu, sigma, nu = caption_output_log_t_params(duration_sec)
    g = _as_numpy_rng(rng)
    z = float(g.standard_t(df=nu))
    y = mu + sigma * z
    return max(1.0, float(math.exp(y)))


def sample_query_output_tokens(*, rng: Rng = None) -> float:
    """
    One random **query** output token count ~ log-t with fixed
    μ=QUERY_OUTPUT_LOG_MU, σ=QUERY_OUTPUT_LOG_SIGMA, ν=QUERY_OUTPUT_LOG_NU.
    """
    g = _as_numpy_rng(rng)
    z = float(g.standard_t(df=QUERY_OUTPUT_LOG_NU))
    y = QUERY_OUTPUT_LOG_MU + QUERY_OUTPUT_LOG_SIGMA * z
    return max(1.0, float(math.exp(y)))


def sample_output_tokens_as_int(token_float: float) -> int:
    """Optional discretization for token counters (round, at least 1)."""
    return max(1, int(round(token_float)))


if __name__ == "__main__":
    # 10-minute clip; reproducible draws with seed 42.
    duration_sec = 600.0
    rng = np.random.default_rng(42)

    cap_in = caption_visual_input_tokens(duration_sec)
    mu, sigma, nu = caption_output_log_t_params(duration_sec)
    cap_out = sample_caption_output_tokens(duration_sec, rng=rng)
    q_out = sample_query_output_tokens(rng=rng)

    print("10 min video, caption vision input tokens:", cap_in)
    print("caption log-t (mu, sigma, nu) at this T:", (mu, sigma, nu))
    print("one caption output token sample:", cap_out)
    print("one query output token sample:", q_out)

    try:
        from sim_env.cost import llm_token_cost_usd

        usd_cap = llm_token_cost_usd(
            "GCP",
            "us-west1",
            "Gemini 2.5 Pro",
            cap_in,
            cap_out,
        )
        usd_q = llm_token_cost_usd(
            "GCP",
            "us-west1",
            "Gemini 2.5 Flash",
            200.0,
            q_out,
        )
        print("example llm cost caption (Gemini 2.5 Pro) USD:", round(usd_cap, 6))
        print("example llm cost query (200 in + sample out, Flash) USD:", round(usd_q, 6))
    except ImportError:
        pass
