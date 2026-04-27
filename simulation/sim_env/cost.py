"""
cost.py
Public cloud cost tables and helpers for the video workflow simulation.
Prices follow SkyXXX Simulation.md; region keys align with config.REGIONS.
All monetary amounts are USD unless noted.
"""

from __future__ import annotations

import math
from typing import Mapping, Tuple

# (provider, region) keys match config.py string literals.
ProviderRegion = Tuple[str, str]

# ---------------------------------------------------------------------------
# LLM: $ per 1M input / output tokens (Vertex / Bedrock / Bailian)
# ---------------------------------------------------------------------------
LLM_TOKEN_PRICE_PER_MILLION: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "GCP": {
        "us-east1": {
            "Gemini 2.5 Pro": (1.25, 10.00),
            "Gemini 2.5 Flash": (0.30, 2.50),
        },
        "us-west1": {
            "Gemini 2.5 Pro": (1.25, 10.00),
            "Gemini 2.5 Flash": (0.30, 2.50),
        },
        "europe-west1": {
            "Gemini 2.5 Pro": (1.25, 10.00),
            "Gemini 2.5 Flash": (0.30, 2.50),
        },
        "asia-east1": {
            "Gemini 2.5 Pro": (1.25, 10.00),
            "Gemini 2.5 Flash": (0.30, 2.50),
        },
    },
    "AWS": {
        "us-west-2": {
            "Claude 3.5 Sonnet": (3.00, 15.00),
            "Claude 3.5 Haiku": (0.80, 4.00),
        },
        "us-east-2": {
            "Claude 3.5 Sonnet": (3.00, 15.00),
            "Claude 3.5 Haiku": (0.80, 4.00),
        },
        "ap-southeast-1": {
            "Claude 3.5 Sonnet": (3.00, 15.00),
            "Claude 3.5 Haiku": (1.00, 5.00),
        },
        "eu-central-1": {
            "Claude 3.5 Sonnet": (3.00, 15.00),
            "Claude 3.5 Haiku": (1.00, 5.00),
        },
    },
    "Aliyun": {
        "cn-beijing": {
            "Qwen3-VL-Plus": (0.144, 1.434),
            "Qwen3-VL-Flash": (0.022, 0.215),
            "Qwen3.5-Plus": (0.115, 0.688),
            "Qwen3.5-Flash": (0.029, 0.287),
        },
        "us-west-1": {
            "Qwen3-VL-Plus": (0.144, 1.434),
            "Qwen3-VL-Flash": (0.022, 0.215),
            "Qwen3.5-Plus": (0.115, 0.688),
            "Qwen3.5-Flash": (0.029, 0.287),
        },
        "ap-southeast-1": {
            "Qwen3-VL-Plus": (0.2, 1.6),
            "Qwen3-VL-Flash": (0.05, 0.4),
            "Qwen3.5-Plus": (0.4, 2.4),
            "Qwen3.5-Flash": (0.1, 0.4),
        },
    },
}

# ---------------------------------------------------------------------------
# Video intelligence: $ per minute of video (where published in the doc)
# ---------------------------------------------------------------------------
VIDEO_SERVICE_USD_PER_MINUTE: dict[str, dict[str, dict[str, float]]] = {
    "GCP": {
        "us-east1": {
            "segment": 0.05,
            "speech_transcription": 0.048,
            "ocr": 0.15,
            "label_detection": 0.10,
        },
        "us-west1": {
            "segment": 0.05,
            "speech_transcription": 0.048,
            "ocr": 0.15,
            "label_detection": 0.10,
        },
        "europe-west1": {
            "segment": 0.05,
            "speech_transcription": 0.048,
            "ocr": 0.15,
            "label_detection": 0.10,
        },
        "asia-east1": {
            "segment": 0.05,
            "speech_transcription": 0.048,
            "ocr": 0.15,
            "label_detection": 0.10,
        },
    },
    "AWS": {
        "us-west-2": {
            "segment": 0.05,
            "ocr": 0.10,
            "label_detection": 0.10,
            "speech_transcription": 0.024,
        },
        "us-east-2": {
            "segment": 0.05,
            "ocr": 0.10,
            "label_detection": 0.10,
            "speech_transcription": 0.024,
        },
        "ap-southeast-1": {
            "segment": 0.0675,
            "ocr": 0.135,
            "label_detection": 0.135,
            "speech_transcription": 0.024,
        },
        "eu-central-1": {
            "segment": 0.06,
            "ocr": 0.12,
            "label_detection": 0.12,
            "speech_transcription": 0.024,
        },
    },
    "Aliyun": {
        # Doc only lists segment for cn-shanghai; other services left undefined.
        "cn-shanghai": {
            "segment": 0.029,
        },
    },
}

# ---------------------------------------------------------------------------
# Split (serverless, 1 vCPU + 1 GiB assumed): USD per minute
# ---------------------------------------------------------------------------
SPLIT_USD_PER_MINUTE: dict[str, dict[str, float]] = {
    "GCP": {
        "us-east1": 0.0012,
        "us-west1": 0.0012,
        "europe-west1": 0.0012,
        "asia-east1": 0.0012,
    },
    "AWS": {
        "us-west-2": 0.002,
        "us-east-2": 0.002,
        "ap-southeast-1": 0.002,
        "eu-central-1": 0.002,
    },
    "Aliyun": {
        "cn-shanghai": 0.0011,
        "cn-beijing": 0.0011,
        "us-west-1": 0.0011,
        "ap-southeast-1": 0.0011,
    },
}

# ---------------------------------------------------------------------------
# Object storage: USD per GB-month (doc); simulator bills in whole days via
# STORAGE_DAYS_PER_MONTH proration below.
# ---------------------------------------------------------------------------
STORAGE_DAYS_PER_MONTH = 30.0

STORAGE_USD_PER_GB_MONTH: dict[str, dict[str, float]] = {
    "GCP": {
        "us-east1": 0.020,
        "us-west1": 0.020,
        "europe-west1": 0.020,
        "asia-east1": 0.020,
    },
    "AWS": {
        "us-west-2": 0.023,
        "us-east-2": 0.023,
        "ap-southeast-1": 0.025,
        "eu-central-1": 0.0245,
    },
    "Aliyun": {
        "cn-shanghai": 0.0173,
        "cn-beijing": 0.0173,
        "us-west-1": 0.0160,
        "ap-southeast-1": 0.0170,
    },
}

# ---------------------------------------------------------------------------
# Internet / cross-region egress: USD per GB (source row -> destination column)
# Order: GCP×4, AWS×4, Aliyun×4, Local
# ---------------------------------------------------------------------------
_EGRESS_KEYS: list[ProviderRegion] = [
    ("GCP", "us-east1"),
    ("GCP", "us-west1"),
    ("GCP", "europe-west1"),
    ("GCP", "asia-east1"),
    ("AWS", "us-west-2"),
    ("AWS", "us-east-2"),
    ("AWS", "ap-southeast-1"),
    ("AWS", "eu-central-1"),
    ("Aliyun", "cn-shanghai"),
    ("Aliyun", "cn-beijing"),
    ("Aliyun", "us-west-1"),
    ("Aliyun", "ap-southeast-1"),
    ("Local", "local"),
]

_EGRESS_MATRIX_PER_GB: list[list[float]] = [
    [0.00, 0.02, 0.05, 0.08, 0.12, 0.12, 0.12, 0.12, 0.22, 0.22, 0.12, 0.12, 0.22],
    [0.02, 0.00, 0.05, 0.08, 0.12, 0.12, 0.12, 0.12, 0.22, 0.22, 0.12, 0.12, 0.22],
    [0.05, 0.05, 0.00, 0.08, 0.12, 0.12, 0.12, 0.12, 0.22, 0.22, 0.12, 0.12, 0.22],
    [0.08, 0.08, 0.08, 0.00, 0.12, 0.12, 0.12, 0.12, 0.22, 0.22, 0.12, 0.12, 0.22],
    [0.09, 0.09, 0.09, 0.09, 0.00, 0.02, 0.02, 0.02, 0.09, 0.09, 0.09, 0.09, 0.09],
    [0.09, 0.09, 0.09, 0.09, 0.02, 0.00, 0.02, 0.02, 0.09, 0.09, 0.09, 0.09, 0.09],
    [0.12, 0.12, 0.12, 0.12, 0.09, 0.09, 0.00, 0.09, 0.12, 0.12, 0.12, 0.12, 0.12],
    [0.09, 0.09, 0.09, 0.09, 0.02, 0.02, 0.02, 0.00, 0.09, 0.09, 0.09, 0.09, 0.09],
    [0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.00, 0.072, 0.80, 0.80, 0.118],
    [0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.118, 0.072, 0.00, 0.80, 0.80, 0.118],
    [0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.80, 0.80, 0.00, 0.08, 0.074],
    [0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.80, 0.80, 0.08, 0.00, 0.100],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
]

_EGRESS_INDEX: dict[ProviderRegion, int] = {k: i for i, k in enumerate(_EGRESS_KEYS)}


def _lookup_llm_prices(provider: str, region: str, model: str) -> tuple[float, float]:
    try:
        inp, out = LLM_TOKEN_PRICE_PER_MILLION[provider][region][model]
        return inp, out
    except KeyError as e:
        raise KeyError(
            f"No LLM token price for provider={provider!r} region={region!r} model={model!r}"
        ) from e


def llm_token_cost_usd(
    provider: str,
    region: str,
    model: str,
    input_tokens: float,
    output_tokens: float,
) -> float:
    """Total LLM charge from token counts (prices are per 1M tokens)."""
    inp_rate, out_rate = _lookup_llm_prices(provider, region, model)
    return (input_tokens / 1_000_000.0) * inp_rate + (output_tokens / 1_000_000.0) * out_rate


def video_service_cost_usd(
    provider: str,
    region: str,
    service: str,
    video_minutes: float,
) -> float:
    """Video API cost from duration in minutes. `service`: segment | speech_transcription | ocr | label_detection."""
    try:
        rate = VIDEO_SERVICE_USD_PER_MINUTE[provider][region][service]
    except KeyError as e:
        raise KeyError(
            f"No video service price for provider={provider!r} region={region!r} service={service!r}"
        ) from e
    return rate * video_minutes


def split_cost_usd(provider: str, region: str, minutes: float = 1.0) -> float:
    """Serverless split cost: table rate is USD per minute (1 vCPU + 1 GiB, per doc)."""
    try:
        return SPLIT_USD_PER_MINUTE[provider][region] * minutes
    except KeyError as e:
        raise KeyError(
            f"No split price for provider={provider!r} region={region!r}"
        ) from e


def storage_cost_usd(provider: str, region: str, gigabytes: float, days: float = 1.0) -> float:
    """
    Object storage charge for `gigabytes` over `days` (calendar simulation time).

    Table values are per GB-month; cost is prorated with STORAGE_DAYS_PER_MONTH.
    Billable duration uses whole days only: any positive fraction of a day counts
    as one full day; zero days yields zero charge.
    """
    if days <= 0 or gigabytes <= 0:
        return 0.0
    billable_days = math.ceil(days)
    try:
        per_gb_month = STORAGE_USD_PER_GB_MONTH[provider][region]
    except KeyError as e:
        raise KeyError(
            f"No storage price for provider={provider!r} region={region!r}"
        ) from e
    per_gb_day = per_gb_month / STORAGE_DAYS_PER_MONTH
    return per_gb_day * gigabytes * billable_days


def egress_rate_usd_per_gb(src: ProviderRegion, dst: ProviderRegion) -> float:
    """Marginal egress price ($/GB) from src to dst; same node is 0."""
    if src == dst:
        return 0.0
    try:
        i = _EGRESS_INDEX[src]
        j = _EGRESS_INDEX[dst]
    except KeyError as e:
        raise KeyError(
            f"Egress table only includes known regions; got src={src!r} dst={dst!r}"
        ) from e
    return _EGRESS_MATRIX_PER_GB[i][j]


def egress_cost_usd(src: ProviderRegion, dst: ProviderRegion, gigabytes: float) -> float:
    """Total egress charge for `gigabytes` transferred src -> dst."""
    return egress_rate_usd_per_gb(src, dst) * gigabytes


def get_egress_table() -> Mapping[ProviderRegion, Mapping[ProviderRegion, float]]:
    """Read-only view: nested map src -> dst -> $/GB (handy for simulators)."""
    return {
        _EGRESS_KEYS[i]: {
            _EGRESS_KEYS[j]: _EGRESS_MATRIX_PER_GB[i][j]
            for j in range(len(_EGRESS_KEYS))
        }
        for i in range(len(_EGRESS_KEYS))
    }
