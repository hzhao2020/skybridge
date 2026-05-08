"""
config.py
Configuration file for Video Processing Workflow Simulation.
Logical Workflow: segment -> split -> caption -> query
"""

import math
import random

# ==========================================
# 1. GLOBAL REGION DEFINITIONS
# ==========================================
REGIONS = {
    "GCP": [
        "us-east1", 
        "us-west1", 
        "europe-west1", 
        "asia-east1"
    ],
    "AWS": [
        "us-west-2", 
        "us-east-2", 
        "ap-southeast-1", 
        "eu-central-1"
    ],
    "Aliyun": [
        "cn-shanghai", 
        "cn-beijing", 
        "us-west-1", 
        "ap-southeast-1"
    ]
}

# ==========================================
# 2. VIDEO SIZE (simulation assumption)
# ==========================================
# Effective compressed size used to relate clip duration to megabytes (e.g. transfer, storage).
VIDEO_MEGABYTES_PER_MINUTE = 120.0

# ==========================================
# 3. DATA CONVERSION RATIO (output size / input size)
# ==========================================
# R ~ LogNormal(μ, σ²) in the usual stats sense: ln(R) ~ N(μ, σ²).
# Use for predicting downstream data volume from an upstream node’s output.
DATA_CONVERSION_RATIO_LOGNORMAL: dict[str, tuple[float, float]] = {
    "segment": (0.00, 0.001),
    "split": (0.00, 0.005),
    "caption": (-11.86, 0.83),
    "query": (-1.72, 0.47),
}

# ==========================================
# 4. OPERATION CAPABILITIES MATRIX
# ==========================================
# Defines the valid physical nodes (Provider + Region [+ Model]) for each logical operation.
# Node ids elsewhere: {operation}_{provider}_{region} or {operation}_{provider}_{region}_{model}.

WORKFLOW_OPERATIONS = {
    
    # ---------------------------------------------------------
    # OPERATION 1: SEGMENT (Video Service)
    # ---------------------------------------------------------
    "segment": {
        "GCP": {
            "supported_regions": REGIONS["GCP"],
        },
        "AWS": {
            "supported_regions": REGIONS["AWS"],
        },
        "Aliyun": {
            # Note: According to the cost table, Aliyun segment is only listed for cn-shanghai
            "supported_regions": ["cn-shanghai"],
        },
    },

    # ---------------------------------------------------------
    # OPERATION 2: SPLIT (Serverless/Compute)
    # ---------------------------------------------------------
    "split": {
        "GCP": {
            "supported_regions": REGIONS["GCP"],
        },
        "AWS": {
            "supported_regions": REGIONS["AWS"],
        },
        "Aliyun": {
            "supported_regions": REGIONS["Aliyun"],
        },
    },

    # ---------------------------------------------------------
    # OPERATION 3: CAPTION (Multimodal/Vision LLMs)
    # ---------------------------------------------------------
    "caption": {
        "GCP": {
            "supported_regions": REGIONS["GCP"],
            "models": [
                "Gemini 2.5 Pro", 
                "Gemini 2.5 Flash"
            ]
        },
        "AWS": {
            "supported_regions": REGIONS["AWS"],
            "models": [
                "Amazon Nova Pro",
                "Amazon Nova Lite",
            ]
        },
        "Aliyun": {
            # Note: LLM pricing table lists Beijing, US West 1, and AP Southeast 1 for Bailian
            "supported_regions": ["cn-beijing", "us-west-1", "ap-southeast-1"],
            "models": [
                "Qwen3-VL-Plus", 
                "Qwen3-VL-Flash"
            ]
        }
    },

    # ---------------------------------------------------------
    # OPERATION 4: QUERY (Text LLMs)
    # ---------------------------------------------------------
    "query": {
        "GCP": {
            "supported_regions": REGIONS["GCP"],
            "models": [
                "Gemini 2.5 Pro", 
                "Gemini 2.5 Flash"
            ]
        },
        "AWS": {
            "supported_regions": REGIONS["AWS"],
            "models": [
                "Amazon Nova Pro",
                "Amazon Nova Lite",
            ]
        },
        "Aliyun": {
            # Note: LLM pricing table lists Beijing, US West 1, and AP Southeast 1 for Bailian
            "supported_regions": ["cn-beijing", "us-west-1", "ap-southeast-1"],
            "models": [
                "Qwen3-VL-Plus",
                "Qwen3-VL-Flash",
            ]
        }
    }
}

# ==========================================
# 5. HELPER FUNCTIONS FOR SIMULATOR
# ==========================================

def video_megabytes_from_duration_sec(duration_sec: float) -> float:
    """Return implied video size in MB given clip length and VIDEO_MEGABYTES_PER_MINUTE."""
    return VIDEO_MEGABYTES_PER_MINUTE * (float(duration_sec) / 60.0)


def video_duration_sec_from_megabytes(megabytes: float) -> float:
    """Clip duration in seconds implied by size and VIDEO_MEGABYTES_PER_MINUTE."""
    if VIDEO_MEGABYTES_PER_MINUTE <= 0:
        raise ValueError("VIDEO_MEGABYTES_PER_MINUTE must be positive")
    return 60.0 * float(megabytes) / VIDEO_MEGABYTES_PER_MINUTE


def data_conversion_ratio_lognormal_params(operation_name: str) -> tuple[float, float]:
    """
    Parameters for output/input ratio R ~ LogNormal: ln(R) ~ N(μ, σ²).

    Returns ``(μ, σ)`` where σ is the standard deviation on the log scale.
    """
    try:
        mu, sigma = DATA_CONVERSION_RATIO_LOGNORMAL[operation_name]
    except KeyError as e:
        raise KeyError(
            f"No data conversion model for operation={operation_name!r}"
        ) from e
    return mu, sigma


def expected_data_conversion_ratio(operation_name: str) -> float:
    """Mean of R under LogNormal ln(R)~N(mu,sigma squared): closed form."""
    mu, sigma = data_conversion_ratio_lognormal_params(operation_name)
    return math.exp(mu + 0.5 * sigma * sigma)


def plugin_mean_data_conversion_ratios(
    *,
    n_calibration_samples: int,
    rng: random.Random,
    operations: tuple[str, str, str, str] = (
        "segment",
        "split",
        "caption",
        "query",
    ),
) -> tuple[float, float, float, float]:
    """
    Plug-in estimate of mean conversion ratios from i.i.d. calibration samples only.

    Uses ``sample_data_conversion_ratio`` (not closed-form ``expected_data_conversion_ratio``),
    so optimizers/algorithms remain consistent with ``info from observed samples``.
    ``n_calibration_samples`` is the Monte Carlo averaging depth for the plug-in mean.
    """
    ops = operations
    if n_calibration_samples <= 0:
        raise ValueError("n_calibration_samples must be positive")

    sums = [0.0, 0.0, 0.0, 0.0]
    inv = 1.0 / float(n_calibration_samples)
    for _ in range(n_calibration_samples):
        for i, op in enumerate(ops):
            sums[i] += sample_data_conversion_ratio(op, rng)

    return tuple(s * inv for s in sums)  # type: ignore[return-value]


def sample_data_conversion_ratio(
    operation_name: str,
    rng: random.Random | None = None,
) -> float:
    """One sample: R = exp(μ + σ·Z) with Z ~ N(0,1)."""
    mu, sigma = data_conversion_ratio_lognormal_params(operation_name)
    rnd = rng or random.Random()
    z = rnd.gauss(0.0, 1.0)
    return math.exp(mu + sigma * z)


def get_supported_providers(operation_name: str) -> list:
    """Returns a list of providers that support a specific operation."""
    if operation_name in WORKFLOW_OPERATIONS:
        return list(WORKFLOW_OPERATIONS[operation_name].keys())
    return []

def get_supported_regions(operation_name: str, provider: str) -> list:
    """Returns the valid regions for a specific operation and provider combination."""
    try:
        return WORKFLOW_OPERATIONS[operation_name][provider]["supported_regions"]
    except KeyError:
        return []

def get_supported_models(operation_name: str, provider: str) -> list:
    """Returns the available LLMs for caption or query operations."""
    if operation_name not in ["caption", "query"]:
        return []
    try:
        return WORKFLOW_OPERATIONS[operation_name][provider].get("models", [])
    except KeyError:
        return []
