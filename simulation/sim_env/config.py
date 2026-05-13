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
        "us-east-1", 
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
# R ~ LogNormal on the original scale: ln(R) ~ N(μ, σ²).
# Table rows give **population mean m** and **population std v** of R; we convert to
# (μ, σ) on the log scale. ``database`` uses a uniform file size (see below), not LogNormal.
# ---------------------------------------------------------------------------
DATABASE_OUTPUT_FILE_BYTES_MIN = 4096
DATABASE_OUTPUT_FILE_BYTES_MAX = 6144


def _ratio_mean_std_to_lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Map target E[R]=mean, Std[R]=std to LogNormal (μ, σ) with ln(R)~N(μ, σ²)."""
    if mean <= 0:
        raise ValueError("ratio mean must be positive")
    if std < 0:
        raise ValueError("ratio std must be non-negative")
    if std == 0.0:
        return (math.log(mean), 0.0)
    var = std * std
    sigma_sq = math.log(1.0 + var / (mean * mean))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - 0.5 * sigma_sq
    return (mu, sigma)


# Keys: workflow1 (segment/split/caption/query) + workflow2-specific ops.
# Video Split / Shot Detection → mean 1, std 0 (degenerate lognormal / deterministic).
_RATIO_TABLE_MEAN_STD: dict[str, tuple[float, float]] = {
    "segment": (1.0, 0.0),
    "split": (1.0, 0.0),
    "shot_detection": (1.0, 0.0),
    "caption": (6.62846139e-04, 0.0006772164768231058),
    "speech_transcription": (1.315495e-05, 1.782099e-05),
    "ocr": (4.911324e-03, 2.880235e-03),
    "label_detection": (1.877817e-04, 4.854159e-05),
    "query": (0.1, 0.05),
    "qa": (0.1, 0.05),
    "answer": (0.1, 0.05),
}

DATA_CONVERSION_RATIO_LOGNORMAL: dict[str, tuple[float, float]] = {
    k: _ratio_mean_std_to_lognormal_params(m, s) for k, (m, s) in _RATIO_TABLE_MEAN_STD.items()
}


def sample_database_output_file_bytes(rng: random.Random) -> float:
    """Uniform database row payload in [DATABASE_OUTPUT_FILE_BYTES_MIN, MAX] bytes."""
    return float(rng.uniform(float(DATABASE_OUTPUT_FILE_BYTES_MIN), float(DATABASE_OUTPUT_FILE_BYTES_MAX)))


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
            # Note: LLM pricing table lists Beijing, US East 1 (Virginia), and AP Southeast 1 for Bailian
            "supported_regions": ["cn-beijing", "us-east-1", "ap-southeast-1"],
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
            # Note: LLM pricing table lists Beijing, US East 1 (Virginia), and AP Southeast 1 for Bailian
            "supported_regions": ["cn-beijing", "us-east-1", "ap-southeast-1"],
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
    """One sample: R = exp(μ + σ·Z) with Z ~ N(0,1); σ=0 yields R ≡ exp(μ)."""
    mu, sigma = data_conversion_ratio_lognormal_params(operation_name)
    rnd = rng or random.Random()
    if sigma == 0.0:
        return math.exp(mu)
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
