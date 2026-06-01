"""Paper pricing tables and data-conversion ratio sampling for SkyFlow sim."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.config import CONFIG_DIR, load_yaml

_PRICING: dict[str, Any] | None = None
_NETWORK: dict[str, Any] | None = None


def load_pricing_config() -> dict[str, Any]:
    global _PRICING
    if _PRICING is None:
        _PRICING = load_yaml(CONFIG_DIR / "pricing.yaml")
    return _PRICING


def load_region_network_config() -> dict[str, Any]:
    global _NETWORK
    if _NETWORK is None:
        _NETWORK = load_yaml(CONFIG_DIR / "region_network.yaml")
    return _NETWORK


def quality_tier(quality_level: str) -> str:
    tiers = load_pricing_config()["quality_tiers"]
    return tiers[quality_level]


def ratio_mean_std_to_lognormal(mean: float, std: float) -> tuple[float, float]:
    if mean <= 0:
        raise ValueError("ratio mean must be positive")
    if std <= 0:
        return math.log(mean), 0.0
    var = std * std
    sigma_sq = math.log(1.0 + var / (mean * mean))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - 0.5 * sigma_sq
    return mu, sigma


def sample_data_conversion_ratio(
    logical_operation: str,
    quality_level: str,
    rng,
) -> float:
    """Sample output/input ratio R for one logical operation and quality level."""
    cfg = load_pricing_config()
    table = cfg["data_conversion_ratio_mean_std"]
    tier = quality_tier(quality_level)
    entry = table[logical_operation]
    if isinstance(entry, dict) and "mean" in entry:
        mean, std = float(entry["mean"]), float(entry["std"])
    elif tier in entry:
        mean = float(entry[tier]["mean"])
        std = float(entry[tier]["std"])
    else:
        raise KeyError(f"No conversion ratio for {logical_operation!r} tier {tier!r}")
    mu, sigma = ratio_mean_std_to_lognormal(mean, std)
    if sigma == 0.0:
        return math.exp(mu)
    z = rng.standard_normal()
    return math.exp(mu + sigma * z)


def expected_data_conversion_ratio(logical_operation: str, quality_level: str) -> float:
    cfg = load_pricing_config()
    table = cfg["data_conversion_ratio_mean_std"]
    tier = quality_tier(quality_level)
    entry = table[logical_operation]
    if isinstance(entry, dict) and "mean" in entry:
        mean, std = float(entry["mean"]), float(entry["std"])
    else:
        mean = float(entry[tier]["mean"])
        std = float(entry[tier]["std"])
    mu, sigma = ratio_mean_std_to_lognormal(mean, std)
    return math.exp(mu + 0.5 * sigma * sigma)


def _lookup_nested(table: dict, provider: str, region: str) -> float | None:
    prov = table.get(provider)
    if prov is None:
        return None
    return prov.get(region)


def tokens_per_mb() -> float:
    cfg = load_pricing_config()
    return float(cfg["tokens_per_kb"]) * 1024.0


def video_mb_per_minute() -> float:
    return float(load_pricing_config()["video_mb_per_minute"])


def llm_model_listed(provider: str, region: str, model: str) -> bool:
    """True only when (provider, region, model) appears in the paper LLM table."""
    cfg = load_pricing_config()
    return model in cfg.get("llm_price_per_million", {}).get(provider, {}).get(region, {})


def llm_prices_per_million(provider: str, region: str, model: str) -> tuple[float, float]:
    cfg = load_pricing_config()
    try:
        inp, out = cfg["llm_price_per_million"][provider][region][model]
        return float(inp), float(out)
    except KeyError as e:
        raise KeyError(
            f"No LLM price for provider={provider!r} region={region!r} model={model!r}"
        ) from e


def llm_performance(provider: str, region: str, model: str) -> tuple[float, float]:
    """Return (TTFT seconds, output tokens/sec) for a listed LLM endpoint."""
    cfg = load_pricing_config()
    try:
        ttft, throughput = cfg["llm_performance"][provider][region][model]
        return float(ttft), float(throughput)
    except KeyError as e:
        raise KeyError(
            f"No LLM performance for provider={provider!r} region={region!r} model={model!r}"
        ) from e


def is_llm_operation(logical_operation: str) -> bool:
    return logical_operation in ("Video Caption", "Q/A")


def llm_cost_usd(
    provider: str,
    region: str,
    model: str,
    input_size_mb: float,
    output_size_mb: float,
) -> float:
    input_price, output_price = llm_prices_per_million(provider, region, model)
    tpm = tokens_per_mb()
    input_tokens = input_size_mb * tpm
    output_tokens = output_size_mb * tpm
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def llm_latency_sec(
    provider: str,
    region: str,
    model: str,
    output_size_mb: float,
) -> float:
    ttft, throughput = llm_performance(provider, region, model)
    output_tokens = output_size_mb * tokens_per_mb()
    return ttft + output_tokens / max(throughput, 1e-9)


def endpoint_cost_fields(
    logical_operation: str,
    provider: str,
    region: str,
    quality_level: str,
    model_name: str | None,
) -> tuple[float, float, float]:
    """
    Map paper pricing to (fixed_cost, cost_per_mb, storage_cost_per_mb).

    Execution cost is modeled as fixed_cost + input_size_mb * cost_per_mb.
    """
    cfg = load_pricing_config()
    mb_per_min = video_mb_per_minute()
    days_per_month = float(cfg["storage_days_per_month"])
    mb_per_image = float(cfg["mb_per_vision_image"])

    storage_month = _lookup_nested(cfg["storage_usd_per_gb_month"], provider, region)
    if storage_month is None:
        raise KeyError(f"No object storage price for {provider}/{region}")
    storage_per_mb_day = (storage_month / days_per_month) / 1024.0

    op = logical_operation
    if op in ("Video Caption", "Q/A"):
        if not model_name:
            raise ValueError(f"{op} requires model_name")
        inp_p, out_p = llm_prices_per_million(provider, region, model_name)
        tpm = tokens_per_mb()
        rho_out = expected_data_conversion_ratio(op, quality_level)
        cost_per_mb = (tpm / 1_000_000.0) * (inp_p + out_p * rho_out)
        return 0.0, cost_per_mb, storage_per_mb_day

    if op == "Shot Detection":
        rate = _lookup_nested(cfg["shot_detection_usd_per_minute"], provider, region)
        if rate is None:
            raise KeyError(f"No shot detection price for {provider}/{region}")
        return 0.0, rate / mb_per_min, storage_per_mb_day

    if op == "Video Split & Sample":
        rate = _lookup_nested(cfg["video_split_usd_per_minute"], provider, region)
        if rate is None:
            raise KeyError(f"No video split price for {provider}/{region}")
        return 0.0, rate / mb_per_min, storage_per_mb_day

    if op == "OCR":
        rate = _lookup_nested(cfg["ocr_usd_per_image"], provider, region)
        if rate is None:
            raise KeyError(f"No OCR price for {provider}/{region}")
        return 0.0, rate / mb_per_image, storage_per_mb_day

    if op == "Label Detection":
        rate = _lookup_nested(cfg["label_detection_usd_per_image"], provider, region)
        if rate is None:
            raise KeyError(f"No label detection price for {provider}/{region}")
        return 0.0, rate / mb_per_image, storage_per_mb_day

    if op == "Speech Transcription":
        rate = _lookup_nested(cfg["speech_transcription_usd_per_minute"], provider, region)
        if rate is None:
            raise KeyError(f"No speech transcription price for {provider}/{region}")
        return 0.0, rate / mb_per_min, storage_per_mb_day

    if op == "Database":
        inst = _lookup_nested(cfg["database_instance_usd_per_month"], provider, region)
        db_stor = _lookup_nested(cfg["database_storage_usd_per_gb_month"], provider, region)
        if inst is None:
            raise KeyError(f"No database instance price for {provider}/{region}")
        fixed = inst / days_per_month
        stor_mb = 0.0
        if db_stor and db_stor > 0:
            stor_mb = (db_stor / days_per_month) / 1024.0
        return fixed, 0.0, stor_mb + storage_per_mb_day

    raise ValueError(f"Unknown logical operation: {op!r}")


def operation_supported(provider: str, region: str, logical_operation: str) -> bool:
    """Return whether pricing exists for this (provider, region, operation)."""
    cfg = load_pricing_config()
    key_map = {
        "Shot Detection": "shot_detection_usd_per_minute",
        "Video Split & Sample": "video_split_usd_per_minute",
        "OCR": "ocr_usd_per_image",
        "Label Detection": "label_detection_usd_per_image",
        "Speech Transcription": "speech_transcription_usd_per_minute",
        "Database": "database_instance_usd_per_month",
    }
    if logical_operation in ("Video Caption", "Q/A"):
        return bool(cfg.get("llm_price_per_million", {}).get(provider, {}).get(region))
    table_key = key_map.get(logical_operation)
    if table_key is None:
        return False
    return _lookup_nested(cfg[table_key], provider, region) is not None


def physical_endpoint_exists(
    provider: str,
    region: str,
    logical_operation: str,
    *,
    model_name: str | None = None,
) -> bool:
    """Physical node exists only if documented in pricing tables (incl. LLM model/region)."""
    if not operation_supported(provider, region, logical_operation):
        return False
    if logical_operation in ("Video Caption", "Q/A"):
        if not model_name:
            return False
        try:
            llm_performance(provider, region, model_name)
        except KeyError:
            return False
        return llm_model_listed(provider, region, model_name)
    return True


def query_generation_params() -> dict[str, Any]:
    cfg = load_pricing_config()
    q = dict(cfg.get("query", {}))
    q.setdefault("requests_per_quality_level", 100)
    q.setdefault("workflow1_workflow2_ratio", [1, 1])
    q.setdefault("video_duration_sec_min", 60)
    q.setdefault("video_duration_sec_max", 1800)
    q["fps"] = float(cfg.get("video_fps", 30))
    return q


def build_region_index() -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    net = load_region_network_config()
    labels = [(str(p), str(r)) for p, r in net["region_labels"]]
    index = {labels[i]: i for i in range(len(labels))}
    return labels, index


def region_network_values(
    src: tuple[str, str],
    dst: tuple[str, str],
) -> tuple[float, float, float]:
    """Return (egress_usd_per_gb, rtt_sec, bandwidth_mb_per_sec) for a region pair."""
    labels, index = build_region_index()
    net = load_region_network_config()
    if src == dst:
        return 0.0, 0.0, float(net["default_bandwidth_mb_per_sec"])
    client = ("Local", "local")
    s = client if src[0] == "client" else src
    d = client if dst[0] == "client" else dst
    if s not in index or d not in index:
        return 0.12, 0.10, float(net["default_bandwidth_mb_per_sec"])
    i, j = index[s], index[d]
    egress = float(net["egress_usd_per_gb"][i][j])
    rtt = float(net["rtt_sec"][i][j])
    bw = float(net["default_bandwidth_mb_per_sec"])
    return egress, rtt, bw
