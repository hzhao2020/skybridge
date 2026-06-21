"""Synthetic data generation for SkyFlow simulation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG_DIR, DATA_DIR, load_default_config
from src.pricing import (
    endpoint_cost_fields,
    expected_data_conversion_ratio,
    caption_output_tokens,
    caption_output_tokens_per_frame_range,
    expected_caption_output_tokens_per_frame,
    llm_performance,
    physical_endpoint_exists,
    q_a_output_token_range,
    query_generation_params,
    region_network_values,
    sample_data_conversion_ratio,
    tokens_per_mb,
    video_mb_per_minute,
    database_output_token_range,
)
from src.workflow import LOGICAL_OPERATIONS, WORKFLOW_OPERATIONS
from src.measurement.network import LinkCategory, classify_link, sample_link_by_index

logger = logging.getLogger(__name__)

VIRTUAL_ENDPOINTS = [
    {
        "endpoint_id": "client_source",
        "logical_operation": "ClientSource",
        "provider": "client",
        "region": "local",
        "quality_level": "Q1",
        "model_name": None,
        "base_latency_sec": 0.0,
        "latency_per_mb": 0.0,
        "fixed_cost": 0.0,
        "cost_per_mb": 0.0,
        "storage_cost_per_mb": 0.0,
        "is_virtual": True,
    },
    {
        "endpoint_id": "client_sink",
        "logical_operation": "ClientSink",
        "provider": "client",
        "region": "local",
        "quality_level": "Q1",
        "model_name": None,
        "base_latency_sec": 0.0,
        "latency_per_mb": 0.0,
        "fixed_cost": 0.0,
        "cost_per_mb": 0.0,
        "storage_cost_per_mb": 0.0,
        "is_virtual": True,
    },
]


def generate_all(output_dir: Path | None = None, seed: int | None = None) -> None:
    cfg = load_default_config()
    rng = np.random.default_rng(seed if seed is not None else cfg.get("random_seed", 42))
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    providers = cfg["providers"]
    quality_models = cfg["quality_models"]
    regions_mode = cfg.get("regions_per_provider", "all")

    endpoints_df = _generate_endpoints(
        rng, providers, quality_models, regions_mode=regions_mode
    )
    network_seed = seed if seed is not None else cfg.get("random_seed", 42)
    network_rng = np.random.default_rng(network_seed)
    network_df = _generate_network(endpoints_df, network_rng)
    queries_df = _generate_queries(rng, cfg)
    scenarios_df = _generate_scenarios(rng, queries_df, cfg)

    endpoints_df.to_csv(out / "endpoints.csv", index=False)
    network_df.to_csv(out / "network.csv", index=False)
    queries_df.to_csv(out / "queries.csv", index=False)
    scenarios_df.to_csv(out / "scenarios.csv", index=False)

    logger.info("Synthetic data written to %s", out)


def _regions_for_provider(
    regions: list[str],
    *,
    regions_mode: str | int,
) -> list[str]:
    if regions_mode == "first":
        return regions[:1]
    if isinstance(regions_mode, int):
        return regions[:regions_mode]
    if isinstance(regions_mode, str) and regions_mode.isdigit():
        return regions[: int(regions_mode)]
    return list(regions)


def _generate_endpoints(
    rng: np.random.Generator,
    providers: dict,
    quality_models: dict,
    *,
    regions_mode: str,
) -> pd.DataFrame:
    rows: list[dict] = list(VIRTUAL_ENDPOINTS)
    quality_levels = ["Q1", "Q2", "Q3"]
    non_quality_latency: dict[tuple[str, str, str], tuple[float, float]] = {}

    for op in LOGICAL_OPERATIONS:
        for ql in quality_levels:
            model_name = quality_models[ql] if op in ("Video Caption", "Q/A") else None
            for prov, regions in providers.items():
                for region in _regions_for_provider(regions, regions_mode=regions_mode):
                    if not physical_endpoint_exists(
                        prov, region, op, model_name=model_name
                    ):
                        continue
                    eid = (
                        f"{op.replace(' ', '_').replace('/', '_').replace('&', 'and')}"
                        f"_{prov}_{region}_{ql}"
                    ).lower()
                    try:
                        fixed_cost, cost_per_mb, storage_cost = endpoint_cost_fields(
                            op, prov, region, ql, model_name
                        )
                    except KeyError:
                        continue

                    latency_key = (op, prov, region)
                    if latency_key not in non_quality_latency:
                        non_quality_latency[latency_key] = (
                            float(rng.uniform(0.5, 5.0)),
                            float(rng.uniform(0.01, 0.1)),
                        )
                    base_lat, lat_per_mb = non_quality_latency[latency_key]
                    if op in ("Video Caption", "Q/A"):
                        ttft, throughput = llm_performance(prov, region, model_name)
                        base_lat = ttft
                        if op == "Video Caption":
                            qcfg = query_generation_params()
                            mean_duration = (
                                float(qcfg["video_duration_sec_min"])
                                + float(qcfg["video_duration_sec_max"])
                            ) / 2.0
                            sampled_video_mb = video_mb_per_minute() * (mean_duration / 60.0)
                            sampled_video_mb *= expected_data_conversion_ratio(
                                "Video Split & Sample",
                                ql,
                            )
                            mean_output_tokens = caption_output_tokens(
                                mean_duration,
                                ql,
                                expected_caption_output_tokens_per_frame(),
                            )
                            lat_per_mb = (
                                mean_output_tokens
                                / max(sampled_video_mb, 1e-9)
                                / max(throughput, 1e-9)
                            )
                        else:
                            rho_out = expected_data_conversion_ratio(op, ql)
                            lat_per_mb = rho_out * tokens_per_mb() / max(throughput, 1e-9)

                    rows.append(
                        {
                            "endpoint_id": eid,
                            "logical_operation": op,
                            "provider": prov,
                            "region": region,
                            "quality_level": ql,
                            "model_name": model_name,
                            "base_latency_sec": base_lat,
                            "latency_per_mb": lat_per_mb,
                            "fixed_cost": fixed_cost,
                            "cost_per_mb": cost_per_mb,
                            "storage_cost_per_mb": storage_cost,
                            "is_virtual": False,
                        }
                    )

    return pd.DataFrame(rows)


def _endpoint_region_key(row: pd.Series) -> tuple[str, str]:
    if row.get("is_virtual"):
        return ("client", "local")
    return str(row["provider"]), str(row["region"])


def _generate_network(
    endpoints_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    link_indices: dict[tuple[str, tuple[str, str], tuple[str, str]], int] = {}
    for _, src_row in endpoints_df.iterrows():
        for _, dst_row in endpoints_df.iterrows():
            src_id = str(src_row["endpoint_id"])
            dst_id = str(dst_row["endpoint_id"])
            if src_id == dst_id:
                continue
            src_reg = _endpoint_region_key(src_row)
            dst_reg = _endpoint_region_key(dst_row)
            egress, _, _ = region_network_values(src_reg, dst_reg)
            category = classify_link(src_reg, dst_reg)
            idx_key = (category.value, src_reg, dst_reg)
            if idx_key not in link_indices:
                link_indices[idx_key] = int(rng.integers(0, 1_000_000))
            measured = sample_link_by_index(
                src_reg,
                dst_reg,
                link_indices[idx_key],
            )
            if category is not LinkCategory.NONE:
                link_indices[idx_key] += 1
            rtt = measured.rtt_sec
            bw = measured.bandwidth_mb_per_sec
            rows.append(
                {
                    "src_endpoint_id": src_id,
                    "dst_endpoint_id": dst_id,
                    "bandwidth_mb_per_sec": bw,
                    "rtt_sec": rtt,
                    "egress_cost_per_gb": egress,
                }
            )
    return pd.DataFrame(rows)


def _generate_queries(
    rng: np.random.Generator,
    cfg: dict,
) -> pd.DataFrame:
    """
    Paper Query: 100 requests per quality; duration ~ Uniform(1min, 30min); fps=30;
    Workflow1 : Workflow2 = 1 : 1.
    """
    qcfg = query_generation_params()
    per_workflow = cfg.get("num_queries_per_workflow_quality")
    if per_workflow is not None:
        n_wf1 = int(per_workflow)
        n_wf2 = int(per_workflow)
    else:
        n_total = int(cfg.get("num_queries_per_quality", qcfg["requests_per_quality_level"]))
        ratio = qcfg.get("workflow1_workflow2_ratio", [1, 1])
        r_sum = sum(int(x) for x in ratio)
        n_wf1 = n_total * int(ratio[0]) // r_sum
        n_wf2 = n_total - n_wf1
    dur_lo = float(qcfg["video_duration_sec_min"])
    dur_hi = float(qcfg["video_duration_sec_max"])
    fps = float(qcfg["fps"])
    mb_per_min = video_mb_per_minute()

    rows: list[dict] = []
    for ql in ["Q1", "Q2", "Q3"]:
        for wf, count in (("workflow1", n_wf1), ("workflow2", n_wf2)):
            for i in range(count):
                qid = f"{wf}_{ql}_q{i:04d}"
                duration = float(rng.uniform(dur_lo, dur_hi))
                video_size = mb_per_min * (duration / 60.0)
                base_sla = 600 + video_size * 0.8 + duration * 0.15
                ql_factor = {"Q1": 2.0, "Q2": 1.5, "Q3": 1.2}[ql]
                sla = base_sla * ql_factor
                rows.append(
                    {
                        "query_id": qid,
                        "workflow": wf,
                        "quality_level": ql,
                        "video_size_mb": video_size,
                        "video_duration_sec": duration,
                        "fps": fps,
                        "sla_sec": sla,
                    }
                )
    return pd.DataFrame(rows)


def _generate_scenarios(
    rng: np.random.Generator,
    queries_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Scenario rows: rho from paper LogNormal means."""
    n_scenarios = int(cfg.get("num_scenarios_per_query", 10)) + int(
        cfg.get("num_heldout_scenarios_per_query", 0)
    )
    rows: list[dict] = []

    for _, qrow in queries_df.iterrows():
        qid = qrow["query_id"]
        ql = str(qrow["quality_level"])
        for s in range(n_scenarios):
            sid = f"{qid}_s{s:03d}"
            row: dict = {
                "query_id": qid,
                "scenario_id": sid,
                "database_output_tokens": float(
                    rng.uniform(*database_output_token_range())
                ),
                "q_a_output_tokens": float(rng.uniform(*q_a_output_token_range())),
                "caption_output_tokens_per_frame": float(
                    rng.uniform(*caption_output_tokens_per_frame_range())
                ),
                "exec_stress": 1.0,
                "bw_stress": 1.0,
                "rtt_stress": 1.0,
            }
            for op in LOGICAL_OPERATIONS:
                if op in ("Database", "Q/A", "Video Caption"):
                    continue
                key = f"rho_{op.replace(' ', '_').replace('/', '_').replace('&', 'and')}"
                row[key] = float(sample_data_conversion_ratio(op, ql, rng))
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all()
