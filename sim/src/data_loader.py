"""Load synthetic or real CSV data into Pydantic models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.schemas import Endpoint, NetworkLink, Query, Scenario
from src.workflow import LOGICAL_OPERATIONS

logger = logging.getLogger(__name__)


def load_endpoints(data_dir: Path | None = None) -> list[Endpoint]:
    path = (data_dir or DATA_DIR) / "endpoints.csv"
    df = pd.read_csv(path)
    endpoints = []
    for _, row in df.iterrows():
        model = row.get("model_name")
        if pd.isna(model):
            model = None
        endpoints.append(
            Endpoint(
                endpoint_id=str(row["endpoint_id"]),
                logical_operation=str(row["logical_operation"]),
                provider=str(row["provider"]),
                region=str(row["region"]),
                quality_level=str(row["quality_level"]),
                model_name=model,
                base_latency_sec=float(row["base_latency_sec"]),
                latency_per_mb=float(row["latency_per_mb"]),
                fixed_cost=float(row["fixed_cost"]),
                cost_per_mb=float(row["cost_per_mb"]),
                storage_cost_per_mb=float(row["storage_cost_per_mb"]),
                is_virtual=bool(row.get("is_virtual", False)),
            )
        )
    return endpoints


def load_network_links(data_dir: Path | None = None) -> list[NetworkLink]:
    path = (data_dir or DATA_DIR) / "network.csv"
    df = pd.read_csv(path)
    return [
        NetworkLink(
            src_endpoint_id=str(row["src_endpoint_id"]),
            dst_endpoint_id=str(row["dst_endpoint_id"]),
            bandwidth_mb_per_sec=float(row["bandwidth_mb_per_sec"]),
            rtt_sec=float(row["rtt_sec"]),
            egress_cost_per_gb=float(row["egress_cost_per_gb"]),
        )
        for _, row in df.iterrows()
    ]


def load_queries(
    data_dir: Path | None = None,
    quality_level: str | None = None,
    workflow: str | None = None,
) -> list[Query]:
    path = (data_dir or DATA_DIR) / "queries.csv"
    df = pd.read_csv(path)
    if quality_level:
        df = df[df["quality_level"] == quality_level]
    if workflow:
        df = df[df["workflow"] == workflow]
    queries: list[Query] = []
    for _, row in df.iterrows():
        fps = float(row["fps"]) if "fps" in row and pd.notna(row.get("fps")) else 30.0
        wf = str(row["workflow"]) if "workflow" in row and pd.notna(row.get("workflow")) else "workflow1"
        queries.append(
            Query(
                query_id=str(row["query_id"]),
                workflow=wf,
                quality_level=str(row["quality_level"]),
                video_size_mb=float(row["video_size_mb"]),
                video_duration_sec=float(row["video_duration_sec"]),
                fps=fps,
                sla_sec=float(row["sla_sec"]),
            )
        )
    return queries


def _normalize_op_key(op: str) -> str:
    return op.replace(" ", "_").replace("/", "_").replace("&", "and")


def load_scenarios(
    data_dir: Path | None = None,
    query_ids: list[str] | None = None,
) -> list[Scenario]:
    path = (data_dir or DATA_DIR) / "scenarios.csv"
    df = pd.read_csv(path)
    if query_ids:
        df = df[df["query_id"].isin(query_ids)]

    op_keys = {_normalize_op_key(op): op for op in LOGICAL_OPERATIONS}

    scenarios: list[Scenario] = []
    for _, row in df.iterrows():
        rho: dict[str, float] = {}
        for col in df.columns:
            if col.startswith("rho_"):
                key = col[4:]
                op_name = op_keys.get(key, key.replace("_", " "))
                for op in LOGICAL_OPERATIONS:
                    if _normalize_op_key(op) == key:
                        op_name = op
                        break
                rho[op_name] = float(row[col])

        exec_mult: dict[str, float] = {}
        bw_mult: dict[str, float] = {}
        rtt_mult: dict[str, float] = {}
        for col in df.columns:
            if col.startswith("exec_mult_"):
                exec_mult[col[10:]] = float(row[col])
            elif col.startswith("bw_mult_"):
                bw_mult[col[8:]] = float(row[col])
            elif col.startswith("rtt_mult_"):
                rtt_mult[col[9:]] = float(row[col])

        scenarios.append(
            Scenario(
                query_id=str(row["query_id"]),
                scenario_id=str(row["scenario_id"]),
                rho=rho,
                database_output_tokens=(
                    float(row["database_output_tokens"])
                    if "database_output_tokens" in row
                    and pd.notna(row.get("database_output_tokens"))
                    else None
                ),
                exec_latency_multiplier=exec_mult,
                bandwidth_multiplier=bw_mult,
                rtt_multiplier=rtt_mult,
                exec_stress=float(row.get("exec_stress", 1.0)),
                bw_stress=float(row.get("bw_stress", 1.0)),
                rtt_stress=float(row.get("rtt_stress", 1.0)),
            )
        )
    return scenarios


def get_virtual_endpoint_map(endpoints: list[Endpoint]) -> dict[str, Endpoint]:
    mapping: dict[str, Endpoint] = {}
    for ep in endpoints:
        if ep.is_virtual:
            mapping[f"virtual_{ep.logical_operation}"] = ep
        if ep.endpoint_id == "client_source":
            mapping["virtual_ClientSource"] = ep
        if ep.endpoint_id == "client_sink":
            mapping["virtual_ClientSink"] = ep
    return mapping
