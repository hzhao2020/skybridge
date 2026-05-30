"""Post-solve evaluation on all query-scenario pairs."""

from __future__ import annotations

import logging

import numpy as np

from src.cost_latency import (
    compute_cvar_value,
    critical_path_latency,
    total_cost,
)
from src.data_loader import get_virtual_endpoint_map, load_network_links
from src.schemas import Endpoint, Query, Scenario, SolverConfig, WorkflowDAG

logger = logging.getLogger(__name__)


def evaluate_deployment(
    workflow: WorkflowDAG,
    assignment: dict[str, Endpoint],
    endpoints: list[Endpoint],
    queries: list[Query],
    scenarios: list[Scenario],
    quality_level: str,
    config: SolverConfig,
    alpha: float = 0.0,
) -> dict:
    """Evaluate deployment on all query-scenario pairs."""
    endpoint_map = {e.endpoint_id: e for e in endpoints}
    virtual_map = get_virtual_endpoint_map(endpoints)
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()}

    assign_full = dict(assignment)
    assign_full.setdefault("ClientSource", virtual_map["virtual_ClientSource"])
    assign_full.setdefault("ClientSink", virtual_map["virtual_ClientSink"])

    scenario_by_q: dict[str, list[Scenario]] = {}
    for s in scenarios:
        scenario_by_q.setdefault(s.query_id, []).append(s)

    per_qs: list[dict] = []
    costs: list[float] = []
    latencies: list[float] = []
    slas: list[float] = []
    violations = 0
    total = 0

    for q in queries:
        if q.quality_level != quality_level:
            continue
        for s in scenario_by_q.get(q.query_id, []):
            cost = total_cost(
                workflow,
                {k: v for k, v in assign_full.items() if not workflow.is_virtual(k)},
                endpoint_map,
                network_index,
                q,
                s,
                config.ablation,
            )
            lat = critical_path_latency(
                workflow,
                assign_full,
                endpoint_map,
                network_index,
                q,
                s,
                config.ablation,
            )
            violated = lat > q.sla_sec
            if violated:
                violations += 1
            total += 1
            costs.append(cost)
            latencies.append(lat)
            slas.append(q.sla_sec)
            per_qs.append(
                {
                    "query_id": q.query_id,
                    "scenario_id": s.scenario_id,
                    "cost": cost,
                    "latency": lat,
                    "sla_sec": q.sla_sec,
                    "violated": violated,
                }
            )

    n = max(total, 1)
    lat_arr = np.array(latencies) if latencies else np.array([0.0])

    cvar = compute_cvar_value(latencies, slas, config.eta) if config.ablation.enable_cvar else 0.0

    return {
        "expected_cost": float(np.mean(costs)) if costs else 0.0,
        "avg_latency": float(np.mean(lat_arr)),
        "p95_latency": float(np.percentile(lat_arr, 95)) if len(lat_arr) else 0.0,
        "p99_latency": float(np.percentile(lat_arr, 99)) if len(lat_arr) else 0.0,
        "violation_rate": violations / n,
        "cvar_value": cvar,
        "per_qs": per_qs,
    }
