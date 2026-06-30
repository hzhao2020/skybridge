"""Export optimization results to files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR
from src.schemas import OptimizationResult

logger = logging.getLogger(__name__)


def export_results(
    result: OptimizationResult,
    output_dir: Path | None = None,
    *,
    metrics_dir: Path | None = None,
) -> None:
    """Write per-run artifacts under ``output_dir``; append summary row to ``metrics_dir``."""
    out = output_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    metrics_root = metrics_dir or out
    metrics_root.mkdir(parents=True, exist_ok=True)

    plan_records = [
        {
            "workflow": result.workflow,
            "quality_level": result.quality_level,
            "logical_node": a.logical_node,
            "endpoint_id": a.endpoint_id,
            "provider": a.provider,
            "region": a.region,
            "model_name": a.model_name or "",
        }
        for a in result.assignments
    ]
    pd.DataFrame(plan_records).to_csv(out / "deployment_plan.csv", index=False)

    metrics_row = {
        "workflow": result.workflow,
        "quality_level": result.quality_level,
        "method": result.method,
        "objective_value": result.objective_value,
        "expected_cost": result.expected_cost,
        "avg_latency": result.avg_latency,
        "p95_latency": result.p95_latency,
        "p99_latency": result.p99_latency,
        "violation_rate": result.violation_rate,
        "cvar_value": result.cvar_value,
        "solver_runtime_sec": result.solver_runtime_sec,
        "status": result.status,
        "selected_initializer": result.selected_initializer or "",
    }
    metrics_path = metrics_root / "metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        combined = pd.concat([existing, pd.DataFrame([metrics_row])], ignore_index=True)
        combined.to_csv(metrics_path, index=False)
    else:
        pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)

    selected = {
        "workflow": result.workflow,
        "quality_level": result.quality_level,
        "method": result.method,
        "assignments": [a.model_dump() for a in result.assignments],
        "objective_value": result.objective_value,
        "metrics": metrics_row,
    }
    if result.selected_initializer is not None:
        selected["selected_initializer"] = result.selected_initializer
    if result.initializer_selection_history:
        selected["initializer_selection_history"] = result.initializer_selection_history
    with open(out / "selected_plan.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    if result.convergence_history:
        conv_df = pd.DataFrame(result.convergence_history)
        conv_df["workflow"] = result.workflow
        conv_df["quality_level"] = result.quality_level
        if "active_path_cut_count" not in conv_df:
            conv_df["active_path_cut_count"] = conv_df["active_scenario_count"]
        if "num_violated_cuts" not in conv_df:
            conv_df["num_violated_cuts"] = conv_df["num_violated_scenarios"]
        cols = [
            "workflow",
            "quality_level",
            "iteration",
            "active_scenario_count",
            "active_path_cut_count",
            "objective_value",
            "max_violation",
            "num_violated_scenarios",
            "num_violated_cuts",
            "runtime_sec",
        ]
        conv_out = conv_df[cols]
        conv_path = out / "convergence.csv"
        if conv_path.exists():
            existing = pd.read_csv(conv_path)
            conv_out = pd.concat([existing, conv_out], ignore_index=True)
        conv_out.to_csv(conv_path, index=False)

    if result.initializer_selection_history:
        init_df = pd.DataFrame(result.initializer_selection_history)
        init_df["workflow"] = result.workflow
        init_df["quality_level"] = result.quality_level
        init_df.to_csv(out / "initializer_selection.csv", index=False)

    logger.info("Results exported to %s", out)
