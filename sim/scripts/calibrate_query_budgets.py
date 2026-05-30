#!/usr/bin/env python3
"""Set per-query latency budgets from calibration-only baseline samples."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINE_SOLVERS, solve_baseline  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR, load_solver_config  # noqa: E402
from src.cost_latency import critical_path_latency  # noqa: E402
from src.data_loader import load_endpoints, load_network_links, load_queries, load_scenarios  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.schemas import Query  # noqa: E402
from src.workflow import get_workflow  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, default=1.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "baseline_calibrated_query_budgets.csv",
    )
    args = parser.parse_args()

    endpoints = load_endpoints()
    endpoint_by_id = {ep.endpoint_id: ep for ep in endpoints}
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()}
    config = load_solver_config()
    qdf = pd.read_csv(DATA_DIR / "queries.csv")
    backup = RESULTS_DIR / "experiment_logs" / "queries_before_budget_calibration.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        qdf.to_csv(backup, index=False)

    records: list[dict] = []
    new_sla: dict[str, float] = {}
    methods = sorted(BASELINE_SOLVERS)
    for workflow_name in ("workflow1", "workflow2"):
        workflow = get_workflow(workflow_name)
        for quality in ("Q1", "Q2", "Q3"):
            logging.info("Calibrating %s %s", workflow_name, quality)
            queries = load_queries(quality_level=quality, workflow=workflow_name)
            scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
            calibration_scenarios, _ = split_scenarios_by_query(scenarios)
            scenario_by_q: dict[str, list] = {}
            for s in calibration_scenarios:
                scenario_by_q.setdefault(s.query_id, []).append(s)

            assignments = {}
            for method in methods:
                result = solve_baseline(
                    method,
                    workflow,
                    endpoints,
                    queries,
                    calibration_scenarios,
                    quality,
                    config,
                )
                assignments[method] = {
                    a.logical_node: endpoint_by_id[a.endpoint_id] for a in result.assignments
                }

            for q in queries:
                method_p90: dict[str, float] = {}
                for method, assignment in assignments.items():
                    latencies = [
                        critical_path_latency(
                            workflow,
                            assignment,
                            endpoint_by_id,
                            network_index,
                            q,
                            s,
                            config.ablation,
                        )
                        for s in scenario_by_q[q.query_id]
                    ]
                    method_p90[method] = float(pd.Series(latencies).quantile(0.90))
                best = min(method_p90, key=method_p90.get)
                budget = args.factor * method_p90[best]
                new_sla[q.query_id] = budget
                records.append(
                    {
                        "workflow": workflow_name,
                        "quality_level": quality,
                        "query_id": q.query_id,
                        "budget_factor": args.factor,
                        "best_baseline": best,
                        "baseline_p90_min": method_p90[best],
                        "sla_sec_old": q.sla_sec,
                        "sla_sec_new": budget,
                        **{f"p90_{m}": v for m, v in method_p90.items()},
                    }
                )

    qdf["sla_sec"] = qdf["query_id"].map(new_sla).fillna(qdf["sla_sec"])
    qdf.to_csv(DATA_DIR / "queries.csv", index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    logging.info("Updated %d query budgets; wrote %s", len(records), args.output)


if __name__ == "__main__":
    main()
