#!/usr/bin/env python3
"""Extend min-P95 query budgets using existing baseline selected plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, RESULTS_DIR, load_default_config, load_solver_config  # noqa: E402
from src.cost_latency import critical_path_latency  # noqa: E402
from src.data_loader import load_endpoints, load_network_links, load_queries, load_scenarios  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.workflow import get_workflow  # noqa: E402


BASELINE_METHODS = ("single_cloud", "greedy", "dpgm", "mtgp")


def _load_plan_assignment(
    plans_root: Path,
    workflow_name: str,
    quality: str,
    method: str,
    endpoint_by_id: dict,
) -> dict:
    path = plans_root / f"{workflow_name}_{quality}" / method / "selected_plan.json"
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    return {
        item["logical_node"]: endpoint_by_id[item["endpoint_id"]]
        for item in plan["assignments"]
    }


def main() -> None:
    cfg = load_default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing-budget",
        type=Path,
        default=RESULTS_DIR
        / "experiment_logs"
        / "baseline_calibrated_query_budgets_Q1000_S50_dbfixed_minp95.csv",
    )
    parser.add_argument(
        "--plans-root",
        type=Path,
        default=RESULTS_DIR / "main_Q1000_S50_dbfixed_minp95_full",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR
        / "experiment_logs"
        / "baseline_calibrated_query_budgets_Q2000_S50_minp95_from_plans.csv",
    )
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=int(cfg.get("num_scenarios_per_query", 50)),
    )
    args = parser.parse_args()

    endpoints = load_endpoints()
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    network_index = {
        (link.src_endpoint_id, link.dst_endpoint_id): link
        for link in load_network_links()
    }
    config = load_solver_config()

    qdf = pd.read_csv(DATA_DIR / "queries.csv")
    existing = pd.read_csv(args.existing_budget)
    done_ids = set(existing["query_id"].astype(str))

    records = existing.to_dict("records")
    new_sla = {
        str(row["query_id"]): float(row["sla_sec_new"])
        for _, row in existing.iterrows()
    }

    for workflow_name in ("workflow1", "workflow2"):
        workflow = get_workflow(workflow_name)
        for quality in ("Q1", "Q2", "Q3"):
            queries = [
                query
                for query in load_queries(quality_level=quality, workflow=workflow_name)
                if query.query_id not in done_ids
            ]
            if not queries:
                continue
            query_ids = [query.query_id for query in queries]
            scenarios = load_scenarios(query_ids=query_ids)
            calibration_scenarios, _ = split_scenarios_by_query(
                scenarios,
                calibration_count=args.calibration_count,
            )
            scenario_by_q: dict[str, list] = {}
            for scenario in calibration_scenarios:
                scenario_by_q.setdefault(scenario.query_id, []).append(scenario)

            assignments = {
                method: _load_plan_assignment(
                    args.plans_root,
                    workflow_name,
                    quality,
                    method,
                    endpoint_by_id,
                )
                for method in BASELINE_METHODS
            }

            print(
                f"Extending budgets for {workflow_name}/{quality}: {len(queries)} queries",
                flush=True,
            )
            for query in queries:
                method_p95: dict[str, float] = {}
                for method, assignment in assignments.items():
                    latencies = [
                        critical_path_latency(
                            workflow,
                            assignment,
                            endpoint_by_id,
                            network_index,
                            query,
                            scenario,
                            config.ablation,
                        )
                        for scenario in scenario_by_q[query.query_id]
                    ]
                    method_p95[method] = float(pd.Series(latencies).quantile(0.95))
                budget_baseline = min(method_p95, key=method_p95.get)
                budget = args.factor * method_p95[budget_baseline]
                new_sla[query.query_id] = budget
                records.append(
                    {
                        "workflow": workflow_name,
                        "quality_level": quality,
                        "query_id": query.query_id,
                        "budget_factor": args.factor,
                        "budget_baseline": budget_baseline,
                        "baseline_p95_min": method_p95[budget_baseline],
                        "sla_sec_old": query.sla_sec,
                        "sla_sec_new": budget,
                        **{f"p95_{method}": value for method, value in method_p95.items()},
                    }
                )

    qdf["sla_sec"] = qdf["query_id"].astype(str).map(new_sla).fillna(qdf["sla_sec"])
    qdf.to_csv(DATA_DIR / "queries.csv", index=False)

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["workflow", "quality_level", "query_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} budget rows to {args.output}")


if __name__ == "__main__":
    main()
