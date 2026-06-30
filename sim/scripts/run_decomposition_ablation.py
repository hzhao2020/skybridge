#!/usr/bin/env python3
"""Compare full SAA-CVaR MILP with critical-path scenario-path cut generation."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import resource
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.evaluator import evaluate_deployment  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.milp_decomposition import solve_decomposition  # noqa: E402
from src.milp_full import solve_full_milp  # noqa: E402
from src.workflow import get_workflow  # noqa: E402

RUNS = [(wf, q) for wf in ("workflow1", "workflow2") for q in ("Q1", "Q2", "Q3")]


def _maxrss_mb() -> float:
    # macOS reports bytes; Linux reports KiB. Normalize by magnitude.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0) if rss > 10_000_000 else rss / 1024.0


def _worker(
    method: str,
    workflow_name: str,
    quality: str,
    solver_overrides: dict,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    try:
        workflow = get_workflow(workflow_name)
        config = load_solver_config(solver_overrides)
        endpoints = load_endpoints()
        queries = load_queries(quality_level=quality, workflow=workflow_name)
        scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
        calibration_scenarios, test_scenarios = split_scenarios_by_query(scenarios)

        if method == "full_milp":
            result = solve_full_milp(
                workflow, endpoints, queries, calibration_scenarios, quality, config
            )
        elif method == "decomposition":
            result = solve_decomposition(
                workflow, endpoints, queries, calibration_scenarios, quality, config
            )
        else:
            raise ValueError(method)

        endpoint_by_id = {ep.endpoint_id: ep for ep in endpoints}
        assignment = {
            a.logical_node: endpoint_by_id[a.endpoint_id] for a in result.assignments
        }
        metrics = evaluate_deployment(
            workflow,
            assignment,
            endpoints,
            queries,
            test_scenarios,
            quality,
            config,
        )
        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "method": method,
                "eta": config.eta,
                "initial_active_fraction": config.initial_active_fraction,
                "initial_active_strategy": config.initial_active_strategy,
                "active_batch_fraction": config.active_batch_fraction,
                "status": result.status,
                "objective_value": result.objective_value,
                "expected_cost": metrics["expected_cost"],
                "violation_rate": metrics["violation_rate"],
                "p95_latency": metrics["p95_latency"],
                "solver_runtime_sec": result.solver_runtime_sec,
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": _maxrss_mb(),
                "num_iterations": result.num_iterations,
                "active_scenario_count": result.active_scenario_count,
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - records external solver failures
        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "method": method,
                "eta": solver_overrides.get("eta", float("nan")),
                "initial_active_fraction": solver_overrides.get(
                    "initial_active_fraction", float("nan")
                ),
                "initial_active_strategy": solver_overrides.get(
                    "initial_active_strategy", ""
                ),
                "active_batch_fraction": solver_overrides.get(
                    "active_batch_fraction", float("nan")
                ),
                "status": "FAILED",
                "objective_value": float("nan"),
                "expected_cost": float("nan"),
                "violation_rate": float("nan"),
                "p95_latency": float("nan"),
                "solver_runtime_sec": float("nan"),
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": _maxrss_mb(),
                "num_iterations": 0,
                "active_scenario_count": 0,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def _run_one(method: str, workflow: str, quality: str, solver_overrides: dict) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker, args=(method, workflow, quality, solver_overrides, queue)
    )
    proc.start()
    proc.join()
    if not queue.empty():
        row = queue.get()
    else:
        row = {
            "workflow": workflow,
            "quality_level": quality,
            "method": method,
            "status": f"FAILED_EXIT_{proc.exitcode}",
            "error": "worker exited before returning a result",
        }
    row["process_exitcode"] = proc.exitcode
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "decomposition_ablation.csv",
    )
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--initial-active-fraction", type=float, default=0.3)
    parser.add_argument(
        "--initial-active-strategy",
        choices=["qbr"],
        default="qbr",
    )
    parser.add_argument("--active-batch-fraction", type=float, default=None)
    args = parser.parse_args()

    solver_overrides = {
        "eta": args.eta,
        "initial_active_fraction": args.initial_active_fraction,
        "initial_active_strategy": args.initial_active_strategy,
    }
    if args.active_batch_fraction is not None:
        solver_overrides["active_batch_fraction"] = args.active_batch_fraction

    rows: list[dict] = []
    for workflow, quality in RUNS:
        for method in ("full_milp", "decomposition"):
            logging.info("Running %s %s %s", method, workflow, quality)
            row = _run_one(method, workflow, quality, solver_overrides)
            logging.info(
                "%s %s %s status=%s wall=%.2fs rss=%.1fMB",
                method,
                workflow,
                quality,
                row.get("status"),
                float(row.get("wall_time_sec", 0.0) or 0.0),
                float(row.get("max_rss_mb", 0.0) or 0.0),
            )
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    with open(args.output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
