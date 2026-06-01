#!/usr/bin/env python3
"""Sweep query count for decomposition with held-out evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_default_config, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.evaluator import evaluate_deployment  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.milp_decomposition import solve_decomposition  # noqa: E402
from src.workflow import get_workflow  # noqa: E402


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _assignment_signature(result) -> str:
    pairs = sorted((a.logical_node, a.endpoint_id) for a in result.assignments)
    return json.dumps(pairs, sort_keys=True)


def _worker(
    workflow_name: str,
    quality: str,
    query_count: int,
    calibration_count: int,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    try:
        workflow = get_workflow(workflow_name)
        config = load_solver_config()
        endpoints = load_endpoints()
        queries = load_queries(quality_level=quality, workflow=workflow_name)[:query_count]
        if len(queries) != query_count:
            raise ValueError(
                f"requested {query_count} queries for {workflow_name}/{quality}, "
                f"but only found {len(queries)}"
            )

        scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
        calibration_scenarios, test_scenarios = split_scenarios_by_query(
            scenarios,
            calibration_count=calibration_count,
        )

        result = solve_decomposition(
            workflow,
            endpoints,
            queries,
            calibration_scenarios,
            quality,
            config,
            stop_on_infeasible=True,
        )

        endpoint_by_id = {ep.endpoint_id: ep for ep in endpoints}
        assignment = {
            a.logical_node: endpoint_by_id[a.endpoint_id] for a in result.assignments
        }
        metrics = evaluate_deployment(
            workflow=workflow,
            assignment=assignment,
            endpoints=endpoints,
            queries=queries,
            scenarios=test_scenarios,
            quality_level=quality,
            config=config,
        )

        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "query_count": query_count,
                "calibration_scenarios_per_query": calibration_count,
                "test_scenarios_per_query": (
                    len(test_scenarios) // query_count if query_count else 0
                ),
                "calibration_scenario_count": len(calibration_scenarios),
                "test_scenario_count": len(test_scenarios),
                "method": "decomposition",
                "status": result.status,
                "reused_last_feasible": "REUSED_LAST_FEASIBLE" in result.status,
                "objective_value": result.objective_value,
                "expected_cost": metrics["expected_cost"],
                "avg_latency": metrics["avg_latency"],
                "p95_latency": metrics["p95_latency"],
                "p99_latency": metrics["p99_latency"],
                "violation_rate": metrics["violation_rate"],
                "solver_runtime_sec": result.solver_runtime_sec,
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": 0.0,
                "num_iterations": result.num_iterations,
                "active_scenario_count": result.active_scenario_count,
                "active_scenario_fraction": (
                    result.active_scenario_count / len(calibration_scenarios)
                    if calibration_scenarios
                    else 0.0
                ),
                "assignment_signature": _assignment_signature(result),
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - records solver/environment failures
        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "query_count": query_count,
                "calibration_scenarios_per_query": calibration_count,
                "test_scenarios_per_query": float("nan"),
                "calibration_scenario_count": query_count * calibration_count,
                "test_scenario_count": float("nan"),
                "method": "decomposition",
                "status": "FAILED",
                "reused_last_feasible": False,
                "objective_value": float("nan"),
                "expected_cost": float("nan"),
                "avg_latency": float("nan"),
                "p95_latency": float("nan"),
                "p99_latency": float("nan"),
                "violation_rate": float("nan"),
                "solver_runtime_sec": float("nan"),
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": 0.0,
                "num_iterations": 0,
                "active_scenario_count": 0,
                "active_scenario_fraction": float("nan"),
                "assignment_signature": "",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def _run_one(
    workflow: str,
    quality: str,
    query_count: int,
    calibration_count: int,
) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker,
        args=(workflow, quality, query_count, calibration_count, queue),
    )
    proc.start()
    peak_rss_mb = 0.0
    try:
        ps_proc = psutil.Process(proc.pid)
        while proc.is_alive():
            try:
                rss = ps_proc.memory_info().rss
                rss += sum(
                    child.memory_info().rss
                    for child in ps_proc.children(recursive=True)
                    if child.is_running()
                )
                peak_rss_mb = max(peak_rss_mb, rss / (1024.0 * 1024.0))
            except psutil.Error:
                pass
            time.sleep(0.2)
    finally:
        proc.join()

    if queue.empty():
        row = {
            "workflow": workflow,
            "quality_level": quality,
            "query_count": query_count,
            "method": "decomposition",
            "status": f"FAILED_EXIT_{proc.exitcode}",
            "wall_time_sec": float("nan"),
            "max_rss_mb": peak_rss_mb,
            "error": "worker exited before returning a result",
        }
    else:
        row = queue.get()
        row["max_rss_mb"] = max(float(row.get("max_rss_mb", 0.0) or 0.0), peak_rss_mb)
    row["process_exitcode"] = proc.exitcode
    return row


def _write_rows(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default="workflow1,workflow2")
    parser.add_argument("--qualities", default="Q1,Q2,Q3")
    parser.add_argument(
        "--query-counts",
        default="100,200,300,400,500,600,700,800,900,1000",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=int(cfg.get("num_scenarios_per_query", 50)),
    )
    parser.add_argument("--budget-factor", type=float, default=1.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "q_scaling_ablation_eta010_gamma110.csv",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workflows = _parse_csv(args.workflows)
    qualities = _parse_csv(args.qualities)
    query_counts = _parse_ints(args.query_counts)

    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if args.resume and args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {
            (str(row["workflow"]), str(row["quality_level"]), int(row["query_count"]))
            for row in rows
        }

    total = len(workflows) * len(qualities) * len(query_counts)
    logging.info(
        "Q-scaling ablation: workflows=%s qualities=%s Q=%s S_cal=%d budget_factor=%.3f",
        workflows,
        qualities,
        query_counts,
        args.calibration_count,
        args.budget_factor,
    )
    for workflow in workflows:
        for quality in qualities:
            for query_count in query_counts:
                key = (workflow, quality, query_count)
                if key in done:
                    logging.info("Skipping completed %s/%s Q=%d", workflow, quality, query_count)
                    continue
                logging.info(
                    "Running decomposition %s/%s Q=%d (%d total planned rows)",
                    workflow,
                    quality,
                    query_count,
                    total,
                )
                row = _run_one(workflow, quality, query_count, args.calibration_count)
                row["budget_factor"] = args.budget_factor
                rows.append(row)
                _write_rows(rows, args.output)
                logging.info(
                    "Finished %s/%s Q=%d status=%s cost=%.6f vio=%.5f iter=%s active=%.3f wall=%.1fs",
                    workflow,
                    quality,
                    query_count,
                    row.get("status"),
                    float(row.get("expected_cost", float("nan"))),
                    float(row.get("violation_rate", float("nan"))),
                    row.get("num_iterations"),
                    float(row.get("active_scenario_fraction", float("nan"))),
                    float(row.get("wall_time_sec", float("nan"))),
                )

    _write_rows(rows, args.output)
    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
