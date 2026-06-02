#!/usr/bin/env python3
"""Sweep query count for full MILP and decomposed MILP with held-out evaluation."""

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

try:
    import psutil
except ImportError:  # pragma: no cover - depends on the active conda environment
    psutil = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_default_config, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.evaluator import evaluate_deployment  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.milp_decomposition import solve_decomposition  # noqa: E402
from src.milp_full import solve_full_milp  # noqa: E402
from src.workflow import get_workflow  # noqa: E402


METHODS = ("full_milp", "decomposed_milp")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _assignment_signature(result) -> str:
    pairs = sorted((a.logical_node, a.endpoint_id) for a in result.assignments)
    return json.dumps(pairs, sort_keys=True)


def _load_solver_config_with_eta(eta: float):
    try:
        return load_solver_config({"eta": eta})
    except TypeError:
        config = load_solver_config()
        if hasattr(config, "model_copy"):
            return config.model_copy(update={"eta": eta})
        config.eta = eta
        return config


def _worker(
    method: str,
    workflow_name: str,
    quality: str,
    query_count: int,
    calibration_count: int,
    eta: float,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    try:
        workflow = get_workflow(workflow_name)
        config = _load_solver_config_with_eta(eta)
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

        if method == "full_milp":
            result = solve_full_milp(
                workflow,
                endpoints,
                queries,
                calibration_scenarios,
                quality,
                config,
            )
        elif method == "decomposed_milp":
            result = solve_decomposition(
                workflow,
                endpoints,
                queries,
                calibration_scenarios,
                quality,
                config,
                stop_on_infeasible=True,
            )
        else:
            raise ValueError(method)

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
                "eta": eta,
                "calibration_scenarios_per_query": calibration_count,
                "test_scenarios_per_query": (
                    len(test_scenarios) // query_count if query_count else 0
                ),
                "calibration_scenario_count": len(calibration_scenarios),
                "test_scenario_count": len(test_scenarios),
                "method": method,
                "solver_method": result.method,
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
                "eta": eta,
                "calibration_scenarios_per_query": calibration_count,
                "test_scenarios_per_query": float("nan"),
                "calibration_scenario_count": query_count * calibration_count,
                "test_scenario_count": float("nan"),
                "method": method,
                "solver_method": "",
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
    method: str,
    workflow: str,
    quality: str,
    query_count: int,
    calibration_count: int,
    eta: float,
) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker,
        args=(method, workflow, quality, query_count, calibration_count, eta, queue),
    )
    proc.start()
    peak_rss_mb = 0.0
    try:
        ps_proc = psutil.Process(proc.pid) if psutil is not None else None
        while proc.is_alive():
            if ps_proc is not None:
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
            "eta": eta,
            "method": method,
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
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=int(cfg.get("num_scenarios_per_query", 50)),
    )
    parser.add_argument("--eta", type=float, default=0.15)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "q_scaling_ablation_eta015_full_decomp.csv",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workflows = _parse_csv(args.workflows)
    qualities = _parse_csv(args.qualities)
    query_counts = _parse_ints(args.query_counts)
    methods = _parse_csv(args.methods)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choose from {list(METHODS)}")

    rows: list[dict] = []
    done: set[tuple[str, str, int, str]] = set()
    if args.resume and args.output.exists():
        rows = pd.read_csv(args.output).to_dict("records")
        done = {
            (
                str(row["workflow"]),
                str(row["quality_level"]),
                int(row["query_count"]),
                str(row["method"]),
            )
            for row in rows
        }

    total = len(workflows) * len(qualities) * len(query_counts) * len(methods)
    logging.info(
        "Q-scaling full/decomp ablation: workflows=%s qualities=%s Q=%s methods=%s S_cal=%d eta=%.3f",
        workflows,
        qualities,
        query_counts,
        methods,
        args.calibration_count,
        args.eta,
    )
    for workflow in workflows:
        for quality in qualities:
            for query_count in query_counts:
                for method in methods:
                    key = (workflow, quality, query_count, method)
                    if key in done:
                        logging.info(
                            "Skipping completed %s/%s Q=%d %s",
                            workflow,
                            quality,
                            query_count,
                            method,
                        )
                        continue
                    logging.info(
                        "Running %s %s/%s Q=%d (%d total planned rows)",
                        method,
                        workflow,
                        quality,
                        query_count,
                        total,
                    )
                    row = _run_one(
                        method,
                        workflow,
                        quality,
                        query_count,
                        args.calibration_count,
                        args.eta,
                    )
                    rows.append(row)
                    _write_rows(rows, args.output)
                    logging.info(
                        "Finished %s %s/%s Q=%d status=%s cost=%.6f vio=%.5f iter=%s active=%.3f wall=%.1fs",
                        method,
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
