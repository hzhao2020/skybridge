#!/usr/bin/env python3
"""Sweep query count for full MILP and decomposed MILP with query-heldout evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import math
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
    initial_active_fraction: float,
    initial_active_strategy: str,
    active_batch_fraction: float | None,
    gurobi_time_limit_sec: float | None,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    try:
        workflow = get_workflow(workflow_name)
        solver_overrides = {
            "eta": eta,
            "initial_active_fraction": initial_active_fraction,
            "initial_active_strategy": initial_active_strategy,
        }
        if active_batch_fraction is not None:
            solver_overrides["active_batch_fraction"] = active_batch_fraction
        if gurobi_time_limit_sec is not None:
            solver_overrides["gurobi_time_limit_sec"] = gurobi_time_limit_sec
        config = load_solver_config(solver_overrides)
        endpoints = load_endpoints()
        train_queries = load_queries(
            quality_level=quality,
            workflow=workflow_name,
            split="train",
        )[:query_count]
        test_queries = load_queries(
            quality_level=quality,
            workflow=workflow_name,
            split="test",
        )[:query_count]
        if len(train_queries) != query_count:
            raise ValueError(
                f"requested {query_count} train queries for {workflow_name}/{quality}, "
                f"but only found {len(train_queries)}"
            )
        if len(test_queries) != query_count:
            raise ValueError(
                f"requested {query_count} test queries for {workflow_name}/{quality}, "
                f"but only found {len(test_queries)}"
            )

        train_scenarios_all = load_scenarios(query_ids=[q.query_id for q in train_queries])
        by_train_query: dict[str, list] = {}
        for scenario in train_scenarios_all:
            by_train_query.setdefault(scenario.query_id, []).append(scenario)
        calibration_scenarios = []
        for query_id in sorted(by_train_query):
            calibration_scenarios.extend(
                sorted(by_train_query[query_id], key=lambda s: s.scenario_id)[:calibration_count]
            )
        test_scenarios = load_scenarios(query_ids=[q.query_id for q in test_queries])

        if method == "full_milp":
            result = solve_full_milp(
                workflow,
                endpoints,
                train_queries,
                calibration_scenarios,
                quality,
                config,
            )
        elif method == "decomposed_milp":
            result = solve_decomposition(
                workflow,
                endpoints,
                train_queries,
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
            queries=test_queries,
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
                "initial_active_fraction": config.initial_active_fraction,
                "initial_active_strategy": config.initial_active_strategy,
                "active_batch_fraction": config.active_batch_fraction,
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
                "success_rate": 1.0 - float(metrics["violation_rate"]),
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
                "initial_active_fraction": initial_active_fraction,
                "initial_active_strategy": initial_active_strategy,
                "active_batch_fraction": (
                    float("nan") if active_batch_fraction is None else active_batch_fraction
                ),
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
                "success_rate": float("nan"),
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
    initial_active_fraction: float,
    initial_active_strategy: str,
    active_batch_fraction: float | None,
    gurobi_time_limit_sec: float | None,
    wall_time_limit_sec: float | None,
) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker,
        args=(
            method,
            workflow,
            quality,
            query_count,
            calibration_count,
            eta,
            initial_active_fraction,
            initial_active_strategy,
            active_batch_fraction,
            gurobi_time_limit_sec,
            queue,
        ),
    )
    proc.start()
    start_time = time.perf_counter()
    peak_rss_mb = 0.0
    timed_out = False
    try:
        ps_proc = psutil.Process(proc.pid) if psutil is not None else None
        while proc.is_alive():
            if wall_time_limit_sec is not None and time.perf_counter() - start_time > wall_time_limit_sec:
                timed_out = True
                if ps_proc is not None:
                    try:
                        for child in ps_proc.children(recursive=True):
                            child.terminate()
                        ps_proc.terminate()
                    except (psutil.Error, PermissionError):
                        proc.terminate()
                else:
                    proc.terminate()
                break
            if ps_proc is not None:
                try:
                    rss = ps_proc.memory_info().rss
                    rss += sum(
                        child.memory_info().rss
                        for child in ps_proc.children(recursive=True)
                        if child.is_running()
                    )
                    peak_rss_mb = max(peak_rss_mb, rss / (1024.0 * 1024.0))
                except (psutil.Error, PermissionError):
                    pass
            time.sleep(0.2)
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()

    if timed_out:
        row = {
            "workflow": workflow,
            "quality_level": quality,
            "query_count": query_count,
            "eta": eta,
            "calibration_scenarios_per_query": calibration_count,
            "initial_active_fraction": initial_active_fraction,
            "initial_active_strategy": initial_active_strategy,
            "active_batch_fraction": (
                float("nan") if active_batch_fraction is None else active_batch_fraction
            ),
            "method": method,
            "solver_method": "",
            "status": "WALL_TIME_LIMIT",
            "reused_last_feasible": False,
            "objective_value": float("nan"),
            "expected_cost": float("nan"),
            "avg_latency": float("nan"),
            "p95_latency": float("nan"),
            "p99_latency": float("nan"),
            "violation_rate": float("nan"),
            "success_rate": float("nan"),
            "solver_runtime_sec": float("nan"),
            "wall_time_sec": wall_time_limit_sec,
            "max_rss_mb": peak_rss_mb,
            "num_iterations": 0,
            "active_scenario_count": 0,
            "active_scenario_fraction": float("nan"),
            "assignment_signature": "",
            "error": f"wall time limit exceeded ({wall_time_limit_sec}s)",
        }
    elif queue.empty():
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


def _finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_pair_summary(rows: list[dict], output: Path) -> None:
    records: list[dict] = []
    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame(records).to_csv(output, index=False)
        return

    for (workflow, quality, query_count), group in df.groupby(
        ["workflow", "quality_level", "query_count"], dropna=False
    ):
        by_method = {str(row["method"]): row for row in group.to_dict("records")}
        full = by_method.get("full_milp", {})
        skyflow = by_method.get("decomposed_milp", {})
        full_cost = _finite(full.get("expected_cost"))
        sky_cost = _finite(skyflow.get("expected_cost"))
        full_wall = _finite(full.get("wall_time_sec"))
        sky_wall = _finite(skyflow.get("wall_time_sec"))
        full_solver = _finite(full.get("solver_runtime_sec"))
        sky_solver = _finite(skyflow.get("solver_runtime_sec"))
        full_mem = _finite(full.get("max_rss_mb"))
        sky_mem = _finite(skyflow.get("max_rss_mb"))

        cost_gap = (
            sky_cost - full_cost
            if sky_cost is not None and full_cost is not None
            else float("nan")
        )
        records.append(
            {
                "workflow": workflow,
                "quality_level": quality,
                "query_count": int(query_count),
                "full_status": full.get("status", ""),
                "skyflow_status": skyflow.get("status", ""),
                "full_expected_cost": full_cost if full_cost is not None else float("nan"),
                "skyflow_expected_cost": sky_cost if sky_cost is not None else float("nan"),
                "skyflow_cost_gap_abs": cost_gap,
                "skyflow_cost_gap_pct": (
                    100.0 * cost_gap / full_cost
                    if full_cost not in (None, 0.0) and math.isfinite(cost_gap)
                    else float("nan")
                ),
                "full_violation_rate": _finite(full.get("violation_rate")),
                "skyflow_violation_rate": _finite(skyflow.get("violation_rate")),
                "full_success_rate": _finite(full.get("success_rate")),
                "skyflow_success_rate": _finite(skyflow.get("success_rate")),
                "full_avg_latency": _finite(full.get("avg_latency")),
                "skyflow_avg_latency": _finite(skyflow.get("avg_latency")),
                "full_p95_latency": _finite(full.get("p95_latency")),
                "skyflow_p95_latency": _finite(skyflow.get("p95_latency")),
                "full_solver_runtime_sec": full_solver if full_solver is not None else float("nan"),
                "skyflow_solver_runtime_sec": sky_solver if sky_solver is not None else float("nan"),
                "solver_runtime_speedup": (
                    full_solver / sky_solver
                    if full_solver is not None and sky_solver not in (None, 0.0)
                    else float("nan")
                ),
                "full_wall_time_sec": full_wall if full_wall is not None else float("nan"),
                "skyflow_wall_time_sec": sky_wall if sky_wall is not None else float("nan"),
                "wall_time_speedup": (
                    full_wall / sky_wall
                    if full_wall is not None and sky_wall not in (None, 0.0)
                    else float("nan")
                ),
                "full_peak_memory_mb": full_mem if full_mem is not None else float("nan"),
                "skyflow_peak_memory_mb": sky_mem if sky_mem is not None else float("nan"),
                "memory_reduction_mb": (
                    full_mem - sky_mem
                    if full_mem is not None and sky_mem is not None
                    else float("nan")
                ),
                "memory_reduction_pct": (
                    100.0 * (full_mem - sky_mem) / full_mem
                    if full_mem not in (None, 0.0) and sky_mem is not None
                    else float("nan")
                ),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values(
        ["workflow", "quality_level", "query_count"]
    ).to_csv(output, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default="workflow1,workflow2,workflow3,workflow4")
    parser.add_argument("--qualities", default="Q1,Q2,Q3")
    parser.add_argument(
        "--query-counts",
        default="200,400,600,800,1000,1200,1400,1600,1800,2000",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=int(cfg.get("num_scenarios_per_query", 50)),
    )
    parser.add_argument("--eta", type=float, default=float(cfg.get("eta", 0.10)))
    parser.add_argument("--initial-active-fraction", type=float, default=0.3)
    parser.add_argument(
        "--initial-active-strategy",
        choices=["qbr"],
        default="qbr",
    )
    parser.add_argument("--active-batch-fraction", type=float, default=None)
    parser.add_argument("--gurobi-time-limit-sec", type=float, default=None)
    parser.add_argument("--wall-time-limit-sec", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "q_scaling_full_vs_decomp_S50_Q200_2000_minp95_v2.csv",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}_paired_summary.csv"
    )

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
        "Q-scaling full/decomp ablation: workflows=%s qualities=%s Q=%s methods=%s S_train=%d eta=%.3f",
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
                        args.initial_active_fraction,
                        args.initial_active_strategy,
                        args.active_batch_fraction,
                        args.gurobi_time_limit_sec,
                        args.wall_time_limit_sec,
                    )
                    rows.append(row)
                    _write_rows(rows, args.output)
                    _write_pair_summary(rows, summary_output)
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
    _write_pair_summary(rows, summary_output)
    logging.info("Wrote %s", args.output)
    logging.info("Wrote %s", summary_output)


if __name__ == "__main__":
    main()
