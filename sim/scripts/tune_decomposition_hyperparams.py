#!/usr/bin/env python3
"""Tune decomposition initialization hyperparameters without touching final test scenarios."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.evaluator import evaluate_deployment  # noqa: E402
from src.milp_decomposition import solve_decomposition  # noqa: E402
from src.workflow import get_workflow  # noqa: E402

WORKFLOWS = ("workflow1", "workflow2")
QUALITIES = ("Q1", "Q2", "Q3")


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _candidate_grid(
    *,
    strategies: list[str] | None = None,
    initial_fractions: list[float] | None = None,
    batch_fractions: list[float] | None = None,
) -> list[dict]:
    if strategies is not None and initial_fractions is not None and batch_fractions is not None:
        return [
            {
                "initial_active_strategy": strategy,
                "initial_active_fraction": fraction,
                "active_batch_fraction": batch,
            }
            for strategy in strategies
            for fraction in initial_fractions
            for batch in batch_fractions
        ]

    configs: list[dict] = []
    for fraction in (0.10, 0.15, 0.20, 0.25):
        for batch in (0.025, 0.05, 0.10):
            configs.append(
                {
                    "initial_active_strategy": "stratified_random",
                    "initial_active_fraction": fraction,
                    "active_batch_fraction": batch,
                }
            )
    for fraction in (0.10, 0.15, 0.20):
        for batch in (0.025, 0.05):
            configs.append(
                {
                    "initial_active_strategy": "stratified_quantile",
                    "initial_active_fraction": fraction,
                    "active_batch_fraction": batch,
                }
            )
    return configs


def _config_id(config: dict) -> str:
    return (
        f"{config['initial_active_strategy']}"
        f"_init{config['initial_active_fraction']:.3f}"
        f"_batch{config['active_batch_fraction']:.3f}"
    )


def _split_tuning_scenarios(scenarios, *, calibration_count: int, train_count: int, val_count: int):
    by_query = defaultdict(list)
    for scenario in scenarios:
        by_query[scenario.query_id].append(scenario)

    tuning_train = []
    tuning_val = []
    for query_id in sorted(by_query):
        group = sorted(by_query[query_id], key=lambda s: s.scenario_id)
        calibration = group[:calibration_count]
        tuning_train.extend(calibration[:train_count])
        tuning_val.extend(calibration[train_count : train_count + val_count])
    return tuning_train, tuning_val


class MemorySampler:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = self.process.memory_info().rss
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            time.sleep(self.interval_sec)

    @property
    def peak_mb(self) -> float:
        return self.peak_rss_bytes / (1024.0 * 1024.0)


def _worker(args: argparse.Namespace) -> None:
    workflow = get_workflow(args.workflow)
    endpoints = load_endpoints()
    queries = load_queries(quality_level=args.quality, workflow=args.workflow)[: args.query_count]
    scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
    tuning_train, tuning_val = _split_tuning_scenarios(
        scenarios,
        calibration_count=args.calibration_count,
        train_count=args.tuning_calibration_count,
        val_count=args.tuning_validation_count,
    )

    config = load_solver_config(
        {
            "initial_active_strategy": args.initial_active_strategy,
            "initial_active_fraction": args.initial_active_fraction,
            "active_batch_fraction": args.active_batch_fraction,
        }
    )

    row = {
        "workflow": args.workflow,
        "quality": args.quality,
        "query_count": args.query_count,
        "tuning_calibration_scenarios_per_query": args.tuning_calibration_count,
        "tuning_validation_scenarios_per_query": args.tuning_validation_count,
        "initial_active_strategy": args.initial_active_strategy,
        "initial_active_fraction": args.initial_active_fraction,
        "active_batch_fraction": args.active_batch_fraction,
        "status": "ERROR",
        "error": "",
        "expected_cost": math.nan,
        "avg_latency": math.nan,
        "p95_latency": math.nan,
        "violation_rate": math.nan,
        "solver_runtime_sec": math.nan,
        "wall_runtime_sec": math.nan,
        "peak_memory_mb": math.nan,
        "iteration_count": 0,
        "active_scenario_count": 0,
        "infeasible": 1,
    }

    t0 = time.perf_counter()
    try:
        with MemorySampler() as memory:
            result = solve_decomposition(
                workflow,
                endpoints,
                queries,
                tuning_train,
                args.quality,
                config,
            )
            assignment = {
                a.logical_node: next(ep for ep in endpoints if ep.endpoint_id == a.endpoint_id)
                for a in result.assignments
            }
            metrics = evaluate_deployment(
                workflow=workflow,
                assignment=assignment,
                endpoints=endpoints,
                queries=queries,
                scenarios=tuning_val,
                quality_level=args.quality,
                config=config,
            )
        row.update(
            {
                "status": result.status,
                "expected_cost": metrics["expected_cost"],
                "avg_latency": metrics["avg_latency"],
                "p95_latency": metrics["p95_latency"],
                "violation_rate": metrics["violation_rate"],
                "solver_runtime_sec": result.solver_runtime_sec,
                "wall_runtime_sec": time.perf_counter() - t0,
                "peak_memory_mb": memory.peak_mb,
                "iteration_count": result.num_iterations,
                "active_scenario_count": result.active_scenario_count,
                "infeasible": int("INFEASIBLE" in result.status.upper()),
            }
        )
    except Exception as exc:  # noqa: BLE001 - worker must serialize failures
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_runtime_sec"] = time.perf_counter() - t0

    print(json.dumps(row, allow_nan=True), flush=True)


def _run_worker(python: str, args: argparse.Namespace, config: dict, workflow: str, quality: str) -> dict:
    cmd = [
        python,
        str(Path(__file__).resolve()),
        "--worker",
        "--workflow",
        workflow,
        "--quality",
        quality,
        "--query-count",
        str(args.query_count),
        "--calibration-count",
        str(args.calibration_count),
        "--tuning-calibration-count",
        str(args.tuning_calibration_count),
        "--tuning-validation-count",
        str(args.tuning_validation_count),
        "--initial-active-strategy",
        config["initial_active_strategy"],
        "--initial-active-fraction",
        str(config["initial_active_fraction"]),
        "--active-batch-fraction",
        str(config["active_batch_fraction"]),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {
        "workflow": workflow,
        "quality": quality,
        "initial_active_strategy": config["initial_active_strategy"],
        "initial_active_fraction": config["initial_active_fraction"],
        "active_batch_fraction": config["active_batch_fraction"],
        "status": "WORKER_FAILED",
        "error": proc.stdout[-4000:],
        "infeasible": 1,
    }


def _pareto_efficient(df: pd.DataFrame, objectives: list[str]) -> pd.Series:
    values = df[objectives].to_numpy(dtype=float)
    efficient = []
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if (other <= row).all() and (other < row).any():
                dominated = True
                break
        efficient.append(not dominated)
    return pd.Series(efficient, index=df.index)


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    results = results[
        results[
            [
                "expected_cost",
                "violation_rate",
                "avg_latency",
                "p95_latency",
                "solver_runtime_sec",
                "peak_memory_mb",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()
    group_cols = [
        "initial_active_strategy",
        "initial_active_fraction",
        "active_batch_fraction",
    ]
    agg = (
        results.groupby(group_cols, dropna=False)
        .agg(
            setting_count=("workflow", "count"),
            mean_cost=("expected_cost", "mean"),
            mean_svr=("violation_rate", "mean"),
            mean_latency=("avg_latency", "mean"),
            mean_p95_latency=("p95_latency", "mean"),
            total_solver_runtime_sec=("solver_runtime_sec", "sum"),
            total_wall_runtime_sec=("wall_runtime_sec", "sum"),
            max_peak_memory_mb=("peak_memory_mb", "max"),
            mean_iteration_count=("iteration_count", "mean"),
            max_iteration_count=("iteration_count", "max"),
            mean_active_scenario_count=("active_scenario_count", "mean"),
            infeasible_rate=("infeasible", "mean"),
        )
        .reset_index()
    )
    objectives = [
        "mean_cost",
        "mean_svr",
        "total_solver_runtime_sec",
        "max_peak_memory_mb",
    ]
    complete = agg["setting_count"] == len(WORKFLOWS) * len(QUALITIES)
    finite = agg[objectives].notna().all(axis=1)
    agg["pareto_efficient"] = False
    mask = complete & finite
    if mask.any():
        agg.loc[mask, "pareto_efficient"] = _pareto_efficient(agg.loc[mask], objectives)
    return agg.sort_values(
        ["pareto_efficient", "mean_svr", "mean_cost", "total_solver_runtime_sec"],
        ascending=[False, True, True, True],
    )


def _parent(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "tuning_runs.csv"
    aggregate_path = out_dir / "tuning_aggregate.csv"
    pareto_path = out_dir / "tuning_pareto.csv"

    configs = _candidate_grid(
        strategies=_parse_str_list(args.strategies),
        initial_fractions=_parse_float_list(args.initial_fractions),
        batch_fractions=_parse_float_list(args.batch_fractions),
    )
    existing = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            done.add(
                (
                    str(row["initial_active_strategy"]),
                    float(row["initial_active_fraction"]),
                    float(row["active_batch_fraction"]),
                    str(row["workflow"]),
                    str(row["quality"]),
                )
            )

    rows = existing.to_dict("records") if not existing.empty else []
    total = len(configs) * len(WORKFLOWS) * len(QUALITIES)
    completed = len(done)
    for config in configs:
        cid = _config_id(config)
        for workflow in WORKFLOWS:
            for quality in QUALITIES:
                key = (
                    config["initial_active_strategy"],
                    float(config["initial_active_fraction"]),
                    float(config["active_batch_fraction"]),
                    workflow,
                    quality,
                )
                if key in done and not args.force:
                    continue
                completed += 1
                label = f"[{completed}/{total}] {cid} {workflow}/{quality}"
                print(f"START {label}", flush=True)
                row = _run_worker(sys.executable, args, config, workflow, quality)
                row["config_id"] = cid
                rows.append(row)
                pd.DataFrame(rows).to_csv(detail_path, index=False)
                aggregate = _aggregate(pd.DataFrame(rows))
                aggregate.to_csv(aggregate_path, index=False)
                aggregate[aggregate["pareto_efficient"]].to_csv(pareto_path, index=False)
                print(
                    "DONE "
                    f"{label} status={row.get('status')} "
                    f"cost={row.get('expected_cost')} svr={row.get('violation_rate')} "
                    f"runtime={row.get('solver_runtime_sec')} mem={row.get('peak_memory_mb')}",
                    flush=True,
                )

    aggregate = _aggregate(pd.DataFrame(rows))
    aggregate.to_csv(aggregate_path, index=False)
    aggregate[aggregate["pareto_efficient"]].to_csv(pareto_path, index=False)
    print(f"detail={detail_path}")
    print(f"aggregate={aggregate_path}")
    print(f"pareto={pareto_path}")
    print(aggregate.head(20).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workflow", choices=WORKFLOWS)
    parser.add_argument("--quality", choices=QUALITIES)
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--calibration-count", type=int, default=50)
    parser.add_argument("--tuning-calibration-count", type=int, default=30)
    parser.add_argument("--tuning-validation-count", type=int, default=20)
    parser.add_argument("--initial-active-strategy", default="stratified_random")
    parser.add_argument("--initial-active-fraction", type=float, default=0.20)
    parser.add_argument("--active-batch-fraction", type=float, default=0.05)
    parser.add_argument("--strategies", default="stratified_random,stratified_quantile")
    parser.add_argument("--initial-fractions", default="0.10,0.15,0.20,0.25")
    parser.add_argument("--batch-fractions", default="0.025,0.05,0.10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "decomposition_hyperparam_tuning_Q200_cal30_val20",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if args.workflow is None or args.quality is None:
            raise SystemExit("--worker requires --workflow and --quality")
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
