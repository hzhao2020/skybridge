#!/usr/bin/env python3
"""Run the main SkyFlow-vs-baselines experiment with progress logging."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import CANONICAL_BASELINE_METHODS, solve_baseline  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR, load_default_config, load_solver_config  # noqa: E402
from src.cost_latency import critical_path_latency  # noqa: E402
from src.data_loader import load_endpoints, load_network_links, load_queries, load_scenarios  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.workflow import get_workflow  # noqa: E402

WORKFLOWS = ("workflow1", "workflow2")
QUALITIES = ("Q1", "Q2", "Q3")
METHODS = ("decomposition", "single_cloud", "greedy", "dpgm", "mtgp")


class Progress:
    def __init__(self, total: int, done: int = 0) -> None:
        self.total = total
        self.done = done
        self.t0 = time.perf_counter()

    def step(self, label: str) -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.t0
        pct = 100.0 * self.done / max(self.total, 1)
        logging.info("[%02d/%02d | %5.1f%% | %8.1fs] %s", self.done, self.total, pct, elapsed, label)


def run_command(
    cmd: list[str],
    label: str,
    progress: Progress,
    *,
    continue_on_error: bool = False,
) -> tuple[bool, str | None]:
    logging.info("START %s: %s", label, " ".join(cmd))
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - t0
        message = f"{label} failed with exit {exc.returncode} in {elapsed:.1f}s"
        if not continue_on_error:
            raise
        logging.exception(message)
        progress.step(message)
        return False, message
    progress.step(f"{label} finished in {time.perf_counter() - t0:.1f}s")
    return True, None


def calibrate_budgets(output: Path, progress: Progress, factor: float, query_count: int) -> None:
    endpoints = load_endpoints()
    endpoint_by_id = {ep.endpoint_id: ep for ep in endpoints}
    network_index = {(l.src_endpoint_id, l.dst_endpoint_id): l for l in load_network_links()}
    config = load_solver_config()
    qdf = pd.read_csv(DATA_DIR / "queries.csv")

    backup = RESULTS_DIR / "experiment_logs" / "queries_before_budget_calibration.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        qdf.to_csv(backup, index=False)

    methods = CANONICAL_BASELINE_METHODS
    records: list[dict] = []
    new_sla: dict[str, float] = {}

    for workflow_name in WORKFLOWS:
        workflow = get_workflow(workflow_name)
        for quality in QUALITIES:
            queries = load_queries(quality_level=quality, workflow=workflow_name)[:query_count]
            scenarios = load_scenarios(query_ids=[q.query_id for q in queries])
            calibration_scenarios, _ = split_scenarios_by_query(
                scenarios,
                calibration_count=int(load_default_config().get("num_scenarios_per_query", 50)),
            )
            scenario_by_q: dict[str, list] = {}
            for scenario in calibration_scenarios:
                scenario_by_q.setdefault(scenario.query_id, []).append(scenario)

            assignments = {}
            for method in methods:
                label = f"calibrate {workflow_name}/{quality}/{method}"
                logging.info("START %s", label)
                t0 = time.perf_counter()
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
                progress.step(f"{label} finished in {time.perf_counter() - t0:.1f}s")

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
                budget = factor * method_p95[budget_baseline]
                new_sla[query.query_id] = budget
                records.append(
                    {
                        "workflow": workflow_name,
                        "quality_level": quality,
                        "query_id": query.query_id,
                        "budget_factor": factor,
                        "budget_baseline": budget_baseline,
                        "baseline_p95_min": method_p95[budget_baseline],
                        "sla_sec_old": query.sla_sec,
                        "sla_sec_new": budget,
                        **{f"p95_{m}": v for m, v in method_p95.items()},
                    }
                )
            progress.step(f"calibrate {workflow_name}/{quality}/budgets updated")

    qdf["sla_sec"] = qdf["query_id"].map(new_sla).fillna(qdf["sla_sec"])
    qdf.to_csv(DATA_DIR / "queries.csv", index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)


def run_experiments(
    results_root: Path,
    progress: Progress,
    *,
    start_at: str | None = None,
    continue_on_error: bool = False,
    failures_output: Path | None = None,
) -> None:
    script = ROOT / "scripts" / "run_simulation.py"
    started = start_at is None
    failures: list[dict[str, str]] = []
    for workflow in WORKFLOWS:
        for quality in QUALITIES:
            for method in METHODS:
                key = f"{workflow}/{quality}/{method}"
                if not started:
                    if key == start_at:
                        started = True
                    else:
                        continue
                out = results_root / f"{workflow}_{quality}" / method
                label = f"run {key}"
                cmd = [
                    sys.executable,
                    str(script),
                    "--workflow",
                    workflow,
                    "--quality",
                    quality,
                    "--method",
                    method,
                    "--results-dir",
                    str(out),
                    "--heldout-eval",
                ]
                ok, message = run_command(
                    cmd,
                    label,
                    progress,
                    continue_on_error=continue_on_error,
                )
                if not ok:
                    failures.append(
                        {
                            "workflow": workflow,
                            "quality": quality,
                            "method": method,
                            "message": message or "",
                        }
                    )
                    if failures_output is not None:
                        failures_output.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(failures).to_csv(failures_output, index=False)

    if start_at is not None and not started:
        raise ValueError(f"--start-at did not match any experiment key: {start_at}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR / "main_Q1000_S50_dbfixed_minp95_full")
    parser.add_argument("--budget-factor", type=float, default=1.0)
    parser.add_argument("--query-count", type=int, default=1000)
    parser.add_argument(
        "--budget-output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "baseline_calibrated_query_budgets_Q1000_S50_dbfixed_minp95.csv",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-budget-calibration", action="store_true")
    parser.add_argument(
        "--start-at",
        help="Resume from an experiment key like workflow1/Q1/single_cloud.",
    )
    parser.add_argument("--initial-done", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "main_Q1000_S50_dbfixed_minp95_full_failures.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    total_steps = 2 + len(WORKFLOWS) * len(QUALITIES) * (len(CANONICAL_BASELINE_METHODS) + 1 + len(METHODS))
    progress = Progress(total_steps, done=args.initial_done)

    if not args.skip_setup:
        run_command([sys.executable, str(ROOT / "scripts" / "generate_synthetic_data.py")], "generate synthetic data", progress)
        run_command([sys.executable, str(ROOT / "scripts" / "populate_from_measurements.py")], "populate measurements", progress)
    if not args.skip_budget_calibration:
        calibrate_budgets(args.budget_output, progress, args.budget_factor, args.query_count)
    run_experiments(
        args.results_root,
        progress,
        start_at=args.start_at,
        continue_on_error=args.continue_on_error,
        failures_output=args.failures_output,
    )

    logging.info("DONE main experiment. Results root: %s", args.results_root)


if __name__ == "__main__":
    main()
