#!/usr/bin/env python3
"""Run one SkyFlow experiment: a single (workflow, quality) pair in isolation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_default_config, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.evaluator import evaluate_deployment  # noqa: E402
from src.exporter import export_results  # noqa: E402
from src.experiment_protocol import split_scenarios_by_query  # noqa: E402
from src.baselines import CANONICAL_BASELINE_METHODS, solve_baseline  # noqa: E402
from src.milp_decomposition import (  # noqa: E402
    solve_decomposition,
    solve_decomposition_with_initializer_selection,
)
from src.milp_full import solve_full_milp  # noqa: E402
from src.pricing import query_generation_params  # noqa: E402
from src.workflow import get_workflow  # noqa: E402

try:
    from src.plotting import generate_plots  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    _PLOTTING_IMPORT_ERROR = exc
    generate_plots = None
else:
    _PLOTTING_IMPORT_ERROR = None


def _expected_queries_per_run(cfg: dict) -> int:
    per_workflow = cfg.get("num_queries_per_workflow_quality")
    if per_workflow is not None:
        return int(per_workflow)
    qcfg = query_generation_params()
    n_total = int(cfg.get("num_queries_per_quality", qcfg["requests_per_quality_level"]))
    ratio = qcfg.get("workflow1_workflow2_ratio", [1, 1])
    r_sum = sum(int(x) for x in ratio)
    return n_total * int(ratio[0]) // r_sum


def _results_dir_for_run(
    workflow: str,
    quality: str,
    method: str,
    override: str | None,
) -> Path:
    if override:
        return Path(override)
    return RESULTS_DIR / f"{workflow}_{quality}" / method


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one experiment for a single workflow and quality level. "
            "Loads only queries tagged with that workflow (50 per quality when using the default 100/quality split)."
        )
    )
    parser.add_argument("--workflow", required=True, choices=["workflow1", "workflow2"])
    parser.add_argument("--quality", required=True, choices=["Q1", "Q2", "Q3"])
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "full_milp",
            "decomposition",
            *CANONICAL_BASELINE_METHODS,
        ],
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Output directory (default: results/<workflow>_<quality>/)",
    )
    parser.add_argument(
        "--heldout-eval",
        action="store_true",
        help="Use the first half of scenarios for selection and report metrics on the held-out half",
    )
    parser.add_argument(
        "--initial-active-fraction",
        type=float,
        default=None,
        help="Override decomposition initial active scenario fraction",
    )
    parser.add_argument(
        "--initial-active-strategy",
        choices=[
            "qbr",
            "qbw",
            "qbb",
            "qbq",
            "qbt",
            "qbu",
            "qbm",
        ],
        default=None,
        help="Override decomposition initial active scenario selection strategy",
    )
    parser.add_argument(
        "--active-batch-fraction",
        type=float,
        default=None,
        help="Override decomposition active scenario batch fraction",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override the SLO violation/CVaR risk threshold",
    )
    parser.add_argument(
        "--disable-initializer-selection",
        action="store_true",
        help="Run decomposition directly with the configured initializer",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_default_config()
    workflow = get_workflow(args.workflow)
    solver_overrides = {}
    if args.initial_active_fraction is not None:
        solver_overrides["initial_active_fraction"] = args.initial_active_fraction
    if args.initial_active_strategy is not None:
        solver_overrides["initial_active_strategy"] = args.initial_active_strategy
    if args.active_batch_fraction is not None:
        solver_overrides["active_batch_fraction"] = args.active_batch_fraction
    if args.eta is not None:
        solver_overrides["eta"] = args.eta
    config = load_solver_config(solver_overrides)
    endpoints = load_endpoints()
    queries = load_queries(quality_level=args.quality, workflow=args.workflow)
    expected_n = _expected_queries_per_run(cfg)
    if len(queries) < expected_n:
        raise SystemExit(
            f"Expected {expected_n} queries for {args.workflow} + {args.quality}, "
            f"found {len(queries)}. Regenerate data: python scripts/generate_synthetic_data.py"
        )
    queries = queries[:expected_n]

    query_ids = [q.query_id for q in queries]
    scenarios = load_scenarios(query_ids=query_ids)
    train_scenarios = scenarios
    eval_scenarios = scenarios
    if args.heldout_eval:
        train_scenarios, eval_scenarios = split_scenarios_by_query(
            scenarios,
            calibration_count=int(cfg.get("num_scenarios_per_query", 10)),
        )
    logging.info(
        "Experiment %s %s: %d queries, %d scenarios, method=%s",
        args.workflow,
        args.quality,
        len(queries),
        len(scenarios),
        args.method,
    )
    if args.heldout_eval:
        logging.info(
            "Held-out protocol: %d calibration scenarios, %d test scenarios",
            len(train_scenarios),
            len(eval_scenarios),
        )

    if args.method == "full_milp":
        result = solve_full_milp(
            workflow, endpoints, queries, train_scenarios, args.quality, config
        )
    elif args.method == "decomposition":
        has_initializer_override = (
            args.initial_active_fraction is not None
            or args.initial_active_strategy is not None
            or args.active_batch_fraction is not None
        )
        if args.disable_initializer_selection or has_initializer_override:
            result = solve_decomposition(
                workflow, endpoints, queries, train_scenarios, args.quality, config
            )
        else:
            result = solve_decomposition_with_initializer_selection(
                workflow, endpoints, queries, train_scenarios, args.quality, config
            )
    else:
        result = solve_baseline(
            args.method,
            workflow,
            endpoints,
            queries,
            train_scenarios,
            args.quality,
            config,
        )

    if args.heldout_eval:
        assignment = {
            a.logical_node: next(ep for ep in endpoints if ep.endpoint_id == a.endpoint_id)
            for a in result.assignments
        }
        metrics = evaluate_deployment(
            workflow=workflow,
            assignment=assignment,
            endpoints=endpoints,
            queries=queries,
            scenarios=eval_scenarios,
            quality_level=args.quality,
            config=config,
        )
        result.expected_cost = metrics["expected_cost"]
        result.avg_latency = metrics["avg_latency"]
        result.p95_latency = metrics["p95_latency"]
        result.p99_latency = metrics["p99_latency"]
        result.violation_rate = metrics["violation_rate"]
        result.cvar_value = metrics["cvar_value"]
        result.per_query_scenario_metrics = metrics["per_qs"]

    run_dir = _results_dir_for_run(
        args.workflow, args.quality, args.method, args.results_dir
    )
    export_results(result, run_dir, metrics_dir=RESULTS_DIR)
    if generate_plots is not None:
        generate_plots(result, run_dir)
    else:
        logging.warning("Skipping plots because plotting dependencies are unavailable: %s", _PLOTTING_IMPORT_ERROR)
    logging.info("Simulation complete: %s %s -> %s", args.workflow, args.quality, run_dir)


if __name__ == "__main__":
    main()
