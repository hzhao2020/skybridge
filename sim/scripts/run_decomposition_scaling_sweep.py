#!/usr/bin/env python3
"""Sweep SAA sample count and query count for decomposition ablation."""

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

from src.config import RESULTS_DIR, load_solver_config  # noqa: E402
from src.data_loader import load_endpoints, load_queries, load_scenarios  # noqa: E402
from src.milp_decomposition import solve_decomposition  # noqa: E402
from src.milp_full import solve_full_milp  # noqa: E402
from src.workflow import get_workflow  # noqa: E402


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _subset_scenarios(scenarios, query_ids: list[str], sample_count: int):
    by_query: dict[str, list] = {}
    for scenario in scenarios:
        by_query.setdefault(scenario.query_id, []).append(scenario)

    selected = []
    for query_id in query_ids:
        group = sorted(by_query.get(query_id, []), key=lambda s: s.scenario_id)
        selected.extend(group[:sample_count])
    return selected


def _assignment_signature(result) -> str:
    pairs = sorted((a.logical_node, a.endpoint_id) for a in result.assignments)
    return json.dumps(pairs, sort_keys=True)


def _worker(
    method: str,
    workflow_name: str,
    quality: str,
    query_count: int,
    sample_count: int,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    try:
        workflow = get_workflow(workflow_name)
        config = load_solver_config()
        endpoints = load_endpoints()
        queries = load_queries(quality_level=quality, workflow=workflow_name)[:query_count]
        query_ids = [q.query_id for q in queries]
        scenarios = _subset_scenarios(load_scenarios(query_ids=query_ids), query_ids, sample_count)

        if method == "full_milp":
            result = solve_full_milp(workflow, endpoints, queries, scenarios, quality, config)
        elif method == "decomposition":
            result = solve_decomposition(workflow, endpoints, queries, scenarios, quality, config)
        else:
            raise ValueError(method)

        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "query_count": query_count,
                "sample_count": sample_count,
                "scenario_count": len(scenarios),
                "method": method,
                "status": result.status,
                "objective_value": result.objective_value,
                "solver_runtime_sec": result.solver_runtime_sec,
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": 0.0,
                "num_iterations": result.num_iterations,
                "active_scenario_count": result.active_scenario_count,
                "active_scenario_fraction": (
                    result.active_scenario_count / len(scenarios) if scenarios else 0.0
                ),
                "assignment_signature": _assignment_signature(result),
                "expected_cost": result.expected_cost,
                "avg_latency": result.avg_latency,
                "p95_latency": result.p95_latency,
                "p99_latency": result.p99_latency,
                "violation_rate": result.violation_rate,
                "error": "",
            }
        )
    except Exception as exc:
        queue.put(
            {
                "workflow": workflow_name,
                "quality_level": quality,
                "query_count": query_count,
                "sample_count": sample_count,
                "scenario_count": query_count * sample_count,
                "method": method,
                "status": "FAILED",
                "objective_value": float("nan"),
                "solver_runtime_sec": float("nan"),
                "wall_time_sec": time.perf_counter() - started,
                "max_rss_mb": 0.0,
                "num_iterations": 0,
                "active_scenario_count": 0,
                "active_scenario_fraction": float("nan"),
                "assignment_signature": "",
                "expected_cost": float("nan"),
                "avg_latency": float("nan"),
                "p95_latency": float("nan"),
                "p99_latency": float("nan"),
                "violation_rate": float("nan"),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def _run_one(method: str, workflow: str, quality: str, query_count: int, sample_count: int) -> dict:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker,
        args=(method, workflow, quality, query_count, sample_count, queue),
    )
    proc.start()
    peak_rss_mb = 0.0
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
        time.sleep(0.05)
    proc.join()
    if queue.empty():
        row = {
            "workflow": workflow,
            "quality_level": quality,
            "query_count": query_count,
            "sample_count": sample_count,
            "scenario_count": query_count * sample_count,
            "method": method,
            "status": f"FAILED_EXIT_{proc.exitcode}",
            "error": "worker exited before returning a result",
        }
    else:
        row = queue.get()
    row["max_rss_mb"] = max(float(row.get("max_rss_mb", 0.0) or 0.0), peak_rss_mb)
    row["process_exitcode"] = proc.exitcode
    return row


def _add_pairwise_comparison(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["workflow", "quality_level", "query_count", "sample_count"]
    full = df[df["method"] == "full_milp"][key_cols + [
        "assignment_signature",
        "objective_value",
        "expected_cost",
        "avg_latency",
        "p95_latency",
        "status",
    ]].rename(
        columns={
            "assignment_signature": "full_assignment_signature",
            "objective_value": "full_objective_value",
            "expected_cost": "full_expected_cost",
            "avg_latency": "full_avg_latency",
            "p95_latency": "full_p95_latency",
            "status": "full_status",
        }
    )
    decomp = df[df["method"] == "decomposition"][key_cols + [
        "assignment_signature",
        "objective_value",
        "expected_cost",
        "avg_latency",
        "p95_latency",
        "status",
    ]].rename(
        columns={
            "assignment_signature": "decomp_assignment_signature",
            "objective_value": "decomp_objective_value",
            "expected_cost": "decomp_expected_cost",
            "avg_latency": "decomp_avg_latency",
            "p95_latency": "decomp_p95_latency",
            "status": "decomp_status",
        }
    )
    cmp_df = full.merge(decomp, on=key_cols, how="outer")
    cmp_df["same_assignment"] = (
        cmp_df["full_assignment_signature"] == cmp_df["decomp_assignment_signature"]
    )
    cmp_df["objective_abs_diff"] = (
        cmp_df["full_objective_value"] - cmp_df["decomp_objective_value"]
    ).abs()
    cmp_df["expected_cost_abs_diff"] = (
        cmp_df["full_expected_cost"] - cmp_df["decomp_expected_cost"]
    ).abs()
    cmp_df["avg_latency_abs_diff"] = (
        cmp_df["full_avg_latency"] - cmp_df["decomp_avg_latency"]
    ).abs()
    cmp_df["p95_latency_abs_diff"] = (
        cmp_df["full_p95_latency"] - cmp_df["decomp_p95_latency"]
    ).abs()
    return cmp_df


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, workflow in zip(axes, sorted(df["workflow"].unique())):
        sub = df[(df["workflow"] == workflow) & (df["status"].isin(["OPTIMAL", "optimal"]))]
        for method, linestyle in (("full_milp", "--"), ("decomposition", "-")):
            method_df = sub[sub["method"] == method]
            for sample_count in sorted(method_df["sample_count"].unique()):
                series = method_df[method_df["sample_count"] == sample_count].sort_values("query_count")
                if series.empty:
                    continue
                ax.plot(
                    series["query_count"],
                    series[metric],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{method}, S={sample_count}",
                )
        ax.set_title(workflow)
        ax.set_xlabel("query count")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_plots(csv_path: Path, out_dir: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "decomposition_scaling_wall_time.png",
        out_dir / "decomposition_scaling_solver_time.png",
        out_dir / "decomposition_scaling_memory.png",
        out_dir / "decomposition_scaling_active_fraction.png",
    ]
    _plot_metric(df, "wall_time_sec", "wall time (sec)", outputs[0])
    _plot_metric(df, "solver_runtime_sec", "solver runtime (sec)", outputs[1])
    _plot_metric(df, "max_rss_mb", "max RSS (MB)", outputs[2])
    _plot_metric(df, "active_scenario_fraction", "active scenario fraction", outputs[3])
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default="workflow1,workflow2")
    parser.add_argument("--quality", default="Q2", choices=["Q1", "Q2", "Q3"])
    parser.add_argument("--query-counts", default="10,20,30,40,50")
    parser.add_argument("--sample-counts", default="2,4,6,8,10")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "decomposition_scaling_sweep.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=RESULTS_DIR / "experiment_logs" / "figures",
    )
    args = parser.parse_args()

    workflows = [part.strip() for part in args.workflows.split(",") if part.strip()]
    query_counts = _parse_ints(args.query_counts)
    sample_counts = _parse_ints(args.sample_counts)

    rows: list[dict] = []
    for workflow in workflows:
        for query_count in query_counts:
            for sample_count in sample_counts:
                for method in ("full_milp", "decomposition"):
                    logging.info(
                        "Running %s workflow=%s quality=%s Q=%d S=%d",
                        method,
                        workflow,
                        args.quality,
                        query_count,
                        sample_count,
                    )
                    row = _run_one(method, workflow, args.quality, query_count, sample_count)
                    logging.info(
                        "%s %s Q=%d S=%d status=%s wall=%.2fs rss=%.1fMB active=%s",
                        method,
                        workflow,
                        query_count,
                        sample_count,
                        row.get("status"),
                        float(row.get("wall_time_sec", 0.0) or 0.0),
                        float(row.get("max_rss_mb", 0.0) or 0.0),
                        row.get("active_scenario_count"),
                    )
                    rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cmp_df = _add_pairwise_comparison(df)
    df.to_csv(args.output, index=False)
    cmp_df.to_csv(args.output.with_name(args.output.stem + "_solution_comparison.csv"), index=False)
    with open(args.output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    outputs = make_plots(args.output, args.plot_dir)
    logging.info("Wrote %s", args.output)
    for output in outputs:
        logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
