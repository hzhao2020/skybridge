"""Matplotlib plotting for SkyFlow results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import RESULTS_DIR
from src.schemas import OptimizationResult

logger = logging.getLogger(__name__)


def generate_plots(
    result: OptimizationResult,
    output_dir: Path | None = None,
) -> None:
    out = output_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    latencies = [m["latency"] for m in result.per_query_scenario_metrics]
    costs = [m["cost"] for m in result.per_query_scenario_metrics]
    slas = [m["sla_sec"] for m in result.per_query_scenario_metrics]

    if latencies:
        _plot_latency_distribution(latencies, slas, out / "latency_distribution.png")
        _plot_cost_latency_summary(costs, latencies, out / "cost_latency_summary.png")

    if result.method == "decomposition" and result.convergence_history:
        _plot_convergence(result, out / "convergence.png")

    logger.info("Plots saved to %s", out)


def _plot_latency_distribution(
    latencies: list[float],
    slas: list[float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latencies, bins=30, alpha=0.7, label="Latency", color="steelblue")
    if slas:
        ax.axvline(np.mean(slas), color="red", linestyle="--", label="Mean SLA")
    ax.set_xlabel("Latency (sec)")
    ax.set_ylabel("Count")
    ax.set_title("End-to-End Latency Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost_latency_summary(costs: list[float], latencies: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(costs, latencies, alpha=0.5, s=15, c="steelblue")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Latency (sec)")
    ax.set_title("Cost vs Latency (all query-scenario pairs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_convergence(result: OptimizationResult, path: Path) -> None:
    hist = result.convergence_history
    iters = [h["iteration"] for h in hist]
    max_viol = [h["max_violation"] for h in hist]
    obj = [h["objective_value"] for h in hist]
    active = [h["active_scenario_count"] for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(iters, max_viol, marker="o", color="crimson")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Max violation")
    axes[0].set_title("Max CVaR violation")

    axes[1].plot(iters, obj, marker="o", color="steelblue")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Objective")
    axes[1].set_title("Objective value")

    axes[2].plot(iters, active, marker="o", color="green")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Active scenarios")
    axes[2].set_title("Active scenario count")

    fig.suptitle(f"Decomposition convergence ({result.workflow}, {result.quality_level})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
