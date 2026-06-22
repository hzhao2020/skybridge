#!/usr/bin/env python3
"""Plot Q1 full MILP vs SkyFlow runtime/memory scaling as a single-column line figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR  # noqa: E402


DEFAULT_CSV = RESULTS_DIR / "experiment_logs" / "q1_full_vs_decomp_eta010_highlimit_minp95.csv"


def _label_method(method: str) -> str:
    if method == "full_milp":
        return "Full MILP"
    if method in {"decomposed_milp", "decomposition"}:
        return "SkyFlow"
    return method


def _label_workflow(workflow: str, quality: str) -> str:
    return f"{workflow.replace('workflow', 'W')}-{quality}"


def _load_source(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"].isin(["OPTIMAL", "optimal"])].copy()
    df = df[df["quality_level"] == "Q1"].copy()
    df = df[df["query_count"].isin([200, 400, 600, 800, 1000])].copy()
    df["method_label"] = df["method"].map(_label_method)
    df["workflow_label"] = [
        _label_workflow(workflow, quality)
        for workflow, quality in zip(df["workflow"], df["quality_level"])
    ]
    df["peak_rss_gb"] = df["max_rss_mb"] / 1024.0
    return df


def _assert_complete(df: pd.DataFrame) -> None:
    expected = {
        (workflow, query_count, method)
        for workflow in ("workflow1", "workflow2")
        for query_count in (200, 400, 600, 800, 1000)
        for method in ("full_milp", "decomposed_milp")
    }
    actual = {
        (str(row.workflow), int(row.query_count), str(row.method))
        for row in df.itertuples(index=False)
    }
    missing = sorted(expected - actual)
    if missing:
        raise SystemExit(f"Missing OPTIMAL rows for: {missing}")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_panel(ax: plt.Axes, sub: pd.DataFrame, metric: str, workflow_label: str) -> None:
    styles = {
        "SkyFlow": {
            "color": "#3b6fb6",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.4,
        },
        "Full MILP": {
            "color": "#8f8f8f",
            "marker": "s",
            "linestyle": "--",
            "linewidth": 2.4,
        },
    }
    for label in ("SkyFlow", "Full MILP"):
        rows = sub[sub["method_label"] == label].sort_values("query_count")
        ax.plot(
            rows["query_count"],
            rows[metric],
            label=label,
            markersize=6.5,
            markeredgecolor="#1a1a1a",
            markeredgewidth=0.6,
            **styles[label],
        )
    ax.text(
        0.02,
        0.92,
        workflow_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    ax.set_xticks([200, 400, 600, 800, 1000])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_line_singlecol(df: pd.DataFrame, out_prefix: Path) -> None:
    _assert_complete(df)
    _style()
    workflows = ["W1-Q1", "W2-Q1"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex="col")

    for row, workflow_label in enumerate(workflows):
        sub = df[df["workflow_label"] == workflow_label]
        _plot_panel(axes[row, 0], sub, "wall_time_sec", workflow_label)
        _plot_panel(axes[row, 1], sub, "peak_rss_gb", workflow_label)

    axes[0, 0].set_title("Runtime", fontweight="bold", pad=8)
    axes[0, 1].set_title("Memory", fontweight="bold", pad=8)
    axes[0, 0].set_ylabel("Wall time (s)", fontweight="bold")
    axes[1, 0].set_ylabel("Wall time (s)", fontweight="bold")
    axes[0, 1].set_ylabel("Peak RSS (GB)", fontweight="bold")
    axes[1, 1].set_ylabel("Peak RSS (GB)", fontweight="bold")
    axes[1, 0].set_xlabel("Query count Q", fontweight="bold")
    axes[1, 1].set_xlabel("Query count Q", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.02),
        handlelength=2.0,
        columnspacing=1.8,
        prop={"weight": "bold", "size": 14},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6, w_pad=1.6)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "fig" / "q_scaling_eta010_q1_highlimit",
    )
    parser.add_argument(
        "--prefix",
        default="q1_Q200_1000_runtime_memory_w1_w2_line_singlecol",
    )
    args = parser.parse_args()

    source = _load_source(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_out = args.out_dir / "q1_Q200_1000_runtime_memory_source.csv"
    source.to_csv(source_out, index=False)
    out_prefix = args.out_dir / args.prefix
    plot_line_singlecol(source, out_prefix)
    print(source_out)
    print(out_prefix.with_suffix(".pdf"))
    print(out_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
