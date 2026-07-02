#!/usr/bin/env python3
"""Plot Q1 full MILP vs SkyFlow runtime/memory scaling as a single-column line figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
        for workflow in ("workflow1", "workflow2", "workflow3", "workflow4")
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "font.size": 11.0,
            "axes.titlesize": 12.6,
            "axes.labelsize": 12.0,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 11.2,
        }
    )


def _plot_panel(ax: plt.Axes, sub: pd.DataFrame, metric: str, workflow_label: str) -> None:
    styles = {
        "SkyFlow": {
            "color": "#3b6fb6",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 1.35,
        },
        "Full MILP": {
            "color": "#8f8f8f",
            "marker": "s",
            "linestyle": "--",
            "linewidth": 1.35,
        },
    }
    for label in ("SkyFlow", "Full MILP"):
        rows = sub[sub["method_label"] == label].sort_values("query_count")
        ax.plot(
            rows["query_count"],
            rows[metric],
            label=label,
            markersize=3.9,
            markeredgecolor="#1a1a1a",
            markeredgewidth=0.45,
            **styles[label],
        )
    ax.text(
        0.02,
        0.92,
        workflow_label,
        transform=ax.transAxes,
        fontsize=12.6,
        fontweight="semibold",
        va="top",
    )
    ax.set_xticks([200, 400, 600, 800, 1000])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.35, alpha=0.75)
    ax.tick_params(axis="x", pad=1.0, length=2.2, width=0.55)
    ax.tick_params(axis="y", pad=1.0, length=2.2, width=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(10.4)


def plot_line_singlecol(df: pd.DataFrame, out_prefix: Path) -> None:
    _assert_complete(df)
    _style()
    workflows = [f"W{workflow}-Q1" for workflow in range(1, 5)]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.55), sharex="col")

    for row, workflow_label in enumerate(workflows):
        sub = df[df["workflow_label"] == workflow_label]
        _plot_panel(axes[row, 0], sub, "wall_time_sec", workflow_label)
        _plot_panel(axes[row, 1], sub, "peak_rss_gb", workflow_label)

    axes[0, 0].set_title("Runtime", fontweight="semibold", pad=3)
    axes[0, 1].set_title("Memory", fontweight="semibold", pad=3)
    axes[0, 0].set_ylabel("Wall time (s)", fontweight="semibold", labelpad=1.2)
    axes[1, 0].set_ylabel("Wall time (s)", fontweight="semibold", labelpad=1.2)
    axes[0, 1].set_ylabel("Peak RSS (GB)", fontweight="semibold", labelpad=1.2)
    axes[1, 1].set_ylabel("Peak RSS (GB)", fontweight="semibold", labelpad=1.2)
    axes[1, 0].set_xlabel("Query count Q", fontweight="semibold", labelpad=1.3)
    axes[1, 1].set_xlabel("Query count Q", fontweight="semibold", labelpad=1.3)
    axes[0, 0].set_ylim(0, 1000)
    axes[0, 1].set_ylim(0, 10)
    axes[1, 0].set_ylim(0, 6000)
    axes[1, 1].set_ylim(0, 30)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.985),
        handlelength=1.5,
        handletextpad=0.32,
        columnspacing=1.0,
        borderaxespad=0.0,
        prop={"weight": "normal", "size": 11.8},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.55, w_pad=0.7)
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
