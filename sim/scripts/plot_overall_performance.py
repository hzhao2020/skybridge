#!/usr/bin/env python3
"""Bar charts comparing methods across main-experiment settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, load_default_config  # noqa: E402

SETTINGS = [
    ("workflow1_Q1", "W1-Q1"),
    ("workflow1_Q2", "W1-Q2"),
    ("workflow1_Q3", "W1-Q3"),
    ("workflow2_Q1", "W2-Q1"),
    ("workflow2_Q2", "W2-Q2"),
    ("workflow2_Q3", "W2-Q3"),
]
METHODS = [
    ("decomposition", "SkyFlow"),
    ("greedy", "Greedy"),
    ("murakkab_profile", "MLS"),
    ("single_cloud", "SC"),
]
FIGSIZE = (3.35, 1.5)
COMBINED_FIGSIZE = (3.35, 1.82)
FONT_WEIGHT = "semibold"
LEGEND_PROP = {"size": 9, "weight": FONT_WEIGHT}
COMBINED_LEGEND_PROP = {"size": 5.8, "weight": "normal"}

PALETTE = {
    "SkyFlow": "#3b6fb6",
    "Greedy": "#64a85b",
    "MLS": "#d9893d",
    "SC": "#8a8a8a",
}
HATCHES = {
    "SkyFlow": "",
    "Greedy": "///",
    "MLS": "\\\\",
    "SC": "xx",
}


def _load_data(result_root: Path) -> pd.DataFrame:
    rows = []
    for setting_key, setting_label in SETTINGS:
        for method_key, method_label in METHODS:
            path = result_root / setting_key / method_key / "selected_plan.json"
            with path.open(encoding="utf-8") as f:
                selected = json.load(f)
            metrics = selected["metrics"]
            rows.append(
                {
                    "setting": setting_label,
                    "workflow": selected["workflow"],
                    "quality_level": selected["quality_level"],
                    "method": method_label,
                    "method_key": method_key,
                    "expected_cost": float(metrics["expected_cost"]),
                    "avg_latency": float(metrics["avg_latency"]),
                    "p95_latency": float(metrics["p95_latency"]),
                    "violation_rate": float(metrics["violation_rate"]),
                    "status": metrics.get("status", ""),
                }
            )
    return pd.DataFrame(rows)


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "font.monospace": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.weight": FONT_WEIGHT,
            "axes.labelweight": FONT_WEIGHT,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.35,
        }
    )


def _decorate_axis(ax, x: np.ndarray, setting_labels: list[str], ylabel: str) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels)
    ax.set_ylabel(ylabel)
    _apply_tick_weight(ax)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _apply_tick_weight(ax) -> None:
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight(FONT_WEIGHT)


def _plot_grouped(
    ax,
    df: pd.DataFrame,
    x: np.ndarray,
    offsets: np.ndarray,
    width: float,
    setting_labels: list[str],
    metric: str,
    ylabel: str,
    *,
    add_eta: bool = False,
    eta: float | None = None,
) -> None:
    for i, (_, method_label) in enumerate(METHODS):
        vals = [
            df[(df["setting"] == setting_label) & (df["method"] == method_label)][metric].iloc[0]
            for _, setting_label in SETTINGS
        ]
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            label=method_label,
            color=PALETTE[method_label],
            edgecolor="black",
            linewidth=0.35,
            hatch=HATCHES[method_label],
        )
    if add_eta:
        eta_value = 0.1 if eta is None else float(eta)
        ax.axhline(
            eta_value,
            color="#b22222",
            linewidth=0.8,
            linestyle="--",
            label=rf"$\eta={eta_value:g}$",
        )
    _decorate_axis(ax, x, setting_labels, ylabel)


def _plot_grouped_percent(
    ax,
    df: pd.DataFrame,
    x: np.ndarray,
    offsets: np.ndarray,
    width: float,
    setting_labels: list[str],
    metric: str,
    ylabel: str,
    *,
    add_eta: bool = False,
    eta: float | None = None,
) -> None:
    percent_df = df.copy()
    percent_df[metric] = percent_df[metric] * 100.0
    _plot_grouped(
        ax,
        percent_df,
        x,
        offsets,
        width,
        setting_labels,
        metric,
        ylabel,
        add_eta=add_eta,
        eta=None if eta is None else eta * 100.0,
    )


def _method_legend_handles() -> tuple[list, list[str]]:
    handles = [
        mpl.patches.Patch(
            facecolor=PALETTE[method_label],
            edgecolor="black",
            linewidth=0.35,
            hatch=HATCHES[method_label],
            label=method_label,
        )
        for _, method_label in METHODS
    ]
    return handles, [method_label for _, method_label in METHODS]


def _combined_legend(
    fig,
    *,
    eta: float | None = None,
    eta_label: str | None = None,
) -> None:
    handles, labels = _method_legend_handles()
    if eta is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#b22222",
                linewidth=0.8,
                linestyle="--",
            )
        )
        labels.append(eta_label or rf"$\eta={eta:g}$")
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.78,
        handlelength=1.55,
        handletextpad=0.28,
        borderaxespad=0.0,
        prop=COMBINED_LEGEND_PROP,
    )


def _apply_scientific_yaxis(ax) -> None:
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontweight(FONT_WEIGHT)
    ax.yaxis.get_offset_text().set_fontweight("normal")
    ax.yaxis.get_offset_text().set_fontsize(5.8)


def _style_combined_axis(
    ax,
    ylabel: str,
    setting_labels: list[str],
    *,
    rotate: bool = False,
) -> None:
    ax.set_ylabel(ylabel, fontsize=6.8, fontweight=FONT_WEIGHT, labelpad=1.3)
    ax.set_xticklabels(setting_labels, fontsize=5.9, rotation=28 if rotate else 0)
    ax.tick_params(axis="x", pad=1.0, length=2.2, width=0.55)
    ax.tick_params(axis="y", labelsize=5.9, pad=1.0, length=2.2, width=0.55)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.35, alpha=0.75)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("normal")


def _set_headroom_ylim(ax, values: pd.Series, *, minimum_top: float = 0.0) -> None:
    top = max(float(values.max()) * 1.18, minimum_top)
    ax.set_ylim(0, top)


def _add_cost_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = ["SkyFlow", "MLS", "Greedy", "SC"]
    ax.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        ncol=2,
        frameon=False,
        loc="upper left",
        columnspacing=0.9,
        handlelength=1.4,
        prop=LEGEND_PROP,
    )


def _add_svr_legend(ax, eta: float) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    eta_label = rf"$\eta={eta:g}$"
    blank = Line2D([], [], linestyle="", marker="", alpha=0.0)
    legend_handles = [
        by_label[eta_label],
        by_label["SkyFlow"],
        by_label["MLS"],
        blank,
        by_label["Greedy"],
        by_label["SC"],
    ]
    legend_labels = [eta_label, "SkyFlow", "MLS", "", "Greedy", "SC"]
    ax.legend(
        legend_handles,
        legend_labels,
        ncol=2,
        frameon=False,
        loc="upper right",
        columnspacing=0.9,
        handlelength=1.4,
        prop=LEGEND_PROP,
    )


def plot_overall_performance(result_root: Path, fig_dir: Path) -> pd.DataFrame:
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = _load_data(result_root)
    df.to_csv(fig_dir / "overall_performance_source.csv", index=False)
    eta = float(load_default_config().get("eta", 0.1))

    _configure_style()
    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
    setting_labels = [label for _, label in SETTINGS]
    combined_setting_labels = [label.replace("-", "\n") for _, label in SETTINGS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_grouped(ax, df, x, offsets, width, setting_labels, "expected_cost", "Expected cost ($)")
    _set_headroom_ylim(ax, df["expected_cost"], minimum_top=1.0)
    _add_cost_legend(ax)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_cost.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_grouped(ax, df, x, offsets, width, setting_labels, "avg_latency", "Mean latency (s)")
    _set_headroom_ylim(ax, df["avg_latency"], minimum_top=1.0)
    _add_cost_legend(ax)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_mean_latency.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_mean_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_grouped(ax, df, x, offsets, width, setting_labels, "p95_latency", "P95 latency (s)")
    _set_headroom_ylim(ax, df["p95_latency"], minimum_top=1.0)
    _add_cost_legend(ax)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_p95_latency.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_p95_latency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_grouped(
        ax,
        df,
        x,
        offsets,
        width,
        setting_labels,
        "violation_rate",
        "SLO violation rate",
        add_eta=True,
        eta=eta,
    )
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.25, 0.5])
    _apply_tick_weight(ax)
    _add_svr_legend(ax, eta)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_svr.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_svr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    _plot_grouped(
        axes[0],
        df,
        x,
        offsets,
        width,
        combined_setting_labels,
        "expected_cost",
        "Cost ($)",
    )
    axes[0].set_ylim(0, 30)
    _plot_grouped_percent(
        axes[1],
        df,
        x,
        offsets,
        width,
        combined_setting_labels,
        "violation_rate",
        "SVR (%)",
        add_eta=True,
        eta=eta,
    )
    axes[1].set_ylim(0, 10)
    axes[1].set_yticks([0, 5, 10])
    _style_combined_axis(axes[0], "Cost ($)", combined_setting_labels)
    _style_combined_axis(axes[1], "SVR (%)", combined_setting_labels)
    _combined_legend(fig, eta=eta, eta_label=rf"$\eta={eta * 100:g}\%$")
    fig.tight_layout(pad=0.15, w_pad=0.28, rect=(0, 0, 1, 0.90))
    fig.savefig(fig_dir / "overall_cost_svr_combined.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_cost_svr_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=COMBINED_FIGSIZE)
    _plot_grouped(
        axes[0],
        df,
        x,
        offsets,
        width,
        combined_setting_labels,
        "avg_latency",
        "Mean (s)",
    )
    axes[0].set_ylim(0, 20000)
    _apply_scientific_yaxis(axes[0])
    _plot_grouped(
        axes[1],
        df,
        x,
        offsets,
        width,
        combined_setting_labels,
        "p95_latency",
        "P95 (s)",
    )
    axes[1].set_ylim(0, 50000)
    _apply_scientific_yaxis(axes[1])
    _style_combined_axis(axes[0], "Mean (s)", combined_setting_labels)
    _style_combined_axis(axes[1], "P95 (s)", combined_setting_labels)
    _combined_legend(fig)
    fig.tight_layout(pad=0.15, w_pad=0.28, rect=(0, 0, 1, 0.90))
    fig.savefig(fig_dir / "overall_latency_combined.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_latency_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULTS_DIR / "main_Q1000_S50_dbfixed_minp95_full",
        help="Directory containing workflow*/method/selected_plan.json",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "fig" / "main_Q1000_S50_dbfixed_minp95_full",
        help="Output directory for PDFs and source CSV",
    )
    args = parser.parse_args()

    df = plot_overall_performance(args.result_root, args.fig_dir)
    for name in (
        "overall_cost.pdf",
        "overall_cost.png",
        "overall_mean_latency.pdf",
        "overall_mean_latency.png",
        "overall_p95_latency.pdf",
        "overall_p95_latency.png",
        "overall_svr.pdf",
        "overall_svr.png",
        "overall_cost_svr_combined.pdf",
        "overall_cost_svr_combined.png",
        "overall_latency_combined.pdf",
        "overall_latency_combined.png",
        "overall_performance_source.csv",
    ):
        print(args.fig_dir / name)
    print(
        df[
            [
                "setting",
                "method",
                "expected_cost",
                "avg_latency",
                "p95_latency",
                "violation_rate",
                "status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
