#!/usr/bin/env python3
"""Redraw SkyFlow eta sensitivity line plots using the compact overall style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIG_DIR = ROOT / "fig" / "sensitivity_eta_minp95_current_qbr030"
DEFAULT_SOURCE = DEFAULT_FIG_DIR / "skyflow_eta_cost_svr_by_setting.csv"

SETTING_LABELS = [
    ("workflow1_Q1", "W1-Q1"),
    ("workflow1_Q2", "W1-Q2"),
    ("workflow1_Q3", "W1-Q3"),
    ("workflow2_Q1", "W2-Q1"),
    ("workflow2_Q2", "W2-Q2"),
    ("workflow2_Q3", "W2-Q3"),
]

# Match the main overall figure palette, with one extra muted red for the sixth setting.
PALETTE = {
    "W1-Q1": "#3b6fb6",  # SkyFlow blue
    "W1-Q2": "#8a8a8a",  # SC gray
    "W1-Q3": "#64a85b",  # Greedy green
    "W2-Q1": "#d9893d",  # DPGM orange
    "W2-Q2": "#8e6ab8",  # MTGP purple
    "W2-Q3": "#c85d5d",
}
MARKERS = {
    "W1-Q1": "o",
    "W1-Q2": "s",
    "W1-Q3": "^",
    "W2-Q1": "D",
    "W2-Q2": "P",
    "W2-Q3": "X",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
        }
    )


def _format_eta_ticks(values: pd.Series) -> list[str]:
    return [f"{value:g}" for value in values]


def _style_axis(ax, ylabel: str, eta_values: pd.Series) -> None:
    ax.set_xlabel(r"Risk tolerance $\eta$", fontsize=6.8, fontweight="semibold", labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=6.8, fontweight="semibold", labelpad=1.6)
    ax.set_xticks(eta_values)
    ax.set_xticklabels(_format_eta_ticks(eta_values), fontsize=5.9, rotation=28)
    ax.tick_params(axis="x", pad=1.0, length=2.2, width=0.55)
    ax.tick_params(axis="y", labelsize=5.9, pad=1.0, length=2.2, width=0.55)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.35, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)


def _plot_setting_lines(ax, df: pd.DataFrame, metric: str) -> None:
    for setting, label in SETTING_LABELS:
        sub = df[df["setting"] == setting].sort_values("eta")
        if sub.empty:
            continue
        ax.plot(
            sub["eta"],
            sub[metric],
            color=PALETTE[label],
            marker=MARKERS[label],
            markersize=3.4,
            linewidth=1.15,
            markeredgewidth=0.35,
            markeredgecolor="black",
            label=label,
        )


def plot_cost_svr(source_csv: Path, out_prefix: Path) -> None:
    _configure_style()
    df = pd.read_csv(source_csv)
    df = df[df["has_result"] == True].copy()  # noqa: E712 - CSV column is boolean-like
    df["eta"] = df["eta"].astype(float)
    eta_values = pd.Series(sorted(df["eta"].unique()))

    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.42))
    _plot_setting_lines(axes[0], df, "expected_cost")
    _plot_setting_lines(axes[1], df, "svr_pct")
    axes[1].plot(
        eta_values,
        eta_values * 100.0,
        color="#b22222",
        linestyle="--",
        linewidth=0.8,
        label=r"$\eta$",
    )

    axes[0].set_title("Cost", fontsize=7.4, fontweight="semibold", pad=2.5)
    axes[1].set_title("SVR", fontsize=7.4, fontweight="semibold", pad=2.5)
    _style_axis(axes[0], "Cost ($)", eta_values)
    _style_axis(axes[1], "SVR (%)", eta_values)
    axes[0].set_ylim(0, max(df["expected_cost"].max() * 1.10, 1.0))
    axes[1].set_ylim(0, 10)
    axes[1].set_yticks([0, 5, 10])

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_labels = [label for _, label in SETTING_LABELS] + [r"$\eta$"]
    legend_handles = [by_label[label] for label in legend_labels if label in by_label]
    fig.legend(
        legend_handles,
        [label for label in legend_labels if label in by_label],
        ncol=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=0.55,
        handlelength=1.25,
        handletextpad=0.22,
        borderaxespad=0.0,
        prop={"size": 5.8, "weight": "normal"},
    )
    fig.tight_layout(pad=0.12, w_pad=0.28, rect=(0, 0, 1, 0.88))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=DEFAULT_FIG_DIR / "skyflow_eta_cost_svr_all_settings",
    )
    args = parser.parse_args()
    plot_cost_svr(args.source_csv, args.out_prefix)
    print(args.out_prefix.with_suffix(".pdf"))
    print(args.out_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
