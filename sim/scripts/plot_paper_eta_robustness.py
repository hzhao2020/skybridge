#!/usr/bin/env python3
"""Draw the main-text eta robustness figure for SkyFlow."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "fig" / "paper_eta_sensitivity" / "eta_sensitivity_main_text_source.csv"
DEFAULT_OUT = ROOT / "fig" / "paper_eta_sensitivity" / "eta_robustness_two_panel_paper"

SETTING_ORDER = [
    (f"workflow{workflow}_Q{quality}", f"W{workflow}-Q{quality}")
    for workflow in range(1, 5)
    for quality in range(1, 4)
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )


def eta_tick_labels(values: list[float]) -> list[str]:
    return [f"{value:g}" for value in values]


def style_axis(ax: plt.Axes, ylabel: str, eta_values: list[float]) -> None:
    ax.set_xlabel(r"Risk tolerance $\eta$", fontsize=8.4, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=8.4, labelpad=4)
    ax.set_xticks(eta_values)
    ax.set_xticklabels(eta_tick_labels(eta_values), rotation=28, ha="right")
    ax.tick_params(axis="both", labelsize=7.4, length=3, pad=2)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.55, linestyle="--", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_setting_lines(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    for setting, _label in SETTING_ORDER:
        sub = df[df["setting"] == setting].sort_values("eta")
        if sub.empty:
            continue
        ax.plot(
            sub["eta"],
            sub[metric],
            color="#b9b9b9",
            marker="o",
            markersize=3.6,
            markerfacecolor="white",
            markeredgecolor="#b9b9b9",
            markeredgewidth=0.9,
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
        )


def plot_mean_line(ax: plt.Axes, df: pd.DataFrame, metric: str) -> None:
    counts = df.groupby("eta")["setting"].nunique()
    complete_etas = counts[counts == len(SETTING_ORDER)].index
    mean_df = (
        df[df["eta"].isin(complete_etas)]
        .groupby("eta", as_index=False)[metric]
        .mean()
        .sort_values("eta")
    )
    ax.plot(
        mean_df["eta"],
        mean_df[metric],
        color="#111111",
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        zorder=3,
    )


def draw_figure(source_csv: Path, out_prefix: Path) -> None:
    configure_style()
    df = pd.read_csv(source_csv)
    df = df[df["has_result"] == True].copy()  # noqa: E712 - CSV stores booleans.
    df["eta"] = df["eta"].astype(float)
    eta_values = sorted(df["eta"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), sharex=False)

    plot_setting_lines(axes[0], df, "cost_change_pct")
    plot_mean_line(axes[0], df, "cost_change_pct")
    axes[0].axhline(0, color="#6b6b6b", linewidth=0.85, linestyle="-", zorder=0)
    axes[0].axvline(0.05, color="#555555", linewidth=1.0, linestyle=(0, (1.5, 1.5)), zorder=2)
    axes[0].text(0.052, 0.33, "default", fontsize=7.1, color="#555555", va="top")
    axes[0].set_title("(a) Normalized cost change", fontsize=10.0, pad=6)
    style_axis(axes[0], r"Change from $\eta=0.05$ (%)", eta_values)
    axes[0].set_ylim(-3.35, 0.55)
    axes[0].set_yticks([-3, -2, -1, 0])
    axes[0].set_yticklabels(["-3%", "-2%", "-1%", "0%"])

    plot_setting_lines(axes[1], df, "svr_pct")
    plot_mean_line(axes[1], df, "svr_pct")
    axes[1].plot(
        eta_values,
        [100 * eta for eta in eta_values],
        color="#666666",
        linewidth=1.2,
        linestyle=(0, (4, 2)),
        zorder=2,
    )
    axes[1].axvline(0.05, color="#555555", linewidth=1.0, linestyle=(0, (1.5, 1.5)), zorder=2)
    axes[1].text(0.052, 9.75, "default", fontsize=7.1, color="#555555", va="top")
    axes[1].set_title("(b) Held-out SVR", fontsize=10.0, pad=6)
    style_axis(axes[1], "SVR (%)", eta_values)
    axes[1].set_ylim(0, 10.6)
    axes[1].set_yticks([0, 2, 4, 6, 8, 10])

    legend_handles = [
        Line2D([0], [0], color="#b9b9b9", marker="o", markerfacecolor="white", lw=1.0, label="Workflow-quality setting"),
        Line2D([0], [0], color="#111111", marker="o", lw=2.0, label="Mean over six settings"),
        Line2D([0], [0], color="#666666", linestyle=(0, (4, 2)), lw=1.2, label=r"Target $100\eta$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.8,
        handlelength=2.0,
        fontsize=7.6,
    )
    fig.text(
        0.5,
        0.02,
        r"Note: $\eta=0.025$ has 4/6 available settings; the mean curve starts from $\eta=0.0375$.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=1.4)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    draw_figure(args.source_csv, args.out_prefix)
    print(args.out_prefix.with_suffix(".pdf"))
    print(args.out_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
