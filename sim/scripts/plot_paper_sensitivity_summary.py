#!/usr/bin/env python3
"""Draw the paper sensitivity summary figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ETA = ROOT / "fig" / "paper_eta_sensitivity" / "eta_sensitivity_main_text_source.csv"
DEFAULT_SLA = (
    ROOT
    / "fig"
    / "sensitivity_sla_multiplier_minp95_current_qbr030"
    / "sla_multiplier_1p0_to_1p5_summary.csv"
)
DEFAULT_OUT = ROOT / "fig" / "paper_sensitivity_summary" / "paper_sensitivity_summary"

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
            "font.size": 8.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.unicode_minus": False,
        }
    )


def load_eta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["has_result"] == True].copy()  # noqa: E712 - CSV stores booleans.
    df["eta"] = df["eta"].astype(float)
    if "cost_change_pct" not in df.columns:
        base = (
            df[df["eta"] == 0.05][["setting", "expected_cost"]]
            .rename(columns={"expected_cost": "base_cost"})
            .copy()
        )
        df = df.merge(base, on="setting", how="left")
        df["cost_change_pct"] = (df["expected_cost"] / df["base_cost"] - 1.0) * 100.0
    return df


def load_sla(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["has_result"] == True].copy()  # noqa: E712 - CSV stores booleans.
    df["multiplier"] = df["multiplier"].astype(float)
    return df


def aggregate_range(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df.groupby(x_col, as_index=False)
        .agg(
            mean=(y_col, "mean"),
            low=(y_col, "min"),
            high=(y_col, "max"),
            count=("setting", "nunique"),
        )
        .sort_values(x_col)
    )


def style_line_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=8.0, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8.0, labelpad=3)
    ax.tick_params(axis="both", labelsize=7.0, length=2.8, pad=2)
    ax.grid(axis="y", color="#d8d8d8", linestyle="--", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)


def plot_eta_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    color: str,
    ylim: tuple[float, float],
    show_target: bool = False,
) -> None:
    summary = aggregate_range(df, "eta", y_col)
    x = summary["eta"].to_numpy()

    for setting, _label in SETTING_ORDER:
        sub = df[df["setting"] == setting].sort_values("eta")
        if sub.empty:
            continue
        ax.plot(
            sub["eta"],
            sub[y_col],
            color="#b7b7b7",
            linewidth=0.65,
            marker="o",
            markersize=2.2,
            markerfacecolor="white",
            markeredgewidth=0.45,
            alpha=0.75,
            zorder=1,
        )

    ax.fill_between(
        x,
        summary["low"].to_numpy(dtype=float),
        summary["high"].to_numpy(dtype=float),
        color=color,
        alpha=0.14,
        linewidth=0,
        zorder=0,
    )
    ax.plot(
        x,
        summary["mean"],
        color=color,
        marker="o",
        markersize=4.2,
        linewidth=1.65,
        zorder=3,
    )
    incomplete = summary[summary["count"] < len(SETTING_ORDER)]
    if not incomplete.empty:
        ax.scatter(
            incomplete["eta"],
            incomplete["mean"],
            s=35,
            facecolors="white",
            edgecolors=color,
            linewidths=1.1,
            zorder=4,
        )
        ax.text(
            incomplete["eta"].iloc[0],
            incomplete["mean"].iloc[0],
            " 4/6",
            fontsize=6.6,
            color="#555555",
            va="center",
        )
    if show_target:
        ax.plot(x, x * 100.0, color="#8c2d2d", linestyle=(0, (4, 2)), linewidth=1.0)
        ax.text(0.083, 8.6, r"$100\eta$", fontsize=6.8, color="#8c2d2d")
    else:
        ax.axhline(0, color="#6a6a6a", linewidth=0.8)

    ax.set_title(title, fontsize=9.0, pad=4)
    ax.set_xticks([0.025, 0.05, 0.075, 0.1])
    ax.set_xticklabels(["0.025", "0.05", "0.075", "0.10"], rotation=18, ha="right")
    ax.set_ylim(*ylim)
    style_line_axis(ax, r"Risk tolerance $\eta$", ylabel)


def plot_eta_combined(ax: plt.Axes, df: pd.DataFrame) -> None:
    cost = aggregate_range(df, "eta", "cost_change_pct")
    svr = aggregate_range(df, "eta", "svr_pct")
    x = cost["eta"].to_numpy()

    ax.fill_between(
        x,
        cost["low"].to_numpy(dtype=float),
        cost["high"].to_numpy(dtype=float),
        color="#28724f",
        alpha=0.14,
        linewidth=0,
    )
    ax.plot(
        x,
        cost["mean"],
        color="#28724f",
        marker="o",
        markersize=3.8,
        linewidth=1.45,
        label="Mean cost change",
    )
    incomplete = cost[cost["count"] < len(SETTING_ORDER)]
    if not incomplete.empty:
        ax.scatter(
            incomplete["eta"],
            incomplete["mean"],
            s=28,
            facecolors="white",
            edgecolors="#28724f",
            linewidths=1.0,
            zorder=4,
        )
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_ylim(-3.0, 0.0)
    ax.set_yticks([-3, -2, -1, 0])
    ax.set_xticks([0.025, 0.05, 0.075, 0.1])
    ax.set_xticklabels(["0.025", "0.05", "0.075", "0.10"], rotation=18, ha="right")
    style_line_axis(ax, r"Risk tolerance $\eta$", r"Cost change (%)")
    ax.set_title("(a) Risk tolerance", fontsize=8.6, pad=3)

    ax2 = ax.twinx()
    ax2.fill_between(
        x,
        svr["low"].to_numpy(dtype=float),
        svr["high"].to_numpy(dtype=float),
        color="#5a51b8",
        alpha=0.12,
        linewidth=0,
    )
    ax2.plot(
        x,
        svr["mean"],
        color="#5a51b8",
        marker="s",
        markersize=3.6,
        linewidth=1.35,
        label="Mean SVR",
    )
    ax2.set_ylabel("SVR (%)", fontsize=8.0, color="#5a51b8", labelpad=3)
    ax2.set_ylim(0, 10)
    ax2.set_yticks([0, 5, 10])
    ax2.tick_params(axis="y", labelsize=7.0, colors="#5a51b8", length=2.8, pad=2)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)


def plot_sla_tradeoff(ax: plt.Axes, df: pd.DataFrame) -> None:
    summary = (
        df.groupby("multiplier", as_index=False)
        .agg(
            mean_svr=("svr_pct", "mean"),
            mean_cost_delta=("cost_delta_pct", "mean"),
            max_svr=("svr_pct", "max"),
        )
        .sort_values("multiplier")
    )
    x = summary["multiplier"].to_numpy()
    ax.axvspan(0.78, 0.95, color="#f0d9d5", alpha=0.9, zorder=0)
    ax.text(0.865, 2.65, "0.8x/0.9x\ninfeasible", ha="center", va="center", fontsize=6.4, color="#8a3b32")
    ax.plot(
        x,
        summary["mean_svr"],
        color="#345f8c",
        marker="o",
        markersize=4.2,
        linewidth=1.7,
        label="Mean SVR",
        zorder=3,
    )
    ax.plot(
        x,
        summary["max_svr"],
        color="#345f8c",
        marker="^",
        markersize=3.8,
        linewidth=1.0,
        linestyle=":",
        alpha=0.95,
        label="Max SVR",
        zorder=3,
    )
    ax.set_xlim(0.78, 1.52)
    ax.set_ylim(0, 6.5)
    ax.set_yticks([0, 2.5, 5, 6.5])
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ax.set_xticklabels(["0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"], rotation=25)
    style_line_axis(ax, "SLA/budget multiplier", "SVR (%)")
    ax.set_title("(b) Budget scaling: aggregate", fontsize=8.6, pad=3)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        -summary["mean_cost_delta"],
        color="#28724f",
        marker="s",
        markersize=3.8,
        linewidth=1.5,
        label="Mean cost saving",
        zorder=4,
    )
    ax2.set_ylabel("Cost saving (%)", fontsize=8.0, color="#28724f", labelpad=4)
    ax2.tick_params(axis="y", labelsize=7.0, colors="#28724f", length=2.8, pad=2)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 3)
    ax2.set_yticks([0, 1, 2, 3])

    handles = [
        Line2D([0], [0], color="#345f8c", marker="o", lw=1.7, label="Mean SVR"),
        Line2D([0], [0], color="#345f8c", marker="^", lw=1.0, linestyle=":", label="Max SVR"),
        Line2D([0], [0], color="#28724f", marker="s", lw=1.5, label="Mean cost saving"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper right", handlelength=1.7)


def draw_figure(eta_csv: Path, sla_csv: Path, out_prefix: Path) -> None:
    configure_style()
    eta = load_eta(eta_csv)
    sla = load_sla(sla_csv)

    fig = plt.figure(figsize=(3.55, 2.35))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.82,
    )
    fig.subplots_adjust(left=0.16, right=0.86, top=0.96, bottom=0.16)
    ax_eta = fig.add_subplot(grid[0, 0])
    ax_sla = fig.add_subplot(grid[1, 0])

    plot_eta_combined(ax_eta, eta)
    plot_sla_tradeoff(ax_sla, sla)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=420, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta-csv", type=Path, default=DEFAULT_ETA)
    parser.add_argument("--sla-csv", type=Path, default=DEFAULT_SLA)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    draw_figure(args.eta_csv, args.sla_csv, args.out_prefix)
    print(args.out_prefix.with_suffix(".pdf"))
    print(args.out_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
