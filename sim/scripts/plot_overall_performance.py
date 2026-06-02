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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR  # noqa: E402

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
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.35,
        }
    )


def _decorate_axis(ax, x: np.ndarray, setting_labels: list[str], ylabel: str) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


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
        ax.axhline(0.1, color="#b22222", linewidth=0.8, linestyle="--", label="η=0.1")
    _decorate_axis(ax, x, setting_labels, ylabel)


def _set_headroom_ylim(ax, values: pd.Series, *, minimum_top: float = 0.0) -> None:
    top = max(float(values.max()) * 1.18, minimum_top)
    ax.set_ylim(0, top)


def plot_overall_performance(result_root: Path, fig_dir: Path) -> pd.DataFrame:
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = _load_data(result_root)
    df.to_csv(fig_dir / "overall_performance_source.csv", index=False)

    _configure_style()
    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
    setting_labels = [label for _, label in SETTINGS]

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    _plot_grouped(ax, df, x, offsets, width, setting_labels, "expected_cost", "Expected cost ($)")
    _set_headroom_ylim(ax, df["expected_cost"], minimum_top=1.0)
    ax.legend(ncol=4, frameon=False, loc="upper left", columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_cost.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
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
    )
    _set_headroom_ylim(ax, df["violation_rate"], minimum_top=0.16)
    ax.legend(ncol=5, frameon=False, loc="upper right", columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "overall_svr.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_svr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULTS_DIR / "main_Q1000_S50_eta010",
        help="Directory containing workflow*/method/selected_plan.json",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=ROOT / "fig",
        help="Output directory for PDFs and source CSV",
    )
    args = parser.parse_args()

    df = plot_overall_performance(args.result_root, args.fig_dir)
    for name in (
        "overall_cost.pdf",
        "overall_cost.png",
        "overall_svr.pdf",
        "overall_svr.png",
        "overall_performance_source.csv",
    ):
        print(args.fig_dir / name)
    print(df[["setting", "method", "expected_cost", "violation_rate", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
