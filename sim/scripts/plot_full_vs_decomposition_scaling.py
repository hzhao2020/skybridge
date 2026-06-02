#!/usr/bin/env python3
"""Compare full MILP vs decomposition: wall time and memory vs query count Q."""

from __future__ import annotations

import argparse
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

DEFAULT_CSVS = [
    RESULTS_DIR / "experiment_logs" / "q_scaling_full_vs_decomp_S50_Q200_2000_minp95_v2.csv",
]


def _load_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise SystemExit("No input CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    if "sample_count" not in df.columns and "calibration_scenarios_per_query" in df.columns:
        df["sample_count"] = df["calibration_scenarios_per_query"]
    df["method"] = df["method"].replace({"decomposed_milp": "decomposition"})
    df = df[df["status"].isin(["OPTIMAL", "optimal"])].copy()
    df["query_count"] = df["query_count"].astype(int)
    df["sample_count"] = df["sample_count"].astype(int)
    return df


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    key = ["workflow", "quality_level", "query_count", "sample_count"]
    full = df[df["method"] == "full_milp"].rename(
        columns={
            "wall_time_sec": "full_wall_sec",
            "solver_runtime_sec": "full_solver_sec",
            "max_rss_mb": "full_rss_mb",
        }
    )
    decomp = df[df["method"] == "decomposition"].rename(
        columns={
            "wall_time_sec": "decomp_wall_sec",
            "solver_runtime_sec": "decomp_solver_sec",
            "max_rss_mb": "decomp_rss_mb",
        }
    )
    merged = full.merge(
        decomp,
        on=key,
        suffixes=("_full", "_decomp"),
    )
    keep = key + [
        "full_wall_sec",
        "decomp_wall_sec",
        "full_solver_sec",
        "decomp_solver_sec",
        "full_rss_mb",
        "decomp_rss_mb",
    ]
    return merged[keep].sort_values(key)


def _plot_grouped_bars(
    paired: pd.DataFrame,
    metric_full: str,
    metric_decomp: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    workflows = sorted(paired["workflow"].unique())
    samples = sorted(paired["sample_count"].unique())
    nrows = len(workflows)
    ncols = len(samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), squeeze=False)

    for i, wf in enumerate(workflows):
        for j, s in enumerate(samples):
            ax = axes[i, j]
            sub = paired[(paired["workflow"] == wf) & (paired["sample_count"] == s)].sort_values(
                "query_count"
            )
            if sub.empty:
                ax.set_visible(False)
                continue
            q = sub["query_count"].to_numpy()
            x = np.arange(len(q), dtype=float)
            width = 0.36
            ax.bar(x - width / 2, sub[metric_full], width, label="full MILP", color="#8a8a8a", edgecolor="black", lw=0.35)
            ax.bar(x + width / 2, sub[metric_decomp], width, label="decomposition", color="#3b6fb6", edgecolor="black", lw=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in q], rotation=45 if len(q) > 6 else 0, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{wf.replace('workflow', 'W')} · S={s}")
            ax.grid(axis="y", alpha=0.25)
            if i == 0 and j == ncols - 1:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio_lines(paired: pd.DataFrame, path: Path) -> None:
    """Full / decomp ratio vs Q for the largest available S per workflow."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (num_col, den_col, ylabel) in zip(
        axes,
        [
            ("full_wall_sec", "decomp_wall_sec", "Wall-time ratio (full / decomp)"),
            ("full_rss_mb", "decomp_rss_mb", "Memory ratio (full / decomp)"),
        ],
    ):
        for wf in sorted(paired["workflow"].unique()):
            sub_wf = paired[paired["workflow"] == wf]
            s = max(sub_wf["sample_count"])
            sub = sub_wf[sub_wf["sample_count"] == s].sort_values("query_count")
            if sub.empty:
                continue
            ratio = sub[num_col] / sub[den_col].replace(0, np.nan)
            ax.plot(sub["query_count"], ratio, marker="o", label=f"{wf.replace('workflow', 'W')} S={s}")
        ax.axhline(1.0, color="#b22222", ls="--", lw=0.8)
        ax.set_xlabel("Query count Q")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Scaling crossover: ratio > 1 means decomposition is faster / uses less memory", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_full_vs_decomposition(
    csv_paths: list[Path],
    out_dir: Path,
    *,
    export_csv: bool = True,
) -> pd.DataFrame:
    df = _load_frames(csv_paths)
    paired = _paired(df)
    if export_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        paired.to_csv(out_dir / "full_vs_decomposition_paired.csv", index=False)

    _plot_grouped_bars(
        paired,
        "full_wall_sec",
        "decomp_wall_sec",
        "Wall time (sec)",
        "Full MILP vs decomposition — wall time vs Q",
        out_dir / "full_vs_decomp_wall_time.png",
    )
    _plot_grouped_bars(
        paired,
        "full_solver_sec",
        "decomp_solver_sec",
        "Gurobi solver time (sec)",
        "Full MILP vs decomposition — solver time vs Q",
        out_dir / "full_vs_decomp_solver_time.png",
    )
    _plot_grouped_bars(
        paired,
        "full_rss_mb",
        "decomp_rss_mb",
        "Peak RSS (MB)",
        "Full MILP vs decomposition — peak memory vs Q",
        out_dir / "full_vs_decomp_memory.png",
    )
    _plot_ratio_lines(paired, out_dir / "full_vs_decomp_ratio.png")
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Input sweep CSV (repeatable). Defaults to sweep + final scaling logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "fig" / "full_vs_decomposition",
        help="Output directory for PNGs and paired CSV",
    )
    args = parser.parse_args()
    paths = args.csv or DEFAULT_CSVS
    paired = plot_full_vs_decomposition(paths, args.out_dir)
    for name in (
        "full_vs_decomposition_paired.csv",
        "full_vs_decomp_wall_time.png",
        "full_vs_decomp_solver_time.png",
        "full_vs_decomp_memory.png",
        "full_vs_decomp_ratio.png",
    ):
        print(args.out_dir / name)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
