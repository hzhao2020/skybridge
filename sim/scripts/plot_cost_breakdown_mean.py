#!/usr/bin/env python3
"""Plot mean cost breakdown across quality levels.

The input CSV must contain workflow, quality, method, execution cost, network
transfer cost, and storage/database cost columns. Several common aliases are
normalized automatically; total cost columns are used only for diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["SkyFlow", "Greedy", "DPGM"]
WORKFLOW_ORDER = ["workflow1", "workflow2", "workflow3", "workflow4"]
QUALITY_ORDER = ["Q1", "Q2", "Q3"]
COMPONENT_COLUMNS = ["execution_cost", "network_cost", "storage_cost"]

COLUMN_ALIASES = {
    "expected_execution_cost": "execution_cost",
    "exec_cost": "execution_cost",
    "c_execution": "execution_cost",
    "transfer_cost": "network_cost",
    "egress_cost": "network_cost",
    "network_transfer_cost": "network_cost",
    "db_cost": "storage_cost",
    "storage_database_cost": "storage_cost",
    "storage_db_cost": "storage_cost",
    "q": "quality",
    "quality_level": "quality",
    "algorithm": "method",
    "approach": "method",
}

METHOD_ALIASES = {
    "decomposition": "SkyFlow",
    "decomposed_milp": "SkyFlow",
    "skyflow": "SkyFlow",
    "greedy": "Greedy",
    "dpgm": "DPGM",
    "murakkab": "DPGM",
    "murakkab-style": "DPGM",
    "murakkab_style": "DPGM",
    "murakkab profile": "DPGM",
    "murakkab_profile": "DPGM",
    "single-cloud": "Single-Cloud",
    "single_cloud": "Single-Cloud",
    "single cloud": "Single-Cloud",
    "sc": "Single-Cloud",
}


def _normalize_method(value: object) -> str:
    raw = str(value).strip()
    key = raw.lower().replace("_", " ").replace("-", " ")
    compact = raw.lower().strip()
    return METHOD_ALIASES.get(compact, METHOD_ALIASES.get(key, raw))


def _pretty_workflow(value: str) -> str:
    if value.startswith("workflow") and value[len("workflow"):].isdigit():
        return f"Workflow {value[len('workflow'):]}"
    return value


def _cost_label(value: float) -> str:
    return f"{value:.3f}" if abs(value) < 1.0 else f"{value:.2f}"


def load_and_normalize_results(path: str | Path) -> pd.DataFrame:
    """Load a result CSV and normalize required columns and method labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})

    required = ["workflow", "quality", "method", *COMPONENT_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns after normalization: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    df["workflow"] = df["workflow"].astype(str).str.strip().str.lower()
    df["quality"] = df["quality"].astype(str).str.strip().str.upper()
    df["method"] = df["method"].map(_normalize_method)

    for col in COMPONENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_costs = df[COMPONENT_COLUMNS].isna().sum()
    bad = missing_costs[missing_costs > 0]
    if not bad.empty:
        raise ValueError(
            "Missing or non-numeric cost component values: "
            + ", ".join(f"{col}={count}" for col, count in bad.items())
        )

    for total_col in ("expected_cost", "total_cost"):
        if total_col in df.columns:
            df[total_col] = pd.to_numeric(df[total_col], errors="coerce")

    df["total_breakdown_cost"] = df[COMPONENT_COLUMNS].sum(axis=1)
    return df


def aggregate_cost_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to paper methods/settings and aggregate over Q1, Q2, Q3."""
    filtered = df[
        df["workflow"].isin(WORKFLOW_ORDER)
        & df["quality"].isin(QUALITY_ORDER)
        & df["method"].isin(METHOD_ORDER)
    ].copy()

    missing_workflows = sorted(set(WORKFLOW_ORDER) - set(filtered["workflow"]))
    missing_methods = sorted(set(METHOD_ORDER) - set(filtered["method"]))
    if missing_workflows:
        print(f"WARNING: missing expected workflow(s): {missing_workflows}", file=sys.stderr)
    if missing_methods:
        print(f"WARNING: missing expected method(s): {missing_methods}", file=sys.stderr)

    expected_pairs = pd.MultiIndex.from_product(
        [WORKFLOW_ORDER, METHOD_ORDER], names=["workflow", "method"]
    )
    present_pairs = pd.MultiIndex.from_frame(filtered[["workflow", "method"]].drop_duplicates())
    missing_pairs = expected_pairs.difference(present_pairs)
    if len(missing_pairs) > 0:
        pairs = [f"{wf}/{method}" for wf, method in missing_pairs]
        print(f"WARNING: missing workflow-method pair(s): {pairs}", file=sys.stderr)

    summary = (
        filtered.groupby(["workflow", "method"], as_index=False)[
            [*COMPONENT_COLUMNS, "total_breakdown_cost"]
        ]
        .mean()
        .sort_values(
            ["workflow", "method"],
            key=lambda s: s.map(
                {**{w: i for i, w in enumerate(WORKFLOW_ORDER)}, **{m: i for i, m in enumerate(METHOD_ORDER)}}
            ).fillna(99),
        )
    )

    if len(summary) != 6:
        print(
            "WARNING: grouped output contains "
            f"{len(summary)} rows; expected 2 workflows x 3 methods = 6 rows "
            "when all methods are available.",
            file=sys.stderr,
        )

    denom = summary["total_breakdown_cost"].replace(0.0, np.nan)
    summary["execution_pct"] = summary["execution_cost"] / denom * 100.0
    summary["network_pct"] = summary["network_cost"] / denom * 100.0
    summary["storage_pct"] = summary["storage_cost"] / denom * 100.0
    return summary


def print_summary(summary_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """Print publication diagnostics and validation summaries."""
    display = summary_df.rename(
        columns={
            "workflow": "Workflow",
            "method": "Method",
            "execution_cost": "Mean Execution Cost",
            "network_cost": "Mean Network Cost",
            "storage_cost": "Mean Storage/DB Cost",
            "total_breakdown_cost": "Mean Total Cost",
            "execution_pct": "Execution %",
            "network_pct": "Network %",
            "storage_pct": "Storage/DB %",
        }
    )
    display["Workflow"] = display["Workflow"].map(_pretty_workflow)
    cols = [
        "Workflow",
        "Method",
        "Mean Execution Cost",
        "Mean Network Cost",
        "Mean Storage/DB Cost",
        "Mean Total Cost",
        "Execution %",
        "Network %",
        "Storage/DB %",
    ]

    print("\nMean Cost Breakdown")
    print(
        display[cols].to_string(
            index=False,
            formatters={
                "Mean Execution Cost": "{:.6f}".format,
                "Mean Network Cost": "{:.6f}".format,
                "Mean Storage/DB Cost": "{:.6f}".format,
                "Mean Total Cost": "{:.6f}".format,
                "Execution %": "{:.2f}".format,
                "Network %": "{:.2f}".format,
                "Storage/DB %": "{:.2f}".format,
            },
        )
    )

    print("\nRank by Mean Total Cost")
    for workflow in WORKFLOW_ORDER:
        sub = summary_df[summary_df["workflow"] == workflow].sort_values("total_breakdown_cost")
        if sub.empty:
            continue
        ranking = " < ".join(
            f"{row.method} ({row.total_breakdown_cost:.6f})" for row in sub.itertuples()
        )
        print(f"{_pretty_workflow(workflow)}: {ranking}")

    print("\nSkyFlow Cost Reduction")
    for workflow in WORKFLOW_ORDER:
        sub = summary_df[summary_df["workflow"] == workflow].set_index("method")
        if "SkyFlow" not in sub.index:
            print(f"{_pretty_workflow(workflow)}: SkyFlow missing")
            continue
        sky_total = float(sub.loc["SkyFlow", "total_breakdown_cost"])
        for baseline in ("Greedy", "DPGM"):
            if baseline not in sub.index:
                print(f"{_pretty_workflow(workflow)} vs {baseline}: baseline missing")
                continue
            base_total = float(sub.loc[baseline, "total_breakdown_cost"])
            reduction = (base_total - sky_total) / base_total * 100.0 if base_total else np.nan
            print(f"{_pretty_workflow(workflow)} vs {baseline}: {reduction:.2f}%")

    print("\nDominant Cost Component")
    component_names = {
        "execution_pct": "Execution",
        "network_pct": "Network Transfer",
        "storage_pct": "Storage/Database",
    }
    for row in summary_df.itertuples():
        pcts = {
            "execution_pct": row.execution_pct,
            "network_pct": row.network_pct,
            "storage_pct": row.storage_pct,
        }
        dominant = max(pcts, key=pcts.get)
        print(f"{_pretty_workflow(row.workflow)} / {row.method}: {component_names[dominant]}")

    total_cols = [c for c in ("expected_cost", "total_cost") if c in original_df.columns]
    if total_cols:
        print("\nTotal Cost Validation")
    for total_col in total_cols:
        valid = original_df.dropna(subset=[total_col]).copy()
        if valid.empty:
            print(f"{total_col}: no numeric values available")
            continue
        valid["abs_diff"] = (valid[total_col] - valid["total_breakdown_cost"]).abs()
        diag = (
            valid[
                valid["workflow"].isin(WORKFLOW_ORDER)
                & valid["method"].isin(METHOD_ORDER)
                & valid["quality"].isin(QUALITY_ORDER)
            ]
            .groupby(["workflow", "method"], as_index=False)["abs_diff"]
            .mean()
        )
        print(f"\nMean absolute difference vs {total_col}")
        print(diag.to_string(index=False, formatters={"abs_diff": "{:.8f}".format}))


def plot_cost_breakdown(summary_df: pd.DataFrame, output_prefix: str | Path) -> None:
    """Create and save the stacked grouped bar chart."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_df = (
        summary_df.set_index(["workflow", "method"])
        .reindex(pd.MultiIndex.from_product([WORKFLOW_ORDER, METHOD_ORDER]))
        .reset_index()
        .rename(columns={"level_0": "workflow", "level_1": "method"})
    )

    x = []
    labels = []
    group_centers = []
    pos = 0.0
    gap = 0.85
    for workflow in WORKFLOW_ORDER:
        group_positions = []
        for method in METHOD_ORDER:
            x.append(pos)
            group_positions.append(pos)
            labels.append(method)
            pos += 1.0
        group_centers.append(float(np.mean(group_positions)))
        pos += gap
    x = np.array(x)

    exec_vals = plot_df["execution_cost"].to_numpy(dtype=float)
    net_vals = plot_df["network_cost"].to_numpy(dtype=float)
    stor_vals = plot_df["storage_cost"].to_numpy(dtype=float)
    totals = plot_df["total_breakdown_cost"].to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.68
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    hatches = ["", "//", ".."]

    bars_exec = ax.bar(
        x,
        exec_vals,
        width,
        label="Execution",
        color=colors[0],
        edgecolor="black",
        linewidth=0.35,
        hatch=hatches[0],
    )
    ax.bar(
        x,
        net_vals,
        width,
        bottom=exec_vals,
        label="Network Transfer",
        color=colors[1],
        edgecolor="black",
        linewidth=0.35,
        hatch=hatches[1],
    )
    ax.bar(
        x,
        stor_vals,
        width,
        bottom=exec_vals + net_vals,
        label="Storage/Database",
        color=colors[2],
        edgecolor="black",
        linewidth=0.35,
        hatch=hatches[2],
    )

    ymax = np.nanmax(totals) if len(totals) else 1.0
    label_pad = ymax * 0.025
    for xpos, total in zip(x, totals):
        if np.isfinite(total):
            ax.text(
                xpos,
                total + label_pad,
                _cost_label(float(total)),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("Mean Cost Breakdown Across Quality Levels", pad=8)
    ax.set_ylabel("Mean expected cost per request")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)

    for center, workflow in zip(group_centers, WORKFLOW_ORDER):
        ax.text(
            center,
            -0.22,
            _pretty_workflow(workflow),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(bottom=0.24)

    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV containing cost component columns")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("cost_breakdown_mean"),
        help="Output path prefix for PDF/PNG/summary CSV",
    )
    args = parser.parse_args()

    try:
        df = load_and_normalize_results(args.input_csv)
        summary = aggregate_cost_breakdown(df)
        print_summary(summary, df)
        plot_cost_breakdown(summary, args.output_prefix)
        summary_path = args.output_prefix.with_name(
            args.output_prefix.name + "_summary.csv"
        )
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved figure: {args.output_prefix.with_suffix('.pdf')}")
        print(f"Saved figure: {args.output_prefix.with_suffix('.png')}")
        print(f"Saved summary: {summary_path}")
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
