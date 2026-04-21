"""
Plot segment / split execution time vs wall-clock from sweep_*.csv; write PDFs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent


def save_pdf(fig: matplotlib.figure.Figure, path: Path) -> Path:
    try:
        fig.savefig(path, bbox_inches="tight", format="pdf")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_alternate{path.suffix}")
        fig.savefig(alt, bbox_inches="tight", format="pdf")
        print(f"Permission denied for {path}; wrote {alt} instead.")
        return alt


def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "batch_id",
            "utc_iso",
            "success",
            "node_segment_execute_sec",
            "node_split_execute_http_observed_sec",
        ],
    )
    df = df.loc[df["success"] == True].copy()  # noqa: E712
    df["t"] = pd.to_datetime(df["utc_iso"], utc=True)
    df = df.sort_values("t").reset_index(drop=True)
    t0 = df["t"].iloc[0]
    df["elapsed_h"] = (df["t"] - t0).dt.total_seconds() / 3600.0
    return df


def plot_one_sweep(csv_path: Path) -> list[Path]:
    df = load_sweep(csv_path)
    batch = str(df["batch_id"].iloc[0])
    out_paths: list[Path] = []

    plt.rcParams.update({"figure.figsize": (11, 5.5), "figure.dpi": 120})

    # Segment
    fig_s, ax_s = plt.subplots()
    ax_s.plot(
        df["elapsed_h"],
        df["node_segment_execute_sec"],
        "o-",
        color="#1f77b4",
        markersize=4,
        linewidth=0.8,
        alpha=0.85,
    )
    ax_s.set_title(f"Segment execution time (batch {batch})")
    ax_s.set_xlabel("Elapsed time (h) since first sample")
    ax_s.set_ylabel("Execution time (s)")
    ax_s.grid(True, alpha=0.35)
    p1 = save_pdf(fig_s, RESULTS_DIR / f"segment_execute_time_{batch}.pdf")
    plt.close(fig_s)
    out_paths.append(p1)

    # Split
    fig_p, ax_p = plt.subplots()
    ax_p.plot(
        df["elapsed_h"],
        df["node_split_execute_http_observed_sec"],
        "o-",
        color="#ff7f0e",
        markersize=4,
        linewidth=0.8,
        alpha=0.85,
    )
    ax_p.set_title(f"Split execution time — HTTP observed (batch {batch})")
    ax_p.set_xlabel("Elapsed time (h) since first sample")
    ax_p.set_ylabel("Execution time (s)")
    ax_p.grid(True, alpha=0.35)
    p2 = save_pdf(fig_p, RESULTS_DIR / f"split_execute_time_{batch}.pdf")
    plt.close(fig_p)
    out_paths.append(p2)

    return out_paths


def main() -> None:
    csvs = sorted(RESULTS_DIR.glob("sweep_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No sweep_*.csv under {RESULTS_DIR}")
    for p in csvs:
        outs = plot_one_sweep(p)
        print(f"{p.name} -> {', '.join(str(x) for x in outs)}")


if __name__ == "__main__":
    main()
