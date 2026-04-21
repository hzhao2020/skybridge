"""
Load RTT / iperf CSVs under results/, plot a common time window, export PDFs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    rtt_glob: str
    bw_glob: str


SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        "cp_cr",
        "cross provider, cross region",
        "cross_provider_cross_region/rtt_*.csv",
        "cross_provider_cross_region/bandwidth_*.csv",
    ),
    SeriesSpec(
        "cp_sr",
        "cross provider, same region",
        "cross_provider_same_region/rtt_*.csv",
        "cross_provider_same_region/bandwidth_*.csv",
    ),
    SeriesSpec(
        "same_cr",
        "same provider, cross region",
        "inter_region/rtt_*.csv",
        "inter_region/bandwidth_*.csv",
    ),
)


def _first_csv(pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No CSV matched: {RESULTS_DIR / pattern}")
    return matches[0]


def load_rtt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: c in ("timestamp_utc", "rtt_avg_ms", "ping_ok"),
    )
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.loc[df["ping_ok"] == 1, ["t", "rtt_avg_ms"]].sort_values("t")
    return df.reset_index(drop=True)


def load_bw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: c
        in (
            "timestamp_utc",
            "bw_out_mbits_per_sec",
            "bw_in_mbits_per_sec",
            "iperf_ok",
        ),
    )
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.loc[df["iperf_ok"] == 1].copy()
    df["bw_mean_mbps"] = (
        df["bw_out_mbits_per_sec"] + df["bw_in_mbits_per_sec"]
    ) / 2.0
    df = df[["t", "bw_mean_mbps"]].sort_values("t")
    return df.reset_index(drop=True)


def common_window(dfs: list[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [d["t"].iloc[0] for d in dfs]
    ends = [d["t"].iloc[-1] for d in dfs]
    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError("No overlapping time window across series")
    return start, end


def clip(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = (df["t"] >= start) & (df["t"] <= end)
    return df.loc[m].reset_index(drop=True)


def elapsed_hours(t: pd.Series, t0: pd.Timestamp) -> pd.Series:
    return (t - t0).dt.total_seconds() / 3600.0


def configure_figure() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11, 5.5),
            "figure.dpi": 120,
        }
    )


def save_pdf(fig: matplotlib.figure.Figure, path: Path) -> Path:
    try:
        fig.savefig(path, bbox_inches="tight", format="pdf")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_alternate{path.suffix}")
        fig.savefig(alt, bbox_inches="tight", format="pdf")
        print(f"Permission denied for {path}; wrote {alt} instead.")
        return alt


def main() -> None:
    configure_figure()

    rtt_frames: list[pd.DataFrame] = []
    bw_frames: list[pd.DataFrame] = []
    labels: list[str] = []

    for spec in SPECS:
        rtt_frames.append(load_rtt(_first_csv(spec.rtt_glob)))
        bw_frames.append(load_bw(_first_csv(spec.bw_glob)))
        labels.append(spec.label)

    w_start, w_end = common_window(rtt_frames)
    rtt_clip = [clip(d, w_start, w_end) for d in rtt_frames]
    w_start2, w_end2 = common_window(bw_frames)
    bw_start = max(w_start, w_start2)
    bw_end = min(w_end, w_end2)
    bw_clip = [clip(d, bw_start, bw_end) for d in bw_frames]

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")

    # RTT
    fig_rtt, ax_rtt = plt.subplots()
    for df, lab, c in zip(rtt_clip, labels, colors):
        x_h = elapsed_hours(df["t"], w_start)
        ax_rtt.plot(x_h, df["rtt_avg_ms"], label=lab, color=c, linewidth=1.2)
    ax_rtt.set_title("24-hour RTT (overlapping sample window)")
    ax_rtt.set_xlabel("Time (h)")
    ax_rtt.set_ylabel("Average RTT (ms)")
    ax_rtt.set_xlim(0, 24)
    ax_rtt.legend(loc="best", framealpha=0.9)
    ax_rtt.grid(True, alpha=0.35)
    rtt_pdf = save_pdf(fig_rtt, RESULTS_DIR / "network_rtt_24h.pdf")
    plt.close(fig_rtt)

    # Bandwidth
    fig_bw, ax_bw = plt.subplots()
    for df, lab, c in zip(bw_clip, labels, colors):
        x_h = elapsed_hours(df["t"], bw_start)
        ax_bw.plot(x_h, df["bw_mean_mbps"], label=lab, color=c, linewidth=1.2)
    ax_bw.set_title(
        "24-hour bandwidth (mean iperf sender-reported up/down, Mbps)"
    )
    ax_bw.set_xlabel("Time (h)")
    ax_bw.set_ylabel("Bandwidth (Mbit/s)")
    ax_bw.set_xlim(0, 24)
    ax_bw.legend(loc="best", framealpha=0.9)
    ax_bw.grid(True, alpha=0.35)
    bw_pdf = save_pdf(fig_bw, RESULTS_DIR / "network_bandwidth_24h.pdf")
    plt.close(fig_bw)

    print("RTT window (UTC):", w_start, "->", w_end)
    print("Bandwidth window (UTC):", bw_start, "->", bw_end)
    print("Wrote:", rtt_pdf)
    print("Wrote:", bw_pdf)


if __name__ == "__main__":
    main()
