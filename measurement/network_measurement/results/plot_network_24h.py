"""
Load RTT / iperf CSVs for three measurement scenarios, plot one 2x1 figure: top = RTT
(3 curves), bottom = bandwidth (3 curves). Per scenario, x is hours from that
scenario's t0 = min(first RTT, first iperf) on 0-24h; each curve is its own 24h
trajectory, not a common wall-clock (see caption in paper).

In LaTeX, use e.g. \\includegraphics for the single PDF, or set FIG_INCHES to match
a two-column (figure*).
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

# All plot text: Times New Roman (embed in PDF; Windows/macOS have TNR; Linux may
# need msttcorefonts or a local Times New Roman TTF).
TNR: str = "Times New Roman"

RESULTS_DIR = Path(__file__).resolve().parent

# 2x1: width similar to a single column; height fits RTT+BW + legend.
FIG_INCHES: tuple[float, float] = (6.5, 4.5)

# Plot first 24 hours from t0; change if your runs are a different design length.
PLOT_HOURS: float = 24.0

# Top (RTT) panel legend, lower right: `loc` pins that corner to `LEGEND_BBOX` in
# the RTT axes (0–1). y = fraction of the upper subplot’s height *above* the
# x-axis: 0.2 = 20% of the panel height from the bottom / x-axis.
LEGEND_LOC: str = "lower right"
LEGEND_BBOX: tuple[float, float] = (0.99, 0.1)  # (right, 20% from x-axis)
LEGEND_COORD_SPACE: str = "axes"  # "axes" (RTT) or "figure" (entire fig)
LEGEND_FONTSIZE: float = 11.0

# After saving, open the PDF in the default viewer (e.g. Edge / Chrome / Sumatra).
# Set to False or set env SKYBRIDGE_NO_PLOT=1 to skip.
OPEN_OUTPUT_PDF: bool = True


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    rtt_glob: str
    bw_glob: str


SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        "cp_cr",
        "GCP asia-east-1, AWS us-west-2",
        "cross_provider_cross_region/rtt_*.csv",
        "cross_provider_cross_region/bandwidth_*.csv",
    ),
    SeriesSpec(
        "cp_sr",
        "GCP us-west-1, AWS us-west-2",
        "cross_provider_same_region/rtt_*.csv",
        "cross_provider_same_region/bandwidth_*.csv",
    ),
    SeriesSpec(
        "same_cr",
        "GCP asia-east-1, GCP us-west-1",
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


def clip_to_hours_from_t0(
    df: pd.DataFrame, t0: pd.Timestamp, hours: float
) -> pd.DataFrame:
    t1 = t0 + pd.Timedelta(hours=hours)
    m = (df["t"] >= t0) & (df["t"] <= t1)
    return df.loc[m].reset_index(drop=True)


def elapsed_hours(t: pd.Series, t0: pd.Timestamp) -> pd.Series:
    return (t - t0).dt.total_seconds() / 3600.0


def prepare_scenario(
    spec_key: str, rtt: pd.DataFrame, bw: pd.DataFrame
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    if rtt.empty or bw.empty:
        raise ValueError(f"{spec_key}: need non-empty RTT and bandwidth after filters")
    t0 = min(rtt["t"].iloc[0], bw["t"].iloc[0])
    rtt_p = clip_to_hours_from_t0(rtt, t0, PLOT_HOURS)
    bw_p = clip_to_hours_from_t0(bw, t0, PLOT_HOURS)
    if rtt_p.empty or bw_p.empty:
        raise ValueError(f"{spec_key}: no samples in [t0, t0+{PLOT_HOURS}h)")
    return t0, rtt_p, bw_p


def configure_figure() -> None:
    # Single font: Times New Roman (no DejaVu/Times fallbacks in rcParams).
    tnr: list[str] = [TNR]
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": tnr,
            "font.sans-serif": tnr,
            "font.monospace": tnr,
            "font.cursive": tnr,
            "font.fantasy": tnr,
            "pdf.fonttype": 42,  # TTF (Times New Roman) embedded in PDF
            "pdf.use14corefonts": False,
            "axes.linewidth": 0.8,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.unicode_minus": False,
        }
    )


def open_output_path(path: Path) -> None:
    """Open the file with the system default app (Windows: startfile, mac: open, Linux: xdg-open)."""
    if not path.is_file():
        print("Skip open: not a file", path)
        return
    p = str(path.resolve())
    try:
        if sys.platform == "win32":
            os.startfile(p)  # nosec: user-requested; opens generated artifact
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
    except OSError as exc:
        print("Could not open in viewer:", exc)


def save_pdf(fig: matplotlib.figure.Figure, path: Path) -> Path:
    try:
        fig.savefig(
            path, bbox_inches="tight", format="pdf", dpi=300, pad_inches=0.02
        )
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_alternate{path.suffix}")
        fig.savefig(
            alt, bbox_inches="tight", format="pdf", dpi=300, pad_inches=0.02
        )
        print(f"Permission denied for {path}; wrote {alt} instead.")
        return alt


def main() -> None:
    configure_figure()

    prepared: list[
        tuple[SeriesSpec, pd.Timestamp, pd.DataFrame, pd.DataFrame]
    ] = []
    for spec in SPECS:
        rtt = load_rtt(_first_csv(spec.rtt_glob))
        bw = load_bw(_first_csv(spec.bw_glob))
        t0, rtt_p, bw_p = prepare_scenario(spec.key, rtt, bw)
        prepared.append((spec, t0, rtt_p, bw_p))
        s_r = (rtt_p["t"].iloc[-1] - t0).total_seconds() / 3600.0
        s_b = (bw_p["t"].iloc[-1] - t0).total_seconds() / 3600.0
        print(f"{spec.key}: t0 (UTC)={t0}  span RTT~{s_r:.2f}h, BW~{s_b:.2f}h")

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    lw = 1.1
    x_ticks = [0, 6, 12, 18, 24]
    fig, (ax_rtt, ax_bw) = plt.subplots(
        2, 1, sharex=True, figsize=FIG_INCHES, constrained_layout=True
    )
    for c, (spec, t0, rtt_p, bw_p) in enumerate(prepared):
        ax_rtt.plot(
            elapsed_hours(rtt_p["t"], t0),
            rtt_p["rtt_avg_ms"],
            color=colors[c],
            linewidth=lw,
            label=spec.label,
        )
        ax_bw.plot(
            elapsed_hours(bw_p["t"], t0),
            bw_p["bw_mean_mbps"],
            color=colors[c],
            linewidth=lw,
        )
    ax_rtt.set_ylabel("Average RTT (ms)")
    ax_rtt.set_ylim(0, 200)
    ax_rtt.set_xlim(0, PLOT_HOURS)
    ax_rtt.set_xticks(x_ticks)
    ax_rtt.grid(True, alpha=0.35)
    _t = (
        fig.transFigure
        if LEGEND_COORD_SPACE.lower() == "figure"
        else ax_rtt.transAxes
    )
    ax_rtt.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX,
        bbox_transform=_t,
        framealpha=0.95,
        prop=fm.FontProperties(family=TNR, size=LEGEND_FONTSIZE),
    )

    ax_bw.set_ylabel("Bandwidth (Mbit/s)")
    ax_bw.set_xlim(0, PLOT_HOURS)
    ax_bw.set_xticks(x_ticks)
    ax_bw.set_xlabel("Time (h)")
    ax_bw.grid(True, alpha=0.35)

    out = save_pdf(fig, RESULTS_DIR / "network_24h_rtt_bandwidth.pdf")
    plt.close(fig)
    print("Wrote:", out)
    if OPEN_OUTPUT_PDF and not os.environ.get("SKYBRIDGE_NO_PLOT", "").strip() in (
        "1",
        "true",
        "yes",
    ):
        open_output_path(out)

if __name__ == "__main__":
    main()
