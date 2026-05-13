"""
Aggregate segment/split measurement JSON and video label/OCR/speech JSON,
then plot mean execution time vs. truncated video duration (x-axis 0–32 min) with ±1 SD error bars.

Figure style matches network_measurement/results/plot_network_24h.py (Times New Roman, 300 DPI PDF).
All plot text is English.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Times New Roman (embed in PDF; Windows/macOS have TNR; Linux may need msttcorefonts).
TNR: str = "Times New Roman"

SCRIPT_DIR = Path(__file__).resolve().parent

# Single panel: width/height aligned with paper figure usage in plot_network_24h.py
FIG_INCHES: tuple[float, float] = (6, 3.5)

LEGEND_LOC: str = "upper left"
LEGEND_BBOX: tuple[float, float] = (0.01, 0.99)
LEGEND_FONTSIZE: float = 14.0

OPEN_OUTPUT_PDF: bool = True


def configure_figure() -> None:
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
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "axes.linewidth": 0.8,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "axes.unicode_minus": False,
        }
    )


def open_output_path(path: Path) -> None:
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


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _collect_segment_split(rows: list[dict], results_dir: Path) -> None:
    for sweep in sorted(results_dir.glob("sweep_*_json")):
        if not sweep.is_dir():
            continue
        for jp in sorted(sweep.glob("*.json")):
            if jp.name.startswith("error"):
                continue
            data = _load_json(jp)
            if not data:
                continue
            meta = data.get("meta") or {}
            timing = data.get("timings_sec") or {}
            dur = meta.get("duration_requested_sec")
            seg = timing.get("segment_execute")
            spl = timing.get("split_execute_http_observed")
            if dur is None or seg is None or spl is None:
                continue
            rows.append(
                {
                    "duration_sec": float(dur),
                    "segment_execute_sec": float(seg),
                    "split_execute_sec": float(spl),
                }
            )


def _collect_video_label(rows: list[dict], results_dir: Path) -> None:
    for sweep in sorted(results_dir.glob("sweep_*_json")):
        if not sweep.is_dir():
            continue
        for jp in sorted(sweep.glob("*.json")):
            if jp.name.startswith("error"):
                continue
            data = _load_json(jp)
            if not data:
                continue
            meta = data.get("meta") or {}
            timing = data.get("timings_sec") or {}
            dur = meta.get("duration_requested_sec")
            lab = timing.get("label_vi_operation_wait_sec")
            ocr = timing.get("ocr_vi_operation_wait_sec")
            sp = timing.get("speech_vi_operation_wait_sec")
            if dur is None or lab is None or ocr is None or sp is None:
                continue
            rows.append(
                {
                    "duration_sec": float(dur),
                    "label_detection_sec": float(lab),
                    "ocr_sec": float(ocr),
                    "speech_sec": float(sp),
                }
            )


def _agg_mean_std(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("duration_min", sort=True)
    parts: list[pd.DataFrame] = []
    for col in value_cols:
        stats = g[col].agg(["mean", "std", "count"])
        stats = stats.rename(columns={"mean": f"{col}_mean", "std": f"{col}_std"})
        stats[f"{col}_std"] = stats[f"{col}_std"].fillna(0.0)
        mask_single = stats["count"] <= 1
        stats.loc[mask_single, f"{col}_std"] = 0.0
        stats = stats.drop(columns=["count"])
        parts.append(stats)
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    return out.sort_index()


def plot_combined(
    segment_split_results: Path,
    video_label_results: Path,
    out_pdf: Path,
) -> Path:
    ss_rows: list[dict] = []
    vl_rows: list[dict] = []
    _collect_segment_split(ss_rows, segment_split_results)
    _collect_video_label(vl_rows, video_label_results)

    df_ss = pd.DataFrame(ss_rows)
    df_vl = pd.DataFrame(vl_rows)
    if df_ss.empty and df_vl.empty:
        raise RuntimeError(
            f"No usable JSON rows under {segment_split_results} or {video_label_results}."
        )

    if not df_ss.empty:
        df_ss["duration_min"] = df_ss["duration_sec"] / 60.0
    if not df_vl.empty:
        df_vl["duration_min"] = df_vl["duration_sec"] / 60.0

    agg_ss = _agg_mean_std(df_ss, ["segment_execute_sec", "split_execute_sec"])
    agg_vl = _agg_mean_std(df_vl, ["label_detection_sec", "ocr_sec", "speech_sec"])

    series_specs: list[tuple[str, str, str, str]] = []
    if not agg_ss.empty:
        series_specs.extend(
            [
                ("segment_execute_sec_mean", "segment_execute_sec_std", "Shot Detection", "#1f77b4"),
                ("split_execute_sec_mean", "split_execute_sec_std", "Video Split", "#ff7f0e"),
            ]
        )
    if not agg_vl.empty:
        series_specs.extend(
            [
                ("label_detection_sec_mean", "label_detection_sec_std", "Label detection", "#2ca02c"),
                ("ocr_sec_mean", "ocr_sec_std", "OCR", "#d62728"),
                ("speech_sec_mean", "speech_sec_std", "Speech transcription", "#9467bd"),
            ]
        )

    lw = 1.1
    x_ticks = [0, 5, 10, 15, 20, 25, 30]

    fig, ax = plt.subplots(figsize=FIG_INCHES, constrained_layout=True)

    for mean_col, std_col, label, color in series_specs:
        if mean_col.startswith("segment") or mean_col.startswith("split"):
            frame = agg_ss
        else:
            frame = agg_vl
        if frame.empty or mean_col not in frame.columns:
            continue
        sub = frame[[mean_col, std_col]].dropna(subset=[mean_col])
        if sub.empty:
            continue
        x = sub.index.to_numpy()
        y = sub[mean_col].to_numpy(dtype=float)
        yerr = np.nan_to_num(sub[std_col].to_numpy(dtype=float), nan=0.0)
        yerr = np.maximum(yerr, 0.0)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=color,
            capsize=3,
            capthick=lw,
            elinewidth=lw,
            markersize=4,
            linewidth=lw,
            label=label,
            alpha=0.92,
        )


    ax.set_ylim(0.0, 800.0)
    ax.set_yticks(np.arange(0.0, 1000.0, 200.0))

    ax.set_xlim(0.0, 32.0)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Video duration (min)")
    ax.set_ylabel("Execution time (s)")
    ax.grid(True, alpha=0.35)
    ax.legend(
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX,
        bbox_transform=ax.transAxes,
        framealpha=0.95,
        prop=fm.FontProperties(family=TNR, size=LEGEND_FONTSIZE),
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    path = save_pdf(fig, out_pdf)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-split-results",
        type=Path,
        default=SCRIPT_DIR / "segment_split_measurement" / "results",
        help="Directory containing sweep_*_json folders from segment_split_measurement.",
    )
    parser.add_argument(
        "--video-label-results",
        type=Path,
        default=SCRIPT_DIR / "video_label_ocr_speech" / "results",
        help="Directory containing sweep_*_json folders from video_label_ocr_speech.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "combined_latency_vs_video_duration.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the PDF after saving (overrides OPEN_OUTPUT_PDF).",
    )
    args = parser.parse_args()

    configure_figure()

    try:
        path = plot_combined(args.segment_split_results, args.video_label_results, args.output)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    print("Wrote:", path)

    if (
        OPEN_OUTPUT_PDF
        and not args.no_open
        and os.environ.get("SKYBRIDGE_NO_PLOT", "").strip().lower()
        not in ("1", "true", "yes")
    ):
        open_output_path(path)


if __name__ == "__main__":
    main()
