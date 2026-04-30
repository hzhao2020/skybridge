#!/usr/bin/env python3
"""Plot wall time (min) and peak memory (GB) for Q=100 ablation rows from ablation_Q_S_grid.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# NotoSansCJK-Regular.ttc 含中日韩字形；显式 addfont 以便 rcParams 生效
_NOTO_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_NOTO_CJK)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=_NOTO_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = Path(__file__).resolve().parent / "ablation_Q_S_grid.csv"
OUT_DIR = Path(__file__).resolve().parent

VARIANT_LABELS = {
    "full": "full（完整分解）",
    "no_warm_start": "no_warm_start（无热启动）",
    "direct_milp": "direct_milp（直接 MILP）",
}


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    sub = df[(df["sweep"] == "fixed_Q_S") & (df["Q"] == 100) & (df["S"].isin([20, 40, 60, 80, 100]))].copy()
    sub = sub.sort_values(["S", "variant"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, style in [
        ("full", "-o"),
        ("no_warm_start", "-s"),
        ("direct_milp", "-^"),
    ]:
        g = sub[sub["variant"] == variant].sort_values("S")
        mins = g["wall_clock_sec"].astype(float) / 60.0
        ax.plot(g["S"], mins, style, label=VARIANT_LABELS.get(variant, variant), markersize=7)
    ax.set_xlabel("S（固定 Q = 100）")
    ax.set_ylabel("求解时间（min）")
    ax.set_title("三种方法的求解时间随 S 变化")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p1 = OUT_DIR / "ablation_Q100_wall_time_min.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, style in [
        ("full", "-o"),
        ("no_warm_start", "-s"),
        ("direct_milp", "-^"),
    ]:
        g = sub[sub["variant"] == variant].sort_values("S")
        gb = g["peak_memory_bytes"].astype(float) / (1024**3)
        ax.plot(g["S"], gb, style, label=VARIANT_LABELS.get(variant, variant), markersize=7)
    ax.set_xlabel("S（固定 Q = 100）")
    ax.set_ylabel("峰值内存（GB）")
    ax.set_title("三种方法的峰值内存随 S 变化")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p2 = OUT_DIR / "ablation_Q100_peak_memory_gb.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
