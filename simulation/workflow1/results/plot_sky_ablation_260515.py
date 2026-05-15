#!/usr/bin/env python3
"""基于 sky_ablation_260515.csv 绘制三种方法对比：墙钟时间（min）与峰值内存（GB）。

参考 plot_ablation_Q100_grid.py 的线型与图例风格；每张图内含两个子图：
  - vary_S_fixed_Q：横轴为 S（Q 固定为 50）
  - vary_Q_fixed_S：横轴为 Q（S 固定为 50）
"""

from __future__ import annotations

import platform
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

CSV_PATH = Path(__file__).resolve().parent / "sky_ablation_260515.csv"
OUT_DIR = Path(__file__).resolve().parent

VARIANT_LABELS = {
    "full": "full（完整分解）",
    "no_warm_start": "no_warm_start（无热启动）",
    "direct_milp": "direct_milp（直接 MILP）",
}


def _setup_cjk_font() -> None:
    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
        return
    _noto = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if _noto.exists():
        font_manager.fontManager.addfont(str(_noto))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_noto)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _load_df() -> pd.DataFrame:
    raw = CSV_PATH.read_text(encoding="utf-8")
    raw = raw.replace("vary_Q_fixedvary_Q_fixed_S", "vary_Q_fixed_S")
    df = pd.read_csv(StringIO(raw), on_bad_lines="skip")
    need = {"sweep", "Q", "S", "variant", "wall_clock_sec", "peak_memory_bytes"}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少列: {sorted(missing_cols)}")
    df = df.dropna(subset=["sweep", "variant", "wall_clock_sec", "peak_memory_bytes"])
    for c in ("Q", "S", "wall_clock_sec", "peak_memory_bytes"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Q", "S"])
    return df


def _plot_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_convert,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    for variant, style in [
        ("full", "-o"),
        ("no_warm_start", "-s"),
        ("direct_milp", "-^"),
    ]:
        g = sub[sub["variant"] == variant].dropna(subset=[x_col, y_col]).sort_values(x_col)
        if g.empty:
            continue
        ys = g[y_col].astype(float).map(y_convert)
        ax.plot(g[x_col], ys, style, label=VARIANT_LABELS.get(variant, variant), markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)


def main() -> None:
    _setup_cjk_font()
    df = _load_df()

    sub_s = df[df["sweep"] == "vary_S_fixed_Q"].copy()
    sub_q = df[df["sweep"] == "vary_Q_fixed_S"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _plot_panel(
        axes[0],
        sub_s,
        "S",
        "wall_clock_sec",
        lambda x: float(x) / 60.0,
        xlabel="S（固定 Q = 50）",
        ylabel="求解时间（min）",
        title="固定 Q：求解时间 vs S",
    )
    _plot_panel(
        axes[1],
        sub_q,
        "Q",
        "wall_clock_sec",
        lambda x: float(x) / 60.0,
        xlabel="Q（固定 S = 50）",
        ylabel="求解时间（min）",
        title="固定 S：求解时间 vs Q",
    )
    fig.suptitle("三种方法求解时间对比（sky_ablation_260515）", fontsize=12, y=1.02)
    fig.tight_layout()
    p1 = OUT_DIR / "sky_ablation_260515_wall_time_min.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_panel(
        axes[0],
        sub_s,
        "S",
        "peak_memory_bytes",
        lambda x: float(x) / (1024**3),
        xlabel="S（固定 Q = 50）",
        ylabel="峰值内存（GB）",
        title="固定 Q：峰值内存 vs S",
    )
    _plot_panel(
        axes[1],
        sub_q,
        "Q",
        "peak_memory_bytes",
        lambda x: float(x) / (1024**3),
        xlabel="Q（固定 S = 50）",
        ylabel="峰值内存（GB）",
        title="固定 S：峰值内存 vs Q",
    )
    fig.suptitle("三种方法峰值内存对比（sky_ablation_260515）", fontsize=12, y=1.02)
    fig.tight_layout()
    p2 = OUT_DIR / "sky_ablation_260515_peak_memory_gb.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
