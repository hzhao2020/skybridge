"""
基于 ``segment_split_measurement`` sweep CSV 的 segment / split **计算段** 延迟（秒）。

将仿真中的 ``video_size_MB`` 映射为视频时长（分钟）：``T = video_size_MB / video_mb_per_min``，
并裁剪到 ``[duration_min, duration_max]``（默认对应测量 2–30 分钟）。对
``node_segment_execute_sec`` 与 ``node_split_execute_http_observed_sec`` 分别作
**对时长的一元线性回归**，从训练残差中有放回抽取随机项，使样本落在测量波动范围内；
``deterministic=True`` 时仅返回回归预测值（非负）。

I/O 仍用 ``SegSplExecTimeParam.io_s_per_MB``，不在此 CSV 中重复建模。
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_CSV_NAME = "sweep_20260415_135645Z.csv"
DEFAULT_VIDEO_MB_PER_MIN = 8.0
DURATION_MIN_MIN = 2.0
DURATION_MAX_MIN = 30.0


@dataclass(frozen=True, slots=True)
class _LinModel:
    """y ≈ b0 + b1 * t_minutes, 与训练集残差列表（用于有放回加噪）。"""

    b0: float
    b1: float
    residuals: tuple[float, ...]

    def pred(self, t_min: float) -> float:
        return self.b0 + self.b1 * t_min

    def sample(self, t_min: float, rng: random.Random, *, deterministic: bool) -> float:
        mu = self.pred(t_min)
        if deterministic:
            return max(0.0, float(mu))
        if not self.residuals:
            return max(0.0, float(mu))
        r = float(rng.choice(self.residuals))
        return max(0.0, mu + r)


def _ols_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("OLS 需要至少 2 个数据点。")
    mx = sum(x) / n
    my = sum(y) / n
    var_x = sum((xi - mx) * (xi - mx) for xi in x)
    if var_x < 1e-18:
        b1 = 0.0
        b0 = my
    else:
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        b1 = cov / var_x
        b0 = my - b1 * mx
    return b0, b1


def _residuals(x: Sequence[float], y: Sequence[float], b0: float, b1: float) -> tuple[float, ...]:
    return tuple(y[i] - (b0 + b1 * x[i]) for i in range(len(x)))


def default_sweep_csv_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "measurement"
        / "segment_split_measurement"
        / "results"
        / DEFAULT_CSV_NAME
    )


class SegmentSplitSweepModel:
    """
    从成功试验行拟合 segment/split 计算时间；:meth:`video_duration_min` 做 MB→分钟映射。
    """

    def __init__(
        self,
        csv_path: Path | str | None = None,
        *,
        video_mb_per_minute: float = DEFAULT_VIDEO_MB_PER_MIN,
        duration_min_minutes: float = DURATION_MIN_MIN,
        duration_max_minutes: float = DURATION_MAX_MIN,
        rng: random.Random | None = None,
    ) -> None:
        p = Path(csv_path) if csv_path else default_sweep_csv_path()
        if not p.is_file():
            raise FileNotFoundError(f"找不到 sweep CSV: {p.resolve()}")
        t_list: list[float] = []
        seg_list: list[float] = []
        spl_list: list[float] = []
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("success", "")).lower() not in ("true", "1", "yes"):
                    continue
                try:
                    dmin = float(row["duration_minutes"])
                    seg = float(row["node_segment_execute_sec"])
                    spl = float(row["node_split_execute_http_observed_sec"])
                except (KeyError, ValueError):
                    continue
                t_list.append(dmin)
                seg_list.append(seg)
                spl_list.append(spl)
        if len(t_list) < 3:
            raise ValueError(f"{p} 中成功样本过少，无法拟合。")
        b0s, b1s = _ols_line(t_list, seg_list)
        b0p, b1p = _ols_line(t_list, spl_list)
        self._segment = _LinModel(
            b0=b0s, b1=b1s, residuals=_residuals(t_list, seg_list, b0s, b1s)
        )
        self._split = _LinModel(
            b0=b0p, b1=b1p, residuals=_residuals(t_list, spl_list, b0p, b1p)
        )
        self._video_mb_per_min = float(video_mb_per_minute)
        self._d_lo = float(duration_min_minutes)
        self._d_hi = float(duration_max_minutes)
        self._rng = rng if rng is not None else random.Random()

    def set_rng(self, rng: random.Random) -> None:
        self._rng = rng

    def video_duration_min(self, video_size_mb: float) -> float:
        if self._video_mb_per_min <= 0:
            raise ValueError("video_mb_per_minute 必须 > 0")
        t = float(video_size_mb) / self._video_mb_per_min
        return max(self._d_lo, min(self._d_hi, t))

    def sample_segment_comp_sec(
        self, video_size_mb: float, *, deterministic: bool
    ) -> float:
        t = self.video_duration_min(video_size_mb)
        return self._segment.sample(t, self._rng, deterministic=deterministic)

    def sample_split_comp_sec(
        self, video_size_mb: float, *, deterministic: bool
    ) -> float:
        t = self.video_duration_min(video_size_mb)
        return self._split.sample(t, self._rng, deterministic=deterministic)


def try_load_sweep_from_config() -> "SegmentSplitSweepModel | None":
    """
    读 ``param.yaml`` 的 ``segment_split_sweep``；未启用或缺文件时返回 ``None``。
    """
    from distribution import config as _cfg  # 延迟避免循环

    sc = _cfg.get("segment_split_sweep")
    if not isinstance(sc, dict) or not sc.get("enabled", False):
        return None
    src_root = Path(__file__).resolve().parents[1]
    rel = str(
        sc.get(
            "csv_rel",
            "measurement/segment_split_measurement/results/sweep_20260415_135645Z.csv",
        )
    )
    path = (src_root / rel).resolve()
    try:
        return SegmentSplitSweepModel(
            path,
            video_mb_per_minute=float(sc.get("video_mb_per_minute", DEFAULT_VIDEO_MB_PER_MIN)),
            duration_min_minutes=float(sc.get("duration_min_minutes", DURATION_MIN_MIN)),
            duration_max_minutes=float(sc.get("duration_max_minutes", DURATION_MAX_MIN)),
        )
    except (OSError, FileNotFoundError, ValueError):
        return None
