"""
从 ``measurement/network_measurement/results`` 下各场景的 RTT / iperf CSV
解析并按 ``timestamp_utc`` 与 plot_network_24h 相同的规则（首 24h、合法样本）
对齐。主要入口：:func:`load_aligned_network_trace` 与 :class:`MeasuredNetworkCursor`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Literal, Sequence

import pandas as pd

MeasuredNetworkScenario = Literal[
    "inter_region",
    "cross_provider_same_region",
    "cross_provider_cross_region",
]

DEFAULT_HOURS: Final[float] = 24.0

_SCENARIO_SUBDIR: dict[str, str] = {
    "inter_region": "inter_region",
    "cross_provider_same_region": "cross_provider_same_region",
    "cross_provider_cross_region": "cross_provider_cross_region",
}


def default_measurement_results_dir() -> Path:
    """``<repo>/src/measurement/network_measurement/results``。"""
    return Path(__file__).resolve().parents[1] / "measurement" / "network_measurement" / "results"


def _first_csv(subdir: str, kind: str, results_dir: Path) -> Path:
    d = results_dir / subdir
    pat = f"{kind}_*.csv"
    matches = sorted(d.glob(pat))
    if not matches:
        raise FileNotFoundError(f"未找到 CSV: {d / pat}")
    return matches[0]


def _load_rtt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: c in ("timestamp_utc", "rtt_avg_ms", "ping_ok"),
    )
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.loc[df["ping_ok"] == 1, ["t", "rtt_avg_ms"]].sort_values("t")
    return df.reset_index(drop=True)


def _load_bw(path: Path) -> pd.DataFrame:
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
    df["bandwidth_mbps"] = (
        df["bw_out_mbits_per_sec"] + df["bw_in_mbits_per_sec"]
    ) / 2.0
    df = df[["t", "bandwidth_mbps"]].sort_values("t")
    return df.reset_index(drop=True)


def _clip_to_hours(
    df: pd.DataFrame, t0: pd.Timestamp, hours: float
) -> pd.DataFrame:
    t1 = t0 + pd.Timedelta(hours=hours)
    m = (df["t"] >= t0) & (df["t"] <= t1)
    return df.loc[m].reset_index(drop=True)


def _prepare_24h(
    rtt: pd.DataFrame, bw: pd.DataFrame, hours: float
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    if rtt.empty or bw.empty:
        raise ValueError("过滤后的 RTT 或带宽表为空，请检查 CSV 与 ping_ok / iperf_ok。")
    t0 = min(rtt["t"].iloc[0], bw["t"].iloc[0])
    r_p = _clip_to_hours(rtt, t0, hours)
    b_p = _clip_to_hours(bw, t0, hours)
    if r_p.empty or b_p.empty:
        raise ValueError(f"在 [t0, t0+{hours}h) 内无有效样本。")
    return t0, r_p, b_p


@dataclass(frozen=True)
class AlignedNetworkTrace:
    """与 ``load_aligned_network_trace`` 返回的对齐时序。"""

    scenario: str
    rtt_ms: tuple[float, ...]
    bandwidth_mbps: tuple[float, ...]
    t0_iso: str
    hours_window: float
    n_samples: int
    rtt_csv: str
    bandwidth_csv: str

    def as_pairs(self) -> list[tuple[float, float]]:
        return list(zip(self.rtt_ms, self.bandwidth_mbps, strict=True))

    def iter_pairs(self) -> Iterator[tuple[float, float]]:
        return zip(self.rtt_ms, self.bandwidth_mbps, strict=True)


def load_aligned_network_trace(
    scenario: MeasuredNetworkScenario = "inter_region",
    *,
    results_dir: Path | str | None = None,
    hours: float = DEFAULT_HOURS,
) -> AlignedNetworkTrace:
    """
    从指定场景目录下**各取时间排序后的第一个** ``rtt_*.csv`` 与 ``bandwidth_*.csv``，
    在 ``hours`` 小时窗口内按时间戳内连接，得到成对的 ``(rtt_ms, bandwidth_mbps)``。
    """
    rdir = Path(results_dir) if results_dir else default_measurement_results_dir()
    sub = _SCENARIO_SUBDIR[scenario]
    rtt_path = _first_csv(sub, "rtt", rdir)
    bw_path = _first_csv(sub, "bandwidth", rdir)
    rtt_df = _load_rtt(rtt_path)
    bw_df = _load_bw(bw_path)
    t0, rtt_p, bw_p = _prepare_24h(rtt_df, bw_df, hours)
    merged = rtt_p.merge(bw_p, on="t", how="inner")
    if merged.empty:
        raise ValueError(f"{scenario}: RTT 与带宽按时间内连接后为空。")
    rtt_ms = tuple(float(x) for x in merged["rtt_avg_ms"].tolist())
    bw_m = tuple(float(x) for x in merged["bandwidth_mbps"].tolist())
    t0s = t0.isoformat()
    return AlignedNetworkTrace(
        scenario=scenario,
        rtt_ms=rtt_ms,
        bandwidth_mbps=bw_m,
        t0_iso=t0s,
        hours_window=hours,
        n_samples=len(merged),
        rtt_csv=str(rtt_path.resolve()),
        bandwidth_csv=str(bw_path.resolve()),
    )


class MeasuredNetworkCursor:
    """
    按时间顺序遍历对齐的 ``(rtt, bandwidth)`` 对。

    每步必须依次调用 :meth:`next_rtt` 与 :meth:`next_bandwidth`；仅调用
    :meth:`next_rtt` 而不接着调用 :meth:`next_bandwidth` 会阻止下一步。
    耗尽时行为由 ``on_exhaust`` 决定（循环或抛错）。
    """

    def __init__(
        self,
        pairs: Sequence[tuple[float, float]],
        *,
        on_exhaust: Literal["cycle", "raise"] = "cycle",
        start_index: int = 0,
    ) -> None:
        if not pairs:
            raise ValueError("空序列，无法构建游标。")
        self._pairs = list(pairs)
        n = len(self._pairs)
        self._i = int(start_index) % n if n else 0
        self._step_open = False
        self._on_exhaust = on_exhaust

    @classmethod
    def from_trace(
        cls,
        trace: AlignedNetworkTrace,
        *,
        on_exhaust: Literal["cycle", "raise"] = "cycle",
        start_index: int = 0,
    ) -> "MeasuredNetworkCursor":
        return cls(
            trace.as_pairs(), on_exhaust=on_exhaust, start_index=start_index
        )

    @property
    def n(self) -> int:
        return len(self._pairs)

    def _pair_at(self) -> tuple[float, float]:
        if self._i >= len(self._pairs):
            if self._on_exhaust == "cycle":
                self._i = 0
            else:
                raise ValueError("测量序列已耗尽，且 on_exhaust='raise'。")
        return self._pairs[self._i]

    def next_rtt(self) -> float:
        if self._step_open:
            raise RuntimeError(
                "上一步尚未调用 next_bandwidth()；请先成对取完 (rtt, bandwidth)。"
            )
        r, _b = self._pair_at()
        self._step_open = True
        return r

    def next_bandwidth(self) -> float:
        if not self._step_open:
            raise RuntimeError("应先调用 next_rtt()，再调用 next_bandwidth()。")
        _r, b = self._pair_at()
        self._step_open = False
        self._i += 1
        if self._i >= len(self._pairs) and self._on_exhaust == "raise":
            # 下一步 next_rtt 时 StopIteration
            pass
        return b

    def next_pair(self) -> tuple[float, float]:
        """同一步内依次取 (rtt_ms, bandwidth_mbps) 并前进索引，等价于 next_rtt + next_bandwidth。"""
        return (self.next_rtt(), self.next_bandwidth())

    def reset(self) -> None:
        self._i = 0
        self._step_open = False
