"""
仿真通信端点与三类**实测**场景对齐（与 ``measurement/network_measurement/results`` 下子目录一一对应）：

- 同云跨地域 → ``inter_region``；
- 跨云同地域 → ``cross_provider_same_region``；
- 跨云跨地域 → ``cross_provider_cross_region``。

任一有向边 ``(a, b)``（``a, b`` 为 ``_node_to_comm_endpoint`` 形式，如 ``local``、``GCP_us-west1``）
使用**独立**的 :class:`MeasuredNetworkCursor`：自该场景 24h 对齐序列中按时间推进；
创建游标时在序列上随机起播偏移，使不同边互不共享「当前时刻」下标。本地相关的边
（任一端为 ``local``）无单独实测，复用 **跨云同地域** 表作为近似；可用
``local_edge_scenario=`` 覆盖。

分类逻辑见 :mod:`link_kind`；
不依赖某条真实云际链路的物理地域是否可测，仅按**厂商 + 归一化地域**落入三类之一。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from link_kind import NetworkLinkKind, classify_network_link

from measured_network_trace import (
    AlignedNetworkTrace,
    DEFAULT_HOURS,
    MeasuredNetworkCursor,
    MeasuredNetworkScenario,
    load_aligned_network_trace,
)

# 与 local 相连时：测量集中无该路径，用「跨云同地域」代替（短路径近似）
_DEFAULT_LOCAL_EDGE_SCENARIO: MeasuredNetworkScenario = "cross_provider_same_region"


def classify_endpoints_to_measured_scenario(
    a: str, b: str, *, local_edge_scenario: MeasuredNetworkScenario | None = None
) -> MeasuredNetworkScenario:
    """
    将通信端点对映射到与 CSV 子目录同名的 MeasuredNetworkScenario。

    若任一端为 ``local``，返回 ``local_edge_scenario`` 或
    默认的 ``_DEFAULT_LOCAL_EDGE_SCENARIO``（跨云同地域表）。
    """
    if a == b:
        raise ValueError("端点相同，无网络传输，不应走实测分类。")
    if a == "local" or b == "local":
        return local_edge_scenario or _DEFAULT_LOCAL_EDGE_SCENARIO
    k = classify_network_link(a, b)
    if k == NetworkLinkKind.INTRA_PROVIDER_CROSS_REGION:
        return "inter_region"
    if k == NetworkLinkKind.INTER_PROVIDER_SAME_REGION:
        return "cross_provider_same_region"
    if k == NetworkLinkKind.INTER_PROVIDER_CROSS_REGION:
        return "cross_provider_cross_region"
    # 理论上传输场景下不应出现同云同地域（会对应 a==b endpoint）
    return "inter_region"


@dataclass
class _ScenarioMeans:
    rtt_ms: float
    bandwidth_mbps: float


class PerEdgeMeasuredNetwork:
    """
    为每条有向边维护独立时序游标；``calculate`` 每触发一次该边的传输，则该边游标前进一步。

    - **deterministic=True**：对应该场景时序的样本均值，不推进游标。
    - **probe_random_pair**（给算法侧MonteCarlo探测）：不推进主游标，从该边对应场景的时序
      中 **均匀有放回** 取一对点。
    """

    def __init__(
        self,
        *,
        results_dir: Path | str | None = None,
        hours: float = DEFAULT_HOURS,
        on_exhaust: Literal["cycle", "raise"] = "cycle",
        per_edge_start_seed: int = 0,
        local_edge_scenario: MeasuredNetworkScenario | None = None,
    ) -> None:
        self._results_dir = results_dir
        self._hours = float(hours)
        self._on_exhaust = on_exhaust
        self._local_edge_scenario = local_edge_scenario
        self._rng = random.Random(int(per_edge_start_seed))
        self._traces: dict[str, AlignedNetworkTrace] = {
            s: load_aligned_network_trace(
                s,  # type: ignore[arg-type]
                results_dir=results_dir,
                hours=self._hours,
            )
            for s in (
                "inter_region",
                "cross_provider_same_region",
                "cross_provider_cross_region",
            )
        }
        self._means: dict[str, _ScenarioMeans] = {}
        for s, t in self._traces.items():
            n = len(t.rtt_ms)
            if n == 0:
                raise ValueError(f"场景 {s} 无时序点。")
            self._means[s] = _ScenarioMeans(
                rtt_ms=sum(t.rtt_ms) / n,
                bandwidth_mbps=sum(t.bandwidth_mbps) / n,
            )
        self._cursors: dict[tuple[str, str], MeasuredNetworkCursor] = {}

    def _classify(self, a: str, b: str) -> MeasuredNetworkScenario:
        return classify_endpoints_to_measured_scenario(
            a, b, local_edge_scenario=self._local_edge_scenario
        )

    def get_rtt_bw_mbps(
        self, a: str, b: str, *, deterministic: bool
    ) -> tuple[float, float]:
        if a == b:
            raise ValueError("端点相同，无需 RTT/带宽。")
        s = self._classify(a, b)
        if deterministic:
            m = self._means[s]
            return (m.rtt_ms, m.bandwidth_mbps)
        key = (a, b)
        if key not in self._cursors:
            tr = self._traces[s]
            n = len(tr.rtt_ms)
            off = self._rng.randrange(0, n) if n else 0
            self._cursors[key] = MeasuredNetworkCursor.from_trace(
                tr, on_exhaust=self._on_exhaust, start_index=off
            )
        rtt_ms, bw = self._cursors[key].next_pair()
        if bw <= 0 and not math.isinf(bw):
            return rtt_ms, 1e-6
        return (rtt_ms, bw)

    def probe_random_pair(
        self, a: str, b: str, rng: random.Random
    ) -> tuple[float, float]:
        """不推进主游标；从该边所属场景的时序中随机取一瞬时值。"""
        if a == b:
            return (0.0, float("inf"))
        s = self._classify(a, b)
        tr = self._traces[s]
        n = len(tr.rtt_ms)
        if n == 0:
            raise ValueError("empty trace")
        i = rng.randrange(0, n)
        return (float(tr.rtt_ms[i]), float(tr.bandwidth_mbps[i]))


@dataclass(frozen=True, slots=True)
class MeasuredNetworkOptions:
    results_dir: Path | str | None = None
    hours: float = DEFAULT_HOURS
    on_exhaust: Literal["cycle", "raise"] = "cycle"
    per_edge_start_seed: int = 0
    local_edge_scenario: MeasuredNetworkScenario | None = None
