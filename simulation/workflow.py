"""
根据 config 拓扑与 ``DistributionParameters`` 构造全量节点；
``Workflow`` 绑定一次 ``sample()``，在 ``calculate`` 中串联 segment→split→caption→query，
并在每次 ``calculate`` 中计算链路上的传输时延与出站费。

**网络传输（默认）**：使用 ``measured_workflow_network.PerEdgeMeasuredNetwork``，按通信端点将链路归入
**三类实测场景**（同云跨地域 / 跨云同地域 / 跨云跨地域，含 ``local`` 边用跨云同地域表替代），
每条有向边有**独立**的测量时序游标。可 ``use_measured_network=False`` 恢复为对 ``params.edge_rtt`` /
``params.edge_bw`` 矩阵的 LogNormal 采样（旧行为）。

``calculate`` 中由 **RTT/BW 实现** 提供本轮各边的具体 RTT(ms)/BW(Mbps)，再与计算/LLM/存储一起汇总
cost / latency / utility。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from param import (
    CLOUD_PROVIDERS,
    CLOUD_REGIONS,
    build_llm_node_name,
    iter_cloud_llm_deployments,
)
from distribution import (
    BW,
    DistributionParameters,
    LogNormalDistribution,
    RTT,
    config as simulation_config,
    sample,
)
from measured_workflow_network import (
    MeasuredNetworkOptions,
    PerEdgeMeasuredNetwork,
)
from segment_split_sweep_model import SegmentSplitSweepModel, try_load_sweep_from_config
from nodes import (
    CaptionCloudNode,
    Edge,
    LocalNode,
    Node,
    QueryCloudNode,
    SegmentNode,
    SplitNode,
    _node_to_comm_endpoint,
    caption_output_tokens_for_query,
)


@dataclass(frozen=True, slots=True)
class WorkflowNodes:
    """按 operation 分类的节点表，key 为节点名（与 distribution 中命名一致）。"""

    segment: dict[str, SegmentNode]
    split: dict[str, SplitNode]
    caption: dict[str, CaptionCloudNode]
    query: dict[str, QueryCloudNode]


@dataclass(frozen=True, slots=True)
class WorkflowBreakdown:
    """
    单次 ``calculate`` 的 cost / latency 分项（与 ``Workflow.calculate`` 内求和顺序一致）。

    金额单位为 USD，时间单位为秒；``mb*`` 为流水线各阶段数据量（MB）。
    """

    # --- Cost (USD) ---
    exec_segment: float
    exec_split: float
    llm_caption: float
    llm_query: float
    egress_local_segment: float
    egress_segment_split: float
    egress_split_caption: float
    egress_to_query: float
    egress_last_to_local: float
    storage: float
    # --- Latency (s) ---
    lat_local_segment: float
    lat_segment_compute: float
    lat_segment_split: float
    lat_split_compute: float
    lat_split_caption: float
    lat_caption_compute: float
    lat_to_query: float
    lat_query_compute: float
    lat_last_to_local: float
    # --- Data sizes (MB) ---
    mb0: float
    mb1: float
    mb2: float
    mb3: float
    mb4: float

    def sum_cost(self) -> float:
        return (
            self.exec_segment
            + self.exec_split
            + self.llm_caption
            + self.llm_query
            + self.egress_local_segment
            + self.egress_segment_split
            + self.egress_split_caption
            + self.egress_to_query
            + self.egress_last_to_local
            + self.storage
        )

    def sum_latency(self) -> float:
        return (
            self.lat_local_segment
            + self.lat_segment_compute
            + self.lat_segment_split
            + self.lat_split_compute
            + self.lat_split_caption
            + self.lat_caption_compute
            + self.lat_to_query
            + self.lat_query_compute
            + self.lat_last_to_local
        )

    def format_lines(self) -> list[str]:
        def row(label: str, value: float, *, unit: str) -> str:
            return f"  {label:<42} {value:>18.6f} {unit}"

        return [
            "Cost (USD)",
            row("segment execution", self.exec_segment, unit=""),
            row("split execution", self.exec_split, unit=""),
            row("caption LLM (tokens)", self.llm_caption, unit=""),
            row("query LLM (tokens)", self.llm_query, unit=""),
            row("egress local → segment (mb0)", self.egress_local_segment, unit=""),
            row("egress segment → split (mb1)", self.egress_segment_split, unit=""),
            row("egress split → caption (mb2)", self.egress_split_caption, unit=""),
            row("egress → query (mb3)", self.egress_to_query, unit=""),
            row("egress last → local (mb4)", self.egress_last_to_local, unit=""),
            row("storage (cloud)", self.storage, unit=""),
            row("── total cost", self.sum_cost(), unit=""),
            "",
            "Latency (s)",
            row("transfer local → segment", self.lat_local_segment, unit=""),
            row("segment compute", self.lat_segment_compute, unit=""),
            row("transfer segment → split", self.lat_segment_split, unit=""),
            row("split compute", self.lat_split_compute, unit=""),
            row("transfer split → caption", self.lat_split_caption, unit=""),
            row("caption LLM", self.lat_caption_compute, unit=""),
            row("transfer → query (mb3)", self.lat_to_query, unit=""),
            row("query LLM", self.lat_query_compute, unit=""),
            row("transfer last → local (mb4)", self.lat_last_to_local, unit=""),
            row("── total latency", self.sum_latency(), unit=""),
            "",
            "Data sizes (MB)",
            f"  mb0={self.mb0:.6f}  mb1={self.mb1:.6f}  mb2={self.mb2:.6f}  mb3={self.mb3:.6f}  mb4={self.mb4:.6f}",
        ]

    def __str__(self) -> str:
        return "\n".join(self.format_lines())


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """单次流水线评估结果。"""

    cost: float
    latency: float
    utility: float
    breakdown: WorkflowBreakdown | None = None


@dataclass(frozen=True, slots=True)
class NodeSelection:
    segment: str
    split: str
    caption: str
    query: str


def _egress_usd_per_gb(params: DistributionParameters, src: Node, dst: Node) -> float:
    eps = params.egress_price["endpoints"]
    matrix = params.egress_price["matrix"]
    i = eps.index(_node_to_comm_endpoint(src))
    j = eps.index(_node_to_comm_endpoint(dst))
    return matrix[i][j]


def _lognormal_sample_from_moments(mean: float, std: float, rng: random.Random) -> float:
    """与 ``algos._lognormal_sample_from_moments`` 一致，避免 workflow 与 algos 循环导入。"""
    if mean <= 0:
        raise ValueError("LogNormal mean must be > 0")
    if std <= 1e-15:
        return float(mean)
    variance = std**2
    phi = math.sqrt(variance + mean**2)
    mu = math.log(mean**2 / phi)
    sigma = math.sqrt(math.log(phi**2 / mean**2))
    return rng.lognormvariate(mu, sigma)


def _utility_weighted(
    segment: SegmentNode,
    split: SplitNode,
    caption: CaptionCloudNode,
    query: QueryCloudNode,
) -> float:
    uw: Any = simulation_config.get("utility_weight") or {}
    return (
        float(uw.get("segment", 0.0)) * segment.utility
        + float(uw.get("split", 0.0)) * split.utility
        + float(uw.get("caption", 0.0)) * caption.utility
        + float(uw.get("query", 0.0)) * query.utility
    )


def build_workflow_nodes(
    params: DistributionParameters,
    *,
    seg_split_sweep: SegmentSplitSweepModel | None = None,
) -> WorkflowNodes:
    segment: dict[str, SegmentNode] = {}
    split: dict[str, SplitNode] = {}
    caption: dict[str, CaptionCloudNode] = {}
    query: dict[str, QueryCloudNode] = {}
    for p in CLOUD_PROVIDERS:
        for r in CLOUD_REGIONS[p]:
            pr = f"{p}_{r}"
            seg_name = f"{pr}_segment"
            spl_name = f"{pr}_split"
            segment[seg_name] = SegmentNode(
                name=seg_name,
                provider=p,
                region=r,
                utility=params.utility[seg_name],
                storage_price_per_gb_month=params.storage_price[seg_name],
                price_per_min=params.segment_price[seg_name],
                exec_time_param=params.seg_spl_exec_time_param[seg_name],
                seg_split_sweep=seg_split_sweep,
            )
            split[spl_name] = SplitNode(
                name=spl_name,
                provider=p,
                region=r,
                utility=params.utility[spl_name],
                storage_price_per_gb_month=params.storage_price[spl_name],
                price_per_min=params.split_price[spl_name],
                exec_time_param=params.seg_spl_exec_time_param[spl_name],
                seg_split_sweep=seg_split_sweep,
            )

    for p, r, model_key in iter_cloud_llm_deployments():
            pr = f"{p}_{r}"
            cap_name = build_llm_node_name(p, r, model_key, "caption")
            qry_name = build_llm_node_name(p, r, model_key, "query")
            cin, cout = params.llm_token_price[cap_name]
            caption[cap_name] = CaptionCloudNode(
                name=cap_name,
                provider=p,
                region=r,
                utility=params.utility[cap_name],
                storage_price_per_gb_month=params.storage_price[cap_name],
                llm_name=model_key,
                input_token_price=cin,
                output_token_price=cout,
                llm_latency_param=params.llm_latency_param[cap_name],
                caption_input_token_num_param=params.caption_input_token_num_param[cap_name],
                caption_output_token_num_param=params.caption_output_token_num_param[cap_name],
            )
            qin, qout = params.llm_token_price[qry_name]
            query[qry_name] = QueryCloudNode(
                name=qry_name,
                provider=p,
                region=r,
                utility=params.utility[qry_name],
                storage_price_per_gb_month=params.storage_price[qry_name],
                llm_name=model_key,
                input_token_price=qin,
                output_token_price=qout,
                llm_latency_param=params.llm_latency_param[qry_name],
                query_output_token_num_param=params.query_output_token_num_param[qry_name],
            )

    return WorkflowNodes(
        segment=segment,
        split=split,
        caption=caption,
        query=query,
    )


def build_workflow_nodes_from_sample() -> WorkflowNodes:
    return build_workflow_nodes(
        sample(), seg_split_sweep=try_load_sweep_from_config()
    )


class Workflow:
    """
    构造时 ``sample()`` 一次并初始化全部节点（除非传入 ``params``）。

    **链路 RTT/BW**：

    - 默认（``use_measured_network=True``）使用三类实测时序、每条有向边独立游标，见
      ``measured_workflow_network`` 模块说明。
    - 若 ``use_measured_network=False``：每条有向边对应 ``(mean, std)``，语义与
      ``distribution.sample_edge_rtt`` / ``sample_edge_bw`` 一致；在 **每次** ``calculate`` 开头
      用 LogNormal(mean, std) 各采样一次。矩阵来自 ``params`` 或 ``edge_rtt=`` / ``edge_bw=``。

    **segment/split 计算时间**：默认读 ``param.yaml`` 的 ``segment_split_sweep``；启用时用
    ``segment_split_sweep_model`` 的测量回归 + 残差采样。可 ``seg_split_sweep=`` 显式传入模型实例。
    """

    def __init__(
        self,
        params: DistributionParameters | None = None,
        *,
        edge_rtt: dict[tuple[str, str], RTT] | None = None,
        edge_bw: dict[tuple[str, str], BW] | None = None,
        use_measured_network: bool = True,
        measured: PerEdgeMeasuredNetwork | None = None,
        measured_options: MeasuredNetworkOptions | None = None,
        seg_split_sweep: SegmentSplitSweepModel | None = None,
    ) -> None:
        self.params = params if params is not None else sample()
        self._seg_split_sweep: SegmentSplitSweepModel | None
        if seg_split_sweep is not None:
            self._seg_split_sweep = seg_split_sweep
        else:
            self._seg_split_sweep = try_load_sweep_from_config()
        self.nodes = build_workflow_nodes(
            self.params, seg_split_sweep=self._seg_split_sweep
        )
        self._local = LocalNode()
        self._rtt_ms: dict[tuple[str, str], float] = {}
        self._bw_mbps: dict[tuple[str, str], float] = {}
        # 用于采样的分布参数矩阵（非本轮 realized 数值；未启用实测时生效）
        self._network_edge_rtt = edge_rtt if edge_rtt is not None else self.params.edge_rtt
        self._network_edge_bw = edge_bw if edge_bw is not None else self.params.edge_bw
        self._use_measured_network = bool(use_measured_network)
        if self._use_measured_network:
            self._per_edge_measured = measured
            if self._per_edge_measured is None:
                opts = measured_options or MeasuredNetworkOptions()
                self._per_edge_measured = PerEdgeMeasuredNetwork(
                    results_dir=opts.results_dir,
                    hours=opts.hours,
                    on_exhaust=opts.on_exhaust,
                    per_edge_start_seed=opts.per_edge_start_seed,
                    local_edge_scenario=opts.local_edge_scenario,
                )
        else:
            self._per_edge_measured = None

    def list_node_names(self) -> dict[str, list[str]]:
        """返回算法可见动作空间（仅节点名称，不暴露分布参数）。"""
        return {
            "segment": sorted(self.nodes.segment.keys()),
            "split": sorted(self.nodes.split.keys()),
            "caption": sorted(self.nodes.caption.keys()),
            "query": sorted(self.nodes.query.keys()),
        }

    def sample_observation(
        self,
        selection: NodeSelection,
        video_size_MB: float,
        *,
        include_breakdown: bool = False,
        deterministic: bool = False,
    ) -> WorkflowResult:
        """黑盒采样接口：按节点名称采样一次观测，不需要读取真实分布参数。"""
        return self.calculate(
            self.nodes.segment[selection.segment],
            self.nodes.split[selection.split],
            self.nodes.caption[selection.caption],
            self.nodes.query[selection.query],
            video_size_MB,
            include_breakdown=include_breakdown,
            deterministic=deterministic,
        )

    def _realize_network(self) -> None:
        """
        由 LogNormal 矩阵采样出**本轮**各边的具体数值（非实测模式）。

        实测模式下传输延迟在每条边被使用时从 ``PerEdgeMeasuredNetwork`` 现取，不在此预填全矩阵。
        """
        if self._per_edge_measured is not None:
            self._rtt_ms = {}
            self._bw_mbps = {}
            return
        self._rtt_ms = {}
        self._bw_mbps = {}
        det = getattr(self, "_deterministic_eval", False)
        for k, r in self._network_edge_rtt.items():
            if det:
                self._rtt_ms[k] = float(r.mean)
            else:
                self._rtt_ms[k] = LogNormalDistribution(float(r.mean), float(r.std)).sample()
        for k, w in self._network_edge_bw.items():
            if det:
                self._bw_mbps[k] = float(w.mean)
            else:
                self._bw_mbps[k] = LogNormalDistribution(float(w.mean), float(w.std)).sample()

    def _rtt_bw_objects(self, src: Node, dst: Node) -> tuple[RTT, BW]:
        a, b = _node_to_comm_endpoint(src), _node_to_comm_endpoint(dst)
        key = (a, b)
        r = RTT()
        w = BW()
        if self._per_edge_measured is not None:
            if a == b:
                r.mean = 0.0
                r.std = 0.0
                w.mean = float("inf")
                w.std = 0.0
            else:
                det = getattr(self, "_deterministic_eval", False)
                rtt_ms, bw = self._per_edge_measured.get_rtt_bw_mbps(
                    a, b, deterministic=det
                )
                r.mean = rtt_ms
                r.std = 0.0
                w.mean = bw
                w.std = 0.0
        else:
            r.mean = self._rtt_ms[key]
            r.std = 0.0
            w.mean = self._bw_mbps[key]
            w.std = 0.0
        return r, w

    def probe_transfer_latency_s(
        self,
        src_ep: str,
        dst_ep: str,
        data_mb: float,
        rng: random.Random,
    ) -> float:
        """
        对一条有向边做一次与 ``Edge.calculate_latency`` 同公式的时延（秒）估计。
        启用实测时从对应场景时序中 **随机有放回** 取一对 (RTT, BW)，不推进主游标；否则从
        ``params.edge_rtt/edge_bw`` 按对数正态各采一值。
        """
        if data_mb < 0:
            raise ValueError("data_mb must be >= 0")
        if src_ep == dst_ep:
            return 0.0
        if self._per_edge_measured is not None:
            rtt_ms, bw_mbps = self._per_edge_measured.probe_random_pair(
                src_ep, dst_ep, rng
            )
        else:
            k = (src_ep, dst_ep)
            er = self.params.edge_rtt[k]
            eb = self.params.edge_bw[k]
            rtt_ms = _lognormal_sample_from_moments(
                float(er.mean), float(er.std), rng
            )
            bw_mbps = _lognormal_sample_from_moments(
                float(eb.mean), float(eb.std), rng
            )
        if math.isinf(bw_mbps):
            return rtt_ms / 2000.0
        bw_mbps = float(bw_mbps)
        if bw_mbps <= 0:
            return float("inf")
        return rtt_ms / 2000.0 + data_mb * 8.0 / bw_mbps

    def _edge(self, src: Node, dst: Node) -> Edge:
        r, bw = self._rtt_bw_objects(src, dst)
        return Edge(src, dst, r, bw, _egress_usd_per_gb(self.params, src, dst))

    def _edge_latency_s(self, src: Node, dst: Node, data_size_MB: float) -> float:
        if _node_to_comm_endpoint(src) == _node_to_comm_endpoint(dst):
            return 0.0
        det = getattr(self, "_deterministic_eval", False)
        return self._edge(src, dst).calculate_latency(data_size_MB, deterministic=det)

    def _edge_egress_usd(self, src: Node, dst: Node, data_size_MB: float) -> float:
        if _node_to_comm_endpoint(src) == _node_to_comm_endpoint(dst):
            return 0.0
        return self._edge(src, dst).calculate_egress_cost(data_size_MB)

    def _egress_and_latency_from_local(self, dst: Node, data_MB: float) -> tuple[float, float]:
        """local → dst：出站 USD（矩阵中 src=local 时多为 0）与链路上传输延迟（秒）。"""
        return (
            self._edge_egress_usd(self._local, dst, data_MB),
            self._edge_latency_s(self._local, dst, data_MB),
        )

    def _egress_and_latency_to_local(self, src: Node, data_MB: float) -> tuple[float, float]:
        """src → local：云侧出站 USD + RTT/BW 传输延迟（终点为 ``local``，须占矩阵一条边）。"""
        return (
            self._edge_egress_usd(src, self._local, data_MB),
            self._edge_latency_s(src, self._local, data_MB),
        )

    def calculate_storage_cost(
        self,
        segment: SegmentNode,
        split: SplitNode,
        caption: CaptionCloudNode,
        query: QueryCloudNode,
        video_size_MB: float,
        *,
        ratio_segment: float,
        ratio_split: float,
        ratio_caption: float,
        ratio_query: float,
    ) -> float:
        """
        全云节点计费：segment 存 mb0、mb1；split 存 mb1、mb2；
        caption 存 mb2、mb3；query 存 mb3、mb4。
        """
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        mb0 = video_size_MB
        mb1 = mb0 * float(ratio_segment)
        mb2 = mb1 * float(ratio_split)
        mb3 = mb2 * float(ratio_caption)
        mb4 = mb3 * float(ratio_query)

        def store(node: Node, mb: float) -> float:
            if not hasattr(node, "calculate_storage_cost"):
                raise ValueError("all nodes must support storage cost in cloud-only mode")
            return node.calculate_storage_cost(mb)

        total = 0.0

        total += store(segment, mb0) + store(segment, mb1)
        total += store(split, mb1) + store(split, mb2)

        total += store(caption, mb2) + store(caption, mb3)
        total += store(query, mb3) + store(query, mb4)

        return total

    def _sample_data_conversion_ratios(self) -> tuple[float, float, float, float]:
        """每次 calculate 采样一次 data conversion ratios（segment/split/caption/query）。"""
        dc = self.params.data_conversion_ratio
        if getattr(self, "_deterministic_eval", False):
            return (
                float(dc["segment"].mean),
                float(dc["split"].mean),
                float(dc["caption"].mean),
                float(dc["query"].mean),
            )
        return (
            float(dc["segment"].sample()),
            float(dc["split"].sample()),
            float(dc["caption"].sample()),
            float(dc["query"].sample()),
        )

    def calculate(
        self,
        segment: SegmentNode,
        split: SplitNode,
        caption: CaptionCloudNode,
        query: QueryCloudNode,
        video_size_MB: float,
        *,
        verbose: bool = False,
        include_breakdown: bool = False,
        deterministic: bool = False,
    ) -> WorkflowResult:
        """
        流水线 **以 local 为起点、以 local 为终点**：上传 mb0 从 local→segment，最终结果 mb4 从
        最后一跳云节点→local；两段均计入传输延迟，出站费按 ``egress_price`` 矩阵
        ``(端点, local)`` 或 ``(local, 端点)`` 计费（与 ``Edge`` / ``_node_to_comm_endpoint`` 一致）。

        全云模式：mb2 经 split→caption，mb3 经 caption→query，最后 mb4 从 query→local。

        默认每次调用对环境随机量重新采样；``deterministic=True`` 时用各分布/链路的 **均值**（及
        Gamma 期望、LLM 无噪声等）求值，对应 Baseline DO 的简化环境。

        ``verbose=True`` 时向 stdout 打印 cost/latency 分项。

        ``include_breakdown=True`` 时在 ``WorkflowResult.breakdown`` 中填入 ``WorkflowBreakdown``：
        各阶段 cost/latency 分项、存储费以及 ``mb0``–``mb4`` 数据量，便于记录或离线分析；
        为 ``False`` 时仅返回 ``cost`` / ``latency`` / ``utility`` 三个标量（``breakdown`` 为 ``None``）。
        ``verbose=True`` 时也会附带 breakdown（与 ``include_breakdown=True`` 等价用于构造分项）。
        """
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")

        self._deterministic_eval = deterministic
        try:
            self._realize_network()

            ratio_segment, ratio_split, ratio_caption, ratio_query = (
                self._sample_data_conversion_ratios()
            )

            mb0 = video_size_MB
            mb1 = mb0 * float(ratio_segment)
            mb2 = mb1 * float(ratio_split)
            mb3 = mb2 * float(ratio_caption)
            mb4 = mb3 * float(ratio_query)

            cost_seg = segment.calculate_execution_cost(mb0, deterministic=deterministic)
            cost_spl = split.calculate_execution_cost(mb1, deterministic=deterministic)
            cost_cap = caption.calculate_token_cost(mb2, deterministic=deterministic)
            n_cap = caption_output_tokens_for_query(caption, mb2, deterministic=deterministic)
            cost_qry = query.calculate_token_cost(
                mb2, input_tokens=n_cap, deterministic=deterministic
            )

            pre_q: Node = caption

            e0, lat_from_local_0 = self._egress_and_latency_from_local(segment, mb0)
            e1 = self._edge_egress_usd(segment, split, mb1)
            e2 = self._edge_egress_usd(split, caption, mb2)
            e3 = self._edge_egress_usd(pre_q, query, mb3)

            last_before_local: Node = query

            e_to_local, lat_to_local = self._egress_and_latency_to_local(
                last_before_local, mb4
            )

            storage = self.calculate_storage_cost(
                segment,
                split,
                caption,
                query,
                video_size_MB,
                ratio_segment=ratio_segment,
                ratio_split=ratio_split,
                ratio_caption=ratio_caption,
                ratio_query=ratio_query,
            )
            total_cost = (
                cost_seg
                + cost_spl
                + cost_cap
                + cost_qry
                + e0
                + e1
                + e2
                + e3
                + e_to_local
                + storage
            )

            lat_segment_compute = segment.calculate_latency(mb0, deterministic=deterministic)
            lat_split_compute = split.calculate_latency(mb1, deterministic=deterministic)
            lat_caption_compute = caption.calculate_latency(mb2, deterministic=deterministic)
            lat_query_compute = query.calculate_latency(mb2, deterministic=deterministic)

            lat_segment_split = self._edge_latency_s(segment, split, mb1)
            lat_split_caption = self._edge_latency_s(split, caption, mb2)
            lat_to_query = self._edge_latency_s(pre_q, query, mb3)

            latency = (
                lat_from_local_0
                + lat_segment_compute
                + lat_segment_split
                + lat_split_compute
                + lat_split_caption
                + lat_caption_compute
                + lat_to_query
                + lat_query_compute
                + lat_to_local
            )

            utility = _utility_weighted(segment, split, caption, query)

            want_breakdown = verbose or include_breakdown
            breakdown: WorkflowBreakdown | None = None
            if want_breakdown:
                breakdown = WorkflowBreakdown(
                    exec_segment=cost_seg,
                    exec_split=cost_spl,
                    llm_caption=cost_cap,
                    llm_query=cost_qry,
                    egress_local_segment=e0,
                    egress_segment_split=e1,
                    egress_split_caption=e2,
                    egress_to_query=e3,
                    egress_last_to_local=e_to_local,
                    storage=storage,
                    lat_local_segment=lat_from_local_0,
                    lat_segment_compute=lat_segment_compute,
                    lat_segment_split=lat_segment_split,
                    lat_split_compute=lat_split_compute,
                    lat_split_caption=lat_split_caption,
                    lat_caption_compute=lat_caption_compute,
                    lat_to_query=lat_to_query,
                    lat_query_compute=lat_query_compute,
                    lat_last_to_local=lat_to_local,
                    mb0=mb0,
                    mb1=mb1,
                    mb2=mb2,
                    mb3=mb3,
                    mb4=mb4,
                )
                if not math.isclose(
                    breakdown.sum_cost(), total_cost, rel_tol=1e-12, abs_tol=1e-9
                ):
                    raise RuntimeError("internal: cost breakdown does not sum to total_cost")
                if not math.isclose(
                    breakdown.sum_latency(), latency, rel_tol=1e-12, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        "internal: latency breakdown does not sum to total latency"
                    )

            if verbose and breakdown is not None:
                print(breakdown)
                print(f"  utility (weighted)                       {utility:>18.6f}")

            return WorkflowResult(
                cost=total_cost,
                latency=latency,
                utility=utility,
                breakdown=breakdown,
            )
        finally:
            del self._deterministic_eval


class SimulationEnvironment:
    """
    面向算法侧的黑盒环境接口：
    - 不直接暴露 DistributionParameters
    - 仅提供动作空间与采样观测接口

    若未传入 ``workflow``，可用 ``params`` / ``edge_rtt`` / ``edge_bw`` / ``use_measured_network`` 等
    传给内部 ``Workflow(...)``。
    """

    def __init__(
        self,
        workflow: Workflow | None = None,
        *,
        params: DistributionParameters | None = None,
        edge_rtt: dict[tuple[str, str], RTT] | None = None,
        edge_bw: dict[tuple[str, str], BW] | None = None,
        use_measured_network: bool = True,
        measured_options: MeasuredNetworkOptions | None = None,
        seg_split_sweep: SegmentSplitSweepModel | None = None,
    ) -> None:
        if workflow is not None:
            if (
                params is not None
                or edge_rtt is not None
                or edge_bw is not None
                or measured_options is not None
                or seg_split_sweep is not None
            ):
                raise ValueError(
                    "provide workflow= XOR constructor kwargs for Workflow, not both"
                )
            self._workflow = workflow
        else:
            self._workflow = Workflow(
                params=params,
                edge_rtt=edge_rtt,
                edge_bw=edge_bw,
                use_measured_network=use_measured_network,
                measured_options=measured_options,
                seg_split_sweep=seg_split_sweep,
            )

    def action_space(self) -> dict[str, list[str]]:
        return self._workflow.list_node_names()

    def sample_once(
        self,
        selection: NodeSelection,
        video_size_MB: float,
        *,
        include_breakdown: bool = False,
    ) -> WorkflowResult:
        return self._workflow.sample_observation(
            selection,
            video_size_MB,
            include_breakdown=include_breakdown,
        )


if __name__ == "__main__":
    env = SimulationEnvironment()
    space = env.action_space()  # dict: segment/split/caption/query -> 节点名列表
    choice = NodeSelection(
        segment=space["segment"][0],
        split=space["split"][0],
        caption=space["caption"][0],
        query=space["query"][0],
    )
    obs = env.sample_once(
        choice,
        video_size_MB=120.0,
        include_breakdown=False,
    )
    print(obs.cost, obs.latency, obs.utility)
