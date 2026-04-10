"""
根据 config 拓扑与 ``DistributionParameters`` 构造全量节点；
``Workflow`` 绑定一次 ``sample()``，在 ``calculate`` 中串联 segment→split→caption→query，
并在 ``calculate`` 中由 ``params.edge_rtt`` / ``edge_bw`` 的分布采样出具体 RTT/BW 后计算
cost / latency / utility。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from config import CLOUD_PROVIDERS, CLOUD_REGIONS, LLM_PROVIDER_TO_OPTIONS
from distribution import (
    BW,
    DistributionParameters,
    LogNormalDistribution,
    RTT,
    config as simulation_config,
    sample,
)
from nodes import (
    CaptionCloudNode,
    CaptionNonCloudNode,
    CloudNode,
    Edge,
    LLM_TOKENS_PER_MILLION,
    LocalNode,
    LlmNode,
    Node,
    QueryCloudNode,
    QueryNonCloudNode,
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
    caption: dict[str, CaptionCloudNode | CaptionNonCloudNode]
    query: dict[str, QueryCloudNode | QueryNonCloudNode]


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
    egress_caption_split: float
    egress_to_query: float
    egress_query_to_cloud: float
    egress_last_to_local: float
    storage: float
    # --- Latency (s) ---
    lat_local_segment: float
    lat_segment_compute: float
    lat_segment_split: float
    lat_split_compute: float
    lat_split_caption: float
    lat_caption_compute: float
    lat_caption_split: float
    lat_to_query: float
    lat_query_compute: float
    lat_query_to_cloud: float
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
            + self.egress_caption_split
            + self.egress_to_query
            + self.egress_query_to_cloud
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
            + self.lat_caption_split
            + self.lat_to_query
            + self.lat_query_compute
            + self.lat_query_to_cloud
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
            row("egress caption → split (mb3)", self.egress_caption_split, unit=""),
            row("egress → query (mb3)", self.egress_to_query, unit=""),
            row("egress query → cloud parent (mb4)", self.egress_query_to_cloud, unit=""),
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
            row("transfer caption → split (mb3)", self.lat_caption_split, unit=""),
            row("transfer → query (mb3)", self.lat_to_query, unit=""),
            row("query LLM", self.lat_query_compute, unit=""),
            row("transfer query → cloud parent (mb4)", self.lat_query_to_cloud, unit=""),
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


def _egress_usd_per_gb(params: DistributionParameters, src: Node, dst: Node) -> float:
    eps = params.egress_price["endpoints"]
    matrix = params.egress_price["matrix"]
    i = eps.index(_node_to_comm_endpoint(src))
    j = eps.index(_node_to_comm_endpoint(dst))
    return matrix[i][j]


def _gamma_mean_comp_time(seg: SegmentNode | SplitNode, video_size_MB: float) -> float:
    """Gamma(α,θ) 计算时间均值 α·θ，θ = θ0 + θ1·S。"""
    p = seg.exec_time_param
    theta = p.theta0 + p.theta1 * video_size_MB
    return p.alpha * theta


def _segment_mean_execution_cost(segment: SegmentNode, video_size_MB: float) -> float:
    ct = _gamma_mean_comp_time(segment, video_size_MB)
    return (ct / 60.0) * segment.price_per_min


def _split_mean_execution_cost(split: SplitNode, video_size_MB: float) -> float:
    return (_gamma_mean_comp_time(split, video_size_MB) / 60.0) * split.price_per_min


def _segment_mean_latency(segment: SegmentNode, video_size_MB: float) -> float:
    p = segment.exec_time_param
    io_time = p.io_s_per_MB * video_size_MB
    return io_time + _gamma_mean_comp_time(segment, video_size_MB)


def _split_mean_latency(split: SplitNode, video_size_MB: float) -> float:
    p = split.exec_time_param
    io_time = p.io_s_per_MB * video_size_MB
    return io_time + _gamma_mean_comp_time(split, video_size_MB)


def _caption_mean_output_tokens(caption: CaptionCloudNode | CaptionNonCloudNode, video_size_MB: float) -> int:
    """Caption 输出 token 数：LogNormal 的均值参数（即 E[X]）取整，与 ``deterministic`` 口径一致。"""
    p = caption.caption_output_token_num_param
    mean = max(float(p.base) + float(p.coef_per_MB) * video_size_MB, 20.0)
    return max(1, int(round(mean)))


def _query_mean_output_tokens(query: QueryCloudNode | QueryNonCloudNode) -> int:
    p = query.query_output_token_num_param
    return max(1, int(round(float(p.mean))))


def _llm_latency_mean_s(llm: LlmNode, n_output_tokens: int) -> float:
    p = llm.llm_latency_param
    if p is None:
        raise ValueError("llm_latency_param is required for LLM latency")
    return max(
        0.0,
        float(p.alpha_ms_per_token) * n_output_tokens + float(p.beta_ms),
    ) / 1000.0


def _utility_weighted(
    segment: SegmentNode,
    split: SplitNode,
    caption: CaptionCloudNode | CaptionNonCloudNode,
    query: QueryCloudNode | QueryNonCloudNode,
) -> float:
    uw: Any = simulation_config.get("utility_weight") or {}
    return (
        float(uw.get("segment", 0.0)) * segment.utility
        + float(uw.get("split", 0.0)) * split.utility
        + float(uw.get("caption", 0.0)) * caption.utility
        + float(uw.get("query", 0.0)) * query.utility
    )


def build_workflow_nodes(params: DistributionParameters) -> WorkflowNodes:
    segment: dict[str, SegmentNode] = {}
    split: dict[str, SplitNode] = {}
    caption: dict[str, CaptionCloudNode | CaptionNonCloudNode] = {}
    query: dict[str, QueryCloudNode | QueryNonCloudNode] = {}
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
            )
            split[spl_name] = SplitNode(
                name=spl_name,
                provider=p,
                region=r,
                utility=params.utility[spl_name],
                storage_price_per_gb_month=params.storage_price[spl_name],
                price_per_min=params.split_price[spl_name],
                exec_time_param=params.seg_spl_exec_time_param[spl_name],
            )

    for p in CLOUD_PROVIDERS:
        for r in CLOUD_REGIONS[p]:
            pr = f"{p}_{r}"
            cap_name = f"{pr}_caption"
            qry_name = f"{pr}_query"
            cin, cout = params.llm_token_price[cap_name]
            caption[cap_name] = CaptionCloudNode(
                name=cap_name,
                provider=p,
                region=r,
                utility=params.utility[cap_name],
                storage_price_per_gb_month=params.storage_price[cap_name],
                llm_name=f"{p}_cloud",
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
                llm_name=f"{p}_cloud",
                input_token_price=qin,
                output_token_price=qout,
                llm_latency_param=params.llm_latency_param[qry_name],
                query_output_token_num_param=params.query_output_token_num_param[qry_name],
            )

    for provider, options in LLM_PROVIDER_TO_OPTIONS.items():
        for opt in options:
            cap_name = f"{provider}_{opt}_caption"
            qry_name = f"{provider}_{opt}_query"
            cin, cout = params.llm_token_price[cap_name]
            caption[cap_name] = CaptionNonCloudNode(
                name=cap_name,
                provider=provider,
                region=None,
                utility=params.utility[cap_name],
                llm_name=opt,
                input_token_price=cin,
                output_token_price=cout,
                llm_latency_param=params.llm_latency_param[cap_name],
                caption_input_token_num_param=params.caption_input_token_num_param[cap_name],
                caption_output_token_num_param=params.caption_output_token_num_param[cap_name],
            )
            qin, qout = params.llm_token_price[qry_name]
            query[qry_name] = QueryNonCloudNode(
                name=qry_name,
                provider=provider,
                region=None,
                utility=params.utility[qry_name],
                llm_name=opt,
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
    return build_workflow_nodes(sample())


class Workflow:
    """
    构造时 ``sample()`` 一次并初始化全部节点；``params.edge_rtt`` / ``params.edge_bw``
    已在 ``distribution.sample()`` 中为每条边给出 LogNormal 的 mean/std（分布矩阵）。
    每次 ``calculate`` 开头在此基础上对每条边再采样一次，得到本次运行的具体
    RTT(ms) 与带宽(Mbps)（数据矩阵）；写入 ``Edge`` 时用 std=0 避免在 ``Edge.calculate_latency`` 内再次随机。
    """

    def __init__(self, params: DistributionParameters | None = None) -> None:
        self.params = params if params is not None else sample()
        self.nodes = build_workflow_nodes(self.params)
        self._local = LocalNode()
        self._rtt_ms: dict[tuple[str, str], float] = {}
        self._bw_mbps: dict[tuple[str, str], float] = {}
        self._deterministic: bool = False

    def _realize_network(self) -> None:
        """由 ``self.params`` 中的 RTT/BW 分布参数矩阵采样出本轮各边的具体数值。"""
        self._rtt_ms = {}
        self._bw_mbps = {}
        for k, r in self.params.edge_rtt.items():
            self._rtt_ms[k] = LogNormalDistribution(float(r.mean), float(r.std)).sample()
        for k, w in self.params.edge_bw.items():
            self._bw_mbps[k] = LogNormalDistribution(float(w.mean), float(w.std)).sample()

    def _realize_network_mean(self) -> None:
        """确定性基线：链路上 RTT/BW 取各边分布的均值（与 ``_realize_network`` 的 mean 字段一致）。"""
        self._rtt_ms = {}
        self._bw_mbps = {}
        for k, r in self.params.edge_rtt.items():
            self._rtt_ms[k] = float(r.mean)
        for k, w in self.params.edge_bw.items():
            self._bw_mbps[k] = float(w.mean)

    def _rtt_bw_objects(self, src: Node, dst: Node) -> tuple[RTT, BW]:
        a, b = _node_to_comm_endpoint(src), _node_to_comm_endpoint(dst)
        key = (a, b)
        r = RTT()
        r.mean = self._rtt_ms[key]
        r.std = 0.0
        w = BW()
        w.mean = self._bw_mbps[key]
        w.std = 0.0
        return r, w

    def _edge(self, src: Node, dst: Node) -> Edge:
        r, bw = self._rtt_bw_objects(src, dst)
        return Edge(src, dst, r, bw, _egress_usd_per_gb(self.params, src, dst))

    def _edge_latency_s(self, src: Node, dst: Node, data_size_MB: float) -> float:
        if _node_to_comm_endpoint(src) == _node_to_comm_endpoint(dst):
            return 0.0
        if self._deterministic:
            a, b = _node_to_comm_endpoint(src), _node_to_comm_endpoint(dst)
            key = (a, b)
            rtt_ms = self._rtt_ms[key]
            bw_mbps = self._bw_mbps[key]
            rtt_half_s = rtt_ms / 2000.0
            if math.isinf(bw_mbps):
                return rtt_half_s
            if bw_mbps <= 0 and not math.isinf(bw_mbps):
                raise ValueError("bandwidth_mbps must be > 0 (or +inf for local/zero transfer time)")
            return rtt_half_s + (data_size_MB * 8.0) / bw_mbps
        return self._edge(src, dst).calculate_latency(data_size_MB)

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

    @staticmethod
    def _pre_query_node(
        split: SplitNode,
        caption: CaptionCloudNode | CaptionNonCloudNode,
    ) -> Node:
        """Query 的输入 mb3 由谁发往 query：云 Caption 则 caption→query；非云 Caption 则 split→query（mb3 已先经 caption→split 回到 split）。"""
        return caption if isinstance(caption, CaptionCloudNode) else split

    @staticmethod
    def _post_query_cloud_storage_node(
        split: SplitNode,
        caption: CaptionCloudNode | CaptionNonCloudNode,
        query: QueryCloudNode | QueryNonCloudNode,
    ) -> CloudNode:
        """非云 query 时 answer（mb4）落盘的云父节点：云 caption 则 caption，否则 split。"""
        if isinstance(caption, CaptionCloudNode):
            return caption
        return split

    def calculate_storage_cost(
        self,
        segment: SegmentNode,
        split: SplitNode,
        caption: CaptionCloudNode | CaptionNonCloudNode,
        query: QueryCloudNode | QueryNonCloudNode,
        video_size_MB: float,
        *,
        ratio_segment: float,
        ratio_split: float,
        ratio_caption: float,
        ratio_query: float,
    ) -> float:
        """
        仅 ``CloudNode`` 计费。云侧先写入 bucket 再读取；无存储的 LLM 节点不付存储费，产物由前置云保存。

        公共：segment 存 mb0（上传）、mb1；split 存 mb1、mb2。

        1) cloud caption + cloud query：caption 存 mb2、mb3；query 存 mb3、mb4。
        2) cloud caption + llm query：caption 存 mb2、mb3；query 无存储，最终 answer（mb4）存回云 caption。
        3) llm caption + cloud query：caption 无存储，mb3 回到 split；split 额外存 mb3；query 存 mb3、mb4。
        4) llm caption + llm query：caption 无存储，mb3 在 split；query 无存储，mb4 存回 split。
        """
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        mb0 = video_size_MB
        mb1 = mb0 * float(ratio_segment)
        mb2 = mb1 * float(ratio_split)
        mb3 = mb2 * float(ratio_caption)
        mb4 = mb3 * float(ratio_query)

        def store(node: CloudNode, mb: float) -> float:
            return node.calculate_storage_cost(mb)

        total = 0.0

        total += store(segment, mb0) + store(segment, mb1)
        total += store(split, mb1) + store(split, mb2)

        if isinstance(caption, CaptionCloudNode):
            total += store(caption, mb2) + store(caption, mb3)
        else:
            total += store(split, mb3)

        if isinstance(query, QueryCloudNode):
            total += store(query, mb3) + store(query, mb4)
        else:
            total += store(
                self._post_query_cloud_storage_node(split, caption, query), mb4
            )

        return total

    def _sample_data_conversion_ratios(
        self, *, deterministic: bool
    ) -> tuple[float, float, float, float]:
        """
        每次 calculate 采样一次 data conversion ratios（segment/split/caption/query）。
        deterministic=True 时取各分布的均值（distribution.LogNormalDistribution.mean）。
        """
        dc = self.params.data_conversion_ratio
        if deterministic:
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
        caption: CaptionCloudNode | CaptionNonCloudNode,
        query: QueryCloudNode | QueryNonCloudNode,
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

        非云 Caption：splitted video（mb2）经 split→caption，caption 产物（mb3）先 **caption→split**
        落盘再 **split→query**，两段链路均计费；云 Caption 则 mb3 留在 caption 侧，仅 **caption→query**。

        ``verbose=True`` 时向 stdout 打印 cost/latency 分项；``include_breakdown=True`` 时在
        ``WorkflowResult.breakdown`` 中返回 ``WorkflowBreakdown``（``verbose`` 为 True 时也会附带）。

        ``deterministic=True``：**确定性基线**——链路与计算时间取分布均值、Gamma 取均值、LLM token
        取期望输出规模、LLM 延迟取无噪声均值；不再对边延迟/执行时间/token 做随机采样。
        """
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")

        self._deterministic = deterministic
        if deterministic:
            self._realize_network_mean()
        else:
            self._realize_network()

        ratio_segment, ratio_split, ratio_caption, ratio_query = self._sample_data_conversion_ratios(
            deterministic=deterministic
        )

        mb0 = video_size_MB
        mb1 = mb0 * float(ratio_segment)
        mb2 = mb1 * float(ratio_split)
        mb3 = mb2 * float(ratio_caption)
        mb4 = mb3 * float(ratio_query)

        if not deterministic:
            cost_seg = segment.calculate_execution_cost(mb0)
            cost_spl = split.calculate_execution_cost(mb1)
            cost_cap = caption.calculate_token_cost(mb2)
            n_cap = caption_output_tokens_for_query(caption, mb2)
            cost_qry = query.calculate_token_cost(mb2, input_tokens=n_cap)
        else:
            cost_seg = _segment_mean_execution_cost(segment, mb0)
            cost_spl = _split_mean_execution_cost(split, mb1)
            n_cap = _caption_mean_output_tokens(caption, mb2)
            n_in_cap = float(caption.input_token_num(mb2))
            cost_cap = (
                n_in_cap * float(caption.input_token_price) + float(n_cap) * float(caption.output_token_price)
            ) / LLM_TOKENS_PER_MILLION
            n_out_q = _query_mean_output_tokens(query)
            cost_qry = (
                float(n_cap) * float(query.input_token_price) + float(n_out_q) * float(query.output_token_price)
            ) / LLM_TOKENS_PER_MILLION

        pre_q = self._pre_query_node(split, caption)

        e0, lat_from_local_0 = self._egress_and_latency_from_local(segment, mb0)
        e1 = self._edge_egress_usd(segment, split, mb1)
        e2 = self._edge_egress_usd(split, caption, mb2)
        # 非云 Caption：mb3 先回到 split，再 split→query；云 Caption：mb3 从 caption→query
        if isinstance(caption, CaptionNonCloudNode):
            e_cap_to_split = self._edge_egress_usd(caption, split, mb3)
            lat_cap_to_split = self._edge_latency_s(caption, split, mb3)
        else:
            e_cap_to_split = 0.0
            lat_cap_to_split = 0.0
        e3 = self._edge_egress_usd(pre_q, query, mb3)

        # 非云 query：answer（mb4）先回云父，再云→local
        if isinstance(query, QueryNonCloudNode):
            post_q = self._post_query_cloud_storage_node(split, caption, query)
            e_ret_q = self._edge_egress_usd(query, post_q, mb4)
            lat_ret_q = self._edge_latency_s(query, post_q, mb4)
            last_before_local: Node = post_q
        else:
            e_ret_q = 0.0
            lat_ret_q = 0.0
            last_before_local = query

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
            + e_cap_to_split
            + e3
            + e_ret_q
            + e_to_local
            + storage
        )

        if not deterministic:
            lat_segment_compute = segment.calculate_latency(mb0)
            lat_split_compute = split.calculate_latency(mb1)
            lat_caption_compute = caption.calculate_latency(mb2)
            lat_query_compute = query.calculate_latency(mb2)
        else:
            lat_segment_compute = _segment_mean_latency(segment, mb0)
            lat_split_compute = _split_mean_latency(split, mb1)
            lat_caption_compute = _llm_latency_mean_s(caption, n_cap)
            lat_query_compute = _llm_latency_mean_s(query, n_out_q)

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
            + lat_cap_to_split
            + lat_to_query
            + lat_query_compute
            + lat_ret_q
            + lat_to_local
        )

        utility = _utility_weighted(segment, split, caption, query)

        self._deterministic = False

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
                egress_caption_split=e_cap_to_split,
                egress_to_query=e3,
                egress_query_to_cloud=e_ret_q,
                egress_last_to_local=e_to_local,
                storage=storage,
                lat_local_segment=lat_from_local_0,
                lat_segment_compute=lat_segment_compute,
                lat_segment_split=lat_segment_split,
                lat_split_compute=lat_split_compute,
                lat_split_caption=lat_split_caption,
                lat_caption_compute=lat_caption_compute,
                lat_caption_split=lat_cap_to_split,
                lat_to_query=lat_to_query,
                lat_query_compute=lat_query_compute,
                lat_query_to_cloud=lat_ret_q,
                lat_last_to_local=lat_to_local,
                mb0=mb0,
                mb1=mb1,
                mb2=mb2,
                mb3=mb3,
                mb4=mb4,
            )
            if not math.isclose(breakdown.sum_cost(), total_cost, rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError("internal: cost breakdown does not sum to total_cost")
            if not math.isclose(breakdown.sum_latency(), latency, rel_tol=1e-12, abs_tol=1e-9):
                raise RuntimeError("internal: latency breakdown does not sum to total latency")

        if verbose and breakdown is not None:
            print(breakdown)
            print(f"  utility (weighted)                       {utility:>18.6f}")

        return WorkflowResult(
            cost=total_cost,
            latency=latency,
            utility=utility,
            breakdown=breakdown,
        )


if __name__ == "__main__":
    wf = Workflow()
    seg = next(iter(wf.nodes.segment.values()))
    spl = next(iter(wf.nodes.split.values()))
    cap = next(iter(wf.nodes.caption.values()))
    qry = next(iter(wf.nodes.query.values()))
    r = wf.calculate(seg, spl, cap, qry, 100.0, verbose=True)
    print("totals: cost", r.cost, "latency", r.latency, "utility", r.utility)
