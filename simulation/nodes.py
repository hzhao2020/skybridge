from __future__ import annotations

from abc import ABC, abstractmethod
import math
import random
from distribution import (
    BW,
    CaptionInputTokenNumParam,
    CaptionOutputTokenNumParam,
    GammaDistribution,
    LlmlatencyParam,
    LogNormalDistribution,
    QueryOutputTokenNumParam,
    RTT,
    SegSplExecTimeParam,
    config as simulation_config,
)


class Node(ABC):
    def __init__(
        self,
        name: str,
        provider: str,
        region: str | None,
        utility: float,
    ):
        self.name = name
        self.provider = provider
        self.region = region
        self.utility = utility

    @abstractmethod
    def calculate_latency(self, *args, **kwargs):
        pass


class LocalNode(Node):
    """流水线起点：边缘/本地端点，映射到通信拓扑中的 ``local``；无独立算子处理延迟。"""

    def __init__(self) -> None:
        super().__init__(
            name="local",
            provider="local",
            region=None,
            utility=0.0,
        )

    def calculate_latency(self, video_size_MB: float) -> float:
        return 0.0


def _node_to_comm_endpoint(node: Node) -> str:
    """
    将 Node 映射到 sample_egress_price / get_comm_endpoints 使用的端点名：
    云侧为 "{provider}_{region}"；本地为 "local"。
    """
    if node.provider == "local":
        return "local"
    if node.region is None:
        raise ValueError(f"node {node.name!r} has no region; cannot map to comm endpoint")
    return f"{node.provider}_{node.region}"


class Edge:
    """
    连接两个 Node 的链路：保存 RTT、BW 参数对象（与 distribution 中每条边的采样一致）。
    calculate_latency(data_size_MB) 从 LogNormal(mean, std) 各采样一次 RTT(ms) 与带宽(Mbps)，
    再返回传输延迟（秒）：RTT/2 + data_MB * 8 / bandwidth_Mbps。
    egress_price 为该有向边已解析好的出站单价 USD/GB（由调用方从 sample_egress_price 矩阵中查表填入）。
    """

    src: Node
    dst: Node
    rtt: RTT
    bw: BW
    egress_price: float

    def __init__(
        self,
        src: Node,
        dst: Node,
        rtt: RTT,
        bw: BW,
        egress_price: float,
    ):
        self.src = src
        self.dst = dst
        self.rtt = rtt
        self.bw = bw
        self.egress_price = egress_price


    def calculate_latency(
        self, data_size_MB: float, *, deterministic: bool = False
    ) -> float:
        if data_size_MB < 0:
            raise ValueError("data_size_MB must be >= 0")
        if deterministic:
            rtt_half_s = float(self.rtt.mean) / 2000.0
            bw_mbps = float(self.bw.mean)
        else:
            rtt_ms = LogNormalDistribution(float(self.rtt.mean), float(self.rtt.std)).sample()
            bandwidth_mbps = LogNormalDistribution(
                float(self.bw.mean), float(self.bw.std)
            ).sample()
            rtt_half_s = rtt_ms / 2000.0
            bw_mbps = bandwidth_mbps
        if bw_mbps <= 0 and not math.isinf(bw_mbps):
            raise ValueError("bandwidth_mbps must be > 0 (or +inf for local/zero transfer time)")
        if math.isinf(bw_mbps):
            return rtt_half_s
        tx_s = (data_size_MB * 8.0) / bw_mbps
        return rtt_half_s + tx_s


    def calculate_egress_cost(self, data_size_MB: float) -> float:
        """GB × egress_price（USD/GB）。"""
        if data_size_MB < 0:
            raise ValueError("data_size_MB must be >= 0")
        data_gb = data_size_MB / 1024.0
        return data_gb * self.egress_price


# --- Cloud 体系 ---
class CloudNode(Node):
    def __init__(
        self,
        name: str,
        provider: str,
        region: str | None,
        utility: float,
        storage_price_per_gb_month: float,
    ):
        super().__init__(name, provider, region, utility)
        self.storage_price_per_gb_month = storage_price_per_gb_month

    def calculate_storage_cost(self, data_size_MB: float) -> float:
        """按 config.storage 默认存放时长，将 USD/GB/月 折算为本次数据量的存储费（USD）。"""
        if data_size_MB < 0:
            raise ValueError("data_size_MB must be >= 0")
        st = simulation_config.get("storage") or {}
        hours = float(st.get("hours", 24))
        hpm = float(st.get("hours_per_month", 720.0))
        if hpm <= 0:
            raise ValueError("storage.hours_per_month must be > 0")
        data_gb = data_size_MB / 1024.0
        fraction_of_month = hours / hpm
        return data_gb * self.storage_price_per_gb_month * fraction_of_month


class SegmentNode(CloudNode):
    exec_time_param: SegSplExecTimeParam

    def __init__(self, name: str, provider: str, region: str | None, utility: float, 
                 storage_price_per_gb_month: float, price_per_min: float,
                 exec_time_param: SegSplExecTimeParam):
        # 修复点：必须向 CloudNode 传递 storage_price_per_gb_month
        super().__init__(name, provider, region, utility, storage_price_per_gb_month)
        self.price_per_min = price_per_min
        self.exec_time_param = exec_time_param


    def calculate_execution_cost(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> float:
        """Gamma 采样计算时间 × 每分钟单价（USD），不含存储费。"""
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        theta = self.exec_time_param.theta0 + self.exec_time_param.theta1 * video_size_MB
        if deterministic:
            comp_time_s = self.exec_time_param.alpha * theta
        else:
            comp_time_s = GammaDistribution(self.exec_time_param.alpha, theta).sample()
        return (comp_time_s / 60.0) * self.price_per_min

    def calculate_latency(self, video_size_MB: float, *, deterministic: bool = False):
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        io_time = self.exec_time_param.io_s_per_MB * video_size_MB
        theta = self.exec_time_param.theta0 + self.exec_time_param.theta1 * video_size_MB
        if deterministic:
            comp_time = self.exec_time_param.alpha * theta
        else:
            comp_time = GammaDistribution(self.exec_time_param.alpha, theta).sample()
        return io_time + comp_time


class SplitNode(CloudNode):
    exec_time_param: SegSplExecTimeParam

    def __init__(self, name: str, provider: str, region: str | None, utility: float, 
                 storage_price_per_gb_month: float, price_per_min: float,
                 exec_time_param: SegSplExecTimeParam):
        super().__init__(name, provider, region, utility, storage_price_per_gb_month)
        self.price_per_min = price_per_min
        self.exec_time_param = exec_time_param

    def calculate_execution_cost(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> float:
        """Gamma 采样计算时间 × 每分钟单价（USD），不含存储费。"""
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        theta = self.exec_time_param.theta0 + self.exec_time_param.theta1 * video_size_MB
        if deterministic:
            comp_time_s = self.exec_time_param.alpha * theta
        else:
            comp_time_s = GammaDistribution(self.exec_time_param.alpha, theta).sample()
        return (comp_time_s / 60.0) * self.price_per_min

    def calculate_latency(self, video_size_MB: float, *, deterministic: bool = False):
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        io_time = self.exec_time_param.io_s_per_MB * video_size_MB
        theta = self.exec_time_param.theta0 + self.exec_time_param.theta1 * video_size_MB
        if deterministic:
            comp_time = self.exec_time_param.alpha * theta
        else:
            comp_time = GammaDistribution(self.exec_time_param.alpha, theta).sample()
        return io_time + comp_time


# --- LLM 体系 ---
# 与常见 API 账单一致：input_token_price / output_token_price 表示 USD / 百万 tokens
LLM_TOKENS_PER_MILLION = 1_000_000.0


class LlmNode:
    """抽象出的 LLM 通用逻辑。"""

    llm_latency_param: LlmlatencyParam | None

    def __init__(
        self,
        llm_name: str,
        input_token_price: float,
        output_token_price: float,
        llm_latency_param: LlmlatencyParam | None = None,
    ):
        self.llm_name = llm_name
        # USD per million input / output tokens
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price
        self.llm_latency_param = llm_latency_param

    def input_token_num(self, video_size_MB: float) -> int:
        """Caption 等子类覆盖；Query 等无输入侧计费的节点默认 0。"""
        return 0

    def output_token_num(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> int:
        raise NotImplementedError

    def calculate_token_cost(
        self,
        video_size_MB: float,
        *,
        input_tokens: int | None = None,
        deterministic: bool = False,
    ) -> float:
        """
        按「百万 token」单价计费：(输入 token×输入单价 + 输出 token×输出单价) / 1e6 → USD。
        ``input_token_price`` / ``output_token_price`` 的单位为 **USD / 百万 tokens**。

        Caption→Query 流水线：上游 Caption 的输出 token 即下游 Query 的输入 token。
        请先对 Caption 采样一次 ``n = caption.output_token_num(video_size_MB)``（或
        ``caption_output_tokens_for_query(caption, video_size_MB)``），再传入
        ``query.calculate_token_cost(video_size_MB, input_tokens=n)``，避免两侧各自采样不一致。
        若 ``input_tokens`` 为 None，则使用本节点的 ``input_token_num``（Query 默认为 0）。
        """
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        if input_tokens is not None:
            if input_tokens < 0:
                raise ValueError("input_tokens must be >= 0")
            n_in = float(input_tokens)
        else:
            n_in = float(self.input_token_num(video_size_MB))
        n_out = float(self.output_token_num(video_size_MB, deterministic=deterministic))
        return (
            n_in * float(self.input_token_price) + n_out * float(self.output_token_price)
        ) / LLM_TOKENS_PER_MILLION

    def sample_ms(self, n_tokens: int, *, deterministic: bool = False) -> float:
        """LLM 执行延迟（毫秒），与 skybridge LlmLatencyParams.sample_ms 一致。"""
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        if self.llm_latency_param is None:
            raise ValueError("llm_latency_param is required to compute LLM latency")
        p = self.llm_latency_param
        base = float(p.alpha_ms_per_token) * n_tokens + float(p.beta_ms)
        if deterministic:
            return max(0.0, base)
        noise = random.gauss(0.0, float(p.noise_sigma_ms))
        return max(0.0, base + noise)

    def _llm_latency_s(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> float:
        """LLM 端到端延迟（秒）；子类（继承 Node）的 calculate_latency 应委托此处以满足 ABC。"""
        return (
            self.sample_ms(
                self.output_token_num(video_size_MB, deterministic=deterministic),
                deterministic=deterministic,
            )
            / 1000.0
        )


def caption_output_tokens_for_query(
    caption: LlmNode, video_size_MB: float, *, deterministic: bool = False
) -> int:
    """Caption 输出 token 数，在 Caption→Query 链路上作为 Query 的 ``input_tokens`` 使用。"""
    return caption.output_token_num(video_size_MB, deterministic=deterministic)


class _CaptionTokenMixin:
    """Caption 节点共用的 input/output token 计数。"""

    caption_input_token_num_param: CaptionInputTokenNumParam
    caption_output_token_num_param: CaptionOutputTokenNumParam

    def input_token_num(self, video_size_MB: float) -> int:
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        tpm = self.caption_input_token_num_param.input_tokens_per_MB
        return max(0, int(round(video_size_MB * float(tpm))))

    def output_token_num(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> int:
        if video_size_MB < 0:
            raise ValueError("video_size_MB must be >= 0")
        p = self.caption_output_token_num_param
        mean = max(float(p.base) + float(p.coef_per_MB) * video_size_MB, 20.0)
        if deterministic:
            return max(1, int(round(mean)))
        sigma = float(p.std)
        if sigma < 0:
            raise ValueError("caption output token sigma must be >= 0")
        x = LogNormalDistribution(mean=mean, std=sigma).sample()
        return max(1, int(round(x)))


class _QueryTokenMixin:
    """Query 节点共用的 output token 采样；输入侧 token 数由上游 Caption 经 ``input_tokens`` 传入。"""

    query_output_token_num_param: QueryOutputTokenNumParam

    def output_token_num(
        self, _video_size_MB: float, *, deterministic: bool = False
    ) -> int:
        p = self.query_output_token_num_param
        if deterministic:
            return max(1, int(round(float(p.mean))))
        return max(
            1,
            int(
                round(
                    LogNormalDistribution(
                        mean=float(p.mean), std=float(p.std)
                    ).sample()
                )
            ),
        )


class LlmCloudNode(CloudNode, LlmNode):
    def __init__(self, name: str, provider: str, region: str | None, utility: float, 
                 storage_price_per_gb_month: float, # 来自 CloudNode
                 llm_name: str, input_token_price: float, output_token_price: float,
                 llm_latency_param: LlmlatencyParam | None = None): # 来自 LlmNodeMixin
        # 初始化 CloudNode 部分
        CloudNode.__init__(self, name, provider, region, utility, storage_price_per_gb_month)
        # 初始化 LLM 部分
        LlmNode.__init__(self, llm_name, input_token_price, output_token_price, llm_latency_param)

    def calculate_latency(
        self, video_size_MB: float, *, deterministic: bool = False
    ) -> float:
        return self._llm_latency_s(video_size_MB, deterministic=deterministic)


# --- Caption / Query specialization ---
class CaptionCloudNode(_CaptionTokenMixin, LlmCloudNode):
    def __init__(
        self,
        name: str,
        provider: str,
        region: str | None,
        utility: float,
        storage_price_per_gb_month: float,
        llm_name: str,
        input_token_price: float,
        output_token_price: float,
        llm_latency_param: LlmlatencyParam | None,
        caption_input_token_num_param: CaptionInputTokenNumParam,
        caption_output_token_num_param: CaptionOutputTokenNumParam,
    ):
        super().__init__(
            name,
            provider,
            region,
            utility,
            storage_price_per_gb_month,
            llm_name,
            input_token_price,
            output_token_price,
            llm_latency_param,
        )
        self.caption_input_token_num_param = caption_input_token_num_param
        self.caption_output_token_num_param = caption_output_token_num_param


class QueryCloudNode(_QueryTokenMixin, LlmCloudNode):
    def __init__(
        self,
        name: str,
        provider: str,
        region: str | None,
        utility: float,
        storage_price_per_gb_month: float,
        llm_name: str,
        input_token_price: float,
        output_token_price: float,
        llm_latency_param: LlmlatencyParam | None,
        query_output_token_num_param: QueryOutputTokenNumParam,
    ):
        super().__init__(
            name,
            provider,
            region,
            utility,
            storage_price_per_gb_month,
            llm_name,
            input_token_price,
            output_token_price,
            llm_latency_param,
        )
        self.query_output_token_num_param = query_output_token_num_param