from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import math
import random

# set global random number seed
random.seed(42)

# ----------------------------
# Node definitions (12+12+20+20)
# ----------------------------

Workflow = Literal["segment", "split", "caption", "query"]
ExecTimeFn = Callable[[float], float]
TokenNumFn = Callable[[], int]
LlmLatencyFn = Callable[[int], float]


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    provider: str
    region: str | None
    can_store: bool
    storage_price_usd_per_gb_month: float | None
    exec_time_s: ExecTimeFn | None
    # Pricing:
    # - segment cloud nodes: USD per minute
    # - split cloud nodes: USD per second
    # - llm nodes: USD per 1M tokens (input/output)
    segment_price_usd_per_min: float | None
    split_price_usd_per_s: float | None
    llm_input_price_usd_per_1m_tokens: float | None
    llm_output_price_usd_per_1m_tokens: float | None
    output_token_num: TokenNumFn | None
    llm_exec_latency_ms: LlmLatencyFn | None
    llm_option: str | None = None  # optional; for LLM nodes / model choice nodes


# Providers are normalized as p1..p5:
# - p1..p3: cloud providers (have bucket storage)
# - p4..p5: LLM-only providers (no storage, no region)
_CLOUD_PROVIDERS: Sequence[str] = ("p1", "p2", "p3")

# Total 6 regions; each cloud provider has 4 regions:
# p1: r1,r2,r3,r4
# p2: r1,r2,r5,r6
# p3: r3,r4,r5,r6
_CLOUD_REGIONS: dict[str, tuple[str, ...]] = {
    "p1": ("r1", "r2", "r3", "r4"),
    "p2": ("r1", "r2", "r5", "r6"),
    "p3": ("r3", "r4", "r5", "r6"),
}

# Language models: total 6 (m1..m6)
# - p4 provides m1..m4
# - p5 provides m3..m6
_LLM_PROVIDER_TO_OPTIONS: dict[str, tuple[str, ...]] = {
    "p4": ("m1", "m2", "m3", "m4"),
    "p5": ("m3", "m4", "m5", "m6"),
}


@dataclass(frozen=True, slots=True)
class ExecTimeParams:
    io_s_per_mb: float
    k: float
    theta0: float
    theta1: float

    def sample_seconds(self, video_size_mb: float) -> float:
        if video_size_mb < 0:
            raise ValueError("video_size_mb must be >= 0")
        io_time = self.io_s_per_mb * video_size_mb
        theta = self.theta0 + self.theta1 * video_size_mb
        comp_time = random.gammavariate(self.k, theta)
        return io_time + comp_time


@dataclass(frozen=True, slots=True)
class RatioParams:
    mean: float
    sigma: float

    def sample_ratio(self) -> float:
        if self.mean <= 0:
            raise ValueError("mean must be > 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        mu = math.log(self.mean) - 0.5 * (self.sigma**2)
        return random.lognormvariate(mu, self.sigma)


@dataclass(frozen=True, slots=True)
class TokenNumParams:
    """
    LogNormal distribution for token counts (positive integers).
    Parameterized by log-space (mu, sigma): log(X) ~ Normal(mu, sigma).
    """

    mu: float
    sigma: float

    @staticmethod
    def from_mean(mean: float, sigma: float) -> "TokenNumParams":
        """
        For LogNormal(mu, sigma): E[X] = exp(mu + 0.5*sigma^2)
        => mu = ln(mean) - 0.5*sigma^2
        """
        if mean <= 0:
            raise ValueError("mean must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        mu = math.log(mean) - 0.5 * (sigma**2)
        return TokenNumParams(mu=mu, sigma=sigma)

    def sample_int(self) -> int:
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        x = random.lognormvariate(self.mu, self.sigma)
        return max(1, int(round(x)))


@dataclass(frozen=True, slots=True)
class LlmLatencyParams:
    """
    LLM execution latency model (milliseconds):

      T = alpha_ms_per_token * N + beta_ms + Normal(0, noise_sigma_ms)
    """

    alpha_ms_per_token: float
    beta_ms: float
    noise_sigma_ms: float

    def sample_ms(self, n_tokens: int) -> float:
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        noise = random.gauss(0.0, self.noise_sigma_ms)
        return max(0.0, self.alpha_ms_per_token * n_tokens + self.beta_ms + noise)


@dataclass(frozen=True, slots=True)
class CommCategoryParams:
    # RTT: LogNormal mean (ms) + sigma; Bandwidth: Pareto xm (Mbps) + alpha
    rtt_mean_ms: float
    rtt_sigma: float
    bw_xm_mbps: float
    bw_alpha: float


@dataclass(frozen=True, slots=True)
class CommParams:
    same_region: CommCategoryParams
    same_cloud: CommCategoryParams
    cross_cloud: CommCategoryParams
    via_local: CommCategoryParams
    via_llm_only: CommCategoryParams


@dataclass(frozen=True, slots=True)
class EgressParams:
    # USD per GB (egress only; ingress is free)
    intra_provider_usd_per_gb: float
    cross_provider_usd_per_gb: float


def sample_ratio_params() -> dict[Workflow, RatioParams]:
    """
    随机采样一次 per-workflow 的 ratio(LogNormal) 分布参数。
    你应该保存返回值并在后续采样时复用它。
    """
    seg_sigma = random.uniform(0.0005, 0.005)
    spl_sigma = random.uniform(0.0005, 0.005)
    cap_mean = random.uniform(0.003, 0.03)
    cap_sigma = random.uniform(0.4, 0.9)
    qry_mean = random.uniform(0.2, 0.8)
    qry_sigma = random.uniform(0.15, 0.45)

    return {
        "segment": RatioParams(mean=1.0, sigma=seg_sigma),
        "split": RatioParams(mean=1.0, sigma=spl_sigma),
        "caption": RatioParams(mean=cap_mean, sigma=cap_sigma),
        "query": RatioParams(mean=qry_mean, sigma=qry_sigma),
    }


def sample_exec_time_params() -> dict[Workflow, ExecTimeParams | None]:
    """
    随机采样一次执行时间分布参数。
    仅对 segment/split 有参数；caption/query 为 None。
    """

    def one() -> ExecTimeParams:
        return ExecTimeParams(
            io_s_per_mb=random.uniform(0.0001, 0.001),
            k=random.uniform(1, 5),
            theta0=random.uniform(0.02, 0.20),
            theta1=random.uniform(0.001, 0.02),
        )

    return {"segment": one(), "split": one(), "caption": None, "query": None}


def sample_comm_params() -> CommParams:
    """
    随机采样一次网络类别参数（RTT=LogNormal，bandwidth=Pareto）。
    你应该保存返回值并用于后续生成矩阵。
    """

    def cat(
        rtt_mean_range: tuple[float, float],
        rtt_sigma_range: tuple[float, float],
        bw_xm_range: tuple[float, float],
        bw_alpha_range: tuple[float, float],
    ) -> CommCategoryParams:
        return CommCategoryParams(
            rtt_mean_ms=random.uniform(*rtt_mean_range),
            rtt_sigma=random.uniform(*rtt_sigma_range),
            bw_xm_mbps=random.uniform(*bw_xm_range),
            bw_alpha=random.uniform(*bw_alpha_range),
        )

    return CommParams(
        same_region=cat((6, 10), (0.10, 0.18), (350, 900), (2.0, 3.0)),
        same_cloud=cat((16, 28), (0.18, 0.28), (800, 1600), (2.2, 3.2)),
        cross_cloud=cat((55, 90), (0.22, 0.35), (80, 250), (1.8, 2.6)),
        via_local=cat((70, 120), (0.25, 0.40), (20, 120), (1.6, 2.3)),
        via_llm_only=cat((75, 130), (0.25, 0.45), (50, 180), (1.7, 2.4)),
    )


def sample_storage_prices(workflow: Workflow) -> dict[str, float]:
    """
    随机采样一次某个 workflow 下云节点的存储价格（按节点名返回）。
    """
    prices: dict[str, float] = {}
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            prices[f"{p}_{r}_{workflow}"] = random.uniform(0.015, 0.025)
    return prices


def sample_egress_params() -> EgressParams:
    """
    随机采样一次出站流量费用参数（USD per GB）。
    - 云内（同 provider）: 0.01 ~ 0.05
    - 跨 provider:         0.08 ~ 0.12
    """
    return EgressParams(
        intra_provider_usd_per_gb=random.uniform(0.01, 0.05),
        cross_provider_usd_per_gb=random.uniform(0.08, 0.12),
    )


def sample_segment_prices() -> dict[str, float]:
    """
    Segment node price: Uniform(0.04, 0.08) USD per minute.
    Returns mapping: node_name -> price.
    """
    prices: dict[str, float] = {}
    workflow: Workflow = "segment"
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            prices[f"{p}_{r}_{workflow}"] = random.uniform(0.04, 0.08)
    return prices


def sample_split_prices() -> dict[str, float]:
    """
    Split node price: Uniform(1e-5, 1e-4) USD per second.
    Returns mapping: node_name -> price.
    """
    prices: dict[str, float] = {}
    workflow: Workflow = "split"
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            prices[f"{p}_{r}_{workflow}"] = random.uniform(0.00001, 0.0001)
    return prices


def sample_llm_token_prices(workflow: Workflow) -> dict[str, tuple[float, float]]:
    """
    LLM price per 1M tokens:
    - input price: Uniform(0.05, 2.0) USD per 1M tokens
    - output price: 4 * input price

    Returns mapping: llm_node_name -> (input_price, output_price).
    Intended workflows: caption/query.
    """
    prices: dict[str, tuple[float, float]] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            name = f"{p}_{opt}_{workflow}"
            inp = random.uniform(0.05, 2.0)
            prices[name] = (inp, 4.0 * inp)
    return prices


def sample_llm_output_token_params(workflow: Workflow) -> dict[str, TokenNumParams]:
    """
    为 LLM 节点采样 output_token_num 的分布参数（LogNormal）。

    - 分布参数只采样一次（你保存返回值并复用）。
    - 对于 query：按固定参数建模：log-space μ=5, σ=1
    - 对于 caption：token 数量量级按“回答文本”设计，取一个常见范围：
      mean: 100 ~ 2000 tokens（用于推导 μ）
      sigma(log-space): 0.3 ~ 0.9
    """
    params: dict[str, TokenNumParams] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            name = f"{p}_{opt}_{workflow}"
            if workflow == "query":
                params[name] = TokenNumParams(mu=5.0, sigma=1.0)
            else:
                params[name] = TokenNumParams.from_mean(
                    mean=random.uniform(100, 2000),
                    sigma=random.uniform(0.3, 0.9),
                )
    return params


def sample_llm_latency_params(workflow: Workflow) -> dict[str, LlmLatencyParams]:
    """
    为 LLM 节点采样 execution latency 参数（只采样一次并复用）。

    模型（ms）: T = alpha * N + beta + noise
    - alpha: 20 ~ 50 ms/token
    - beta:  400 ~ 800 ms
    - noise: Normal(0, sigma)

    噪声 sigma 这里取一个相对保守的范围：30 ~ 120 ms（你可以后续再调）。
    """
    params: dict[str, LlmLatencyParams] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            name = f"{p}_{opt}_{workflow}"
            params[name] = LlmLatencyParams(
                alpha_ms_per_token=random.uniform(20.0, 50.0),
                beta_ms=random.uniform(400.0, 800.0),
                noise_sigma_ms=random.uniform(30.0, 120.0),
            )
    return params


def _make_exec_time_fn() -> ExecTimeFn:
    """
    Returns a function: exec_time_s(video_size_mb) = io_time + computation_time.

    - io_time is proportional to data size (seconds per MB).
    - computation_time | size ~ Gamma(k, theta(size)) where theta increases with size.
    All parameters are sampled once when the function is created (seeded by global RNG).
    """
    io_s_per_mb = random.uniform(0.0001, 0.001)  # 0.5ms~5ms per MB
    k = random.uniform(1,5)  # shape
    theta0 = random.uniform(0.02, 0.20)  # base scale (seconds)
    theta1 = random.uniform(0.001, 0.02)  # scale slope per MB (seconds/MB)

    def f(video_size_mb: float) -> float:
        if video_size_mb < 0:
            raise ValueError("video_size_mb must be >= 0")
        io_time = io_s_per_mb * video_size_mb
        theta = theta0 + theta1 * video_size_mb
        comp_time = random.gammavariate(k, theta)
        return io_time + comp_time

    return f


def _build_cloud_nodes(
    workflow: Workflow,
    *,
    storage_prices: dict[str, float] | None = None,
    exec_params: ExecTimeParams | None = None,
    segment_prices: dict[str, float] | None = None,
    split_prices: dict[str, float] | None = None,
) -> list[Node]:
    nodes: list[Node] = []
    exec_fn: ExecTimeFn | None = (exec_params.sample_seconds if exec_params is not None else None)
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            name = f"{p}_{r}_{workflow}"
            nodes.append(
                Node(
                    name=name,
                    provider=p,
                    region=r,
                    can_store=True,
                    storage_price_usd_per_gb_month=(
                        storage_prices[name] if storage_prices is not None else random.uniform(0.015, 0.025)
                    ),
                    exec_time_s=exec_fn,
                    segment_price_usd_per_min=(
                        (segment_prices[name] if segment_prices is not None else random.uniform(0.04, 0.08))
                        if workflow == "segment"
                        else None
                    ),
                    split_price_usd_per_s=(
                        (split_prices[name] if split_prices is not None else random.uniform(0.00001, 0.0001))
                        if workflow == "split"
                        else None
                    ),
                    llm_input_price_usd_per_1m_tokens=None,
                    llm_output_price_usd_per_1m_tokens=None,
                    output_token_num=None,
                    llm_exec_latency_ms=None,
                )
            )
    return nodes  # 3 * 4 = 12


def _build_llm_nodes(workflow: Workflow) -> list[Node]:
    nodes: list[Node] = []
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            # LLM-only providers (e.g. OpenAI) are modeled as having no storage.
            inp = random.uniform(0.05, 2.0)
            nodes.append(
                Node(
                    name=f"{p}_{opt}_{workflow}",
                    provider=p,
                    region=None,
                    can_store=False,
                    storage_price_usd_per_gb_month=None,
                    exec_time_s=None,
                    segment_price_usd_per_min=None,
                    split_price_usd_per_s=None,
                    llm_input_price_usd_per_1m_tokens=inp,
                    llm_output_price_usd_per_1m_tokens=4.0 * inp,
                    output_token_num=(
                        TokenNumParams(mu=5.0, sigma=1.0).sample_int
                        if workflow == "query"
                        else TokenNumParams.from_mean(
                            mean=random.uniform(100, 2000),
                            sigma=random.uniform(0.3, 0.9),
                        ).sample_int
                    ),
                    llm_exec_latency_ms=LlmLatencyParams(
                        alpha_ms_per_token=random.uniform(20.0, 50.0),
                        beta_ms=random.uniform(400.0, 800.0),
                        noise_sigma_ms=random.uniform(30.0, 120.0),
                    ).sample_ms,
                    llm_option=opt,
                )
            )
    return nodes  # 2 * 4 = 8


def build_nodes(workflow: Workflow) -> list[Node]:
    """
    Returns nodes used by a workflow.

    Counts:
    - segment: 12 (storage only)
    - split:   12 (storage only)
    - caption: 20 (12 storage + 8 llm)
    - query:   20 (12 storage + 8 llm)
    """
    cloud = _build_cloud_nodes(workflow)
    if workflow in ("segment", "split"):
        return cloud
    if workflow in ("caption", "query"):
        return cloud + _build_llm_nodes(workflow)
    raise ValueError(f"Unknown workflow: {workflow}")


def build_nodes_with_params(
    workflow: Workflow,
    *,
    storage_prices: dict[str, float] | None = None,
    exec_time_params: dict[Workflow, ExecTimeParams | None] | None = None,
    segment_prices: dict[str, float] | None = None,
    split_prices: dict[str, float] | None = None,
    llm_token_prices: dict[str, tuple[float, float]] | None = None,
    llm_output_token_params: dict[str, TokenNumParams] | None = None,
    llm_latency_params: dict[str, LlmLatencyParams] | None = None,
) -> list[Node]:
    """
    简化用法（推荐）：
    1) 你先调用 sample_storage_prices / sample_exec_time_params 采样一次参数
    2) 后续把参数传进来，这里就会复用同一套参数构建节点
    """
    exec_params = exec_time_params[workflow] if exec_time_params is not None else None
    cloud = _build_cloud_nodes(
        workflow,
        storage_prices=storage_prices,
        exec_params=exec_params,
        segment_prices=segment_prices,
        split_prices=split_prices,
    )
    if workflow in ("segment", "split"):
        return cloud
    if workflow in ("caption", "query"):
        llm_nodes = _build_llm_nodes(workflow)
        if llm_token_prices is not None:
            fixed: list[Node] = []
            for n in llm_nodes:
                inp, outp = llm_token_prices[n.name]
                fixed.append(
                    Node(
                        name=n.name,
                        provider=n.provider,
                        region=n.region,
                        can_store=n.can_store,
                        storage_price_usd_per_gb_month=n.storage_price_usd_per_gb_month,
                        exec_time_s=n.exec_time_s,
                        segment_price_usd_per_min=n.segment_price_usd_per_min,
                        split_price_usd_per_s=n.split_price_usd_per_s,
                        llm_input_price_usd_per_1m_tokens=inp,
                        llm_output_price_usd_per_1m_tokens=outp,
                        output_token_num=n.output_token_num,
                        llm_option=n.llm_option,
                    )
                )
            llm_nodes = fixed

        if llm_output_token_params is not None:
            llm_nodes = [
                Node(
                    name=n.name,
                    provider=n.provider,
                    region=n.region,
                    can_store=n.can_store,
                    storage_price_usd_per_gb_month=n.storage_price_usd_per_gb_month,
                    exec_time_s=n.exec_time_s,
                    segment_price_usd_per_min=n.segment_price_usd_per_min,
                    split_price_usd_per_s=n.split_price_usd_per_s,
                    llm_input_price_usd_per_1m_tokens=n.llm_input_price_usd_per_1m_tokens,
                    llm_output_price_usd_per_1m_tokens=n.llm_output_price_usd_per_1m_tokens,
                    output_token_num=llm_output_token_params[n.name].sample_int,
                    llm_exec_latency_ms=n.llm_exec_latency_ms,
                    llm_option=n.llm_option,
                )
                for n in llm_nodes
            ]

        if llm_latency_params is not None:
            llm_nodes = [
                Node(
                    name=n.name,
                    provider=n.provider,
                    region=n.region,
                    can_store=n.can_store,
                    storage_price_usd_per_gb_month=n.storage_price_usd_per_gb_month,
                    exec_time_s=n.exec_time_s,
                    segment_price_usd_per_min=n.segment_price_usd_per_min,
                    split_price_usd_per_s=n.split_price_usd_per_s,
                    llm_input_price_usd_per_1m_tokens=n.llm_input_price_usd_per_1m_tokens,
                    llm_output_price_usd_per_1m_tokens=n.llm_output_price_usd_per_1m_tokens,
                    output_token_num=n.output_token_num,
                    llm_exec_latency_ms=llm_latency_params[n.name].sample_ms,
                    llm_option=n.llm_option,
                )
                for n in llm_nodes
            ]

        return cloud + llm_nodes
    raise ValueError(f"Unknown workflow: {workflow}")


def _assert_node_counts() -> None:
    expected = {"segment": 12, "split": 12, "caption": 20, "query": 20}
    for wf, n in expected.items():
        got = len(build_nodes(wf))  # type: ignore[arg-type]
        if got != n:
            raise AssertionError(f"{wf}: expected {n}, got {got}")


_assert_node_counts()


# data size, 均匀采样
def get_data_size_mb(min_mb: float = 5.0, max_mb: float = 500.0) -> float:
    if min_mb <= 0 or max_mb <= 0:
        raise ValueError("min_mb/max_mb must be > 0")
    if max_mb < min_mb:
        raise ValueError("max_mb must be >= min_mb")
    return random.uniform(min_mb, max_mb)


def get_data_conversion_ratio(workflow: Workflow) -> float:
    """
    Data conversion ratio = output_size / input_size.

    Sampled from a LogNormal distribution to keep ratio > 0.
    Configured per-workflow by a target mean ratio and a log-space sigma.
    """

    # 兼容旧接口：直接调用会“每次重新采样一套参数再采样一次 ratio”。
    # 更推荐：params = sample_ratio_params(); sample_data_conversion_ratio(workflow, params)
    return sample_ratio_params()[workflow].sample_ratio()


def sample_data_conversion_ratio(workflow: Workflow, params: dict[Workflow, RatioParams]) -> float:
    """
    推荐用法：
    - params = sample_ratio_params()  # 只运行一次
    - ratio  = sample_data_conversion_ratio("caption", params)
    """
    return params[workflow].sample_ratio()


# ----------------------------
# Network communication matrix (15 x 15)
# ----------------------------


@dataclass(frozen=True, slots=True)
class CommEndpoint:
    name: str  # e.g. "local", "p1_r1", "p4"
    provider: str  # "local" | "p1".."p5"
    region: str | None  # r1..r6 for cloud, None for local/p4/p5


@dataclass(frozen=True, slots=True)
class CommMatrix:
    endpoints: tuple[CommEndpoint, ...]
    rtt_ms: tuple[tuple[float, ...], ...]
    bandwidth_mbps: tuple[tuple[float, ...], ...]
    index_by_name: dict[str, int]

    def idx(self, node: int | str | CommEndpoint) -> int:
        if isinstance(node, int):
            return node
        if isinstance(node, str):
            try:
                return self.index_by_name[node]
            except KeyError as e:
                raise KeyError(f"Unknown endpoint name: {node}") from e
        # CommEndpoint
        try:
            return self.index_by_name[node.name]
        except KeyError as e:
            raise KeyError(f"Endpoint not in matrix: {node.name}") from e

    def link(self, a: int | str | CommEndpoint, b: int | str | CommEndpoint) -> tuple[float, float]:
        """
        Returns (rtt_ms, bandwidth_mbps) for the link a<->b.
        a/b can be index, endpoint name, or CommEndpoint.
        """
        i = self.idx(a)
        j = self.idx(b)
        return self.rtt_ms[i][j], self.bandwidth_mbps[i][j]


@dataclass(frozen=True, slots=True)
class EgressMatrix:
    endpoints: tuple[CommEndpoint, ...]
    usd_per_gb: tuple[tuple[float, ...], ...]  # non-symmetric
    index_by_name: dict[str, int]

    def idx(self, node: int | str | CommEndpoint) -> int:
        if isinstance(node, int):
            return node
        if isinstance(node, str):
            try:
                return self.index_by_name[node]
            except KeyError as e:
                raise KeyError(f"Unknown endpoint name: {node}") from e
        try:
            return self.index_by_name[node.name]
        except KeyError as e:
            raise KeyError(f"Endpoint not in matrix: {node.name}") from e

    def cost(self, src: int | str | CommEndpoint, dst: int | str | CommEndpoint) -> float:
        """
        Outbound egress cost (USD per GB) from src -> dst.
        Note: non-symmetric; inbound is free.
        """
        i = self.idx(src)
        j = self.idx(dst)
        return self.usd_per_gb[i][j]


def get_comm_endpoints() -> list[CommEndpoint]:
    """
    Communication endpoints (15 total):
    - local
    - p1's 4 regions, p2's 4 regions, p3's 4 regions (12)
    - p4, p5 (no region)
    """
    eps: list[CommEndpoint] = [CommEndpoint(name="local", provider="local", region=None)]
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            eps.append(CommEndpoint(name=f"{p}_{r}", provider=p, region=r))
    eps.append(CommEndpoint(name="p4", provider="p4", region=None))
    eps.append(CommEndpoint(name="p5", provider="p5", region=None))

    if len(eps) != 15:
        raise AssertionError(f"Expected 15 comm endpoints, got {len(eps)}")
    return eps


def _lognormal_ms(mean_ms: float, sigma: float) -> float:
    if mean_ms <= 0:
        raise ValueError("mean_ms must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    mu = math.log(mean_ms) - 0.5 * (sigma**2)
    return random.lognormvariate(mu, sigma)


def _pareto_scale(xm: float, alpha: float) -> float:
    """
    Returns xm * Pareto(alpha) where random.paretovariate(alpha) has support [1, inf).
    """
    if xm <= 0:
        raise ValueError("xm must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    return xm * random.paretovariate(alpha)


def build_comm_matrices(comm_params: CommParams) -> tuple[list[CommEndpoint], list[list[float]], list[list[float]]]:
    """
    Build symmetric 15x15 matrices:
    - rtt_ms[i][j]: round-trip time in milliseconds (LogNormal)
    - bw_mbps[i][j]: bandwidth in Mbps (Pareto)

    Constraints:
    - Same cloud provider (p1/p2/p3) => higher bandwidth, lower RTT
    - Same region label (e.g. r1) => lower RTT (even across p1/p2/p3)
    """
    eps = get_comm_endpoints()
    n = len(eps)
    rtt_ms: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    bw_mbps: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]

    rtt_cfg = {
        "same_region": (comm_params.same_region.rtt_mean_ms, comm_params.same_region.rtt_sigma),
        "same_cloud": (comm_params.same_cloud.rtt_mean_ms, comm_params.same_cloud.rtt_sigma),
        "cross_cloud": (comm_params.cross_cloud.rtt_mean_ms, comm_params.cross_cloud.rtt_sigma),
        "via_local": (comm_params.via_local.rtt_mean_ms, comm_params.via_local.rtt_sigma),
        "via_llm_only": (comm_params.via_llm_only.rtt_mean_ms, comm_params.via_llm_only.rtt_sigma),
    }
    bw_cfg = {
        "same_region": (comm_params.same_region.bw_xm_mbps, comm_params.same_region.bw_alpha),
        "same_cloud": (comm_params.same_cloud.bw_xm_mbps, comm_params.same_cloud.bw_alpha),
        "cross_cloud": (comm_params.cross_cloud.bw_xm_mbps, comm_params.cross_cloud.bw_alpha),
        "via_local": (comm_params.via_local.bw_xm_mbps, comm_params.via_local.bw_alpha),
        "via_llm_only": (comm_params.via_llm_only.bw_xm_mbps, comm_params.via_llm_only.bw_alpha),
    }

    def is_cloud(p: str) -> bool:
        return p in _CLOUD_PROVIDERS

    def is_llm_only(p: str) -> bool:
        return p in ("p4", "p5")

    for i in range(n):
        for j in range(i, n):
            if i == j:
                rtt_ms[i][j] = 0.0
                bw_mbps[i][j] = float("inf")
                continue

            a, b = eps[i], eps[j]

            involves_local = (a.provider == "local") or (b.provider == "local")
            involves_llm_only = is_llm_only(a.provider) or is_llm_only(b.provider)

            same_provider = a.provider == b.provider
            same_region = (a.region is not None) and (a.region == b.region)

            # Pick category with priority to satisfy constraints:
            # same_region lowest RTT; same_cloud next; then others.
            if same_region and is_cloud(a.provider) and is_cloud(b.provider):
                rtt_mean, rtt_sigma = rtt_cfg["same_region"]
                bw_xm, bw_alpha = bw_cfg["same_region"]
            elif same_provider and is_cloud(a.provider):
                rtt_mean, rtt_sigma = rtt_cfg["same_cloud"]
                bw_xm, bw_alpha = bw_cfg["same_cloud"]
            elif involves_local:
                rtt_mean, rtt_sigma = rtt_cfg["via_local"]
                bw_xm, bw_alpha = bw_cfg["via_local"]
            elif involves_llm_only:
                rtt_mean, rtt_sigma = rtt_cfg["via_llm_only"]
                bw_xm, bw_alpha = bw_cfg["via_llm_only"]
            else:
                rtt_mean, rtt_sigma = rtt_cfg["cross_cloud"]
                bw_xm, bw_alpha = bw_cfg["cross_cloud"]

            # Pair-specific jitter (keeps ordering but adds diversity).
            rtt = _lognormal_ms(rtt_mean * random.uniform(0.9, 1.1), rtt_sigma)
            bw = _pareto_scale(bw_xm * random.uniform(0.9, 1.1), bw_alpha)

            rtt_ms[i][j] = rtt_ms[j][i] = rtt
            bw_mbps[i][j] = bw_mbps[j][i] = bw

    return eps, rtt_ms, bw_mbps


def build_comm_matrix(comm_params: CommParams) -> CommMatrix:
    """
    Convenience wrapper around build_comm_matrices() that supports lookup:

    - by indices: matrix.link(0, 3)
    - by names:   matrix.link("p1_r1", "p2_r1")
    - by endpoint objects
    """
    eps, rtt, bw = build_comm_matrices(comm_params)
    endpoints = tuple(eps)
    n = len(endpoints)
    index_by_name = {ep.name: i for i, ep in enumerate(endpoints)}
    rtt_t = tuple(tuple(rtt[i][j] for j in range(n)) for i in range(n))
    bw_t = tuple(tuple(bw[i][j] for j in range(n)) for i in range(n))
    return CommMatrix(endpoints=endpoints, rtt_ms=rtt_t, bandwidth_mbps=bw_t, index_by_name=index_by_name)


def build_egress_matrix(egress_params: EgressParams) -> EgressMatrix:
    """
    Build a non-symmetric 15x15 egress-cost matrix (USD per GB).

    Rules:
    - Ingress is free; only egress (outbound) is charged.
    - Only sources in cloud providers (p1/p2/p3) are charged.
    - If src is cloud:
      - dst in same provider => intra_provider_usd_per_gb
      - dst in different provider (including local/p4/p5) => cross_provider_usd_per_gb
    - If src is not cloud (local/p4/p5) => 0 everywhere.
    """
    eps = tuple(get_comm_endpoints())
    n = len(eps)
    index_by_name = {ep.name: i for i, ep in enumerate(eps)}

    def is_cloud(p: str) -> bool:
        return p in _CLOUD_PROVIDERS

    mat: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        src = eps[i]
        if not is_cloud(src.provider):
            continue  # all zeros
        for j in range(n):
            if i == j:
                mat[i][j] = 0.0
                continue
            dst = eps[j]
            if dst.provider == src.provider:
                mat[i][j] = egress_params.intra_provider_usd_per_gb
            else:
                mat[i][j] = egress_params.cross_provider_usd_per_gb

    usd_per_gb = tuple(tuple(mat[i][j] for j in range(n)) for i in range(n))
    return EgressMatrix(endpoints=eps, usd_per_gb=usd_per_gb, index_by_name=index_by_name)



