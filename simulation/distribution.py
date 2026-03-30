from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import math
import random

from config import get_simulation_config

# set global random number seed
# 全局随机种子：保证每次运行采样/计算结果可复现
random.seed(42)

# ----------------------------
# Node definitions (12+12+20+20)
# 节点数量说明：segment/split 各 12 个（仅云存储），caption/query 各 20 个（含 8 个 LLM 节点）
# ----------------------------

Operation = Literal["segment", "split", "caption", "query"]
ExecTimeFn = Callable[[float], float]
TokenNumFn = Callable[[float], int]  # video_size_mb -> token count
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
    utility: float | None
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
# p1,p2,p3 各表示一个模型（云节点 llm_option 为 p1_cloud 等，模型 key 为 p1,p2,p3）
_LLM_MODEL_KEYS: tuple[str, ...] = ("m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3")


@dataclass(frozen=True, slots=True)
class ExecTimeParams:
    io_s_per_mb: float
    k: float
    theta0: float
    theta1: float

    def sample_seconds(self, video_size_mb: float) -> float:
        if video_size_mb < 0:
            raise ValueError("video_size_mb must be >= 0")
        # I/O 延迟：按数据量线性增长（秒/MB * MB）
        io_time = self.io_s_per_mb * video_size_mb
        # 计算延迟：用 Gamma 分布模拟（参数随输入规模变化）
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
        # ratio 使用 LogNormal：log(X) ~ Normal(mu, sigma)
        mu = math.log(self.mean) - 0.5 * (self.sigma**2)
        return random.lognormvariate(mu, self.sigma)


@dataclass(frozen=True, slots=True)
class TokenNumParams:
    """
    LogNormal distribution for token counts (positive integers).
    Parameterized by log-space (mu, sigma): log(X) ~ Normal(mu, sigma).

    中文：token 数使用对数正态分布（保证为正且通常为“长尾”整数）。
    其中 log(X) 服从 Normal(mu, sigma)，用于由给定的（均值/方差）推导 log-space 参数。
    """

    mu: float
    sigma: float

    @staticmethod
    def from_mean(mean: float, sigma: float) -> "TokenNumParams":
        """
        For LogNormal(mu, sigma): E[X] = exp(mu + 0.5*sigma^2)
        => mu = ln(mean) - 0.5*sigma^2

        中文：由 LogNormal 的期望公式反推 mu，使得 E[X] 约等于传入的 mean。
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

    def as_video_fn(self) -> TokenNumFn:
        """Query: token count 与 video_size 无关，返回忽略参数的 wrapper。"""
        s = self.sample_int

        def fn(_video_size_mb: float) -> int:
            return s()

        return fn


@dataclass(frozen=True, slots=True)
class CaptionTokenNumParams:
    """
    Caption output tokens 与 video_size_mb 正相关：mean = base + coef_per_mb * video_size_mb。
    LogNormal(mean, sigma) 采样。
    """

    base: float  # 基础 token 数（video_size≈0 时）
    coef_per_mb: float  # 每 MB 增加的 token 数
    sigma: float  # log-space std

    def sample_int(self, video_size_mb: float) -> int:
        if video_size_mb < 0:
            raise ValueError("video_size_mb must be >= 0")
        mean = self.base + self.coef_per_mb * video_size_mb
        mean = max(mean, 20.0)
        mu = math.log(mean) - 0.5 * (self.sigma**2)
        x = random.lognormvariate(mu, self.sigma)
        return max(1, int(round(x)))

    def as_video_fn(self) -> TokenNumFn:
        return self.sample_int


@dataclass(frozen=True, slots=True)
class LlmLatencyParams:
    """
    LLM execution latency model (milliseconds):

      T = alpha_ms_per_token * N + beta_ms + Normal(0, noise_sigma_ms)

    中文：LLM 执行延迟（毫秒）模型：
    - 与 token 数 N 成线性关系（alpha）
    - 固定开销（beta）
    - 额外噪声（Normal(0, noise_sigma_ms)）
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
class UtilityParams:
    A: float
    B: float


def sample_utility_params() -> dict[Operation, UtilityParams]:
    """
    为每个 operation 采样 utility 参数 (A, B)。

    公式：Accuracy = A + B * ln(P + 1)，高价 → 高 utility。
    B 取较大值使价格差异有适度体现（不致过平或过陡）。
    结果 clamp 到 [0, 1]，不做归一化。
    """
    cfg = get_simulation_config()
    u = cfg.utility
    return {
        "segment": None,
        "split": None,
        "caption": UtilityParams(
            A=random.uniform(*u.caption_query_A),
            B=random.uniform(*u.caption_query_B),
        ),
        "query": UtilityParams(
            A=random.uniform(*u.caption_query_A),
            B=random.uniform(*u.caption_query_B),
        ),
    }


def _price_for_utility(operation: Operation, node: Node) -> float:
    """
    Choose the 'Price' used in utility formula for a node.

    中文：utility 公式里的“Price”取决于 operation：
    - segment：用 segment 的 USD/min
    - split：用 split 的 USD/s
    - caption/query：
      - LLM 节点：用 LLM 输入 token 的单价
      - 云存储节点：用存储价格（USD/GB/月）
    """
    if operation == "segment":
        if node.segment_price_usd_per_min is None:
            raise ValueError(f"Missing segment price for node {node.name}")
        return node.segment_price_usd_per_min
    if operation == "split":
        if node.split_price_usd_per_s is None:
            raise ValueError(f"Missing split price for node {node.name}")
        return node.split_price_usd_per_s

    # caption/query:
    # - LLM nodes: token input price
    # - cloud nodes: storage price
    if node.llm_option is not None:
        if node.llm_input_price_usd_per_1m_tokens is None:
            raise ValueError(f"Missing llm input price for node {node.name}")
        return node.llm_input_price_usd_per_1m_tokens
    if node.storage_price_usd_per_gb_month is None:
        raise ValueError(f"Missing storage price for node {node.name}")
    return node.storage_price_usd_per_gb_month


def _llm_model_key(node: Node, operation: Operation) -> str:
    """
    节点对应的模型 key：LLM 节点用 llm_option(m1..m6)，云 caption/query 用 provider(p1,p2,p3) 表示各模型。
    """
    if node.llm_option is not None:
        if node.llm_option in ("m1", "m2", "m3", "m4", "m5", "m6"):
            return node.llm_option
        # p1_cloud, p2_cloud, p3_cloud -> p1, p2, p3
        return node.provider
    return node.provider


def _utility_group_key(operation: Operation, node: Node) -> str:
    """
    同一 group 内节点 utility 相同：
    - segment/split：按 provider（同 provider 不同 region 相同）
    - caption/query 云节点：按 provider
    - caption/query LLM 节点：按 llm_option（同模型不同 provider 相同）
    """
    if operation in ("segment", "split"):
        return node.provider
    if node.llm_option is not None:
        return f"llm_{node.llm_option}"
    return node.provider


def compute_and_normalize_utilities(
    operation: Operation,
    nodes: list[Node],
    params: UtilityParams,
    *,
    segment_provider_centers: dict[str, float] | None = None,
    split_provider_centers: dict[str, float] | None = None,
    llm_utility: dict[str, float] | None = None,
) -> list[Node]:
    """
    Compute per-node utility:

    - split：所有节点 utility = 1（固定）
    - segment：同 provider 不同 region 的 utility 相同，按 provider 分组。
      utility 从 [0.6, 1] 区间随机采样（每个 provider 采样一次）。
    - caption/query：同 provider 的云节点 utility 相同；同 llm_option 的 LLM 节点 utility 相同。
      公式：mu_k = min(0.99, A + B * log10(Price))，按 group 用代表价格计算。

    对于 segment/split：若传入 provider_centers，则用中心价作为 group 价格；否则用第一个节点的价格。
    """
    if not nodes:
        return nodes

    # split：utility 固定为 1
    if operation == "split":
        return [
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
                llm_exec_latency_ms=n.llm_exec_latency_ms,
                utility=1.0,
                llm_option=n.llm_option,
            )
            for n in nodes
        ]

    if operation == "segment":
        cfg = get_simulation_config()
        provider_to_utility: dict[str, float] = {}
        for n in nodes:
            if n.provider not in provider_to_utility:
                provider_to_utility[n.provider] = random.uniform(*cfg.utility.segment_provider)
        norm = [provider_to_utility[n.provider] for n in nodes]
    else:
        # caption/query：有 llm_utility 则按模型取 [0.7,1] 均匀采样值，否则按价格公式
        if llm_utility is not None:
            norm = [llm_utility[_llm_model_key(n, operation)] for n in nodes]
        else:
            group_to_price: dict[str, float] = {}
            for n in nodes:
                key = _utility_group_key(operation, n)
                if key not in group_to_price:
                    price = _price_for_utility(operation, n)
                    if price <= 0:
                        raise ValueError(f"Price must be > 0 for utility; got {price} ({n.name})")
                    group_to_price[key] = price

            group_to_raw: dict[str, float] = {}
            for k, p in group_to_price.items():
                raw = params.A + params.B * math.log(p + 1.0)
                group_to_raw[k] = max(0.0, min(1.0, raw))

            norm = [group_to_raw[_utility_group_key(operation, n)] for n in nodes]

    out: list[Node] = []
    for n, u in zip(nodes, norm, strict=True):
        out.append(
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
                llm_exec_latency_ms=n.llm_exec_latency_ms,
                utility=u,
                llm_option=n.llm_option,
            )
        )
    return out


# 模块加载时采样一次，供 build_nodes / build_nodes_with_params 做 utility 归一化
DEFAULT_UTILITY_PARAMS: dict[Operation, UtilityParams] = sample_utility_params()


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


def sample_ratio_params() -> dict[Operation, RatioParams]:
    """
    随机采样一次 per-operation 的 ratio(LogNormal) 分布参数。
    你应该保存返回值并在后续采样时复用它。
    """
    cfg = get_simulation_config()
    r = cfg.ratio
    seg_sigma = random.uniform(*r.seg_sigma)
    spl_sigma = random.uniform(*r.spl_sigma)
    cap_mean = random.uniform(*r.cap_mean)
    cap_sigma = random.uniform(*r.cap_sigma)
    qry_mean = random.uniform(*r.qry_mean)
    qry_sigma = random.uniform(*r.qry_sigma)

    return {
        "segment": RatioParams(mean=1.0, sigma=seg_sigma),
        "split": RatioParams(mean=1.0, sigma=spl_sigma),
        "caption": RatioParams(mean=cap_mean, sigma=cap_sigma),
        "query": RatioParams(mean=qry_mean, sigma=qry_sigma),
    }


def sample_exec_time_params() -> dict[Operation, dict[str, ExecTimeParams] | None]:
    """
    随机采样一次执行时间分布参数。
    仅对 segment/split 有参数；caption/query 为 None。
    每个 provider+region 组合独立采样，不同节点参数不同。

    校准参考：results/timing_logs，video ~10–15 MB 时：
    - segment: ~60–68 s
    - split: ~11–24 s
    - segment 时长约为 split 的 5–6 倍
    """

    cfg = get_simulation_config()
    es, esp = cfg.exec_time_segment, cfg.exec_time_split

    def _segment_params() -> ExecTimeParams:
        return ExecTimeParams(
            io_s_per_mb=random.uniform(*es.io_s_per_mb),
            k=random.uniform(*es.k),
            theta0=random.uniform(*es.theta0),
            theta1=random.uniform(*es.theta1),
        )

    def _split_params() -> ExecTimeParams:
        return ExecTimeParams(
            io_s_per_mb=random.uniform(*esp.io_s_per_mb),
            k=random.uniform(*esp.k),
            theta0=random.uniform(*esp.theta0),
            theta1=random.uniform(*esp.theta1),
        )

    return {
        "segment": {f"{p}_{r}_segment": _segment_params() for p in _CLOUD_PROVIDERS for r in _CLOUD_REGIONS[p]},
        "split": {f"{p}_{r}_split": _split_params() for p in _CLOUD_PROVIDERS for r in _CLOUD_REGIONS[p]},
        "caption": None,
        "query": None,
    }


DEFAULT_EXEC_TIME_PARAMS: dict[Operation, dict[str, ExecTimeParams] | None] = sample_exec_time_params()


def sample_comm_params() -> CommParams:
    """
    随机采样一次网络类别参数（RTT=LogNormal，bandwidth=Pareto）。
    你应该保存返回值并用于后续生成矩阵。
    """

    cfg = get_simulation_config()
    c = cfg.comm

    def cat(cat_cfg: Any) -> CommCategoryParams:
        return CommCategoryParams(
            rtt_mean_ms=random.uniform(*cat_cfg.rtt_mean),
            rtt_sigma=random.uniform(*cat_cfg.rtt_sigma),
            bw_xm_mbps=random.uniform(*cat_cfg.bw_xm),
            bw_alpha=random.uniform(*cat_cfg.bw_alpha),
        )

    return CommParams(
        same_region=cat(c.same_region),
        same_cloud=cat(c.same_cloud),
        cross_cloud=cat(c.cross_cloud),
        via_local=cat(c.via_local),
        via_llm_only=cat(c.via_llm_only),
    )


def sample_storage_prices(operation: Operation) -> dict[str, float]:
    """
    随机采样一次某个 operation 下云节点的存储价格（按节点名返回）。
    已废弃：请用 sample_storage_prices_by_provider_region 保证同 provider+region 价格一致。
    """
    cfg = get_simulation_config()
    lo, hi = cfg.storage_price
    prices: dict[str, float] = {}
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            prices[f"{p}_{r}_{operation}"] = random.uniform(lo, hi)
    return prices


def sample_storage_prices_by_provider_region() -> dict[str, float]:
    """
    按 (provider, region) 采样存储价格，同 provider+region 在所有 operation 下价格一致。
    返回: {"p1_r1": 0.02, "p1_r2": ..., ...}
    """
    cfg = get_simulation_config()
    lo, hi = cfg.storage_price
    prices: dict[str, float] = {}
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            key = f"{p}_{r}"
            prices[key] = random.uniform(lo, hi)
    return prices


def sample_egress_params() -> EgressParams:
    """
    随机采样一次出站流量费用参数（USD per GB）。
    """
    cfg = get_simulation_config()
    return EgressParams(
        intra_provider_usd_per_gb=random.uniform(*cfg.egress_intra),
        cross_provider_usd_per_gb=random.uniform(*cfg.egress_cross),
    )


def sample_segment_prices() -> tuple[dict[str, float], dict[str, float]]:
    """
    Segment 价格：每个 provider 有中心价，各 region 在中心价附近波动。

    Returns:
        (node_prices, provider_centers)
        - node_prices: node_name -> 价格（用于成本计算）
        - provider_centers: provider -> 中心价（用于 utility 分组）
    """
    cfg = get_simulation_config()
    operation: Operation = "segment"
    lo, hi = cfg.segment_center
    provider_centers: dict[str, float] = {
        p: random.uniform(lo, hi) for p in _CLOUD_PROVIDERS
    }
    node_prices: dict[str, float] = {}
    fluc = cfg.price_fluctuation
    for p in _CLOUD_PROVIDERS:
        center = provider_centers[p]
        for r in _CLOUD_REGIONS[p]:
            name = f"{p}_{r}_{operation}"
            fluctuation = random.uniform(-fluc, fluc)
            node_prices[name] = max(0.001, center * (1.0 + fluctuation))
    return node_prices, provider_centers


def sample_split_prices() -> tuple[dict[str, float], dict[str, float]]:
    """
    Split 价格：每个 provider 有中心价，各 region 在中心价附近波动。

    Returns:
        (node_prices, provider_centers)
    """
    cfg = get_simulation_config()
    operation: Operation = "split"
    lo, hi = cfg.split_center
    provider_centers: dict[str, float] = {
        p: random.uniform(lo, hi) for p in _CLOUD_PROVIDERS
    }
    node_prices: dict[str, float] = {}
    fluc = cfg.price_fluctuation
    for p in _CLOUD_PROVIDERS:
        center = provider_centers[p]
        for r in _CLOUD_REGIONS[p]:
            name = f"{p}_{r}_{operation}"
            fluctuation = random.uniform(-fluc, fluc)
            node_prices[name] = max(0.000001, center * (1.0 + fluctuation))
    return node_prices, provider_centers


# 模块加载时采样一次
_DEFAULT_SEGMENT_PRICES, _DEFAULT_SEGMENT_PROVIDER_CENTERS = sample_segment_prices()
_DEFAULT_SPLIT_PRICES, _DEFAULT_SPLIT_PROVIDER_CENTERS = sample_split_prices()


def sample_llm_token_prices(operation: Operation) -> dict[str, tuple[float, float]]:
    """
    LLM price per 1M tokens. Returns mapping: llm_node_name -> (input_price, output_price).
    """
    cfg = get_simulation_config()
    lo, hi = cfg.llm_input_price
    mult = cfg.llm_output_multiplier
    prices: dict[str, tuple[float, float]] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            name = f"{p}_{opt}_{operation}"
            inp = random.uniform(lo, hi)
            prices[name] = (inp, mult * inp)
    return prices


def _sample_llm_output_token_params_by_model(
    operation: Operation,
) -> dict[str, CaptionTokenNumParams | TokenNumParams]:
    """按模型 key 采样，同模型相同。"""
    cfg = get_simulation_config()
    lot = cfg.llm_output_token
    by_model: dict[str, CaptionTokenNumParams | TokenNumParams] = {}
    for mk in _LLM_MODEL_KEYS:
        if operation == "query":
            by_model[mk] = TokenNumParams(
                mu=random.uniform(*lot.query_mu),
                sigma=random.uniform(*lot.query_sigma),
            )
        else:
            by_model[mk] = CaptionTokenNumParams(
                base=random.uniform(*lot.caption_base),
                coef_per_mb=random.uniform(*lot.caption_coef_per_mb),
                sigma=random.uniform(*lot.caption_sigma),
            )
    return by_model


def sample_llm_output_token_params(
    operation: Operation,
    *,
    by_model: dict[str, CaptionTokenNumParams | TokenNumParams] | None = None,
) -> dict[str, CaptionTokenNumParams | TokenNumParams]:
    """
    同模型 output_token_params 相同。by_model 可选，不传则内部采样。
    """
    model_params = by_model or _sample_llm_output_token_params_by_model(operation)
    params: dict[str, CaptionTokenNumParams | TokenNumParams] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            params[f"{p}_{opt}_{operation}"] = model_params[opt]
    return params


def sample_llm_utility(operation: Operation) -> dict[str, float]:
    """模型 utility：均匀采样，同模型相同。"""
    cfg = get_simulation_config()
    lo, hi = cfg.utility.llm_model
    return {mk: random.uniform(lo, hi) for mk in _LLM_MODEL_KEYS}


def sample_cloud_llm_base_params() -> tuple[
    dict[str, tuple[float, float]],
    dict[str, LlmLatencyParams],
]:
    """
    同 provider+region 的 cloud LLM：无论 caption 还是 query，price 与 latency 参数一致。
    采样一次，caption/query 复用。返回 (token_prices_by_pr, latency_params_by_pr)，
    key 为 "p1_r1" 等。
    """
    cfg = get_simulation_config()
    lo, hi = cfg.llm_input_price
    mult = cfg.llm_output_multiplier
    ll = cfg.llm_latency
    token_prices: dict[str, tuple[float, float]] = {}
    latency_params: dict[str, LlmLatencyParams] = {}
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            pr_key = f"{p}_{r}"
            inp = random.uniform(lo, hi)
            token_prices[pr_key] = (inp, mult * inp)
            latency_params[pr_key] = LlmLatencyParams(
                alpha_ms_per_token=random.uniform(*ll.alpha_ms_per_token),
                beta_ms=random.uniform(*ll.beta_ms),
                noise_sigma_ms=random.uniform(*ll.noise_sigma_ms),
            )
    return token_prices, latency_params


def sample_cloud_llm_params(
    operation: Operation,
    *,
    by_model: dict[str, CaptionTokenNumParams | TokenNumParams] | None = None,
    base_token_prices: dict[str, tuple[float, float]] | None = None,
    base_latency_params: dict[str, LlmLatencyParams] | None = None,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, CaptionTokenNumParams | TokenNumParams],
    dict[str, LlmLatencyParams],
]:
    """
    同 provider+region 的 price/latency 一致（跨 caption/query）。output_token_params 按任务不同。
    base_token_prices, base_latency_params 若传入则复用（来自 sample_cloud_llm_base_params）。
    """
    if by_model is None:
        by_model = _sample_llm_output_token_params_by_model(operation)
    token_prices: dict[str, tuple[float, float]] = {}
    output_params: dict[str, CaptionTokenNumParams | TokenNumParams] = {}
    latency_params: dict[str, LlmLatencyParams] = {}
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            pr_key = f"{p}_{r}"
            name = f"{p}_{r}_{operation}"
            if base_token_prices is not None:
                token_prices[name] = base_token_prices[pr_key]
            else:
                cfg = get_simulation_config()
                inp = random.uniform(*cfg.llm_input_price)
                token_prices[name] = (inp, cfg.llm_output_multiplier * inp)
            output_params[name] = by_model[p]
            if base_latency_params is not None:
                latency_params[name] = base_latency_params[pr_key]
            else:
                ll = get_simulation_config().llm_latency
                latency_params[name] = LlmLatencyParams(
                    alpha_ms_per_token=random.uniform(*ll.alpha_ms_per_token),
                    beta_ms=random.uniform(*ll.beta_ms),
                    noise_sigma_ms=random.uniform(*ll.noise_sigma_ms),
                )
    return token_prices, output_params, latency_params


def sample_llm_latency_params(operation: Operation) -> dict[str, LlmLatencyParams]:
    """为 LLM 节点采样 execution latency 参数（只采样一次并复用）。"""
    cfg = get_simulation_config()
    ll = cfg.llm_latency
    params: dict[str, LlmLatencyParams] = {}
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            name = f"{p}_{opt}_{operation}"
            params[name] = LlmLatencyParams(
                alpha_ms_per_token=random.uniform(*ll.alpha_ms_per_token),
                beta_ms=random.uniform(*ll.beta_ms),
                noise_sigma_ms=random.uniform(*ll.noise_sigma_ms),
            )
    return params


def _make_exec_time_fn() -> ExecTimeFn:
    """
    Returns a function: exec_time_s(video_size_mb) = io_time + computation_time.

    - io_time is proportional to data size (seconds per MB).
    - computation_time | size ~ Gamma(k, theta(size)) where theta increases with size.
    All parameters are sampled once when the function is created (seeded by global RNG).

    中文：构造一个“执行时长函数” `exec_time_s(video_size_mb)`：
    - I/O 时间随输入数据量线性增长（秒/MB * MB）
    - 计算时间用 Gamma 分布建模，且随输入规模增长（theta 随 size 上升）
    - 函数创建时采样的随机参数会被固定复用（来自全局 RNG 的状态）
    """
    cfg = get_simulation_config()
    cef = cfg.cloud_exec_fallback
    io_s_per_mb = random.uniform(*cef.io_s_per_mb)
    k = random.uniform(*cef.k)
    theta0 = random.uniform(*cef.theta0)
    theta1 = random.uniform(*cef.theta1)

    def f(video_size_mb: float) -> float:
        if video_size_mb < 0:
            raise ValueError("video_size_mb must be >= 0")
        io_time = io_s_per_mb * video_size_mb
        theta = theta0 + theta1 * video_size_mb
        comp_time = random.gammavariate(k, theta)
        return io_time + comp_time

    return f


def _build_cloud_nodes(
    operation: Operation,
    *,
    storage_prices: dict[str, float] | None = None,
    exec_params: dict[str, ExecTimeParams] | None = None,
    segment_prices: dict[str, float] | None = None,
    split_prices: dict[str, float] | None = None,
    cloud_llm_token_prices: dict[str, tuple[float, float]] | None = None,
    cloud_llm_output_token_params: dict[str, CaptionTokenNumParams | TokenNumParams] | None = None,
    cloud_llm_latency_params: dict[str, LlmLatencyParams] | None = None,
) -> list[Node]:
    """
    exec_params: 按节点名 (provider_region_operation) 映射到 ExecTimeParams。
    仅 segment/split 需要；caption/query 传 None。

    caption/query 云节点：若传入 cloud_llm_* 则复用，否则现场采样。
    """
    nodes: list[Node] = []
    for p in _CLOUD_PROVIDERS:
        for r in _CLOUD_REGIONS[p]:
            name = f"{p}_{r}_{operation}"
            params = exec_params.get(name) if exec_params is not None else None
            exec_fn: ExecTimeFn | None = params.sample_seconds if params is not None else None

            # caption/query 云节点：LLM 属性（传入则复用，否则采样）
            llm_inp: float | None = None
            llm_out: float | None = None
            out_tok: TokenNumFn | None = None
            llm_lat: LlmLatencyFn | None = None
            llm_opt: str | None = None
            if operation in ("caption", "query"):
                if (
                    cloud_llm_token_prices is not None
                    and cloud_llm_output_token_params is not None
                    and cloud_llm_latency_params is not None
                    and name in cloud_llm_token_prices
                ):
                    inp, outp = cloud_llm_token_prices[name]
                    llm_inp, llm_out = inp, outp
                    out_tok = cloud_llm_output_token_params[name].as_video_fn()
                    llm_lat = cloud_llm_latency_params[name].sample_ms
                else:
                    cfg = get_simulation_config()
                    llm_inp = random.uniform(*cfg.llm_input_price)
                    llm_out = cfg.llm_output_multiplier * llm_inp
                    lot = cfg.llm_output_token
                    if operation == "query":
                        out_tok = TokenNumParams(
                            mu=lot.query_fallback_mu,
                            sigma=lot.query_fallback_sigma,
                        ).as_video_fn()
                    else:
                        out_tok = CaptionTokenNumParams(
                            base=random.uniform(*lot.caption_base),
                            coef_per_mb=random.uniform(*lot.caption_coef_per_mb),
                            sigma=random.uniform(*lot.caption_sigma),
                        ).as_video_fn()
                    ll = cfg.llm_latency
                    llm_lat = LlmLatencyParams(
                        alpha_ms_per_token=random.uniform(*ll.alpha_ms_per_token),
                        beta_ms=random.uniform(*ll.beta_ms),
                        noise_sigma_ms=random.uniform(*ll.noise_sigma_ms),
                    ).sample_ms
                llm_opt = f"{p}_cloud"

            nodes.append(
                Node(
                    name=name,
                    provider=p,
                    region=r,
                    can_store=True,
                    storage_price_usd_per_gb_month=(
                        storage_prices[name]
                        if storage_prices is not None
                        else random.uniform(*get_simulation_config().storage_price)
                    ),
                    exec_time_s=exec_fn,
                    segment_price_usd_per_min=(
                        (segment_prices[name] if segment_prices is not None else random.uniform(*get_simulation_config().segment_center))
                        if operation == "segment"
                        else None
                    ),
                    split_price_usd_per_s=(
                        (split_prices[name] if split_prices is not None else random.uniform(*get_simulation_config().split_center))
                        if operation == "split"
                        else None
                    ),
                    llm_input_price_usd_per_1m_tokens=llm_inp,
                    llm_output_price_usd_per_1m_tokens=llm_out,
                    output_token_num=out_tok,
                    llm_exec_latency_ms=llm_lat,
                    utility=None,
                    llm_option=llm_opt,
                )
            )
    return nodes  # 3 * 4 = 12


def _build_llm_nodes(operation: Operation) -> list[Node]:
    cfg = get_simulation_config()
    lot, ll = cfg.llm_output_token, cfg.llm_latency
    inp_lo, inp_hi = cfg.llm_input_price
    nodes: list[Node] = []
    for p, provider_options in _LLM_PROVIDER_TO_OPTIONS.items():
        for opt in provider_options:
            inp = random.uniform(inp_lo, inp_hi)
            nodes.append(
                Node(
                    name=f"{p}_{opt}_{operation}",
                    provider=p,
                    region=None,
                    can_store=False,
                    storage_price_usd_per_gb_month=None,
                    exec_time_s=None,
                    segment_price_usd_per_min=None,
                    split_price_usd_per_s=None,
                    llm_input_price_usd_per_1m_tokens=inp,
                    llm_output_price_usd_per_1m_tokens=cfg.llm_output_multiplier * inp,
                    output_token_num=(
                        TokenNumParams(mu=lot.query_fallback_mu, sigma=lot.query_fallback_sigma).as_video_fn()
                        if operation == "query"
                        else CaptionTokenNumParams(
                            base=random.uniform(*lot.caption_base),
                            coef_per_mb=random.uniform(*lot.caption_coef_per_mb),
                            sigma=random.uniform(*lot.caption_sigma),
                        ).as_video_fn()
                    ),
                    llm_exec_latency_ms=LlmLatencyParams(
                        alpha_ms_per_token=random.uniform(*ll.alpha_ms_per_token),
                        beta_ms=random.uniform(*ll.beta_ms),
                        noise_sigma_ms=random.uniform(*ll.noise_sigma_ms),
                    ).sample_ms,
                    utility=None,
                    llm_option=opt,
                )
            )
    return nodes  # 2 * 4 = 8


def build_nodes(operation: Operation) -> list[Node]:
    """
    Returns nodes used by a operation.

    Counts:
    - segment: 12 (storage only)
    - split:   12 (storage only)
    - caption: 20 (12 storage + 8 llm)
    - query:   20 (12 storage + 8 llm)

    中文：根据 operation 返回可用的计算节点集合。
    - segment/split：只有云存储节点（用于 I/O + 云端处理），共 12 个
    - caption/query：云存储节点 + LLM 节点（用于生成 caption / query），共 20 个
    """
    cloud = _build_cloud_nodes(
        operation,
        exec_params=DEFAULT_EXEC_TIME_PARAMS[operation],
        segment_prices=_DEFAULT_SEGMENT_PRICES if operation == "segment" else None,
        split_prices=_DEFAULT_SPLIT_PRICES if operation == "split" else None,
    )
    if operation in ("segment", "split"):
        return compute_and_normalize_utilities(
            operation,
            cloud,
            DEFAULT_UTILITY_PARAMS[operation],
            segment_provider_centers=_DEFAULT_SEGMENT_PROVIDER_CENTERS if operation == "segment" else None,
            split_provider_centers=_DEFAULT_SPLIT_PROVIDER_CENTERS if operation == "split" else None,
        )
    if operation in ("caption", "query"):
        nodes = cloud + _build_llm_nodes(operation)
        return compute_and_normalize_utilities(operation, nodes, DEFAULT_UTILITY_PARAMS[operation])
    raise ValueError(f"Unknown operation: {operation}")


def _infer_provider_centers_from_node_prices(
    operation: Operation, node_prices: dict[str, float]
) -> dict[str, float]:
    """从 node_prices 推断各 provider 的中心价（取同 provider 内 region 的均值）。"""
    provider_to_prices: dict[str, list[float]] = {}
    for name, price in node_prices.items():
        if not name.endswith(f"_{operation}"):
            continue
        # name 格式: p1_r1_segment
        provider = name.split("_")[0]
        provider_to_prices.setdefault(provider, []).append(price)
    return {p: sum(prices) / len(prices) for p, prices in provider_to_prices.items()}


def build_nodes_with_params(
    operation: Operation,
    *,
    storage_prices: dict[str, float] | None = None,
    exec_time_params: dict[Operation, dict[str, ExecTimeParams] | None] | None = None,
    segment_prices: dict[str, float] | None = None,
    split_prices: dict[str, float] | None = None,
    segment_provider_centers: dict[str, float] | None = None,
    split_provider_centers: dict[str, float] | None = None,
    llm_token_prices: dict[str, tuple[float, float]] | None = None,
    llm_output_token_params: dict[str, CaptionTokenNumParams | TokenNumParams] | None = None,
    llm_latency_params: dict[str, LlmLatencyParams] | None = None,
    cloud_llm_token_prices: dict[str, tuple[float, float]] | None = None,
    cloud_llm_output_token_params: dict[str, CaptionTokenNumParams | TokenNumParams] | None = None,
    cloud_llm_latency_params: dict[str, LlmLatencyParams] | None = None,
    llm_utility: dict[str, float] | None = None,
) -> list[Node]:
    """
    简化用法（推荐）：
    1) 你先调用 sample_storage_prices / sample_exec_time_params / sample_segment_prices 等采样一次参数
    2) 后续把参数传进来，这里就会复用同一套参数构建节点

    segment/split 价格：sample_segment_prices() 返回 (node_prices, provider_centers)。
    若只传 segment_prices 未传 segment_provider_centers，则从 node_prices 推断中心价。
    """
    exec_params = (
        exec_time_params[operation]
        if exec_time_params is not None
        else DEFAULT_EXEC_TIME_PARAMS[operation]
    )
    cloud_llm_tp = cloud_llm_token_prices if operation in ("caption", "query") else None
    cloud_llm_otp = cloud_llm_output_token_params if operation in ("caption", "query") else None
    cloud_llm_lp = cloud_llm_latency_params if operation in ("caption", "query") else None
    cloud = _build_cloud_nodes(
        operation,
        storage_prices=storage_prices,
        exec_params=exec_params,
        segment_prices=segment_prices or (_DEFAULT_SEGMENT_PRICES if operation == "segment" else None),
        split_prices=split_prices or (_DEFAULT_SPLIT_PRICES if operation == "split" else None),
        cloud_llm_token_prices=cloud_llm_tp,
        cloud_llm_output_token_params=cloud_llm_otp,
        cloud_llm_latency_params=cloud_llm_lp,
    )
    if operation in ("segment", "split"):
        seg_centers = segment_provider_centers
        spl_centers = split_provider_centers
        if operation == "segment" and seg_centers is None and segment_prices is not None:
            seg_centers = _infer_provider_centers_from_node_prices("segment", segment_prices)
        elif operation == "segment" and seg_centers is None:
            seg_centers = _DEFAULT_SEGMENT_PROVIDER_CENTERS
        if operation == "split" and spl_centers is None and split_prices is not None:
            spl_centers = _infer_provider_centers_from_node_prices("split", split_prices)
        elif operation == "split" and spl_centers is None:
            spl_centers = _DEFAULT_SPLIT_PROVIDER_CENTERS
        return compute_and_normalize_utilities(
            operation,
            cloud,
            DEFAULT_UTILITY_PARAMS[operation],
            segment_provider_centers=seg_centers,
            split_provider_centers=spl_centers,
        )
    if operation in ("caption", "query"):
        llm_nodes = _build_llm_nodes(operation)
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
                        llm_exec_latency_ms=n.llm_exec_latency_ms,
                        utility=n.utility,
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
                    output_token_num=llm_output_token_params[n.name].as_video_fn(),
                    llm_exec_latency_ms=n.llm_exec_latency_ms,
                    utility=n.utility,
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
                    utility=n.utility,
                    llm_option=n.llm_option,
                )
                for n in llm_nodes
            ]

        nodes = cloud + llm_nodes
        return compute_and_normalize_utilities(
            operation, nodes, DEFAULT_UTILITY_PARAMS[operation], llm_utility=llm_utility
        )
    raise ValueError(f"Unknown operation: {operation}")


@dataclass
class SimulationParams:
    """
    仿真参数包：采样一次后复用，保证 build_nodes 结果确定。

    用法：
      params = sample_simulation_params()
      seg = params.build_nodes("segment")
      ...
    """

    storage_prices: dict[Operation, dict[str, float]]
    exec_time_params: dict[Operation, dict[str, ExecTimeParams] | None]
    segment_prices: dict[str, float]
    split_prices: dict[str, float]
    segment_provider_centers: dict[str, float]
    split_provider_centers: dict[str, float]
    llm_token_prices: dict[Operation, dict[str, tuple[float, float]]]
    llm_output_token_params: dict[Operation, dict[str, CaptionTokenNumParams | TokenNumParams]]
    llm_latency_params: dict[Operation, dict[str, LlmLatencyParams]]
    cloud_llm_token_prices: dict[Operation, dict[str, tuple[float, float]]]
    cloud_llm_output_token_params: dict[Operation, dict[str, CaptionTokenNumParams | TokenNumParams]]
    cloud_llm_latency_params: dict[Operation, dict[str, LlmLatencyParams]]
    llm_utility: dict[Operation, dict[str, float]]

    def build_nodes(self, operation: Operation) -> list[Node]:
        """用固定参数构建 operation 的节点。"""
        return build_nodes_with_params(
            operation,
            storage_prices=self.storage_prices.get(operation),
            exec_time_params=self.exec_time_params,
            segment_prices=self.segment_prices if operation == "segment" else None,
            split_prices=self.split_prices if operation == "split" else None,
            segment_provider_centers=self.segment_provider_centers if operation == "segment" else None,
            split_provider_centers=self.split_provider_centers if operation == "split" else None,
            llm_token_prices=self.llm_token_prices.get(operation),
            llm_output_token_params=self.llm_output_token_params.get(operation),
            llm_latency_params=self.llm_latency_params.get(operation),
            cloud_llm_token_prices=self.cloud_llm_token_prices.get(operation),
            cloud_llm_output_token_params=self.cloud_llm_output_token_params.get(operation),
            cloud_llm_latency_params=self.cloud_llm_latency_params.get(operation),
            llm_utility=self.llm_utility.get(operation),
        )


_FIXED_SIMULATION_PARAMS: SimulationParams | None = None


def get_fixed_simulation_params() -> SimulationParams:
    """
    返回固定仿真参数（懒加载，首次调用时采样一次，后续复用）。
    test.py、solver.py 等统一使用此函数，保证同一进程内评估结果可复现。
    """
    global _FIXED_SIMULATION_PARAMS
    if _FIXED_SIMULATION_PARAMS is None:
        _FIXED_SIMULATION_PARAMS = sample_simulation_params()
    return _FIXED_SIMULATION_PARAMS


def sample_simulation_params() -> SimulationParams:
    """
    采样一次仿真参数，供 build_nodes 复用。
    调用一次后，用 params.build_nodes(operation) 获取节点，同一 params 下结果确定。
    """
    exec_time_params = sample_exec_time_params()
    segment_prices, segment_provider_centers = sample_segment_prices()
    split_prices, split_provider_centers = sample_split_prices()

    # 同 provider+region 存储价格一致，按 (provider, region) 采样一次
    storage_by_pr = sample_storage_prices_by_provider_region()
    storage_prices: dict[Operation, dict[str, float]] = {}
    for op in ("segment", "split", "caption", "query"):
        storage_prices[op] = {}
        for p in _CLOUD_PROVIDERS:
            for r in _CLOUD_REGIONS[p]:
                name = f"{p}_{r}_{op}"
                pr_key = f"{p}_{r}"
                storage_prices[op][name] = storage_by_pr[pr_key]

    llm_token_prices: dict[Operation, dict[str, tuple[float, float]]] = {}
    llm_output_token_params: dict[Operation, dict[str, CaptionTokenNumParams | TokenNumParams]] = {}
    llm_latency_params: dict[Operation, dict[str, LlmLatencyParams]] = {}
    cloud_llm_token_prices: dict[Operation, dict[str, tuple[float, float]]] = {}
    cloud_llm_output_token_params: dict[Operation, dict[str, CaptionTokenNumParams | TokenNumParams]] = {}
    cloud_llm_latency_params: dict[Operation, dict[str, LlmLatencyParams]] = {}
    llm_utility: dict[Operation, dict[str, float]] = {}
    cloud_base_tp, cloud_base_lp = sample_cloud_llm_base_params()
    for op in ("caption", "query"):  # type: ignore[assignment]
        by_model = _sample_llm_output_token_params_by_model(op)
        llm_token_prices[op] = sample_llm_token_prices(op)  # type: ignore[assignment]
        llm_output_token_params[op] = sample_llm_output_token_params(op, by_model=by_model)  # type: ignore[assignment]
        llm_latency_params[op] = sample_llm_latency_params(op)  # type: ignore[assignment]
        tp, otp, lp = sample_cloud_llm_params(
            op,
            by_model=by_model,
            base_token_prices=cloud_base_tp,
            base_latency_params=cloud_base_lp,
        )  # type: ignore[arg-type]
        cloud_llm_token_prices[op] = tp
        cloud_llm_output_token_params[op] = otp
        cloud_llm_latency_params[op] = lp
        llm_utility[op] = sample_llm_utility(op)

    return SimulationParams(
        storage_prices=storage_prices,
        exec_time_params=exec_time_params,
        segment_prices=segment_prices,
        split_prices=split_prices,
        segment_provider_centers=segment_provider_centers,
        split_provider_centers=split_provider_centers,
        llm_token_prices=llm_token_prices,
        llm_output_token_params=llm_output_token_params,
        llm_latency_params=llm_latency_params,
        cloud_llm_token_prices=cloud_llm_token_prices,
        cloud_llm_output_token_params=cloud_llm_output_token_params,
        cloud_llm_latency_params=cloud_llm_latency_params,
        llm_utility=llm_utility,
    )


def _assert_node_counts() -> None:
    expected = {"segment": 12, "split": 12, "caption": 20, "query": 20}
    for wf, n in expected.items():
        got = len(build_nodes(wf))  # type: ignore[arg-type]
        if got != n:
            raise AssertionError(f"{wf}: expected {n}, got {got}")


_assert_node_counts()


# ----------------------------
# Latency / Cost budget (per query/video, proportional to video size)
# ----------------------------
# 系数见 config.yaml budget 段


def get_latency_budget_s(video_size_mb: float) -> float:
    """每个 query/video 的 latency budget（秒），与 video size 成正比。"""
    if video_size_mb < 0:
        raise ValueError("video_size_mb must be >= 0")
    cfg = get_simulation_config().budget
    return cfg.latency_intercept_s + cfg.latency_slope_per_mb * video_size_mb


def get_cost_budget_usd(video_size_mb: float) -> float:
    """每个 query/video 的 cost budget（USD），与 video size 成正比。"""
    if video_size_mb < 0:
        raise ValueError("video_size_mb must be >= 0")
    cfg = get_simulation_config().budget
    return cfg.cost_intercept_usd + cfg.cost_slope_per_mb * video_size_mb


def get_data_size_mb(min_mb: float | None = None, max_mb: float | None = None) -> float:
    """均匀采样数据量 MB。未指定时使用 config.yaml data_size 范围。"""
    cfg = get_simulation_config()
    lo = min_mb if min_mb is not None else cfg.data_size[0]
    hi = max_mb if max_mb is not None else cfg.data_size[1]
    if lo <= 0 or hi <= 0:
        raise ValueError("min_mb/max_mb must be > 0")
    if hi < lo:
        raise ValueError("max_mb must be >= min_mb")
    return random.uniform(lo, hi)


def get_data_conversion_ratio(operation: Operation) -> float:
    """
    Data conversion ratio = output_size / input_size.

    Sampled from a LogNormal distribution to keep ratio > 0.
    Configured per-operation by a target mean ratio and a log-space sigma.

    中文：数据转换比例 `ratio = output_size / input_size`。
    - 用 LogNormal 采样保证 ratio > 0
    - 每个 operation 有各自的目标均值与 log-space 方差
    """

    # 兼容旧接口：直接调用会“每次重新采样一套参数再采样一次 ratio”。
    # 更推荐：params = sample_ratio_params(); sample_data_conversion_ratio(operation, params)
    return sample_ratio_params()[operation].sample_ratio()


def sample_data_conversion_ratio(operation: Operation, params: dict[Operation, RatioParams]) -> float:
    """
    推荐用法：
    - params = sample_ratio_params()  # 只运行一次
    - ratio  = sample_data_conversion_ratio("caption", params)
    """
    return params[operation].sample_ratio()


def sample_data_conversion_ratios_all(
    params: dict[Operation, RatioParams] | None = None,
) -> dict[Operation, float]:
    """
    一次采样四个 operation 的 data conversion ratio，保证 Latency 与 Cost 计算使用同一套数据量。

    推荐用法：在需要同时计算 latency 和 cost 时，先调用此函数采样一次，
    再将返回值传给 compute_end_to_end_latency_s_breakdown 和 compute_end_to_end_cost_usd_breakdown。
    """
    p = params if params is not None else DEFAULT_RATIO_PARAMS
    return {
        op: sample_data_conversion_ratio(op, p)
        for op in ("segment", "split", "caption", "query")
    }


# ----------------------------
# Network communication matrix (15 x 15)
# ----------------------------


@dataclass(frozen=True, slots=True)
class CommEndpoint:
    name: str  # e.g. "local", "p1_r1", "p4"（端点名，用于矩阵查表）
    provider: str  # "local" | "p1".."p5"（提供方/类别）
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

        中文：返回链路 a<->b 的：
        - RTT（毫秒，往返时延）
        - 带宽（Mbps）
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

        中文：出站 egress 费用（USD/GB），从 `src -> dst`。
        - 非对称：`A->B` 和 `B->A` 的价格可能不同
        - 入站免费：只计 outbound
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

    中文：通信端点共 15 个：
    - `local`：本地端
    - `p1/p2/p3`：每个 provider 4 个 region，总计 12 个云端点
    - `p4/p5`：LLM-only，没有 region
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

    中文：Pareto(α) 采样（支撑 [1, +inf)），再乘以 xm 进行缩放，用于模拟“长尾带宽”。
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

    中文：构建对称的 15x15 网络矩阵：
    - RTT：用 LogNormal 生成往返时延（ms）
    - 带宽：用 Pareto 生成链路带宽（Mbps，长尾分布）
    - 约束/优先级：同 region 通常 RTT 更低；同 provider 往往带宽更高/RTT 更低
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
            jitter_lo, jitter_hi = get_simulation_config().comm.jitter
            rtt = _lognormal_ms(rtt_mean * random.uniform(jitter_lo, jitter_hi), rtt_sigma)
            bw = _pareto_scale(bw_xm * random.uniform(jitter_lo, jitter_hi), bw_alpha)

            rtt_ms[i][j] = rtt_ms[j][i] = rtt
            bw_mbps[i][j] = bw_mbps[j][i] = bw

    return eps, rtt_ms, bw_mbps


def build_comm_matrix(comm_params: CommParams) -> CommMatrix:
    """
    Convenience wrapper around build_comm_matrices() that supports lookup:

    - by indices: matrix.link(0, 3)
    - by names:   matrix.link("p1_r1", "p2_r1")
    - by endpoint objects

    中文：构建带查表能力的 `CommMatrix`：
    - 支持通过下标、端点名字或端点对象来获取 RTT/带宽
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

    中文：构建非对称的 egress 成本矩阵（USD/GB）。
    - 只对 outbound 计费（ingress 免费）
    - 只有云（p1/p2/p3）作为 source 时才计费；local/p4/p5 source 全为 0
    - 同 provider 使用 intra_provider 费率；不同 provider 使用 cross_provider 费率
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


# ----------------------------
# Module-default sampled params (sample once, then reuse)
# ----------------------------

DEFAULT_RATIO_PARAMS: dict[Operation, RatioParams] = sample_ratio_params()
DEFAULT_COMM_PARAMS: CommParams = sample_comm_params()
DEFAULT_COMM_MATRIX: CommMatrix = build_comm_matrix(DEFAULT_COMM_PARAMS)
DEFAULT_EGRESS_PARAMS: EgressParams = sample_egress_params()
DEFAULT_EGRESS_MATRIX: EgressMatrix = build_egress_matrix(DEFAULT_EGRESS_PARAMS)


def compute_end_to_end_latency_s_breakdown(
    *,
    video_size_mb: float,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
    data_conversion_ratios: dict[Operation, float] | None = None,
) -> dict[str, float]:
    """
    计算端到端延迟（秒），并返回各组成部分的 breakdown，便于 debug。

    返回的 dict 包含：
    - local_to_segment_transfer_s, segment_exec_s
    - segment_to_split_transfer_s, split_exec_s
    - split_to_caption_transfer_s, caption_exec_s
    - caption_to_query_transfer_s, query_exec_s
    - query_to_local_transfer_s
    - total_s

    data_conversion_ratios: 若提供，则使用预采样的 ratio，保证与 cost 计算数据一致；
        若为 None，则内部采样（单独调用时兼容旧行为）。
    """
    if video_size_mb < 0:
        raise ValueError("video_size_mb must be >= 0")

    def endpoint_name(node: Node) -> str:
        if node.region is not None:
            return f"{node.provider}_{node.region}"
        return node.provider

    comm_matrix = DEFAULT_COMM_MATRIX
    ratio_params = DEFAULT_RATIO_PARAMS

    def transfer_time_s(src_name: str, dst_name: str, data_mb: float) -> float:
        rtt_ms, bw_mbps = comm_matrix.link(src_name, dst_name)
        if bw_mbps == 0.0:
            raise ValueError(f"Bandwidth is 0 for link {src_name}->{dst_name}")
        if math.isinf(bw_mbps):
            tx_s = 0.0
        else:
            tx_s = (data_mb * 8.0) / bw_mbps
        rtt_half_s = rtt_ms / 2000.0
        return rtt_half_s + tx_s

    # 各节点输出大小均按 conversion ratio 推导（使用预采样或内部采样，保证与 cost 一致）
    if data_conversion_ratios is not None:
        seg_ratio = data_conversion_ratios["segment"]
        split_ratio = data_conversion_ratios["split"]
        cap_ratio = data_conversion_ratios["caption"]
        qry_ratio = data_conversion_ratios["query"]
    else:
        seg_ratio = sample_data_conversion_ratio("segment", ratio_params)
        split_ratio = sample_data_conversion_ratio("split", ratio_params)
        cap_ratio = sample_data_conversion_ratio("caption", ratio_params)
        qry_ratio = sample_data_conversion_ratio("query", ratio_params)
    segment_output_mb = video_size_mb * seg_ratio
    split_output_mb = segment_output_mb * split_ratio
    caption_output_mb = split_output_mb * cap_ratio
    answer_size_mb = caption_output_mb * qry_ratio

    local = "local"
    seg_ep = endpoint_name(segment_node)
    split_ep = endpoint_name(split_node)
    caption_ep = endpoint_name(caption_node)
    query_ep = endpoint_name(query_node)

    bd: dict[str, float] = {}

    # 1) local -> segment
    bd["local_to_segment_transfer_s"] = transfer_time_s(local, seg_ep, video_size_mb)
    if segment_node.exec_time_s is None:
        raise ValueError("segment_node.exec_time_s is None; segment/split nodes must be built with exec_time_params.")
    bd["segment_exec_s"] = segment_node.exec_time_s(video_size_mb)

    # 2) segment -> split（传输 segment 输出）
    bd["segment_to_split_transfer_s"] = transfer_time_s(seg_ep, split_ep, segment_output_mb)
    if split_node.exec_time_s is None:
        raise ValueError("split_node.exec_time_s is None; segment/split nodes must be built with exec_time_params.")
    bd["split_exec_s"] = split_node.exec_time_s(segment_output_mb)

    # 3) split -> caption（传输 split 输出）
    bd["split_to_caption_transfer_s"] = transfer_time_s(split_ep, caption_ep, split_output_mb)
    if caption_node.output_token_num is None or caption_node.llm_exec_latency_ms is None:
        raise ValueError("caption_node must be an LLM node with output_token_num and llm_exec_latency_ms.")
    cap_tokens = caption_node.output_token_num(video_size_mb)
    bd["caption_exec_s"] = caption_node.llm_exec_latency_ms(cap_tokens) / 1000.0

    # 4) caption -> query（传输 caption 输出）
    bd["caption_to_query_transfer_s"] = transfer_time_s(caption_ep, query_ep, caption_output_mb)
    if query_node.output_token_num is None or query_node.llm_exec_latency_ms is None:
        raise ValueError("query_node must be an LLM node with output_token_num and llm_exec_latency_ms.")
    qry_tokens = query_node.output_token_num(video_size_mb)
    bd["query_exec_s"] = query_node.llm_exec_latency_ms(qry_tokens) / 1000.0

    # 5) query -> local (return answer)
    bd["query_to_local_transfer_s"] = transfer_time_s(query_ep, local, answer_size_mb)

    bd["total_s"] = sum(bd.values())
    return bd


def compute_end_to_end_latency_s(
    *,
    video_size_mb: float,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
    data_conversion_ratios: dict[Operation, float] | None = None,
) -> float:
    """
    计算端到端延迟（秒）。

    严格按你描述的流程（数据经过每个节点后的大小均用 conversion ratio 推导）：
    1) 本地 -> segment：上传 video（数据量 video_size_mb）
       + segment 处理延迟
    2) segment -> split：传输 segment 输出（video_size_mb × ratio_segment）
       + split 处理延迟（输入为 segment 输出）
    3) split -> caption：传输 split 输出（× ratio_split）
       + caption 处理延迟（基于 token 数输出）
    4) caption -> query：传输 caption 输出（× ratio_caption）
       + query 处理延迟（基于 token 数输出）
    5) query -> 本地：返回结果（× ratio_query）

    传输延迟模型：
    RTT/2 + data_size/bandwidth

    单位处理：
    - RTT：ms -> s（除以 1000，再除以 2）
    - bandwidth：Mbps -> MB/s（用 8 bit/byte 进行换算：data_MB * 8 / bw_Mbps）
    """
    return compute_end_to_end_latency_s_breakdown(
        video_size_mb=video_size_mb,
        segment_node=segment_node,
        split_node=split_node,
        caption_node=caption_node,
        query_node=query_node,
        data_conversion_ratios=data_conversion_ratios,
    )["total_s"]


def compute_end_to_end_cost_usd_breakdown(
    *,
    video_size_mb: float,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
    data_conversion_ratios: dict[Operation, float] | None = None,
) -> dict[str, float]:
    """
    计算端到端成本（USD），并返回各组成部分的 breakdown，便于 debug。

    返回的 dict 包含各 cost 项及 total_usd。

    data_conversion_ratios: 若提供，则使用预采样的 ratio，保证与 latency 计算数据一致；
        若为 None，则内部采样（单独调用时兼容旧行为）。
    """
    if video_size_mb < 0:
        raise ValueError("video_size_mb must be >= 0")

    ratio_params = DEFAULT_RATIO_PARAMS
    egress_matrix = DEFAULT_EGRESS_MATRIX

    sim_cfg = get_simulation_config()
    storage_months = sim_cfg.storage_months

    def endpoint_name(node: Node) -> str:
        if node.region is not None:
            return f"{node.provider}_{node.region}"
        return node.provider

    def mb_to_gb(mb: float) -> float:
        return mb / 1024.0

    def storage_cost_usd(node: Node, size_mb: float) -> float:
        if not node.can_store:
            return 0.0
        if node.storage_price_usd_per_gb_month is None:
            return 0.0
        return node.storage_price_usd_per_gb_month * mb_to_gb(size_mb) * storage_months

    def egress_cost_usd(src_ep: str, dst_ep: str, size_mb: float) -> float:
        usd_per_gb = egress_matrix.cost(src_ep, dst_ep)
        return usd_per_gb * mb_to_gb(size_mb)

    tokens_per_mb = sim_cfg.token.tokens_per_mb

    # 各节点输出大小均按 conversion ratio 推导（使用预采样或内部采样，保证与 latency 一致）
    if data_conversion_ratios is not None:
        seg_ratio = data_conversion_ratios["segment"]
        split_ratio = data_conversion_ratios["split"]
        cap_ratio = data_conversion_ratios["caption"]
        qry_ratio = data_conversion_ratios["query"]
    else:
        seg_ratio = sample_data_conversion_ratio("segment", ratio_params)
        split_ratio = sample_data_conversion_ratio("split", ratio_params)
        cap_ratio = sample_data_conversion_ratio("caption", ratio_params)
        qry_ratio = sample_data_conversion_ratio("query", ratio_params)
    segment_output_mb = video_size_mb * seg_ratio
    split_output_mb = segment_output_mb * split_ratio
    caption_output_mb = split_output_mb * cap_ratio

    bd: dict[str, float] = {}

    # 1) Storage: video on segment bucket for 1 hour
    bd["segment_storage_usd"] = storage_cost_usd(segment_node, video_size_mb)

    # 2) Segment compute cost (USD/min)
    if segment_node.exec_time_s is None or segment_node.segment_price_usd_per_min is None:
        raise ValueError("segment_node must have exec_time_s and segment_price_usd_per_min.")
    seg_time_s = segment_node.exec_time_s(video_size_mb)
    bd["segment_compute_usd"] = segment_node.segment_price_usd_per_min * (seg_time_s / 60.0)

    # 3) segment -> split egress（传输 segment 输出）, special rule: same provider + same region => 0
    seg_ep = endpoint_name(segment_node)
    split_ep = endpoint_name(split_node)
    if (
        segment_node.provider == split_node.provider
        and segment_node.region is not None
        and segment_node.region == split_node.region
    ):
        bd["segment_to_split_egress_usd"] = 0.0
    else:
        bd["segment_to_split_egress_usd"] = egress_cost_usd(seg_ep, split_ep, segment_output_mb)

    # 4) Storage: split 输入（segment 输出）on split bucket for 1 hour
    bd["split_storage_usd"] = storage_cost_usd(split_node, segment_output_mb)

    # 5) Split compute cost (USD/s)，输入为 segment 输出
    if split_node.exec_time_s is None or split_node.split_price_usd_per_s is None:
        raise ValueError("split_node must have exec_time_s and split_price_usd_per_s.")
    split_time_s = split_node.exec_time_s(segment_output_mb)
    bd["split_compute_usd"] = split_node.split_price_usd_per_s * split_time_s

    # 6) split -> caption egress（传输 split 输出）
    caption_ep = endpoint_name(caption_node)
    bd["split_to_caption_egress_usd"] = egress_cost_usd(split_ep, caption_ep, split_output_mb)

    # 7) Caption LLM cost (token-based)，输入为 split 输出
    if (
        caption_node.llm_input_price_usd_per_1m_tokens is None
        or caption_node.llm_output_price_usd_per_1m_tokens is None
        or caption_node.output_token_num is None
    ):
        raise ValueError("caption_node must be an LLM node with token prices and output_token_num.")
    caption_in_tokens = int(round(max(0.0, split_output_mb * tokens_per_mb)))
    caption_out_tokens = caption_node.output_token_num(video_size_mb)
    bd["caption_llm_usd"] = (
        (caption_in_tokens / 1_000_000.0) * caption_node.llm_input_price_usd_per_1m_tokens
        + (caption_out_tokens / 1_000_000.0) * caption_node.llm_output_price_usd_per_1m_tokens
    )

    query_ep = endpoint_name(query_node)
    caption_bucket_node = caption_node if caption_node.can_store else split_node
    caption_bucket_ep = caption_ep if caption_node.can_store else split_ep

    # Storage: caption text stored for 1 hour on the selected bucket node
    bd["caption_storage_usd"] = storage_cost_usd(caption_bucket_node, caption_output_mb)

    # 8) Provide caption to query: query reads from the bucket holding caption text（传输 caption 输出）
    bd["caption_to_query_egress_usd"] = egress_cost_usd(caption_bucket_ep, query_ep, caption_output_mb)

    # 9) Query LLM cost (token-based)
    if (
        query_node.llm_input_price_usd_per_1m_tokens is None
        or query_node.llm_output_price_usd_per_1m_tokens is None
        or query_node.output_token_num is None
    ):
        raise ValueError("query_node must be an LLM node with token prices and output_token_num.")
    query_in_tokens = caption_out_tokens
    query_out_tokens = query_node.output_token_num(video_size_mb)
    bd["query_llm_usd"] = (
        (query_in_tokens / 1_000_000.0) * query_node.llm_input_price_usd_per_1m_tokens
        + (query_out_tokens / 1_000_000.0) * query_node.llm_output_price_usd_per_1m_tokens
    )

    bd["total_usd"] = sum(v for k, v in bd.items())
    return bd


def compute_end_to_end_cost_usd(
    *,
    video_size_mb: float,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
    data_conversion_ratios: dict[Operation, float] | None = None,
) -> float:
    """
    计算端到端成本（USD），按你描述的顺序累加：

    - local 上传到云端：无 cost
    - 存储成本：按 storage_price_usd_per_gb_month，假设存储时间=1小时
      - video 在 segment bucket 存储 1 小时
      - video 在 split bucket 存储 1 小时
      - 若 caption 节点是云节点（can_store=True），则 caption 输出也按 1 小时存储
    - 计算成本：
      - segment：按 USD/min（segment_price_usd_per_min）对处理时长计费
      - split：按 USD/s（split_price_usd_per_s）对处理时长计费
      - caption/query：按 token 计费（USD per 1M tokens）
    - 传输/出站成本（USD per GB）：
      - segment -> split：如果同 provider 且同 region，则 0；否则按出站费用
      - split -> caption：传输 video，按出站费用
      - caption -> query：如果 caption 是云节点，则按出站费用；否则 0
    """
    return compute_end_to_end_cost_usd_breakdown(
        video_size_mb=video_size_mb,
        segment_node=segment_node,
        split_node=split_node,
        caption_node=caption_node,
        query_node=query_node,
        data_conversion_ratios=data_conversion_ratios,
    )["total_usd"]


def compute_end_to_end_utility_score_breakdown(
    *,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
) -> dict[str, float]:
    """
    端到端 utility score 各组成部分的 breakdown，便于 debug。

    返回的 dict 包含：
    - segment_contribution, caption_contribution, query_contribution
    - total（split 权重为 0，故无 contribution）
    """
    uw = get_simulation_config().utility_weights
    weights = {
        "segment": uw.segment,
        "split": uw.split,
        "caption": uw.caption,
        "query": uw.query,
    }

    def req(u: float | None, name: str) -> float:
        if u is None:
            raise ValueError(f"{name}.utility is None; build nodes via build_nodes/build_nodes_with_params first.")
        return u

    bd: dict[str, float] = {}
    bd["segment_contribution"] = weights["segment"] * req(segment_node.utility, "segment_node")
    bd["split_contribution"] = 0.0
    bd["caption_contribution"] = weights["caption"] * req(caption_node.utility, "caption_node")
    bd["query_contribution"] = weights["query"] * req(query_node.utility, "query_node")
    bd["total"] = (
        bd["segment_contribution"]
        + bd["split_contribution"]
        + bd["caption_contribution"]
        + bd["query_contribution"]
    )
    return bd


def compute_end_to_end_utility_score(
    *,
    segment_node: Node,
    split_node: Node,
    caption_node: Node,
    query_node: Node,
) -> float:
    """
    端到端 utility score = 各 operation 节点 utility 的加权和。

    权重见 config.yaml 中 utility_weights。
    """
    return compute_end_to_end_utility_score_breakdown(
        segment_node=segment_node,
        split_node=split_node,
        caption_node=caption_node,
        query_node=query_node,
    )["total"]



