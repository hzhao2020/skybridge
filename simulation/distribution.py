import random
import math
from typing import Any, Callable, Dict, Literal, Sequence, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from config import *
from config import get_llm_node_name
# 全局种子,确保实验可以复现
random.seed(42)

config = get_simulation_config()
node_names = get_node_name()
cloud_region_names = get_cloud_region_name()


class LogNormalDistribution:
    def __init__(self, mean: float, std: float):
        """
        初始化 LogNormal 分布
        :param mean: 变量X的均值 (E[X])
        :param std: 变量X的标准差 (sqrt(Var[X]))
        """
        if mean <= 0:
            raise ValueError("mean must be > 0")
        if std < 0:
            raise ValueError("std must be >= 0")
        # 记录原始 mean/std，便于 deterministic 口径直接取均值
        self.mean = float(mean)
        self.std = float(std)
        # 根据 LogNormal 分布性质，推导 mu, sigma
        variance = std ** 2
        phi = math.sqrt(variance + mean ** 2)
        self.mu = math.log(mean ** 2 / phi)
        self.sigma = math.sqrt(math.log(phi ** 2 / mean ** 2))

    def sample(self) -> float:
        """
        从对数正态分布中采样一个值
        """
        return random.lognormvariate(self.mu, self.sigma)


class GammaDistribution:
    def __init__(self, alpha: float, theta: float):
        """
        初始化 Gamma 分布
        :param alpha: 形状参数 (k 或 alpha)
        :param theta: scale 参数 (θ)
        """
        if alpha <= 0:
            raise ValueError("alpha (shape) must be > 0")
        if theta <= 0:
            raise ValueError("theta (scale) must be > 0")
        self.alpha = alpha
        self.theta = theta

    def sample(self) -> float:
        """
        从 Gamma 分布中采样一个值
        """
        return random.gammavariate(self.alpha, self.theta)


class UniformDistribution:
    def __init__(self, min_val: float, max_val: float):
        """
        初始化均匀分布
        :param min_val: 最小值 (包含)
        :param max_val: 最大值 (包含)
        """
        if min_val > max_val:
            raise ValueError("min_val must be less than or equal to max_val")
        self.min_val = min_val
        self.max_val = max_val

    def sample(self) -> float:
        """
        从均匀分布中采样一个值
        """
        return random.uniform(self.min_val, self.max_val)


class SegSplExecTimeParam:
    io_s_per_mb: float
    alpha: float
    theta0: float
    theta1: float


class QueryOutputTokenNumParam:
    mean: float
    std: float


class CaptionOutputTokenNumParam:
    base: float
    coef_per_mb: float
    std: float


class CaptionInputTokenNumParam:
    input_tokens_per_mb: float


class LlmlatencyParam:
    alpha_ms_per_token: float
    beta_ms: float
    noise_sigma_ms: float


class Query:
    data_size_mb: float
    cost_budget: float # USD
    latency_budget: float # s


class RTT:
    mean: float
    std: float


class BW:
    mean: float
    std: float


def sample_query_output_token_num_param() -> Dict[str, QueryOutputTokenNumParam]:
    # 对所有的llm node生成QueryOutputTokenNumParam,基于config的参数,对于同一个llm而言,参数是一样的
    mean_rng = config.get("query_mean")
    std_rng = config.get("query_std")
    if not isinstance(mean_rng, (list, tuple)) or len(mean_rng) != 2:
        raise ValueError("config.query_mean must be a 2-item list like [min, max]")
    if not isinstance(std_rng, (list, tuple)) or len(std_rng) != 2:
        raise ValueError("config.query_std must be a 2-item list like [min, max]")

    mean_lo, mean_hi = float(mean_rng[0]), float(mean_rng[1])
    std_lo, std_hi = float(std_rng[0]), float(std_rng[1])
    if mean_hi < mean_lo:
        raise ValueError("config.query_mean must satisfy max >= min")
    if std_hi < std_lo:
        raise ValueError("config.query_std must satisfy max >= min")

    mean_dist = UniformDistribution(mean_lo, mean_hi)
    std_dist = UniformDistribution(std_lo, std_hi)

    # 1) per-model-key sampled params
    by_model: Dict[str, QueryOutputTokenNumParam] = {}
    for mk in LLM_MODEL_KEYS:
        p = QueryOutputTokenNumParam()
        p.mean = mean_dist.sample()
        p.std = std_dist.sample()
        if p.mean <= 0:
            raise ValueError("sampled query_mean must be > 0")
        if p.std < 0:
            raise ValueError("sampled query_std must be >= 0")
        by_model[mk] = p

    # 2) assign to nodes
    out: Dict[str, QueryOutputTokenNumParam] = {}

    # cloud query nodes: p?_r?_query -> model key is provider (p1/p2/p3)
    for pr in cloud_region_names:
        provider = pr.split("_", 1)[0]
        out[f"{pr}_query"] = by_model[provider]

    # llm-only query nodes: p4_m3_query / p5_m3_query -> model key is m3
    for name in get_llm_node_name():
        if not name.endswith("_query"):
            continue
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected llm node name format: {name!r}")
        model = parts[1]
        out[name] = by_model[model]

    return out



def sample_data_conversion_ratio() -> Dict[Operation, LogNormalDistribution]:
    # 返回的是分布,然后需要根据分布来获取对应的数值(是随机的,不是确定的)
    cfg = config.get("data_conversion_ratio", {})
    ops: Sequence[Operation] = ("segment", "split", "caption", "query")

    out: Dict[Operation, LogNormalDistribution] = {}
    for op in ops:
        mean_key = f"{op}_mean"
        std_key = f"{op}_std"
        if mean_key not in cfg or std_key not in cfg:
            raise KeyError(
                f"Missing data_conversion_ratio config keys: {mean_key}, {std_key}"
            )
        out[op] = LogNormalDistribution(mean=float(cfg[mean_key]), std=float(cfg[std_key]))
    return out


def sample_utility():
    uv = config.get("utility_value")
    ops: Sequence[Operation] = ("segment", "split", "caption", "query")
    if not isinstance(uv, dict):
        raise ValueError("config.utility_value must be a dict mapping operation -> [min, max]")

    op_to_dist: Dict[Operation, UniformDistribution] = {}
    for op in ops:
        rng = uv.get(op)
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(f"config.utility_value[{op!r}] must be a 2-item list like [min, max]")
        lo, hi = float(rng[0]), float(rng[1])
        if hi < lo:
            raise ValueError(f"config.utility_value[{op!r}] must satisfy max >= min")
        op_to_dist[op] = UniformDistribution(lo, hi)

    utilities: Dict[str, float] = {}
    for name in node_names:
        op = name.rsplit("_", 1)[-1]
        if op not in ops:
            raise ValueError(f"Unknown operation parsed from node name {name!r}: {op!r}")
        utilities[name] = op_to_dist[op].sample()  # type: ignore[index]
    return utilities


def sample_llm_token_price():
    llm = config.get("llm", {})
    inp_rng = llm.get("input_price")
    mult = llm.get("output_multiplier")
    if not isinstance(inp_rng, (list, tuple)) or len(inp_rng) != 2:
        raise ValueError("config.llm.input_price must be a 2-item list like [min, max]")
    if mult is None:
        raise ValueError("config.llm.output_multiplier is required")

    lo, hi = float(inp_rng[0]), float(inp_rng[1])
    if hi < lo:
        raise ValueError("config.llm.input_price must satisfy max >= min")
    mult_f = float(mult)

    # Return mapping: llm_node_name -> (input_price, output_price)，单位为 USD / 百万 tokens
    # caption/query share the same sampled prices (per cloud provider+region or llm provider+model).
    llm_token_price: Dict[str, tuple[float, float]] = {}

    # Cloud LLM nodes (provider+region): share across caption/query
    for pr in cloud_region_names:
        inp = random.uniform(lo, hi)
        outp = mult_f * inp
        llm_token_price[f"{pr}_caption"] = (inp, outp)
        llm_token_price[f"{pr}_query"] = (inp, outp)

    # LLM-only nodes (provider+model): share across caption/query
    provider_model_seen: set[str] = set()
    for name in get_llm_node_name():
        # name format: p4_m1_caption
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected llm node name format: {name!r}")
        pm = f"{parts[0]}_{parts[1]}"
        provider_model_seen.add(pm)

    for pm in sorted(provider_model_seen):
        inp = random.uniform(lo, hi)
        outp = mult_f * inp
        llm_token_price[f"{pm}_caption"] = (inp, outp)
        llm_token_price[f"{pm}_query"] = (inp, outp)

    return llm_token_price


def sample_storage_price():
    # 每一个(cloud provider, region) 对应一个存储价格,但是key是具体的node
    # 这需要是确定的量
    rng = config.get("storage_price")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError("config.storage_price must be a 2-item list like [min, max]")
    lo, hi = float(rng[0]), float(rng[1])
    if hi < lo:
        raise ValueError("config.storage_price must satisfy max >= min")

    dist = UniformDistribution(lo, hi)
    prices: Dict[str, float] = {}
    ops: Sequence[Operation] = ("segment", "split", "caption", "query")
    # 先按 (provider, region) 采样一次，保证跨 operation 复用同一价格
    pr_to_price: Dict[str, float] = {pr: dist.sample() for pr in cloud_region_names}
    # 再展开为具体 node key: "{provider}_{region}_{operation}"
    for pr, price in pr_to_price.items():
        for op in ops:
            prices[f"{pr}_{op}"] = price
    return prices


def sample_egress_price():
    # 生成一个矩阵,节点包括 *cloud_region_names, p4, p5, local, 一共15个节点,
    # 也就是生成15*15的矩阵
    # 元素(a,b)表示从a出站到b的价格,对于local,p4,p5,没有出站费用
    # 对于元素(a,b)只有a是cloud node的时候,也就是a属于p1-3的时候,是非0数值,对角线为0,(a,b)表示从a到b的出站费用,此时,你需要考虑a和b是否属于同一个provider,如果是,从intra_provider的参数中均匀采样,如果不是,从cross_provider的参数中均匀采样
    cfg = config.get("egress_price", {})
    intra_rng = cfg.get("intra_provider")
    cross_rng = cfg.get("cross_provider")
    if not isinstance(intra_rng, (list, tuple)) or len(intra_rng) != 2:
        raise ValueError("config.egress_price.intra_provider must be a 2-item list like [min, max]")
    if not isinstance(cross_rng, (list, tuple)) or len(cross_rng) != 2:
        raise ValueError("config.egress_price.cross_provider must be a 2-item list like [min, max]")
    intra_lo, intra_hi = float(intra_rng[0]), float(intra_rng[1])
    cross_lo, cross_hi = float(cross_rng[0]), float(cross_rng[1])
    if intra_hi < intra_lo:
        raise ValueError("config.egress_price.intra_provider must satisfy max >= min")
    if cross_hi < cross_lo:
        raise ValueError("config.egress_price.cross_provider must satisfy max >= min")

    intra_dist = UniformDistribution(intra_lo, intra_hi)
    cross_dist = UniformDistribution(cross_lo, cross_hi)

    endpoints: list[str] = [*cloud_region_names, "p4", "p5", "local"]
    if len(endpoints) != 15:
        raise ValueError(f"Expected 15 endpoints, got {len(endpoints)}")

    def provider_of(ep: str) -> str:
        # cloud endpoints are like "p1_r1"
        if ep in ("p4", "p5", "local"):
            return ep
        return ep.split("_", 1)[0]

    def is_cloud(ep: str) -> bool:
        # cloud endpoints are exactly the 12 provider+region names (p1..p3)
        return ep not in ("p4", "p5", "local")

    # matrix[src][dst] = outbound price from src -> dst (USD/GB)
    matrix: list[list[float]] = [[0.0 for _ in endpoints] for _ in endpoints]
    for i, src in enumerate(endpoints):
        for j, dst in enumerate(endpoints):
            if i == j:
                matrix[i][j] = 0.0
                continue
            if not is_cloud(src):
                matrix[i][j] = 0.0
                continue
            src_p = provider_of(src)
            dst_p = provider_of(dst)
            # 每个 (src, dst) 独立采样一次：
            # - same provider => intra_provider range
            # - different provider (including local/p4/p5) => cross_provider range
            matrix[i][j] = intra_dist.sample() if (is_cloud(dst) and src_p == dst_p) else cross_dist.sample()

    # 返回 endpoints + 15x15 matrix，便于用下标或名字查表
    return {"endpoints": endpoints, "matrix": matrix}


def sample_segment_price():
    # 一共有12中segment的node,即(provider_region_segment),为每一个节点分别进行均匀采样(依据config中的参数)
    rng = config.get("segment_center")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError("config.segment_center must be a 2-item list like [min, max]")
    lo, hi = float(rng[0]), float(rng[1])
    if hi < lo:
        raise ValueError("config.segment_center must satisfy max >= min")
    dist = UniformDistribution(lo, hi)

    prices: Dict[str, float] = {}
    for pr in cloud_region_names:
        prices[f"{pr}_segment"] = dist.sample()
    return prices


def sample_split_price():
    # 一共有12中split的node,即(provider_region_split),为每一个节点分别进行均匀采样(依据config中的参数)
    rng = config.get("split_center")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError("config.split_center must be a 2-item list like [min, max]")
    lo, hi = float(rng[0]), float(rng[1])
    if hi < lo:
        raise ValueError("config.split_center must satisfy max >= min")
    dist = UniformDistribution(lo, hi)

    prices: Dict[str, float] = {}
    for pr in cloud_region_names:
        prices[f"{pr}_split"] = dist.sample()
    return prices


def sample_seg_spl_exec_time_param() -> Dict[str, SegSplExecTimeParam]:
    # 为24个split和segment的node分别进行均匀采样(依据config中的exec_time参数)
    et = config.get("exec_time", {})
    seg = et.get("segment", {})
    spl = et.get("split", {})

    def _req_range(obj: Any, key: str) -> tuple[float, float]:
        rng = obj.get(key) if isinstance(obj, dict) else None
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(f"config.exec_time.*.{key} must be a 2-item list like [min, max]")
        lo, hi = float(rng[0]), float(rng[1])
        if hi < lo:
            raise ValueError(f"config.exec_time.*.{key} must satisfy max >= min")
        return lo, hi

    seg_io = UniformDistribution(*_req_range(seg, "io_s_per_mb"))
    seg_alpha = UniformDistribution(*_req_range(seg, "alpha"))
    seg_t0 = UniformDistribution(*_req_range(seg, "theta0"))
    seg_t1 = UniformDistribution(*_req_range(seg, "theta1"))

    spl_io = UniformDistribution(*_req_range(spl, "io_s_per_mb"))
    spl_alpha = UniformDistribution(*_req_range(spl, "alpha"))
    spl_t0 = UniformDistribution(*_req_range(spl, "theta0"))
    spl_t1 = UniformDistribution(*_req_range(spl, "theta1"))

    params: Dict[str, SegSplExecTimeParam] = {}
    for pr in cloud_region_names:
        seg_node = f"{pr}_segment"
        p1 = SegSplExecTimeParam()
        p1.io_s_per_mb = seg_io.sample()
        p1.alpha = seg_alpha.sample()
        p1.theta0 = seg_t0.sample()
        p1.theta1 = seg_t1.sample()
        params[seg_node] = p1

        spl_node = f"{pr}_split"
        p2 = SegSplExecTimeParam()
        p2.io_s_per_mb = spl_io.sample()
        p2.alpha = spl_alpha.sample()
        p2.theta0 = spl_t0.sample()
        p2.theta1 = spl_t1.sample()
        params[spl_node] = p2

    return params


def sample_caption_output_token_num_param() -> Dict[str, CaptionOutputTokenNumParam]:
    # 为所有的LLM_MODEL_KEYS采样分布的参数,然后应用到所有对应的节点中
    # 认为每一个cloud provider的llm是一样的,比如p1_r1-4 都用的是一个模型
    # 然后对于p4_m3和p5_m3也是一个模型,参数也是相同的

    base_rng = config.get("caption_base")
    coef_rng = config.get("caption_coef_per_mb")
    std_rng = config.get("caption_std")
    if not isinstance(base_rng, (list, tuple)) or len(base_rng) != 2:
        raise ValueError("config.caption_base must be a 2-item list like [min, max]")
    if not isinstance(coef_rng, (list, tuple)) or len(coef_rng) != 2:
        raise ValueError("config.caption_coef_per_mb must be a 2-item list like [min, max]")
    if not isinstance(std_rng, (list, tuple)) or len(std_rng) != 2:
        raise ValueError("config.caption_std must be a 2-item list like [min, max]")

    base_dist = UniformDistribution(float(base_rng[0]), float(base_rng[1]))
    coef_dist = UniformDistribution(float(coef_rng[0]), float(coef_rng[1]))
    std_dist = UniformDistribution(float(std_rng[0]), float(std_rng[1]))

    # 1) per-model-key sampled params
    by_model: Dict[str, CaptionOutputTokenNumParam] = {}
    for mk in LLM_MODEL_KEYS:
        p = CaptionOutputTokenNumParam()
        p.base = base_dist.sample()
        p.coef_per_mb = coef_dist.sample()
        p.std = std_dist.sample()
        by_model[mk] = p

    # 2) assign to nodes
    out: Dict[str, CaptionOutputTokenNumParam] = {}

    # cloud caption nodes: p?_r?_caption -> model key is provider (p1/p2/p3)
    for pr in cloud_region_names:
        provider = pr.split("_", 1)[0]
        node = f"{pr}_caption"
        out[node] = by_model[provider]

    # llm-only caption nodes: p4_m3_caption / p5_m3_caption -> model key is m3
    for name in get_llm_node_name():
        if not name.endswith("_caption"):
            continue
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected llm node name format: {name!r}")
        model = parts[1]
        out[name] = by_model[model]

    return out


def sample_caption_input_token_num_param() -> Dict[str, CaptionInputTokenNumParam]:
    rng = config.get("input_tokens_per_mb")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ValueError("config.input_tokens_per_mb must be a 2-item list like [min, max]")
    lo, hi = float(rng[0]), float(rng[1])
    if hi < lo:
        raise ValueError("config.input_tokens_per_mb must satisfy max >= min")

    dist = UniformDistribution(lo, hi)

    # 1) per-model-key sampled params
    by_model: Dict[str, CaptionInputTokenNumParam] = {}
    for mk in LLM_MODEL_KEYS:
        p = CaptionInputTokenNumParam()
        p.input_tokens_per_mb = dist.sample()
        if p.input_tokens_per_mb <= 0:
            raise ValueError("sampled input_tokens_per_mb must be > 0")
        by_model[mk] = p

    # 2) assign to caption nodes
    out: Dict[str, CaptionInputTokenNumParam] = {}

    # cloud caption nodes: p?_r?_caption -> model key is provider (p1/p2/p3)
    for pr in cloud_region_names:
        provider = pr.split("_", 1)[0]
        out[f"{pr}_caption"] = by_model[provider]

    # llm-only caption nodes: p4_m3_caption / p5_m3_caption -> model key is m3
    for name in get_llm_node_name():
        if not name.endswith("_caption"):
            continue
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected llm node name format: {name!r}")
        model = parts[1]
        out[name] = by_model[model]

    return out


def sample_llm_latency_param() -> Dict[str, LlmlatencyParam]:
    # 对于每一个llm node进行均匀采样参数
    ll = config.get("llm_latency", {})
    alpha_rng = ll.get("alpha_ms_per_token")
    beta_rng = ll.get("beta_ms")
    noise_rng = ll.get("noise_sigma_ms")
    if not isinstance(alpha_rng, (list, tuple)) or len(alpha_rng) != 2:
        raise ValueError("config.llm_latency.alpha_ms_per_token must be a 2-item list like [min, max]")
    if not isinstance(beta_rng, (list, tuple)) or len(beta_rng) != 2:
        raise ValueError("config.llm_latency.beta_ms must be a 2-item list like [min, max]")
    if not isinstance(noise_rng, (list, tuple)) or len(noise_rng) != 2:
        raise ValueError("config.llm_latency.noise_sigma_ms must be a 2-item list like [min, max]")

    alpha_dist = UniformDistribution(float(alpha_rng[0]), float(alpha_rng[1]))
    beta_dist = UniformDistribution(float(beta_rng[0]), float(beta_rng[1]))
    noise_dist = UniformDistribution(float(noise_rng[0]), float(noise_rng[1]))

    llm_nodes: list[str] = []
    for pr in cloud_region_names:
        llm_nodes.append(f"{pr}_caption")
        llm_nodes.append(f"{pr}_query")
    llm_nodes.extend(get_llm_node_name())  # p4/p5 caption/query nodes

    out: Dict[str, LlmlatencyParam] = {}
    for node in llm_nodes:
        p = LlmlatencyParam()
        p.alpha_ms_per_token = alpha_dist.sample()
        p.beta_ms = beta_dist.sample()
        p.noise_sigma_ms = noise_dist.sample()
        out[node] = p
    return out


def get_comm_endpoints() -> list[str]:
    """通信端点 15 个：local + p1/p2/p3 各 4 region + p4 + p5。"""
    eps = ["local"]
    for p in CLOUD_PROVIDERS:
        for r in CLOUD_REGIONS[p]:
            eps.append(f"{p}_{r}")
    eps.extend(["p4", "p5"])
    if len(eps) != 15:
        raise ValueError(f"Expected 15 comm endpoints, got {len(eps)}")
    return eps


def _classify_network_edge(a: str, b: str) -> str:
    """
    返回 config.network 下的类别 key：
    - edge_to_cloud: 涉及 local
    - inter_region_intra_provider: 同云 provider、不同 region（仅 p1–p3 的 region 端点）
    - intra_region_inter_provider: 同 region 名、不同云 provider
    - inter_region_inter_provider: 其余（含 p4/p5 与云、或 p4 与 p5）
    """
    if a == b:
        raise ValueError("edge endpoints must be distinct")
    if a == "local" or b == "local":
        return "edge_to_cloud"
    if a in cloud_region_names and b in cloud_region_names:
        pa, ra = a.split("_", 1)
        pb, rb = b.split("_", 1)
        if pa == pb and ra != rb:
            return "inter_region_intra_provider"
        if ra == rb and pa != pb:
            return "intra_region_inter_provider"
        return "inter_region_inter_provider"
    return "inter_region_inter_provider"


def _network_category_cfg() -> dict[str, Any]:
    net = config.get("network")
    if not isinstance(net, dict):
        raise ValueError("config.network must be a dict")
    for k in (
        "inter_region_intra_provider",
        "intra_region_inter_provider",
        "inter_region_inter_provider",
        "edge_to_cloud",
    ):
        if k not in net:
            raise KeyError(f"config.network missing key {k!r}")
    return net


def _sample_rtt_params_for_category(net: dict[str, Any], category: str) -> tuple[float, float]:
    cat = net[category]
    mrng = cat.get("rtt_mean_ms")
    srng = cat.get("rtt_std_ms")
    if not isinstance(mrng, (list, tuple)) or len(mrng) != 2:
        raise ValueError(f"config.network.{category}.rtt_mean_ms must be [min, max]")
    if not isinstance(srng, (list, tuple)) or len(srng) != 2:
        raise ValueError(f"config.network.{category}.rtt_std_ms must be [min, max]")
    mean = UniformDistribution(float(mrng[0]), float(mrng[1])).sample()
    std = UniformDistribution(float(srng[0]), float(srng[1])).sample()
    if mean <= 0:
        raise ValueError("sampled RTT mean must be > 0")
    if std < 0:
        raise ValueError("sampled RTT std must be >= 0")
    return mean, std


def _sample_bw_params_for_category(net: dict[str, Any], category: str) -> tuple[float, float]:
    cat = net[category]
    mrng = cat.get("bw_mean_mbps")
    srng = cat.get("bw_std_mbps")
    if not isinstance(mrng, (list, tuple)) or len(mrng) != 2:
        raise ValueError(f"config.network.{category}.bw_mean_mbps must be [min, max]")
    if not isinstance(srng, (list, tuple)) or len(srng) != 2:
        raise ValueError(f"config.network.{category}.bw_std_mbps must be [min, max]")
    mean = UniformDistribution(float(mrng[0]), float(mrng[1])).sample()
    std = UniformDistribution(float(srng[0]), float(srng[1])).sample()
    if mean <= 0:
        raise ValueError("sampled BW mean must be > 0")
    if std < 0:
        raise ValueError("sampled BW std must be >= 0")
    return mean, std


def sample_edge_rtt() -> Dict[Tuple[str, str], RTT]:
    """
    15 个端点两两无向边：对每条边按类别从 network.* 均匀采样 RTT 的 mean/std（用于 LogNormal 刻画）。
    同时写入 (a,b) 与 (b,a)，参数相同。
    """
    net = _network_category_cfg()
    endpoints = get_comm_endpoints()
    out: Dict[Tuple[str, str], RTT] = {}
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a, b = endpoints[i], endpoints[j]
            cat = _classify_network_edge(a, b)
            mean, std = _sample_rtt_params_for_category(net, cat)
            r = RTT()
            r.mean = mean
            r.std = std
            out[(a, b)] = r
            out[(b, a)] = r
    return out


def sample_edge_bw() -> Dict[Tuple[str, str], BW]:
    """同上，对带宽（Mbps）的 mean/std 采样。"""
    net = _network_category_cfg()
    endpoints = get_comm_endpoints()
    out: Dict[Tuple[str, str], BW] = {}
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a, b = endpoints[i], endpoints[j]
            cat = _classify_network_edge(a, b)
            mean, std = _sample_bw_params_for_category(net, cat)
            w = BW()
            w.mean = mean
            w.std = std
            out[(a, b)] = w
            out[(b, a)] = w
    return out


@dataclass(frozen=True, slots=True)
class DistributionParameters:
    data_conversion_ratio: Dict[Operation, LogNormalDistribution]
    utility: Dict[str, float]
    llm_token_price: Dict[str, tuple[float, float]]
    storage_price: Dict[str, float]
    segment_price: Dict[str, float]
    split_price: Dict[str, float]
    egress_price: Any
    seg_spl_exec_time_param: Dict[str, SegSplExecTimeParam]
    query_output_token_num_param: Dict[str, QueryOutputTokenNumParam]
    caption_output_token_num_param: Dict[str, CaptionOutputTokenNumParam]
    caption_input_token_num_param: Dict[str, CaptionInputTokenNumParam]
    llm_latency_param: Dict[str, LlmlatencyParam]
    edge_rtt: Dict[Tuple[str, str], RTT]
    edge_bw: Dict[Tuple[str, str], BW]








def sample() -> DistributionParameters:
    data_conversion_ratio = sample_data_conversion_ratio()
    utility = sample_utility()
    llm_token_price = sample_llm_token_price()
    storage_price = sample_storage_price()
    segment_price = sample_segment_price()
    split_price = sample_split_price()
    egress_price = sample_egress_price()
    seg_spl_exec_time_param = sample_seg_spl_exec_time_param()
    query_output_token_num_param = sample_query_output_token_num_param()
    caption_output_token_num_param = sample_caption_output_token_num_param()
    caption_input_token_num_param = sample_caption_input_token_num_param()
    llm_latency_param = sample_llm_latency_param()
    edge_rtt = sample_edge_rtt()
    edge_bw = sample_edge_bw()
    return DistributionParameters(
        data_conversion_ratio=data_conversion_ratio,
        utility=utility,
        llm_token_price=llm_token_price,
        storage_price=storage_price,
        segment_price=segment_price,
        split_price=split_price,
        egress_price=egress_price,
        seg_spl_exec_time_param=seg_spl_exec_time_param,
        query_output_token_num_param=query_output_token_num_param,
        caption_output_token_num_param=caption_output_token_num_param,
        caption_input_token_num_param=caption_input_token_num_param,
        llm_latency_param=llm_latency_param,
        edge_rtt=edge_rtt,
        edge_bw=edge_bw,
    )


def sample_query_with_budget(num: float) -> list[Query]:
    # num 表示 query 的数量
    n = int(num)
    if n <= 0:
        raise ValueError("num must be a positive integer")

    b = config.get("budget", {})
    ds = config.get("data_size")
    if not isinstance(ds, (list, tuple)) or len(ds) != 2:
        raise ValueError("config.data_size must be a 2-item list like [min, max] MB")

    # 兼容两种写法：
    # 1) 旧版：budget.latency_intercept_s / latency_slope_per_mb / cost_intercept_usd / cost_slope_per_mb
    # 2) 数据驱动：budget.baseline.{...} + budget.slack_factor.{latency,cost}
    if isinstance(b, dict) and "baseline" in b:
        base = b.get("baseline") or {}
        slack = b.get("slack_factor") or {}
        lat_sf = float(slack.get("latency", 1.0))
        cost_sf = float(slack.get("cost", 1.0))
        if lat_sf <= 0 or cost_sf <= 0:
            raise ValueError("budget.slack_factor.latency/cost must be > 0")
        lat_i = float(base.get("latency_intercept_s")) * lat_sf
        lat_s = float(base.get("latency_slope_per_mb")) * lat_sf
        cost_i = float(base.get("cost_intercept_usd")) * cost_sf
        cost_s = float(base.get("cost_slope_per_mb")) * cost_sf
    else:
        lat_i = float(b.get("latency_intercept_s"))
        lat_s = float(b.get("latency_slope_per_mb"))
        cost_i = float(b.get("cost_intercept_usd"))
        cost_s = float(b.get("cost_slope_per_mb"))

    lo, hi = float(ds[0]), float(ds[1])
    if hi < lo:
        raise ValueError("config.data_size must satisfy max >= min")

    size_dist = UniformDistribution(lo, hi)
    out: list[Query] = []
    for _ in range(n):
        data_size_mb = size_dist.sample()
        q = Query()
        q.data_size_mb = data_size_mb
        q.latency_budget = lat_i + lat_s * data_size_mb
        q.cost_budget = cost_i + cost_s * data_size_mb
        out.append(q)
    return out




if __name__ == "__main__":
    queries = sample_query_with_budget(10)
    print(queries)
