import random
import math
from typing import Any, Dict, Literal, Sequence, Tuple
from dataclasses import dataclass
from param import *
from param import build_llm_node_name, get_llm_node_name, iter_cloud_llm_deployments
# 全局种子,确保实验可以复现
random.seed(42)

config = get_simulation_config()
cloud_region_names = get_cloud_region_name()


def _lookup_caption_model_param(model_params: dict[str, Any], model_key: str) -> dict[str, Any]:
    mp = model_params.get(model_key)
    if isinstance(mp, dict):
        return mp
    alt = model_key.replace("-", " ")
    mp = model_params.get(alt)
    if isinstance(mp, dict):
        return mp
    raise ValueError(f"config.caption_model_params[{model_key!r}] is required")


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
        # 记录原始 mean/std（用于从同一 LogNormal 参数采样）
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
    io_s_per_MB: float
    alpha: float
    theta0: float
    theta1: float


class QueryOutputTokenNumParam:
    mean: float
    std: float


class CaptionOutputTokenNumParam:
    base: float
    coef_per_MB: float
    std: float


class CaptionInputTokenNumParam:
    input_tokens_per_MB: float


class LlmlatencyParam:
    alpha_ms_per_token: float
    beta_ms: float
    noise_sigma_ms: float


class Query:
    data_size_MB: float
    cost_budget: float # USD
    latency_budget: float # s


class RTT:
    mean: float
    std: float


class BW:
    mean: float
    std: float


def sample_query_output_token_num_param() -> Dict[str, QueryOutputTokenNumParam]:
    # 对所有 llm 节点生成 QueryOutputTokenNumParam（按节点独立采样）
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

    out: Dict[str, QueryOutputTokenNumParam] = {}
    for provider, region, model_key in iter_cloud_llm_deployments():
        p = QueryOutputTokenNumParam()
        p.mean = mean_dist.sample()
        p.std = std_dist.sample()
        if p.mean <= 0:
            raise ValueError("sampled query_mean must be > 0")
        if p.std < 0:
            raise ValueError("sampled query_std must be >= 0")
        out[build_llm_node_name(provider, region, model_key, "query")] = p

    for name in get_llm_node_name():
        if not name.endswith("_query") or name in out:
            continue
        p = QueryOutputTokenNumParam()
        p.mean = mean_dist.sample()
        p.std = std_dist.sample()
        if p.mean <= 0:
            raise ValueError("sampled query_mean must be > 0")
        if p.std < 0:
            raise ValueError("sampled query_std must be >= 0")
        out[name] = p

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
    um = config.get("utility_by_node")
    if not isinstance(um, dict):
        raise ValueError("config.utility_by_node must be a dict with keys: segment/split/caption/query")

    def _read_op_map(op: str) -> Dict[str, float]:
        m = um.get(op)
        if not isinstance(m, dict):
            raise ValueError(f"config.utility_by_node.{op} must be a dict keyed by '<provider>_<region>'")
        out: Dict[str, float] = {}
        for pr in cloud_region_names:
            if pr not in m:
                raise ValueError(f"config.utility_by_node.{op} missing key {pr!r}")
            out[pr] = float(m[pr])
        return out

    seg_u = _read_op_map("segment")
    spl_u = _read_op_map("split")
    utilities: Dict[str, float] = {}
    for pr in cloud_region_names:
        utilities[f"{pr}_segment"] = seg_u[pr]
        utilities[f"{pr}_split"] = spl_u[pr]

    cap_raw = um.get("caption")
    qry_raw = um.get("query")
    if not isinstance(cap_raw, dict) or not isinstance(qry_raw, dict):
        raise ValueError("config.utility_by_node.caption/query must be dict")

    # 优先读取模型粒度 key（<provider>_<region>__<model_key>）。
    has_model_granularity = True
    for provider, region, model_key in iter_cloud_llm_deployments():
        nk = build_llm_node_name(provider, region, model_key, "caption").removesuffix("_caption")
        if nk not in cap_raw or nk not in qry_raw:
            has_model_granularity = False
            break

    if has_model_granularity:
        for provider, region, model_key in iter_cloud_llm_deployments():
            nk = build_llm_node_name(provider, region, model_key, "caption").removesuffix("_caption")
            utilities[build_llm_node_name(provider, region, model_key, "caption")] = float(cap_raw[nk])
            utilities[build_llm_node_name(provider, region, model_key, "query")] = float(qry_raw[nk])
        return utilities

    # 否则回退到 llm_profiles 里的 accuracy（兼容旧配置）。
    llm_profiles = config.get("llm_profiles")
    if not isinstance(llm_profiles, dict):
        raise ValueError("config.llm_profiles must be a dict keyed by '<provider>_<region>'")
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        prof_by_model = llm_profiles.get(pr)
        if not isinstance(prof_by_model, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}] is required")
        mp = prof_by_model.get(model_key)
        if not isinstance(mp, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}] is required")
        acc = mp.get("accuracy")
        if acc is None:
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}].accuracy is required")
        u = float(acc)
        utilities[build_llm_node_name(provider, region, model_key, "caption")] = u
        utilities[build_llm_node_name(provider, region, model_key, "query")] = u
    return utilities


def sample_llm_token_price():
    # Return mapping: llm_node_name -> (input_price, output_price)，单位 USD / 百万 tokens
    llm_profiles = config.get("llm_profiles")
    if not isinstance(llm_profiles, dict):
        raise ValueError("config.llm_profiles must be a dict keyed by '<provider>_<region>'")
    llm_token_price: Dict[str, tuple[float, float]] = {}
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        prof_by_model = llm_profiles.get(pr)
        if not isinstance(prof_by_model, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}] is required")
        prof = prof_by_model.get(model_key)
        if not isinstance(prof, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}] is required")
        inp = float(prof.get("input_price_per_m"))
        outp = float(prof.get("output_price_per_m"))
        llm_token_price[build_llm_node_name(provider, region, model_key, "caption")] = (inp, outp)
        llm_token_price[build_llm_node_name(provider, region, model_key, "query")] = (inp, outp)
    return llm_token_price


def sample_storage_price():
    # 每个 (provider,region) 对应一个固定存储单价；跨 operation 复用
    m = config.get("storage_price_per_gb_month")
    if not isinstance(m, dict):
        raise ValueError("config.storage_price_per_gb_month must be a dict keyed by '<provider>_<region>'")
    prices: Dict[str, float] = {}
    ops: Sequence[Operation] = ("segment", "split")
    for pr in cloud_region_names:
        if pr not in m:
            raise ValueError(f"config.storage_price_per_gb_month missing key {pr!r}")
        price = float(m[pr])
        for op in ops:
            prices[f"{pr}_{op}"] = price
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        price = float(m[pr])
        prices[build_llm_node_name(provider, region, model_key, "caption")] = price
        prices[build_llm_node_name(provider, region, model_key, "query")] = price
    return prices


def sample_egress_price():
    cfg = config.get("egress_price_matrix", {})
    endpoints = cfg.get("endpoints")
    matrix = cfg.get("matrix")
    if not isinstance(endpoints, list) or not all(isinstance(x, str) for x in endpoints):
        raise ValueError("config.egress_price_matrix.endpoints must be a string list")
    if not isinstance(matrix, list) or len(matrix) != len(endpoints):
        raise ValueError("config.egress_price_matrix.matrix row count must match endpoints length")
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(endpoints):
            raise ValueError("config.egress_price_matrix.matrix must be a square matrix")
    return {"endpoints": endpoints, "matrix": [[float(v) for v in row] for row in matrix]}


def sample_segment_price():
    m = config.get("segment_price_per_min")
    if not isinstance(m, dict):
        raise ValueError("config.segment_price_per_min must be a dict keyed by '<provider>_<region>'")
    prices: Dict[str, float] = {}
    for pr in cloud_region_names:
        if pr not in m:
            raise ValueError(f"config.segment_price_per_min missing key {pr!r}")
        prices[f"{pr}_segment"] = float(m[pr])
    return prices


def sample_split_price():
    m = config.get("split_price_per_min")
    if not isinstance(m, dict):
        raise ValueError("config.split_price_per_min must be a dict keyed by '<provider>_<region>'")
    prices: Dict[str, float] = {}
    for pr in cloud_region_names:
        if pr not in m:
            raise ValueError(f"config.split_price_per_min missing key {pr!r}")
        prices[f"{pr}_split"] = float(m[pr])
    return prices


def sample_seg_spl_exec_time_param() -> Dict[str, SegSplExecTimeParam]:
    # 为 24 个 split/segment 节点生成执行时间参数（逐节点确定值）。
    et = config.get("exec_time", {})
    seg_by_node = et.get("segment_by_node", {}) if isinstance(et, dict) else {}
    spl_by_node = et.get("split_by_node", {}) if isinstance(et, dict) else {}
    if not isinstance(seg_by_node, dict) or not isinstance(spl_by_node, dict):
        raise ValueError("config.exec_time.segment_by_node / split_by_node must be dict")

    params: Dict[str, SegSplExecTimeParam] = {}
    for pr in cloud_region_names:
        seg_node = f"{pr}_segment"
        p1 = SegSplExecTimeParam()
        seg_cfg = seg_by_node.get(pr) if isinstance(seg_by_node, dict) else None
        if not isinstance(seg_cfg, dict):
            raise ValueError(f"config.exec_time.segment_by_node missing key {pr!r}")
        p1.io_s_per_MB = float(seg_cfg["io_s_per_MB"])
        p1.alpha = float(seg_cfg["alpha"])
        p1.theta0 = float(seg_cfg["theta0"])
        p1.theta1 = float(seg_cfg["theta1"])
        params[seg_node] = p1

        spl_node = f"{pr}_split"
        p2 = SegSplExecTimeParam()
        spl_cfg = spl_by_node.get(pr) if isinstance(spl_by_node, dict) else None
        if not isinstance(spl_cfg, dict):
            raise ValueError(f"config.exec_time.split_by_node missing key {pr!r}")
        p2.io_s_per_MB = float(spl_cfg["io_s_per_MB"])
        p2.alpha = float(spl_cfg["alpha"])
        p2.theta0 = float(spl_cfg["theta0"])
        p2.theta1 = float(spl_cfg["theta1"])
        params[spl_node] = p2

    return params


def sample_caption_output_token_num_param() -> Dict[str, CaptionOutputTokenNumParam]:
    llm_profiles = config.get("llm_profiles")
    model_params = config.get("caption_model_params")
    if not isinstance(llm_profiles, dict):
        raise ValueError("config.llm_profiles must be a dict keyed by '<provider>_<region>'")
    if not isinstance(model_params, dict):
        raise ValueError("config.caption_model_params must be a dict keyed by model name")

    out: Dict[str, CaptionOutputTokenNumParam] = {}
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        prof_by_model = llm_profiles.get(pr)
        if not isinstance(prof_by_model, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}] is required")
        if model_key not in prof_by_model:
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}] is required")
        mp = _lookup_caption_model_param(model_params, model_key)
        p = CaptionOutputTokenNumParam()
        p.base = float(mp["base"])
        p.coef_per_MB = float(mp["coef_per_MB"])
        p.std = float(mp["std"])
        out[build_llm_node_name(provider, region, model_key, "caption")] = p

    return out


def sample_caption_input_token_num_param() -> Dict[str, CaptionInputTokenNumParam]:
    llm_profiles = config.get("llm_profiles")
    model_params = config.get("caption_model_params")
    if not isinstance(llm_profiles, dict):
        raise ValueError("config.llm_profiles must be a dict keyed by '<provider>_<region>'")
    if not isinstance(model_params, dict):
        raise ValueError("config.caption_model_params must be a dict keyed by model name")

    out: Dict[str, CaptionInputTokenNumParam] = {}
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        prof_by_model = llm_profiles.get(pr)
        if not isinstance(prof_by_model, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}] is required")
        if model_key not in prof_by_model:
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}] is required")
        mp = _lookup_caption_model_param(model_params, model_key)
        p = CaptionInputTokenNumParam()
        p.input_tokens_per_MB = float(mp["input_tokens_per_MB"])
        if p.input_tokens_per_MB <= 0:
            raise ValueError("caption_model_params.input_tokens_per_MB must be > 0")
        out[build_llm_node_name(provider, region, model_key, "caption")] = p

    return out


def sample_llm_latency_param() -> Dict[str, LlmlatencyParam]:
    # 由 llm_profiles 直接给出 TTFT/TPT，噪声可在 llm_latency.default_noise_sigma_ms 设常数
    llm_profiles = config.get("llm_profiles")
    if not isinstance(llm_profiles, dict):
        raise ValueError("config.llm_profiles must be a dict keyed by '<provider>_<region>'")
    ll = config.get("llm_latency", {}) or {}
    noise_sigma_ms = float(ll.get("default_noise_sigma_ms", 50.0))

    out: Dict[str, LlmlatencyParam] = {}
    for provider, region, model_key in iter_cloud_llm_deployments():
        pr = f"{provider}_{region}"
        prof_by_model = llm_profiles.get(pr)
        if not isinstance(prof_by_model, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}] is required")
        prof = prof_by_model.get(model_key)
        if not isinstance(prof, dict):
            raise ValueError(f"config.llm_profiles[{pr!r}][{model_key!r}] is required")
        p = LlmlatencyParam()
        p.alpha_ms_per_token = float(prof.get("tpt_ms_per_token"))
        p.beta_ms = float(prof.get("ttft_ms"))
        p.noise_sigma_ms = noise_sigma_ms
        out[build_llm_node_name(provider, region, model_key, "caption")] = p
        out[build_llm_node_name(provider, region, model_key, "query")] = p
    return out


def get_comm_endpoints() -> list[str]:
    """通信端点列表，优先使用 network_matrix.endpoints。"""
    nm = config.get("network_matrix", {})
    eps = nm.get("endpoints") if isinstance(nm, dict) else None
    if isinstance(eps, list) and all(isinstance(x, str) for x in eps):
        return list(eps)
    fallback = ["local"]
    for p in CLOUD_PROVIDERS:
        for r in CLOUD_REGIONS[p]:
            fallback.append(f"{p}_{r}")
    return fallback


def sample_edge_rtt() -> Dict[Tuple[str, str], RTT]:
    """按 network_matrix.rtt_mean_std 读取每对端点的 RTT(mean,std)。"""
    nm = config.get("network_matrix", {})
    if not isinstance(nm, dict):
        raise ValueError("config.network_matrix must be a dict")
    endpoints = get_comm_endpoints()
    mat = nm.get("rtt_mean_std")
    if not isinstance(mat, list) or len(mat) != len(endpoints):
        raise ValueError("config.network_matrix.rtt_mean_std must be a square matrix")
    for row in mat:
        if not isinstance(row, list) or len(row) != len(endpoints):
            raise ValueError("config.network_matrix.rtt_mean_std must be a square matrix")
    out: Dict[Tuple[str, str], RTT] = {}
    for i, a in enumerate(endpoints):
        for j, b in enumerate(endpoints):
            if i == j:
                continue
            pair = mat[i][j]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("each RTT matrix cell must be [mean_ms, std_ms]")
            mean, std = float(pair[0]), float(pair[1])
            r = RTT()
            r.mean = mean
            r.std = std
            out[(a, b)] = r
    return out


def sample_edge_bw() -> Dict[Tuple[str, str], BW]:
    """按 network_matrix.bw_mean_std 读取每对端点的 BW(mean,std)。"""
    nm = config.get("network_matrix", {})
    if not isinstance(nm, dict):
        raise ValueError("config.network_matrix must be a dict")
    endpoints = get_comm_endpoints()
    mat = nm.get("bw_mean_std")
    if not isinstance(mat, list) or len(mat) != len(endpoints):
        raise ValueError("config.network_matrix.bw_mean_std must be a square matrix")
    for row in mat:
        if not isinstance(row, list) or len(row) != len(endpoints):
            raise ValueError("config.network_matrix.bw_mean_std must be a square matrix")
    out: Dict[Tuple[str, str], BW] = {}
    for i, a in enumerate(endpoints):
        for j, b in enumerate(endpoints):
            if i == j:
                continue
            pair = mat[i][j]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("each BW matrix cell must be [mean_mbps, std_mbps]")
            mean, std = float(pair[0]), float(pair[1])
            w = BW()
            w.mean = mean
            w.std = std
            out[(a, b)] = w
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








def build_distribution_parameters() -> DistributionParameters:
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


def sample() -> DistributionParameters:
    # 兼容旧调用名
    return build_distribution_parameters()


def sample_query_with_budget(num: float) -> list[Query]:
    # num 表示 query 的数量
    n = int(num)
    if n <= 0:
        raise ValueError("num must be a positive integer")

    b = config.get("budget", {})
    ds = config.get("data_size")
    if not isinstance(ds, (list, tuple)) or len(ds) != 2:
        raise ValueError("config.data_size must be a 2-item list like [min, max] MB")

    if isinstance(b, dict) and "baseline" in b:
        base = b.get("baseline") or {}
        slack = b.get("slack_factor") or {}
        lat_sf = float(slack.get("latency", 1.0))
        cost_sf = float(slack.get("cost", 1.0))
        if lat_sf <= 0 or cost_sf <= 0:
            raise ValueError("budget.slack_factor.latency/cost must be > 0")
        lat_i = float(base.get("latency_intercept_s")) * lat_sf
        lat_s = float(base.get("latency_slope_per_MB")) * lat_sf
        cost_i = float(base.get("cost_intercept_usd")) * cost_sf
        cost_s = float(base.get("cost_slope_per_MB")) * cost_sf
    else:
        lat_i = float(b.get("latency_intercept_s"))
        lat_s = float(b.get("latency_slope_per_MB"))
        cost_i = float(b.get("cost_intercept_usd"))
        cost_s = float(b.get("cost_slope_per_MB"))

    lo, hi = float(ds[0]), float(ds[1])
    if hi < lo:
        raise ValueError("config.data_size must satisfy max >= min")

    size_dist = UniformDistribution(lo, hi)
    out: list[Query] = []
    for _ in range(n):
        data_size_MB = size_dist.sample()
        q = Query()
        q.data_size_MB = data_size_MB
        q.latency_budget = lat_i + lat_s * data_size_MB
        q.cost_budget = cost_i + cost_s * data_size_MB
        out.append(q)
    return out
