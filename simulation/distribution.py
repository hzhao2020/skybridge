import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


# 固定随机种子：调用方不传 rng 时，默认使用这个全局 RNG
DEFAULT_RANDOM_SEED = 123
_DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


@dataclass
class RttLognormalParams:
    """
    RTT 对数正态分布参数（单位：毫秒）。

    建模：RTT ~ LogNormal(mu, sigma)
    - 你可以直接设置 mu / sigma；
    - 或者用经验均值 m 和标准差 s 反推 mu / sigma（在外部自己算好再传进来）。
    """

    mu: float = math.log(30.0)  # 默认大致 30ms 量级
    sigma: float = 0.4


@dataclass
class BandwidthParetoParams:
    """
    帕累托分布参数（单位以 Mbps 为尺度）

    说明：
    - numpy.pareto(a) 生成 Y，其密度 ~ (1 + y)^(-a-1)，y >= 0
    - 实际带宽 X = xm * (1 + Y)，X >= xm
    - alpha 越大，尾部 “肥” 程度越小（分布更集中）
    """

    # 最小带宽（尺度参数 xm），单位 Mbps
    xm_mbps: float = 100.0
    # 形状参数 alpha：越大分布越集中，尾部越不“肥”
    alpha: float = 2.5


@dataclass
class OperationSizeLognormalParams:
    """
    各种操作的数据体积转化率（输出大小 / 输入大小）建模参数。

    使用对数正态分布：
    - ratio ~ LogNormal(mu, sigma)
    - 期望 E[ratio] ≈ exp(mu + 0.5 * sigma^2)

    默认设置：
    - video_segment: 均值约 1，方差很小（sigma_small）
    - video_split:   均值约 1，方差很小（sigma_small）
    - caption:       均值约 0.1，方差中等（sigma_medium）
    - query:         均值约 1，方差中等（sigma_medium）
    """

    # 控制整体“方差级别”的形状参数
    sigma_small: float = 0.08   # 很小的离散程度
    sigma_medium: float = 0.30  # 中等的离散程度

    # 目标均值（在给定 sigma 下大致逼近）
    mean_segment: float = 1.0
    mean_split: float = 1.0
    mean_caption: float = 0.1
    mean_query: float = 1.0

    @property
    def segment_mu(self) -> float:
        return math.log(self.mean_segment) - 0.5 * self.sigma_small**2

    @property
    def split_mu(self) -> float:
        return math.log(self.mean_split) - 0.5 * self.sigma_small**2

    @property
    def caption_mu(self) -> float:
        return math.log(self.mean_caption) - 0.5 * self.sigma_medium**2

    @property
    def query_mu(self) -> float:
        return math.log(self.mean_query) - 0.5 * self.sigma_medium**2


@dataclass
class LlmTokenLognormalParams:
    """
    LLM 生成 token 数目的对数正态分布参数。

    建模思路：
    - 先用所有输入视频的“规模”构造一个标量（例如总时长分钟数）；
    - 根据该规模线性调整 mu，再从 LogNormal(mu, sigma) 采样 token 数。

    默认仅给出一个相对中性的线性模型，你可以按需要重写参数：
    - base_mu:    基础 μ（当总视频规模接近 0 时）
    - mu_per_unit: 每单位视频规模（比如每分钟）额外增加的 μ
    - sigma:      控制 token 数目的离散程度
    """

    base_mu: float = math.log(512.0)  # 规模很小时，大致几百 token
    mu_per_unit: float = 0.05         # 每单位视频规模带来的 μ 增量
    sigma: float = 0.6


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """内部工具函数：统一获取 numpy Generator。"""
    return _DEFAULT_RNG if rng is None else rng


def sample_rtt_ms(
    params: Optional[RttLognormalParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    采样单次 RTT（毫秒）。

    使用对数正态分布采样一次 RTT（毫秒）。
    所有“同云 / 跨云 / 远近”等语义由你在外部通过不同的参数对象来区分。
    """
    if params is None:
        params = RttLognormalParams()
    rng = _get_rng(rng)

    rtt = float(rng.lognormal(mean=params.mu, sigma=params.sigma))
    # 防止极端过小的 RTT（比如 < 0.1ms），做一个下界
    return max(rtt, 0.1)


def sample_bandwidth_mbps(
    params: Optional[BandwidthParetoParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    采样单次带宽（Mbps）。

    使用帕累托分布采样一次带宽（Mbps）。
    - 你可以为不同类型的链路使用不同的 `BandwidthParetoParams` 实例。
    """
    if params is None:
        params = BandwidthParetoParams()
    rng = _get_rng(rng)

    # numpy.pareto 返回 Y >= 0，实际带宽 X = xm * (1 + Y) >= xm
    y = float(rng.pareto(params.alpha))
    bw = params.xm_mbps * (1.0 + y)
    return max(bw, 1.0)


def sample_link(
    rtt_params: Optional[RttLognormalParams] = None,
    bw_params: Optional[BandwidthParetoParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """
    同时采样一条链路的 RTT 与带宽。

    返回：
        (rtt_ms, bandwidth_mbps)
    """
    rng = _get_rng(rng)

    rtt = sample_rtt_ms(params=rtt_params, rng=rng)
    bw = sample_bandwidth_mbps(params=bw_params, rng=rng)
    return rtt, bw


# -----------------------------
# Node definitions
# -----------------------------

NODES = [
    "local",
    "gcp_us",
    "gcp_tw",
    "gcp_sg",
    "aws_us",
    "aws_sg",
    "aliyun_us",
    "aliyun_se",
]


# -----------------------------
# RTT mean table (ms)
# -----------------------------

RTT_MEAN_MS: Dict[Tuple[str, str], float] = {
    ("local", "gcp_tw"): 35,
    ("local", "gcp_sg"): 60,
    ("local", "aws_sg"): 60,
    ("local", "aliyun_se"): 60,
    ("local", "gcp_us"): 140,
    ("local", "aws_us"): 140,
    ("local", "aliyun_us"): 200,
    ("gcp_tw", "gcp_sg"): 45,
    ("gcp_tw", "aws_sg"): 45,
    ("gcp_tw", "aliyun_se"): 45,
    ("gcp_tw", "gcp_us"): 130,
    ("gcp_tw", "aws_us"): 130,
    ("gcp_tw", "aliyun_us"): 190,
    ("gcp_sg", "aws_sg"): 3,
    ("gcp_sg", "aliyun_se"): 3,
    ("aws_sg", "aliyun_se"): 3,
    ("gcp_sg", "gcp_us"): 140,
    ("gcp_sg", "aws_us"): 140,
    ("gcp_sg", "aliyun_us"): 200,
    ("aws_sg", "gcp_us"): 140,
    ("aws_sg", "aws_us"): 140,
    ("aws_sg", "aliyun_us"): 200,
    ("aliyun_se", "gcp_us"): 140,
    ("aliyun_se", "aws_us"): 140,
    ("aliyun_se", "aliyun_us"): 200,
    ("gcp_us", "aws_us"): 2,
    ("gcp_us", "aliyun_us"): 75,
    ("aws_us", "aliyun_us"): 75,
}

RTT_SIGMA = 0.25


# -----------------------------
# Bandwidth minimum (Mbps)
# -----------------------------

BW_XM_MBPS: Dict[Tuple[str, str], float] = {
    ("local", "gcp_tw"): 200,
    ("local", "gcp_sg"): 120,
    ("local", "aws_sg"): 120,
    ("local", "aliyun_se"): 120,
    ("local", "gcp_us"): 80,
    ("local", "aws_us"): 80,
    ("local", "aliyun_us"): 60,
    ("gcp_tw", "gcp_sg"): 400,
    ("gcp_tw", "aws_sg"): 400,
    ("gcp_tw", "aliyun_se"): 400,
    ("gcp_tw", "gcp_us"): 150,
    ("gcp_tw", "aws_us"): 150,
    ("gcp_tw", "aliyun_us"): 120,
    ("gcp_sg", "aws_sg"): 1500,
    ("gcp_sg", "aliyun_se"): 1200,
    ("gcp_sg", "gcp_us"): 200,
    ("gcp_sg", "aws_us"): 200,
    ("gcp_sg", "aliyun_us"): 150,
    ("aws_sg", "gcp_us"): 200,
    ("aws_sg", "aws_us"): 200,
    ("aws_sg", "aliyun_us"): 150,
    ("aliyun_se", "gcp_us"): 200,
    ("aliyun_se", "aws_us"): 200,
    ("aliyun_se", "aliyun_us"): 150,
    ("gcp_us", "aws_us"): 2000,
    ("gcp_us", "aliyun_us"): 800,
    ("aws_us", "aliyun_us"): 800,
}

BW_ALPHA = 2.5


# -----------------------------
# Internal helpers
# -----------------------------

def _lookup(table: Dict[Tuple[str, str], float], src: str, dst: str) -> float:
    if src == dst:
        raise ValueError("No lookup needed when src == dst")

    key = (src, dst)
    rev = (dst, src)

    if key in table:
        return table[key]

    if rev in table:
        return table[rev]

    raise ValueError(f"No parameter defined for link {src} <-> {dst}")


# -----------------------------
# Parameter builders
# -----------------------------

def get_rtt_params(src: str, dst: str) -> RttLognormalParams:
    if src == dst:
        mean = 2
    else:
        mean = _lookup(RTT_MEAN_MS, src, dst)

    sigma = RTT_SIGMA
    mu = math.log(mean) - 0.5 * sigma**2

    return RttLognormalParams(mu=mu, sigma=sigma)


def get_bw_params(src: str, dst: str) -> BandwidthParetoParams:
    if src == dst:
        xm = 2000
    else:
        xm = _lookup(BW_XM_MBPS, src, dst)

    return BandwidthParetoParams(
        xm_mbps=xm,
        alpha=BW_ALPHA,
    )


# -----------------------------
# Main sampling API
# -----------------------------

def sample_link_between(
    src: str,
    dst: str,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    rng = _get_rng(rng)

    rtt_params = get_rtt_params(src, dst)
    bw_params = get_bw_params(src, dst)

    rtt = sample_rtt_ms(rtt_params, rng)
    bw = sample_bandwidth_mbps(bw_params, rng)

    return rtt, bw


def sample_operation_size_ratio(
    operation: str,
    params: Optional[OperationSizeLognormalParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    采样一次“数据转化率”（输出大小 / 输入大小）。

    Args:
        operation: 操作名称，支持的值：
            - 'video_segment' / 'segment'
            - 'video_split'   / 'split'
            - 'caption'
            - 'llm_query'     / 'query'
        params: 可选，自定义各操作的对数正态分布参数
        rng: 可选，共享的随机数发生器

    Returns:
        ratio (float): 本次采样得到的大小比例
    """
    if params is None:
        params = OperationSizeLognormalParams()
    rng = _get_rng(rng)

    op = operation.lower()
    if op in {"video_segment", "segment"}:
        mu = params.segment_mu
        sigma = params.sigma_small
    elif op in {"video_split", "split"}:
        mu = params.split_mu
        sigma = params.sigma_small
    elif op == "caption":
        mu = params.caption_mu
        sigma = params.sigma_medium
    elif op in {"llm_query", "query"}:
        mu = params.query_mu
        sigma = params.sigma_medium
    else:
        raise ValueError(
            f"Unknown operation: {operation}. "
            "Expected one of: 'video_segment', 'segment', 'video_split', 'split', 'caption', 'llm_query', 'query'."
        )

    ratio = float(rng.lognormal(mean=mu, sigma=sigma))
    return ratio


def sample_llm_tokens_from_videos(
    videos,
    *,
    params: Optional[LlmTokenLognormalParams] = None,
    rng: Optional[np.random.Generator] = None,
    unit_extractor=None,
) -> int:
    """
    根据输入的视频集合，采样一次 LLM 生成的 token 数量（对数正态）。

    Args:
        videos: 表示视频的可迭代对象。推荐几种简单用法：
            1) 直接传总时长（秒）或总大小（MB）的浮点数；
            2) 传入一个 list，每个元素是该视频的“规模”（例如时长秒数）；
        params: LlmTokenLognormalParams，自定义 μ 的线性模型和 σ；
        rng: 可选，共享的随机数发生器；
        unit_extractor: 可选函数，用于从自定义 video 对象中提取“规模”标量。
            如果提供，如: lambda v: v.duration_seconds。

    返回:
        采样得到的 token 数（int，至少为 1）。
    """
    if params is None:
        params = LlmTokenLognormalParams()
    rng = _get_rng(rng)

    # 统一把输入 videos 映射为一个正的“规模标量” total_units
    if isinstance(videos, (int, float)):
        total_units = max(float(videos), 0.0)
    else:
        total_units = 0.0
        for v in videos:
            if unit_extractor is not None:
                val = float(unit_extractor(v))
            else:
                # 默认为元素本身就是标量（例如时长秒数）
                val = float(v)
            if val > 0:
                total_units += val

    # 将“规模”转为对 μ 的线性影响；例如你可以约定 total_units 是分钟数
    mu = params.base_mu + params.mu_per_unit * total_units
    sigma = params.sigma

    tokens = rng.lognormal(mean=mu, sigma=sigma)
    # 向最近整数取整，并确保至少为 1
    return max(int(round(tokens)), 1)


def sample_video_size_mb(
    *,
    min_mb: float = 5.0,
    max_mb: float = 500.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    从均匀分布 U[min_mb, max_mb] 中采样单个视频大小（单位：MB）。

    默认：
        min_mb = 5 MB
        max_mb = 500 MB
    """
    rng = _get_rng(rng)
    low = float(min_mb)
    high = float(max_mb)
    if high < low:
        raise ValueError(f"max_mb ({max_mb}) must be >= min_mb ({min_mb})")
    return float(rng.uniform(low, high))


@dataclass
class OperationGammaTimeParams:
    """
    使用 Gamma 条件分布建模单个 operation 的执行时间（并可叠加线性 IO 时间）：

        T_compute | S ~ Gamma(shape=k, scale=theta(S))
        E[T_compute|S] = k * theta(S) = base_time * f(S)
        T_io(S) = io_time_per_mb * S
        T_total = T_compute + T_io

    - 规模 S：使用 video_size_mb（MB）
    - 确定性部分 f(S)：可选择 O(S) 或 O(S log S)
    - 噪声：通过 Gamma 的 shape=k 控制（k 越大抖动越小）

    参数含义：
    - shape: Gamma 的形状参数 k
    - base_time_per_mb: 当 complexity='linear' 时，f(S)=S，对应每 MB 的平均时间（秒/MB）
    - io_time_per_mb: IO 时间系数（秒/MB），总 IO 时间与输入大小线性成正比
    - complexity: 'linear' 或 's_log_s'
    """

    shape: float = 3.0
    base_time_per_mb: float = 0.05
    io_time_per_mb: float = 0.0
    complexity: str = "linear"  # 'linear' | 's_log_s'


DEFAULT_EXECUTION_TIME_PARAMS: dict[str, OperationGammaTimeParams] = {
    # -----------------------------
    # Video Segmentation (S log S)
    # -----------------------------
    "seg_google_us": OperationGammaTimeParams(
        shape=3.5,
        base_time_per_mb=0.030,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_google_tw": OperationGammaTimeParams(
        shape=3.5,
        base_time_per_mb=0.032,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_google_sg": OperationGammaTimeParams(
        shape=3.5,
        base_time_per_mb=0.032,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_aws_us": OperationGammaTimeParams(
        shape=3.0,
        base_time_per_mb=0.036,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_aws_sg": OperationGammaTimeParams(
        shape=3.0,
        base_time_per_mb=0.036,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_aliyun_us": OperationGammaTimeParams(
        shape=2.8,
        base_time_per_mb=0.040,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),
    "seg_aliyun_se": OperationGammaTimeParams(
        shape=2.8,
        base_time_per_mb=0.040,
        io_time_per_mb=0.002,
        complexity="s_log_s",
    ),

    # -----------------------------
    # Video Splitting (linear)
    # -----------------------------
    "split_google_us": OperationGammaTimeParams(
        shape=4.0,
        base_time_per_mb=0.020,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_google_tw": OperationGammaTimeParams(
        shape=4.0,
        base_time_per_mb=0.022,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_google_sg": OperationGammaTimeParams(
        shape=4.0,
        base_time_per_mb=0.022,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_aws_us": OperationGammaTimeParams(
        shape=4.0,
        base_time_per_mb=0.024,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_aws_sg": OperationGammaTimeParams(
        shape=4.0,
        base_time_per_mb=0.024,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_aliyun_us": OperationGammaTimeParams(
        shape=3.8,
        base_time_per_mb=0.026,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
    "split_aliyun_se": OperationGammaTimeParams(
        shape=3.8,
        base_time_per_mb=0.026,
        io_time_per_mb=0.002,
        complexity="linear",
    ),
}


def sample_execution_time_seconds(
    task_name: str,
    operation_name: str,
    video_size_mb: float,
    *,
    params_overrides: Optional[dict[str, OperationGammaTimeParams]] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    使用 Gamma 分布估计某个 operation 的执行时间（秒）。

    Args:
        task_name: 任务名称（目前仅作为语义信息，不参与计算），例如：
            - 'Video Segmentation'
            - 'Video Splitting'
        operation_name: 具体 operation 名称，例如：
            - 'seg_aws_us', 'seg_aws_sg', 'seg_google_us', 'seg_google_tw'
            - 'split_aws_us', 'split_aws_sg', 'split_google_us', 'split_google_sg'
        video_size_mb: 视频大小（MB）
        params_overrides: 可选，用于覆盖默认的每个 operation Gamma 参数：
            例如：
                {
                  "seg_aws_us": OperationGammaTimeParams(shape=2.5, mean_time_per_mb=0.03),
                  ...
                }
        rng: 可选，共享的随机数发生器。

    Returns:
        本次采样得到的执行时间（秒）。
    """
    rng = _get_rng(rng)
    cfg_map = DEFAULT_EXECUTION_TIME_PARAMS if params_overrides is None else {
        **DEFAULT_EXECUTION_TIME_PARAMS,
        **params_overrides,
    }

    op = operation_name.strip()
    if op not in cfg_map:
        raise ValueError(
            f"Unknown operation_name for execution time: {operation_name}. "
            "请检查 DEFAULT_EXECUTION_TIME_PARAMS 或通过 params_overrides 提供该 operation 的参数。"
        )

    cfg = cfg_map[op]
    size_mb = max(float(video_size_mb), 0.0)

    # 确定性复杂度部分 f(S)
    if cfg.complexity == "linear":
        f_s = size_mb
    elif cfg.complexity == "s_log_s":
        # 用 ln(1+S) 避免 S=0 时的 log(0)
        f_s = size_mb * math.log1p(size_mb)
    else:
        raise ValueError(
            f"Unknown complexity '{cfg.complexity}' for operation '{op}'. "
            "Expected 'linear' or 's_log_s'."
        )

    # 条件均值：E[T|S] = base_time * f(S)
    mean = cfg.base_time_per_mb * f_s
    mean = max(mean, 1e-6)  # 防止 0

    k = cfg.shape
    theta = mean / k

    compute_seconds = float(rng.gamma(shape=k, scale=theta))
    io_seconds = cfg.io_time_per_mb * size_mb
    return max(compute_seconds + io_seconds, 0.0)


__all__ = [
    "RttLognormalParams",
    "BandwidthParetoParams",
    "OperationSizeLognormalParams",
    "LlmTokenLognormalParams",
    "OperationGammaTimeParams",
    "NODES",
    "RTT_MEAN_MS",
    "RTT_SIGMA",
    "BW_XM_MBPS",
    "BW_ALPHA",
    "get_rtt_params",
    "get_bw_params",
    "sample_link_between",
    "sample_rtt_ms",
    "sample_bandwidth_mbps",
    "sample_link",
    "sample_operation_size_ratio",
    "sample_llm_tokens_from_videos",
    "sample_video_size_mb",
    "sample_execution_time_seconds",
]





































