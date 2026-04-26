"""
算法侧工具：

- 通过 K 次采样/观测估计环境中的随机量，并对分布参数做拟合（矩估计等）。
- 论文 Baselines：Single-Cloud (SC)、Logical-Optimal (LO)、Greedy-Locality (GL) 等由 ``selection_*`` /
  ``selections_sc_lo_gl`` 仅输出 ``NodeSelection``；指标由 ``evaluate_static_deployment`` 单独计算。

典型用法（环境拟合）：
- 对同一组 ``DistributionParameters`` 下的随机通道重复观测 K 次（如对数正态的换算比），
  用样本反推 ``LogNormalDistribution`` 的 mean/std。
- 重复抽取 K 个「世界」``build_distribution_parameters()``，对跨世界波动的量（如 query 输出 token的 mean/std 超参）用样本矩估计其均匀分布区间。

测评公平性（共同随机场景）：
- 部署算法只应输出 ``NodeSelection``；对动态环境的蒙特卡洛测评宜使用 **共同随机数（CRN）**：
  对每条 (query, run) 在调用 ``Workflow.sample_observation`` 前按固定规则重置 ``random`` 种子，
  使所有 baseline 在同一组「环境实现」下比较；见 ``evaluate_static_deployment`` 的 ``fair_eval_seed``。
"""

from __future__ import annotations
import pulp
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, Literal, Mapping, Sequence
from nodes import caption_output_tokens_for_query

from param import (
    CLOUD_PROVIDERS,
    CLOUD_REGIONS,
    PROVIDER_LLM_MODELS,
    Operation,
    build_llm_node_name,
)
from distribution import (
    DistributionParameters,
    LogNormalDistribution,
    Query,
    QueryOutputTokenNumParam,
    UniformDistribution,
    build_distribution_parameters,
)
from tqdm import tqdm

from nodes import LocalNode, _node_to_comm_endpoint
from workflow import NodeSelection, Workflow, _utility_weighted


def _sample_mean_var(xs: Sequence[float], *, ddof: int = 1) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        raise ValueError("xs must be non-empty")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    denom = n - ddof if ddof == 1 else n
    if denom <= 0:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / denom
    return m, v


def fit_lognormal_from_samples(xs: Sequence[float]) -> LogNormalDistribution:
    """
    对样本拟合对数正态分布，参数为随机变量 X 的均值与标准差（与 ``LogNormalDistribution`` 一致）。
    """
    if any(x <= 0 for x in xs):
        raise ValueError("LogNormal samples must be > 0")
    m, s = _sample_mean_var(xs, ddof=1)
    if s < 0:
        s = 0.0
    s = max(s, 1e-12)
    if m <= 0:
        m = max(m, 1e-12)
    return LogNormalDistribution(mean=m, std=s)


def fit_uniform_moments_from_samples(
    xs: Sequence[float],
    *,
    non_negative: bool = False,
) -> UniformDistribution:
    """
    用样本矩估计 Uniform(a,b)：E[X]=(a+b)/2，Var[X]=(b-a)^2/12。
    若 ``non_negative`` 为 True（例如标准差），将 a 截断为 >= 0。
    """
    m, v = _sample_mean_var(xs, ddof=1)
    if v < 0:
        v = 0.0
    span = (12.0 * v) ** 0.5
    a = m - span / 2.0
    b = m + span / 2.0
    if non_negative:
        a = max(a, 0.0)
        if b < a:
            b = a
    if a > b:
        a, b = b, a
    if b - a < 1e-12:
        eps = 1e-6 * max(abs(m), 1.0)
        a, b = m - eps, m + eps
        if non_negative:
            a = max(a, 0.0)
            if b < a:
                b = a
    return UniformDistribution(a, b)


def fit_normal_moments_from_samples(xs: Sequence[float]) -> tuple[float, float]:
    """返回 (mean, std)，std 为样本标准差（ddof=1）。"""
    m, v = _sample_mean_var(xs, ddof=1)
    return m, v**0.5


def observe_and_fit_lognormal(dist: LogNormalDistribution, K: int) -> LogNormalDistribution:
    """从给定对数正态分布独立采样 K 次并重新拟合。"""
    if K <= 0:
        raise ValueError("K must be positive")
    xs = [dist.sample() for _ in range(K)]
    return fit_lognormal_from_samples(xs)


def observe_and_fit_data_conversion_ratios(
    params: DistributionParameters,
    K: int,
) -> Dict[Operation, LogNormalDistribution]:
    """
    对每个 operation 的 data_conversion_ratio 分布独立观测 K 次，各拟合一个 LogNormal。
    """
    if K <= 0:
        raise ValueError("K must be positive")
    out: Dict[Operation, LogNormalDistribution] = {}
    for op, dist in params.data_conversion_ratio.items():
        xs = [dist.sample() for _ in range(K)]
        out[op] = fit_lognormal_from_samples(xs)
    return out


@dataclass(frozen=True, slots=True)
class UniformHyperparamFit:
    min_val: float
    max_val: float

    def to_distribution(self) -> UniformDistribution:
        return UniformDistribution(self.min_val, self.max_val)


@dataclass(frozen=True, slots=True)
class QueryOutputHyperparameterFit:
    """对 query 输出 token 数超参（跨「世界」的 mean / std）的均匀区间估计。"""

    mean: UniformHyperparamFit
    std: UniformHyperparamFit


@dataclass(frozen=True, slots=True)
class EnvironmentFitResult:
    K: int
    query_output_hyperparams: Dict[str, QueryOutputHyperparameterFit]


def fit_query_output_hyperparams_from_worlds(
    worlds: Sequence[DistributionParameters],
) -> Dict[str, QueryOutputHyperparameterFit]:
    """给定多次 ``build_distribution_parameters()`` 的结果，按节点拟合 mean/std 的均匀超参。"""
    if not worlds:
        raise ValueError("worlds must be non-empty")
    keys = sorted(worlds[0].query_output_token_num_param.keys())
    out: Dict[str, QueryOutputHyperparameterFit] = {}
    for name in keys:
        means = [w.query_output_token_num_param[name].mean for w in worlds]
        stds = [w.query_output_token_num_param[name].std for w in worlds]
        mean_fit = fit_uniform_moments_from_samples(means, non_negative=False)
        std_fit = fit_uniform_moments_from_samples(stds, non_negative=True)
        out[name] = QueryOutputHyperparameterFit(
            mean=UniformHyperparamFit(mean_fit.min_val, mean_fit.max_val),
            std=UniformHyperparamFit(std_fit.min_val, std_fit.max_val),
        )
    return out


def point_estimate_query_output_from_worlds(
    worlds: Sequence[DistributionParameters],
) -> Dict[str, QueryOutputTokenNumParam]:
    """对 K 个世界做简单点估计：各节点 mean/std 取样本均值。"""
    if not worlds:
        raise ValueError("worlds must be non-empty")
    keys = sorted(worlds[0].query_output_token_num_param.keys())
    n = float(len(worlds))
    out: Dict[str, QueryOutputTokenNumParam] = {}
    for name in keys:
        m = sum(w.query_output_token_num_param[name].mean for w in worlds) / n
        s = sum(w.query_output_token_num_param[name].std for w in worlds) / n
        p = QueryOutputTokenNumParam()
        p.mean = m
        p.std = s
        out[name] = p
    return out


def merge_distribution_parameters_with_query_estimate(
    base: DistributionParameters,
    query_estimate: Mapping[str, QueryOutputTokenNumParam],
) -> DistributionParameters:
    """在其它字段不变的前提下，替换 query 输出 token 参数字典。"""
    return replace(base, query_output_token_num_param=dict(query_estimate))


class EnvironmentMonteCarloEstimator:
    """
    蒙特卡洛式环境估计：重复 K 次 ``build_distribution_parameters()``，
    对跨世界随机超参做均匀分布拟合，并可生成点估计版 ``DistributionParameters``。
    """

    def __init__(self, K: int) -> None:
        if K <= 0:
            raise ValueError("K must be positive")
        self.K = K

    def draw_worlds(self) -> list[DistributionParameters]:
        return [build_distribution_parameters() for _ in range(self.K)]

    def fit_from_worlds(
        self, worlds: Sequence[DistributionParameters]
    ) -> EnvironmentFitResult:
        return EnvironmentFitResult(
            K=len(worlds),
            query_output_hyperparams=fit_query_output_hyperparams_from_worlds(worlds),
        )

    def estimate(self) -> EnvironmentFitResult:
        return self.fit_from_worlds(self.draw_worlds())

    def point_estimate_parameters(self) -> DistributionParameters:
        worlds = self.draw_worlds()
        q = point_estimate_query_output_from_worlds(worlds)
        return merge_distribution_parameters_with_query_estimate(worlds[0], q)


# --- Baselines (SC / LO / GL)：静态部署 + 指标 ---


def _static_weighted_utility(workflow: Workflow, sel: NodeSelection) -> float:
    n = workflow.nodes
    return _utility_weighted(
        n.segment[sel.segment],
        n.split[sel.split],
        n.caption[sel.caption],
        n.query[sel.query],
    )


def _sample_one_transfer_latency_s(
    workflow: Workflow,
    src_ep: str,
    dst_ep: str,
    data_mb: float,
    rng: random.Random,
) -> float:
    """
    对 local→segment 类边做一次随机探测，与 :meth:`Workflow.probe_transfer_latency_s` 一致：
    启用实测时从时序有放回取 (RTT,BW)；否则从 ``params`` 的 LogNormal 各采一值。算法侧
    只应通过此类探测的样本做决策，不得把标称均值直接当作实际延迟。
    """
    return workflow.probe_transfer_latency_s(src_ep, dst_ep, data_mb, rng)


def _caption_to_query_name(caption_name: str) -> str:
    if not caption_name.endswith("_caption"):
        raise ValueError(f"expected caption node name, got {caption_name!r}")
    return caption_name[: -len("caption")] + "query"


def selection_logical_optimal(workflow: Workflow) -> NodeSelection:
    """
    Logical-Optimal (LO)：各逻辑阶段独立选取 capability（utility）最高的物理端点。
    """
    n = workflow.nodes
    seg = max(n.segment.values(), key=lambda x: (x.utility, x.name))
    spl = max(n.split.values(), key=lambda x: (x.utility, x.name))
    cap = max(n.caption.values(), key=lambda x: (x.utility, x.name))
    qry = max(n.query.values(), key=lambda x: (x.utility, x.name))
    return NodeSelection(
        segment=seg.name,
        split=spl.name,
        caption=cap.name,
        query=qry.name,
    )


def selection_single_cloud(workflow: Workflow, *, provider: str | None = None) -> NodeSelection:
    """
    Single-Cloud (SC)：同一云厂商、同一区域内完成整条流水线；caption/query 共用同一模型 key。
    在可行链上最大化静态加权效用 ``_utility_weighted``。

    ``provider``：若指定（如 ``\"GCP\"``），仅在该云厂商内枚举；若为 ``None``，在所有厂商中找最优链。
    """
    n = workflow.nodes
    best: NodeSelection | None = None
    best_u = float("-inf")
    if provider is not None:
        if provider not in CLOUD_PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}, expected one of {CLOUD_PROVIDERS}"
            )
        providers: tuple[str, ...] = (provider,)
    else:
        providers = CLOUD_PROVIDERS
    for p in providers:
        for r in CLOUD_REGIONS[p]:
            seg_name = f"{p}_{r}_segment"
            spl_name = f"{p}_{r}_split"
            if seg_name not in n.segment or spl_name not in n.split:
                continue
            for model_key in PROVIDER_LLM_MODELS[p]:
                cap_name = build_llm_node_name(p, r, model_key, "caption")
                qry_name = build_llm_node_name(p, r, model_key, "query")
                if cap_name not in n.caption or qry_name not in n.query:
                    continue
                sel = NodeSelection(
                    segment=seg_name,
                    split=spl_name,
                    caption=cap_name,
                    query=qry_name,
                )
                u = _static_weighted_utility(workflow, sel)
                if u > best_u:
                    best_u = u
                    best = sel
    if best is None:
        raise RuntimeError("no valid Single-Cloud chain in current workflow topology")
    return best


def selection_greedy_locality(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    ref_size: Literal["mean", "max"] = "mean",
    network_probe_samples: int = 32,
    locality_rng_seed: int = 123456791,
) -> NodeSelection:
    """
    Greedy-Locality (GL)：数据就近；segment 在每条候选边上对 local→segment 做
    ``network_probe_samples`` 次与 ``Edge`` 一致的随机探测，用 **样本平均时延** 比较，
    取估计最小者（同估计时延再按 segment utility 决胜）。后续阶段与 segment 同云同区；
    在同区 caption 候选中选 utility 最高的模型，query 为配对节点名。

    ``ref_size``：用 query 列表上 ``data_size_MB`` 的均值或最大值作为 segment 侧比较的参考体积。
    ``locality_rng_seed``：仅用于上述网络探测的独立 RNG，与测评 CRN 分离。
    """
    if not queries:
        raise ValueError("queries must be non-empty for Greedy-Locality (need reference data size)")
    if network_probe_samples <= 0:
        raise ValueError("network_probe_samples must be positive")
    sizes = [float(q.data_size_MB) for q in queries]
    mb0 = max(sizes) if ref_size == "max" else sum(sizes) / len(sizes)

    local = LocalNode()
    local_ep = _node_to_comm_endpoint(local)
    rng = random.Random(locality_rng_seed)
    n = workflow.nodes
    best_seg_name: str | None = None
    best_lat = float("inf")
    best_tie_u = float("-inf")

    for seg in n.segment.values():
        lat = 0.0
        for _ in range(network_probe_samples):
            lat += _sample_one_transfer_latency_s(
                workflow,
                local_ep,
                _node_to_comm_endpoint(seg),
                mb0,
                rng,
            )
        lat /= float(network_probe_samples)
        tie_u = seg.utility
        if lat < best_lat or (math.isclose(lat, best_lat) and tie_u > best_tie_u):
            best_lat = lat
            best_tie_u = tie_u
            best_seg_name = seg.name

    if best_seg_name is None:
        raise RuntimeError("no segment nodes available")

    seg_node = n.segment[best_seg_name]
    p, r = seg_node.provider, seg_node.region
    if r is None:
        raise RuntimeError(f"segment node {best_seg_name!r} has no region")
    spl_name = f"{p}_{r}_split"
    if spl_name not in n.split:
        raise RuntimeError(f"missing split node {spl_name!r} for Greedy-Locality")

    prefix = f"{p}_{r}__"
    cap_candidates = [c for name, c in n.caption.items() if name.startswith(prefix)]
    if not cap_candidates:
        raise RuntimeError(f"no caption nodes under {prefix!r}")
    cap_node = max(cap_candidates, key=lambda x: (x.utility, x.name))
    qry_name = _caption_to_query_name(cap_node.name)
    if qry_name not in n.query:
        raise RuntimeError(f"missing query node {qry_name!r}")

    return NodeSelection(
        segment=best_seg_name,
        split=spl_name,
        caption=cap_node.name,
        query=qry_name,
    )


@dataclass(frozen=True, slots=True)
class SloViolationRates:
    """SLO Violation Rate：分别统计成本与延迟超预算的观测比例。"""

    cost_violation_ratio: float
    latency_violation_ratio: float


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """
    四类评估指标（与论文 Evaluation 一致）：
    (1) 聚合效用 U：对所选 physical 节点按 ``utility_weight`` 加权求和，与 query 及随机观测无关；
    (2) 平均成本 C；(3) 平均端到端延迟 T；(4) SVR（两项比率）。
    字段名 ``mean_aggregate_utility`` 保留以兼容旧表头，语义上即静态 U。
    """

    mean_aggregate_utility: float
    mean_cost_usd: float
    mean_latency_s: float
    slo_violation: SloViolationRates
    num_observations: int


def evaluate_static_deployment(
    workflow: Workflow,
    queries: Sequence[Query],
    selection: NodeSelection,
    *,
    runs_per_query: int = 1,
    fair_eval_seed: int | None = None,
    rng_seeds: Sequence[int] | None = None,
) -> BaselineMetrics:
    """
    对固定 ``NodeSelection``，在每条 query 上重复 ``runs_per_query`` 次 ``sample_observation``，
    汇总指标：成本与延迟对观测取平均；效用 U 仅为 ``_static_weighted_utility``（与 query/随机性无关）。

    ``fair_eval_seed`` 若给定，则在每次观测前按 ``(query 下标, run 下标)`` 重置全局 ``random`` 种子，
    使不同 ``NodeSelection`` 在 **同一组** 动态环境实现下测评（共同随机数，减小比较方差）。

    ``rng_seeds`` 若给定，须长度等于 ``len(queries) * runs_per_query``，按展平顺序
    （先 query 下标、再 run 下标）在每次观测前 ``random.seed(rng_seeds[k])``；与 ``fair_eval_seed``
    互斥。二者均为 ``None`` 时不重播种，与旧版「顺序消耗全局 RNG」行为一致。
    """
    if runs_per_query <= 0:
        raise ValueError("runs_per_query must be positive")
    if not queries:
        raise ValueError("queries must be non-empty")
    if rng_seeds is not None and fair_eval_seed is not None:
        raise ValueError("pass only one of rng_seeds and fair_eval_seed")
    n_expected = len(queries) * runs_per_query
    if rng_seeds is not None and len(rng_seeds) != n_expected:
        raise ValueError(
            f"rng_seeds length {len(rng_seeds)} != {n_expected} (queries × runs_per_query)"
        )

    u_static = _static_weighted_utility(workflow, selection)
    tot_c = tot_l = 0.0
    viol_c = viol_l = 0
    n_obs = 0
    k_seed = 0

    for qi, q in enumerate(queries):
        for ri in range(runs_per_query):
            if rng_seeds is not None:
                random.seed(rng_seeds[k_seed])
                k_seed += 1
            elif fair_eval_seed is not None:
                random.seed(fair_eval_seed + 1_000_003 * qi + 1_048_573 * ri)
            obs = workflow.sample_observation(selection, float(q.data_size_MB))
            tot_c += obs.cost
            tot_l += obs.latency
            n_obs += 1
            if obs.cost > float(q.cost_budget):
                viol_c += 1
            if obs.latency > float(q.latency_budget):
                viol_l += 1

    inv = 1.0 / float(n_obs)
    return BaselineMetrics(
        mean_aggregate_utility=u_static,
        mean_cost_usd=tot_c * inv,
        mean_latency_s=tot_l * inv,
        slo_violation=SloViolationRates(
            cost_violation_ratio=viol_c / float(n_obs),
            latency_violation_ratio=viol_l / float(n_obs),
        ),
        num_observations=n_obs,
    )


def selections_sc_lo_gl(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    gl_ref_size: Literal["mean", "max"] = "mean",
    sc_provider: str | None = None,
    gl_network_probe_samples: int = 32,
    gl_locality_rng_seed: int = 123456791,
) -> dict[str, NodeSelection]:
    """SC / LO / GL 的节点选择（不含测评）。"""
    return {
        "SC": selection_single_cloud(workflow, provider=sc_provider),
        "LO": selection_logical_optimal(workflow),
        "GL": selection_greedy_locality(
            workflow,
            queries,
            ref_size=gl_ref_size,
            network_probe_samples=gl_network_probe_samples,
            locality_rng_seed=gl_locality_rng_seed,
        ),
    }


def iter_paired_selections(workflow: Workflow):
    """
    枚举 (segment, split, caption, query) 组合，且 query 名与 caption 成对（``..._caption`` → ``..._query``）。
    规模约为 |seg|×|spl|×|cap|，用于 DO / Proposed 的穷举搜索。
    """
    n = workflow.nodes
    for seg_name in sorted(n.segment.keys()):
        for spl_name in sorted(n.split.keys()):
            for cap_name in sorted(n.caption.keys()):
                qn = _caption_to_query_name(cap_name)
                if qn not in n.query:
                    continue
                yield NodeSelection(seg_name, spl_name, cap_name, qn)


_DO_SEARCH_CRN_OFFSET = 5_003_009


def _selection_feasible_empirical_chance_per_query(
    workflow: Workflow,
    selection: NodeSelection,
    queries: Sequence[Query],
    *,
    samples_per_query: int,
    eta_c: float,
    eta_t: float,
    fair_eval_seed: int | None,
) -> bool:
    """
    对每个 query q 独立用 ``samples_per_query`` 次 ``sample_observation`` 估计违反率；
    当且仅当每条 query 上成本违反率 ≤ η_C、延迟违反率 ≤ η_T 时可行（经验近似于
    ``P(C ≤ Theta_C^q) ≥ 1-η_C``、``P(T ≤ Theta_T^q) ≥ 1-η_T``）。

    ``fair_eval_seed`` 非空时，对 (query 下标, 采样下标) 使用与测评相同的线性同余种子，
    并加 ``_DO_SEARCH_CRN_OFFSET``，使搜索阶段与最终测评场景分离；候选 ``NodeSelection``
    之间对同一 (q,k) 复现同一环境实现（CRN）。
    """
    if samples_per_query <= 0:
        raise ValueError("samples_per_query must be positive")
    base = (
        (fair_eval_seed + _DO_SEARCH_CRN_OFFSET)
        if fair_eval_seed is not None
        else None
    )
    for qi, q in enumerate(queries):
        viol_c = viol_t = 0
        for k in range(samples_per_query):
            if base is not None:
                random.seed(base + 1_000_003 * qi + 1_048_573 * k)
            obs = workflow.sample_observation(selection, float(q.data_size_MB))
            if obs.cost > float(q.cost_budget):
                viol_c += 1
            if obs.latency > float(q.latency_budget):
                viol_t += 1
        rate_c = viol_c / float(samples_per_query)
        rate_t = viol_t / float(samples_per_query)
        if rate_c > eta_c or rate_t > eta_t:
            return False
    return True


def selection_chance_constrained_optimal(
    workflow: Workflow,
    queries: Sequence[Query],
    eta_c: float,
    eta_t: float,
    *,
    samples_per_query: int,
    fair_eval_seed: int | None = None,
    show_progress: bool = True,
) -> NodeSelection:
    """
    DO（机会约束、采样版）：在拓扑配对约束下枚举 ``NodeSelection``，仅用随机
    ``Workflow.sample_observation`` 得到样本，在经验约束（逐 query 违反率 ≤ η_C / η_T）
    下最大化静态效用 ``U``；无可行解时退化为全体配对中效用最高者。对应论文中
    ``P(C ≤ Theta_C^q) ≥ 1-η_C``、``P(T ≤ Theta_T^q) ≥ 1-η_T`` 的蒙特卡洛可行判别
    （此处为穷举 + 采样，非 MILP 求解器）。

    ``show_progress``：是否在枚举候选部署时显示 tqdm 进度条。
    """
    if not queries:
        raise ValueError("queries must be non-empty")
    if samples_per_query <= 0:
        raise ValueError("samples_per_query must be positive")
    paired = list(iter_paired_selections(workflow))
    best_any: NodeSelection | None = None
    best_u_any = float("-inf")
    feasible_best: NodeSelection | None = None
    feasible_u = float("-inf")
    bar = tqdm(
        paired,
        desc="DO 机会约束",
        unit="候选",
        disable=not show_progress,
    )
    for sel in bar:
        u = _static_weighted_utility(workflow, sel)
        if u > best_u_any:
            best_u_any = u
            best_any = sel
        if _selection_feasible_empirical_chance_per_query(
            workflow,
            sel,
            queries,
            samples_per_query=samples_per_query,
            eta_c=eta_c,
            eta_t=eta_t,
            fair_eval_seed=fair_eval_seed,
        ):
            if u > feasible_u:
                feasible_u = u
                feasible_best = sel
    if best_any is None:
        raise RuntimeError("no paired selections in workflow")
    if feasible_best is not None:
        return feasible_best
    return best_any


def selection_proposed_empirical_slo(
    workflow: Workflow,
    queries: Sequence[Query],
    eta_c: float,
    eta_t: float,
    *,
    search_runs_per_query: int,
    fair_eval_seed: int | None = None,
) -> NodeSelection:
    """
    所提方法（理论实现版）：基于 CVaR 的 MILP 重构 (论文 Section IV.A)
    在求解过程中对每个 query 实时进行 K 次环境采样 (SAA)，提取当前场景的具体常数，
    并通过 McCormick Envelopes 线性化端到端物理链路的组合代价。
    """
    if not queries:
        raise ValueError("queries must be non-empty")
    if search_runs_per_query <= 0:
        raise ValueError("search_runs_per_query must be positive")

    # 1. 初始化 MILP 问题 (最大化加权效用)
    prob = pulp.LpProblem("Deployment_Optimization", pulp.LpMaximize)

    # 2. 定义决策变量 (x: 节点选择, y: 边选择)
    x_seg = {u: pulp.LpVariable(f"x_seg_{u.replace('-', '_')}", cat="Binary") for u in workflow.nodes.segment}
    x_spl = {v: pulp.LpVariable(f"x_spl_{v.replace('-', '_')}", cat="Binary") for v in workflow.nodes.split}

    # 逻辑拓扑要求 Caption 和 Query 必须成对出现，将其合并为联合决策空间
    cap_qry_pairs = []
    for c in workflow.nodes.caption:
        q = c.removesuffix("_caption") + "_query"
        if q in workflow.nodes.query:
            cap_qry_pairs.append((c, q))

    x_cq = {c: pulp.LpVariable(f"x_cq_{c.replace('-', '_')}", cat="Binary") for c, q in cap_qry_pairs}

    # McCormick 包络辅助变量 (y_ij = x_i * x_j)
    y_seg_spl = {(u, v): pulp.LpVariable(f"y_seg_spl_{u.replace('-', '_')}_{v.replace('-', '_')}", 0, 1, cat="Continuous")
                 for u in x_seg for v in x_spl}
    y_spl_cq = {(v, c): pulp.LpVariable(f"y_spl_cq_{v.replace('-', '_')}_{c.replace('-', '_')}", 0, 1, cat="Continuous")
                for v in x_spl for c in x_cq}

    # 3. 拓扑约束：每个逻辑节点仅能实例化在一个物理节点上 (Eq. 16)
    prob += pulp.lpSum(x_seg.values()) == 1
    prob += pulp.lpSum(x_spl.values()) == 1
    prob += pulp.lpSum(x_cq.values()) == 1

    # 4. McCormick 线性化包络约束 (Eq. 17)
    for u in x_seg:
        for v in x_spl:
            prob += y_seg_spl[u, v] >= x_seg[u] + x_spl[v] - 1
            prob += y_seg_spl[u, v] <= x_seg[u]
            prob += y_seg_spl[u, v] <= x_spl[v]

    for v in x_spl:
        for c, _ in cap_qry_pairs:
            prob += y_spl_cq[v, c] >= x_spl[v] + x_cq[c] - 1
            prob += y_spl_cq[v, c] <= x_spl[v]
            prob += y_spl_cq[v, c] <= x_cq[c]

    # 5. 目标函数：最大化聚合效用 (Eq. 15 / Eq. 26)
    from distribution import config as simulation_config
    uw = simulation_config.get("utility_weight") or {}
    w_seg, w_spl = float(uw.get("segment", 0.0)), float(uw.get("split", 0.0))
    w_cap, w_qry = float(uw.get("caption", 0.0)), float(uw.get("query", 0.0))

    prob += pulp.lpSum([
        x_seg[u] * (w_seg * workflow.nodes.segment[u].utility) for u in x_seg
    ]) + pulp.lpSum([
        x_spl[v] * (w_spl * workflow.nodes.split[v].utility) for v in x_spl
    ]) + pulp.lpSum([
        x_cq[c] * (w_cap * workflow.nodes.caption[c].utility + w_qry * workflow.nodes.query[q].utility)
        for c, q in cap_qry_pairs
    ])

    # 6. CVaR SAA 辅助变量 (Eq. 23 - 25)
    K = search_runs_per_query
    num_queries = len(queries)
    alpha_c = pulp.LpVariable("alpha_c")
    alpha_t = pulp.LpVariable("alpha_t")

    z_c = pulp.LpVariable.dicts("z_c", ((qi, k) for qi in range(num_queries) for k in range(K)), lowBound=0)
    z_t = pulp.LpVariable.dicts("z_t", ((qi, k) for qi in range(num_queries) for k in range(K)), lowBound=0)

    # 7. SAA 场景展开: 在算法内部直接采样 K 次（不看底层分布参数，仅获取采样观测值）
    for qi, query_obj in enumerate(queries):
        for k in range(K):
            if fair_eval_seed is not None:
                # 偏移基准种子，防止搜索空间的随机环境与最终 Eval 测试环境发生种子碰撞（避免“数据穿越”）
                base = fair_eval_seed + 5_003_009
                random.seed(base + 1_000_003 * qi + 1_048_573 * k)

            # 触发底层网络和数据压缩率的黑盒采样
            workflow._realize_network()
            r_seg, r_spl, r_cap, r_qry = workflow._sample_data_conversion_ratios()
            
            mb0 = float(query_obj.data_size_MB)
            mb1 = mb0 * r_seg
            mb2 = mb1 * r_spl
            mb3 = mb2 * r_cap
            mb4 = mb3 * r_qry

            # --- A. 节点代数展开 (记录 Cost / Latency 的常数采样子项) ---
            c_seg_vals, t_seg_vals = {}, {}
            for u, seg_node in workflow.nodes.segment.items():
                c_seg_vals[u] = (
                    seg_node.calculate_execution_cost(mb0) +
                    seg_node.calculate_storage_cost(mb0) + seg_node.calculate_storage_cost(mb1) +
                    workflow._edge_egress_usd(workflow._local, seg_node, mb0)
                )
                t_seg_vals[u] = (
                    seg_node.calculate_latency(mb0) + 
                    workflow._edge_latency_s(workflow._local, seg_node, mb0)
                )

            c_spl_vals, t_spl_vals = {}, {}
            for v, spl_node in workflow.nodes.split.items():
                c_spl_vals[v] = (
                    spl_node.calculate_execution_cost(mb1) +
                    spl_node.calculate_storage_cost(mb1) + spl_node.calculate_storage_cost(mb2)
                )
                t_spl_vals[v] = spl_node.calculate_latency(mb1)

            c_cq_vals, t_cq_vals = {}, {}
            for c, q in cap_qry_pairs:
                cap_node, qry_node = workflow.nodes.caption[c], workflow.nodes.query[q]
                n_cap = caption_output_tokens_for_query(cap_node, mb2)
                
                c_cq_vals[c] = (
                    cap_node.calculate_token_cost(mb2) +
                    qry_node.calculate_token_cost(mb2, input_tokens=n_cap) +
                    cap_node.calculate_storage_cost(mb2) + cap_node.calculate_storage_cost(mb3) +
                    qry_node.calculate_storage_cost(mb3) + qry_node.calculate_storage_cost(mb4) +
                    workflow._edge_egress_usd(cap_node, qry_node, mb3) +
                    workflow._edge_egress_usd(qry_node, workflow._local, mb4)
                )
                t_cq_vals[c] = (
                    cap_node.calculate_latency(mb2) +
                    qry_node.calculate_latency(mb2) +
                    workflow._edge_latency_s(cap_node, qry_node, mb3) +
                    workflow._edge_latency_s(qry_node, workflow._local, mb4)
                )

            # --- B. 物理边跨域代价展开 (Edge Egress / Latency) ---
            c_edge1_vals, t_edge1_vals = {}, {}
            for u, seg_node in workflow.nodes.segment.items():
                for v, spl_node in workflow.nodes.split.items():
                    c_edge1_vals[u, v] = workflow._edge_egress_usd(seg_node, spl_node, mb1)
                    t_edge1_vals[u, v] = workflow._edge_latency_s(seg_node, spl_node, mb1)

            c_edge2_vals, t_edge2_vals = {}, {}
            for v, spl_node in workflow.nodes.split.items():
                for c, _ in cap_qry_pairs:
                    cap_node = workflow.nodes.caption[c]
                    c_edge2_vals[v, c] = workflow._edge_egress_usd(spl_node, cap_node, mb2)
                    t_edge2_vals[v, c] = workflow._edge_latency_s(spl_node, cap_node, mb2)

            # --- C. 构建当前场景 k 下的系统总代价表达式 ---
            total_cost_expr = (
                pulp.lpSum([x_seg[u] * c_seg_vals[u] for u in x_seg]) +
                pulp.lpSum([x_spl[v] * c_spl_vals[v] for v in x_spl]) +
                pulp.lpSum([x_cq[c] * c_cq_vals[c] for c in x_cq]) +
                pulp.lpSum([y_seg_spl[u, v] * c_edge1_vals[u, v] for u, v in y_seg_spl]) +
                pulp.lpSum([y_spl_cq[v, c] * c_edge2_vals[v, c] for v, c in y_spl_cq])
            )

            total_lat_expr = (
                pulp.lpSum([x_seg[u] * t_seg_vals[u] for u in x_seg]) +
                pulp.lpSum([x_spl[v] * t_spl_vals[v] for v in x_spl]) +
                pulp.lpSum([x_cq[c] * t_cq_vals[c] for c in x_cq]) +
                pulp.lpSum([y_seg_spl[u, v] * t_edge1_vals[u, v] for u, v in y_seg_spl]) +
                pulp.lpSum([y_spl_cq[v, c] * t_edge2_vals[v, c] for v, c in y_spl_cq])
            )

            # 单次超出预算的 slack 变量约束 (Eq. 23 & 24)
            prob += z_c[qi, k] >= total_cost_expr - float(query_obj.cost_budget) - alpha_c
            prob += z_t[qi, k] >= total_lat_expr - float(query_obj.latency_budget) - alpha_t

    # 8. 全局 CVaR 机会约束边界 (Eq. 21 & 22)
    # 将 slack 变量的均值与 alpha 的组合强行约束在 0 以下，实现 1-eta 概率的安全界限
    eta_c_denom = eta_c * num_queries * K
    eta_t_denom = eta_t * num_queries * K

    prob += alpha_c + (1.0 / eta_c_denom) * pulp.lpSum(z_c.values()) <= 0
    prob += alpha_t + (1.0 / eta_t_denom) * pulp.lpSum(z_t.values()) <= 0

    # 9. 求解 MILP
    # msg=False 屏蔽求解器的冗长日志，timeLimit 防止极端情况卡死
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    prob.solve(solver)

    # 10. 结果解析与降级后备
    if prob.status == pulp.LpStatusOptimal:
        best_seg = [u for u in x_seg if pulp.value(x_seg[u]) and pulp.value(x_seg[u]) > 0.5][0]
        best_spl = [v for v in x_spl if pulp.value(x_spl[v]) and pulp.value(x_spl[v]) > 0.5][0]
        best_cap = [c for c in x_cq if pulp.value(x_cq[c]) and pulp.value(x_cq[c]) > 0.5][0]
        best_qry = best_cap.removesuffix("_caption") + "_query"
        return NodeSelection(segment=best_seg, split=best_spl, caption=best_cap, query=best_qry)

    # 如果无法找到在给定 eta_c/eta_t 下满足的解，退化为逻辑最优 (LO) Baseline
    from algos import selection_logical_optimal
    return selection_logical_optimal(workflow)