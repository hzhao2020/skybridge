"""
算法侧工具：

- 通过 K 次采样/观测估计环境中的随机量，并对分布参数做拟合（矩估计等）。
- 论文 Baselines：Single-Cloud (SC)、Logical-Optimal (LO)、Greedy-Locality (GL)，
  输入 ``queries``（``data_size_MB``、``cost_budget``、``latency_budget``）与 ``Workflow``，
  输出聚合效用、平均成本、平均延迟及 SLO 违反率（成本/延迟两项比率）。

典型用法（环境拟合）：
- 对同一组 ``DistributionParameters`` 下的随机通道重复观测 K 次（如对数正态的换算比），
  用样本反推 ``LogNormalDistribution`` 的 mean/std。
- 重复抽取 K 个「世界」``build_distribution_parameters()``，对跨世界波动的量（如 query 输出 token的 mean/std 超参）用样本矩估计其均匀分布区间。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Literal, Mapping, Sequence

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


def _mean_transfer_latency_s(
    workflow: Workflow,
    src_ep: str,
    dst_ep: str,
    data_mb: float,
) -> float:
    """用边参数矩阵中的 RTT/BW **均值** 估计传输时延（秒），与 ``Edge`` 公式一致但不采样。"""
    key = (src_ep, dst_ep)
    r = workflow.params.edge_rtt[key]
    w = workflow.params.edge_bw[key]
    rtt_ms = float(r.mean)
    bw = float(w.mean)
    if bw <= 0:
        return float("inf")
    return rtt_ms / 2000.0 + data_mb * 8.0 / bw


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
) -> NodeSelection:
    """
    Greedy-Locality (GL)：数据就近；segment 选使 local→segment **均值网络**传输延迟最小的端点，
    后续阶段与 segment 同云同区；在同区 caption 候选中选 utility 最高的模型，query 为配对节点名。

    ``ref_size``：用 query 列表上 ``data_size_MB`` 的均值或最大值作为 segment 侧比较的参考体积。
    """
    if not queries:
        raise ValueError("queries must be non-empty for Greedy-Locality (need reference data size)")
    sizes = [float(q.data_size_MB) for q in queries]
    mb0 = max(sizes) if ref_size == "max" else sum(sizes) / len(sizes)

    local = LocalNode()
    n = workflow.nodes
    best_seg_name: str | None = None
    best_lat = float("inf")
    best_tie_u = float("-inf")

    for seg in n.segment.values():
        lat = _mean_transfer_latency_s(
            workflow,
            _node_to_comm_endpoint(local),
            _node_to_comm_endpoint(seg),
            mb0,
        )
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
) -> BaselineMetrics:
    """
    对固定 ``NodeSelection``，在每条 query 上重复 ``runs_per_query`` 次 ``sample_observation``，
    汇总指标：成本与延迟对观测取平均；效用 U 仅为 ``_static_weighted_utility``（与 query/随机性无关）。
    """
    if runs_per_query <= 0:
        raise ValueError("runs_per_query must be positive")
    if not queries:
        raise ValueError("queries must be non-empty")

    u_static = _static_weighted_utility(workflow, selection)
    tot_c = tot_l = 0.0
    viol_c = viol_l = 0
    n_obs = 0

    for q in queries:
        for _ in range(runs_per_query):
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


def run_baseline_sc(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    runs_per_query: int = 1,
    provider: str | None = None,
) -> tuple[NodeSelection, BaselineMetrics]:
    sel = selection_single_cloud(workflow, provider=provider)
    return sel, evaluate_static_deployment(workflow, queries, sel, runs_per_query=runs_per_query)


def run_baseline_lo(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    runs_per_query: int = 1,
) -> tuple[NodeSelection, BaselineMetrics]:
    sel = selection_logical_optimal(workflow)
    return sel, evaluate_static_deployment(workflow, queries, sel, runs_per_query=runs_per_query)


def run_baseline_gl(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    runs_per_query: int = 1,
    ref_size: Literal["mean", "max"] = "mean",
) -> tuple[NodeSelection, BaselineMetrics]:
    sel = selection_greedy_locality(workflow, queries, ref_size=ref_size)
    return sel, evaluate_static_deployment(workflow, queries, sel, runs_per_query=runs_per_query)


def evaluate_sc_lo_gl(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    runs_per_query: int = 1,
    gl_ref_size: Literal["mean", "max"] = "mean",
    sc_provider: str | None = None,
) -> dict[str, tuple[NodeSelection, BaselineMetrics]]:
    """依次运行 SC / LO / GL，返回 ``{"SC": (sel, m), "LO": ..., "GL": ...}``。"""
    return {
        "SC": run_baseline_sc(
            workflow, queries, runs_per_query=runs_per_query, provider=sc_provider
        ),
        "LO": run_baseline_lo(workflow, queries, runs_per_query=runs_per_query),
        "GL": run_baseline_gl(
            workflow, queries, runs_per_query=runs_per_query, ref_size=gl_ref_size
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


def selection_deterministic_optimal(
    workflow: Workflow, queries: Sequence[Query]
) -> NodeSelection:
    """
    Baseline DO（Deterministic-Optimal）：在 **均值确定性环境**（``Workflow.calculate(..., deterministic=True)``）
    下，对每条 query 检查 cost/latency 是否不超过预算；在所有满足约束的部署中取静态加权效用最高者；
    若无可行解，退化为全体组合中效用最高者（仍用确定性环境判定约束，仅放宽「必须全部满足」）。
    """
    if not queries:
        raise ValueError("queries must be non-empty")
    nn = workflow.nodes
    best_any: NodeSelection | None = None
    best_u_any = float("-inf")
    feasible_best: NodeSelection | None = None
    feasible_u = float("-inf")
    for sel in iter_paired_selections(workflow):
        u = _static_weighted_utility(workflow, sel)
        if u > best_u_any:
            best_u_any = u
            best_any = sel
        ok_all = True
        for q in queries:
            obs = workflow.calculate(
                nn.segment[sel.segment],
                nn.split[sel.split],
                nn.caption[sel.caption],
                nn.query[sel.query],
                float(q.data_size_MB),
                deterministic=True,
            )
            if obs.cost > float(q.cost_budget) or obs.latency > float(q.latency_budget):
                ok_all = False
                break
        if ok_all and u > feasible_u:
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
) -> NodeSelection:
    """
    所提方法（经验版）：在候选部署上作蒙特卡洛评估，选取满足    ``viol_c ≤ η_C`` 且 ``viol_t ≤ η_T`` 且静态加权效用最大的 ``NodeSelection``；
    若无满足者，退化为 LO。
    """
    if not queries:
        raise ValueError("queries must be non-empty")
    if search_runs_per_query <= 0:
        raise ValueError("search_runs_per_query must be positive")
    best_sel: NodeSelection | None = None
    best_u = float("-inf")
    for sel in iter_paired_selections(workflow):
        m = evaluate_static_deployment(
            workflow,
            queries,
            sel,
            runs_per_query=search_runs_per_query,
        )
        if m.slo_violation.cost_violation_ratio > eta_c:
            continue
        if m.slo_violation.latency_violation_ratio > eta_t:
            continue
        u = _static_weighted_utility(workflow, sel)
        if u > best_u:
            best_u = u
            best_sel = sel
    if best_sel is not None:
        return best_sel
    return selection_logical_optimal(workflow)


def run_baseline_do(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    runs_per_query: int = 1,
) -> tuple[NodeSelection, BaselineMetrics]:
    sel = selection_deterministic_optimal(workflow, queries)
    return sel, evaluate_static_deployment(workflow, queries, sel, runs_per_query=runs_per_query)


def run_proposed(
    workflow: Workflow,
    queries: Sequence[Query],
    eta_c: float,
    eta_t: float,
    *,
    runs_per_query: int = 1,
    search_runs_per_query: int = 12,
) -> tuple[NodeSelection, BaselineMetrics]:
    sel = selection_proposed_empirical_slo(
        workflow,
        queries,
        eta_c,
        eta_t,
        search_runs_per_query=search_runs_per_query,
    )
    return sel, evaluate_static_deployment(workflow, queries, sel, runs_per_query=runs_per_query)
