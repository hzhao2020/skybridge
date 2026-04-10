"""
与论文评估一致的 **Baselines** 与指标汇总。

Baselines
---------
- **single_cloud**：同一 ``(provider, region)`` 上 segment/split/caption/query 均为该云侧点。
- **logical_optimal**：各逻辑阶段独立选取 ``utility``（精度代理）最大的物理端点。
- **greedy**：与源数据同 ``(provider, region)`` 优先；若全局存在端点使精度相对同区最优提升超过阈值，则跨区/跨云切换。
- **deterministic**：调用 ``Workflow.calculate(..., deterministic=True)``，随机量用分布均值替代。

Metrics
-------
- **utility / cost / latency**：在随机基线上对多次 ``calculate`` 与多个校准请求取平均；确定性基线为单次均值口径。
- **cost_violation_rate**：``cost > cost_budget`` 的样本占比（所有请求 × 所有 run 合并计数）。
- **latency_violation_rate**：``latency > latency_budget`` 的样本占比。
"""

from __future__ import annotations

from dataclasses import dataclass
from config import CLOUD_PROVIDERS, CLOUD_REGIONS

from distribution import Query, config as simulation_config, sample_query_with_budget
from nodes import (
    CaptionCloudNode,
    CaptionNonCloudNode,
    QueryCloudNode,
    QueryNonCloudNode,
    SegmentNode,
    SplitNode,
)
from workflow import Workflow, WorkflowNodes


@dataclass(frozen=True, slots=True)
class BaselineMetrics:
    """单次 baseline 在若干校准请求与重复运行下的汇总。"""

    name: str
    mean_utility: float
    mean_cost: float
    mean_latency: float
    cost_violation_rate: float
    latency_violation_rate: float
    n_samples: int


def _utility_weights() -> tuple[float, float, float, float]:
    uw = simulation_config.get("utility_weight") or {}
    return (
        float(uw.get("segment", 0.0)),
        float(uw.get("split", 0.0)),
        float(uw.get("caption", 0.0)),
        float(uw.get("query", 0.0)),
    )


def _pick_single_cloud(
    nodes: WorkflowNodes,
    provider: str,
    region: str,
) -> tuple[SegmentNode, SplitNode, CaptionCloudNode | CaptionNonCloudNode, QueryCloudNode | QueryNonCloudNode]:
    pr = f"{provider}_{region}"
    return (
        nodes.segment[f"{pr}_segment"],
        nodes.split[f"{pr}_split"],
        nodes.caption[f"{pr}_caption"],
        nodes.query[f"{pr}_query"],
    )


def best_single_cloud_by_node_utility(
    nodes: WorkflowNodes,
) -> tuple[str, str, SegmentNode, SplitNode, CaptionCloudNode | CaptionNonCloudNode, QueryCloudNode | QueryNonCloudNode]:
    """在全部 ``(provider, region)`` 组合中，按加权 utility 最大选一条单云流水线。"""
    w_seg, w_spl, w_cap, w_qry = _utility_weights()
    best: tuple[float, str, str] | None = None
    for p in CLOUD_PROVIDERS:
        for r in CLOUD_REGIONS[p]:
            seg, spl, cap, qry = _pick_single_cloud(nodes, p, r)
            s = (
                w_seg * seg.utility
                + w_spl * spl.utility
                + w_cap * cap.utility
                + w_qry * qry.utility
            )
            if best is None or s > best[0]:
                best = (s, p, r)
    assert best is not None
    _, p, r = best
    seg, spl, cap, qry = _pick_single_cloud(nodes, p, r)
    return p, r, seg, spl, cap, qry


def logical_optimal(nodes: WorkflowNodes) -> tuple[SegmentNode, SplitNode, CaptionCloudNode | CaptionNonCloudNode, QueryCloudNode | QueryNonCloudNode]:
    """各阶段独立选「加权 utility」最大的端点（忽略跨阶段耦合与网络）。"""
    w_seg, w_spl, w_cap, w_qry = _utility_weights()
    seg = max(nodes.segment.values(), key=lambda n: w_seg * n.utility)
    spl = max(nodes.split.values(), key=lambda n: w_spl * n.utility)
    cap = max(nodes.caption.values(), key=lambda n: w_cap * n.utility)
    qry = max(nodes.query.values(), key=lambda n: w_qry * n.utility)
    return seg, spl, cap, qry


def greedy(
    nodes: WorkflowNodes,
    source_provider: str,
    source_region: str,
    accuracy_improvement_threshold: float = 0.05,
) -> tuple[SegmentNode, SplitNode, CaptionCloudNode | CaptionNonCloudNode, QueryCloudNode | QueryNonCloudNode]:
    """
    与源数据同区优先；若全局某端点相对「同区最优」的**加权 utility**增益超过 ``accuracy_improvement_threshold``，
    则该阶段改为全局最优端点（允许跨云 / LLM-only）。
    """
    _, _, w_cap, w_qry = _utility_weights()
    pr = f"{source_provider}_{source_region}"
    seg = nodes.segment[f"{pr}_segment"]
    spl = nodes.split[f"{pr}_split"]

    def same_region_caption() -> list:
        return [n for name, n in nodes.caption.items() if name.startswith(f"{pr}_") and name.endswith("_caption")]

    def same_region_query() -> list:
        return [n for name, n in nodes.query.items() if name.startswith(f"{pr}_") and name.endswith("_query")]

    sc = same_region_caption()
    sq = same_region_query()
    cap_same = max(sc, key=lambda n: w_cap * n.utility) if sc else max(nodes.caption.values(), key=lambda n: w_cap * n.utility)
    qry_same = max(sq, key=lambda n: w_qry * n.utility) if sq else max(nodes.query.values(), key=lambda n: w_qry * n.utility)
    cap_all = max(nodes.caption.values(), key=lambda n: w_cap * n.utility)
    qry_all = max(nodes.query.values(), key=lambda n: w_qry * n.utility)

    cap = cap_all if (w_cap * (cap_all.utility - cap_same.utility)) > accuracy_improvement_threshold else cap_same
    qry = qry_all if (w_qry * (qry_all.utility - qry_same.utility)) > accuracy_improvement_threshold else qry_same
    return seg, spl, cap, qry


def evaluate_pipeline(
    wf: Workflow,
    seg: SegmentNode,
    spl: SplitNode,
    cap: CaptionCloudNode | CaptionNonCloudNode,
    qry: QueryCloudNode | QueryNonCloudNode,
    queries: list[Query],
    n_runs: int,
    *,
    deterministic: bool = False,
) -> tuple[float, float, float, float, float, int]:
    """
    返回 (mean_utility, mean_cost, mean_latency, cost_violation_rate, latency_violation_rate, n_samples)。
    每个 ``(query, run)`` 计一次样本；违例按单次 run 是否超预算计数。
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    utilities: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    cost_viol = 0
    lat_viol = 0
    n_samples = 0

    for q in queries:
        for _ in range(n_runs):
            r = wf.calculate(
                seg,
                spl,
                cap,
                qry,
                q.data_size_MB,
                deterministic=deterministic,
            )
            utilities.append(r.utility)
            costs.append(r.cost)
            latencies.append(r.latency)
            if r.cost > q.cost_budget:
                cost_viol += 1
            if r.latency > q.latency_budget:
                lat_viol += 1
            n_samples += 1

    mean_u = sum(utilities) / len(utilities)
    mean_c = sum(costs) / len(costs)
    mean_l = sum(latencies) / len(latencies)
    cr = cost_viol / n_samples if n_samples else 0.0
    lr = lat_viol / n_samples if n_samples else 0.0
    return mean_u, mean_c, mean_l, cr, lr, n_samples


def run_single_cloud(
    wf: Workflow,
    queries: list[Query],
    n_runs: int,
) -> BaselineMetrics:
    """单云：四算子均在 ``utility`` 之和最大的 ``(provider, region)`` 上。"""
    _, _, seg, spl, cap, qry = best_single_cloud_by_node_utility(wf.nodes)
    mu, mc, ml, cr, lr, n = evaluate_pipeline(wf, seg, spl, cap, qry, queries, n_runs, deterministic=False)
    return BaselineMetrics("single_cloud", mu, mc, ml, cr, lr, n)


def run_logical_optimal(
    wf: Workflow,
    queries: list[Query],
    n_runs: int,
) -> BaselineMetrics:
    seg, spl, cap, qry = logical_optimal(wf.nodes)
    mu, mc, ml, cr, lr, n = evaluate_pipeline(wf, seg, spl, cap, qry, queries, n_runs, deterministic=False)
    return BaselineMetrics("logical_optimal", mu, mc, ml, cr, lr, n)


def run_greedy(
    wf: Workflow,
    queries: list[Query],
    n_runs: int,
    *,
    source_provider: str = "GCP",
    source_region: str = "us-east1",
    accuracy_improvement_threshold: float = 0.05,
) -> BaselineMetrics:
    seg, spl, cap, qry = greedy(
        wf.nodes,
        source_provider,
        source_region,
        accuracy_improvement_threshold,
    )
    mu, mc, ml, cr, lr, n = evaluate_pipeline(wf, seg, spl, cap, qry, queries, n_runs, deterministic=False)
    return BaselineMetrics("greedy", mu, mc, ml, cr, lr, n)


def run_deterministic(
    wf: Workflow,
    queries: list[Query],
    *,
    use_single_cloud_pipeline: bool = False,
) -> BaselineMetrics:
    """
    确定性：每个请求单次评估（无重复采样）；违例率为「请求中超预算的比例」。

    ``use_single_cloud_pipeline=True`` 时与 ``single_cloud`` 使用同一套端点，仅随机性改为均值；
    否则默认与 ``logical_optimal`` 相同端点，便于隔离「均值化」效应。
    """
    if use_single_cloud_pipeline:
        _, _, seg, spl, cap, qry = best_single_cloud_by_node_utility(wf.nodes)
    else:
        seg, spl, cap, qry = logical_optimal(wf.nodes)
    mu, mc, ml, cr, lr, n = evaluate_pipeline(wf, seg, spl, cap, qry, queries, 1, deterministic=True)
    return BaselineMetrics("deterministic", mu, mc, ml, cr, lr, n)


def compare_baselines(
    wf: Workflow | None = None,
    num_calibration_queries: int = 20,
    n_runs: int = 30,
    *,
    greedy_source_provider: str = "GCP",
    greedy_source_region: str = "us-east1",
    greedy_accuracy_delta: float = 0.05,
    deterministic_single_cloud_pipeline: bool = False,
) -> list[BaselineMetrics]:
    """
    一次跑完所有 baselines。校准请求由 ``distribution.sample_query_with_budget`` 生成（与 ``config.budget`` / ``data_size`` 一致）。
    """
    if wf is None:
        wf = Workflow()
    queries = sample_query_with_budget(num_calibration_queries)
    return [
        run_single_cloud(wf, queries, n_runs),
        run_logical_optimal(wf, queries, n_runs),
        run_greedy(
            wf,
            queries,
            n_runs,
            source_provider=greedy_source_provider,
            source_region=greedy_source_region,
            accuracy_improvement_threshold=greedy_accuracy_delta,
        ),
        run_deterministic(wf, queries, use_single_cloud_pipeline=deterministic_single_cloud_pipeline),
    ]


def format_metrics_table(rows: list[BaselineMetrics]) -> str:
    """便于打印的表格字符串。"""
    header = (
        f"{'baseline':<18} {'mean_U':>10} {'mean_cost':>12} {'mean_lat(s)':>12} "
        f"{'viol_c':>8} {'viol_t':>8} {'n':>6}"
    )
    lines = [header, "-" * len(header)]
    for m in rows:
        lines.append(
            f"{m.name:<18} {m.mean_utility:>10.4f} {m.mean_cost:>12.4f} {m.mean_latency:>12.2f} "
            f"{m.cost_violation_rate:>8.3f} {m.latency_violation_rate:>8.3f} {m.n_samples:>6d}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run baseline comparisons (utility, cost, latency, violation rates).")
    ap.add_argument("--queries", type=int, default=15, help="number of calibration queries")
    ap.add_argument("--runs", type=int, default=25, help="Monte Carlo runs per query (stochastic baselines)")
    ap.add_argument("--greedy-src", default="GCP", help="greedy: assumed upload provider")
    ap.add_argument("--greedy-region", default="us-east1", help="greedy: assumed upload region")
    ap.add_argument("--greedy-delta", type=float, default=0.05, help="greedy: cross-cloud if utility gain exceeds this")
    ap.add_argument(
        "--deterministic-single-cloud",
        action="store_true",
        help="deterministic baseline uses same pipeline as single_cloud (default: same as logical_optimal)",
    )
    args = ap.parse_args()

    rows = compare_baselines(
        num_calibration_queries=args.queries,
        n_runs=args.runs,
        greedy_source_provider=args.greedy_src,
        greedy_source_region=args.greedy_region,
        greedy_accuracy_delta=args.greedy_delta,
        deterministic_single_cloud_pipeline=args.deterministic_single_cloud,
    )
    print(format_metrics_table(rows))
