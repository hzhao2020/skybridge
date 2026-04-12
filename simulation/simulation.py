"""
试运行：统一测评管线如下：

1. 从 ``distribution.sample_query_with_budget`` 采样 queries 与预算；
2. 构建 ``Workflow``，依次调用各算法得到 ``NodeSelection``（与测评解耦）；
3. 在动态环境下 **一次性** 定义测评场景（共同随机数 CRN：每条 (query, run) 对应固定 ``rng_seed``）；
4. 对所有算法的节点选择，在 **同一组** 场景上调用 ``evaluate_static_deployment`` 汇总指标。

与论文 chance constraint 对应：给定 η_C、η_T，若经验违反率 viol_c ≤ η_C 且 viol_t ≤ η_T，
则认为该 baseline 在该次仿真下满足 SLO 口径（经验版）。

用法示例::

    python simulation.py --queries 20 --runs-per-query 2
    python simulation.py --eta-c 0.1 --eta-t 0.05

默认测评使用 ``--fair-eval-seed``（默认 42）生成场景种子，使各 baseline 共享同一批动态环境实现。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from param import CLOUD_PROVIDERS
from distribution import Query, build_distribution_parameters, sample_query_with_budget
from workflow import Workflow
from algos import (
    BaselineMetrics,
    NodeSelection,
    evaluate_static_deployment,
    selection_chance_constrained_optimal,
    selection_greedy_locality,
    selection_logical_optimal,
    selection_proposed_empirical_slo,
    selection_single_cloud,
)


@dataclass(frozen=True, slots=True)
class DynamicEvalScenario:
    """单条测评场景：对应 ``queries[query_index]`` 的第 ``run_index`` 次动态环境重放。"""

    query_index: int
    run_index: int
    rng_seed: int


def sample_dynamic_eval_scenarios(
    queries: Sequence[Query],
    runs_per_query: int,
    fair_eval_seed: int,
) -> tuple[DynamicEvalScenario, ...]:
    """
    为 CRN 测评生成场景列表（与 ``algos.evaluate_static_deployment`` 的展平顺序一致：
    先 ``query_index`` 升序，再 ``run_index`` 升序）。
    """
    if runs_per_query <= 0:
        raise ValueError("runs_per_query must be positive")
    out: list[DynamicEvalScenario] = []
    for qi in range(len(queries)):
        for ri in range(runs_per_query):
            seed = fair_eval_seed + 1_000_003 * qi + 1_048_573 * ri
            out.append(DynamicEvalScenario(query_index=qi, run_index=ri, rng_seed=seed))
    return tuple(out)


def collect_baseline_selections(
    workflow: Workflow,
    queries: Sequence[Query],
    *,
    eta_c: float,
    eta_t: float,
    search_runs: int,
    do_samples_per_query: int,
    fair_eval_seed_for_search: int,
) -> list[tuple[str, NodeSelection]]:
    """仅收集各 baseline 的节点选择（不含蒙特卡洛测评）。"""
    labeled: list[tuple[str, NodeSelection]] = []

    labeled.append(("LO", selection_logical_optimal(workflow)))

    labeled.append(
        (
            "GL",
            selection_greedy_locality(
                workflow,
                queries,
                ref_size="mean",
                locality_rng_seed=fair_eval_seed_for_search + 500_001,
            ),
        )
    )

    labeled.append(("SC(global)", selection_single_cloud(workflow, provider=None)))
    for p in CLOUD_PROVIDERS:
        labeled.append((f"SC({p})", selection_single_cloud(workflow, provider=p)))

    # labeled.append(
    #     (
    #         "DO",
    #         selection_chance_constrained_optimal(
    #             workflow,
    #             queries,
    #             eta_c,
    #             eta_t,
    #             samples_per_query=do_samples_per_query,
    #             fair_eval_seed=fair_eval_seed_for_search,
    #         ),
    #     )
    # )

    labeled.append(
        (
            "Proposed",
            selection_proposed_empirical_slo(
                workflow,
                queries,
                eta_c,
                eta_t,
                search_runs_per_query=search_runs,
                fair_eval_seed=fair_eval_seed_for_search,
            ),
        )
    )
    return labeled


def evaluate_all_selections_on_scenarios(
    workflow: Workflow,
    queries: Sequence[Query],
    labeled_selections: Sequence[tuple[str, NodeSelection]],
    *,
    runs_per_query: int,
    eval_scenarios: tuple[DynamicEvalScenario, ...],
) -> list[tuple[str, NodeSelection, BaselineMetrics]]:
    """对同一 ``queries`` 与同一批 CRN 场景（导出 ``rng_seeds``）评估多个 ``NodeSelection``。"""
    if len(eval_scenarios) != len(queries) * runs_per_query:
        raise ValueError("eval_scenarios length does not match queries × runs_per_query")
    rng_seeds = tuple(s.rng_seed for s in eval_scenarios)

    rows: list[tuple[str, NodeSelection, BaselineMetrics]] = []
    for label, sel in labeled_selections:
        m = evaluate_static_deployment(
            workflow,
            queries,
            sel,
            runs_per_query=runs_per_query,
            fair_eval_seed=None,
            rng_seeds=rng_seeds,
        )
        rows.append((label, sel, m))
    return rows


def _slo_ok(m: BaselineMetrics, eta_c: float, eta_t: float) -> bool:
    return (
        m.slo_violation.cost_violation_ratio <= eta_c
        and m.slo_violation.latency_violation_ratio <= eta_t
    )


def _fmt_row(
    label: str,
    m: BaselineMetrics,
    *,
    eta_c: float,
    eta_t: float,
    show_selection: bool,
    sel: NodeSelection | None,
) -> str:
    ok = _slo_ok(m, eta_c, eta_t)
    line = (
        f"{label:<22}"
        f"  {m.mean_aggregate_utility:>10.4f}"
        f"  {m.mean_cost_usd:>12.4f}"
        f"  {m.mean_latency_s:>12.2f}"
        f"  {m.slo_violation.cost_violation_ratio:>8.3f}"
        f"  {m.slo_violation.latency_violation_ratio:>8.3f}"
        f"  {m.num_observations:>6d}"
        f"  {'Y' if ok else 'N':>5}"
    )
    if show_selection and sel is not None:
        line += (
            f"\n{'':22}  seg={sel.segment}\n"
            f"{'':22}  spl={sel.split}\n"
            f"{'':22}  cap={sel.caption}\n"
            f"{'':22}  qry={sel.query}"
        )
    return line


def _print_comparison_summary(
    rows: list[tuple[str, NodeSelection, BaselineMetrics]],
    *,
    eta_c: float,
    eta_t: float,
) -> None:
    """在满足 SLO 经验阈值的方法子集内对比效用/成本/延迟，并给出全体极值参考。"""
    slo_rows = [(lbl, m) for lbl, _, m in rows if _slo_ok(m, eta_c, eta_t)]
    if not slo_rows:
        print("\n== 对比小结 ==")
        print(" 无满足 SLO_ok 的方法（请放宽 η_C/η_T 或增加 --runs-per-query）。")
        return

    def min_by_metric(
        pairs: list[tuple[str, BaselineMetrics]], key
    ) -> tuple[str, BaselineMetrics]:
        return min(pairs, key=lambda x: key(x[1]))

    def max_by_metric(
        pairs: list[tuple[str, BaselineMetrics]], key
    ) -> tuple[str, BaselineMetrics]:
        return max(pairs, key=lambda x: key(x[1]))

    best_u = max_by_metric(slo_rows, lambda m: m.mean_aggregate_utility)
    best_cost = min_by_metric(slo_rows, lambda m: m.mean_cost_usd)
    best_lat = min_by_metric(slo_rows, lambda m: m.mean_latency_s)

    print("\n== 对比小结（仅含 SLO_ok=Y 的方法） ==")
    print(
        f"  最高 mean_U:   {best_u[0]!r}  (U={best_u[1].mean_aggregate_utility:.4f})"
    )
    print(
        f"  最低 mean_cost: {best_cost[0]!r}  (cost={best_cost[1].mean_cost_usd:.4f} USD)"
    )
    print(
        f"  最低 mean_lat:  {best_lat[0]!r}  (T={best_lat[1].mean_latency_s:.2f} s)"
    )

    # 按效用排序（SLO_ok 子集）
    ranked = sorted(slo_rows, key=lambda x: (-x[1].mean_aggregate_utility, x[0]))
    print("  按 mean_U 降序: " + " > ".join(lbl for lbl, _ in ranked))

    # 全体极值（含可能违反 SLO）
    all_u = max(rows, key=lambda x: x[2].mean_aggregate_utility)
    all_cost = min(rows, key=lambda x: x[2].mean_cost_usd)
    all_lat = min(rows, key=lambda x: x[2].mean_latency_s)
    print("\n== 全体极值参考（含 SLO_ok=N） ==")
    print(
        f"  mean_U 最大: {all_u[0]!r}  U={all_u[2].mean_aggregate_utility:.4f}  "
        f"SLO_ok={'Y' if _slo_ok(all_u[2], eta_c, eta_t) else 'N'}"
    )
    print(
        f"  mean_cost 最小: {all_cost[0]!r}  cost={all_cost[2].mean_cost_usd:.4f}  "
        f"SLO_ok={'Y' if _slo_ok(all_cost[2], eta_c, eta_t) else 'N'}"
    )
    print(
        f"  mean_lat 最小: {all_lat[0]!r}  T={all_lat[2].mean_latency_s:.2f} s  "
        f"SLO_ok={'Y' if _slo_ok(all_lat[2], eta_c, eta_t) else 'N'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 LO / GL / SC / DO / Proposed 全方法并输出指标表与对比小结"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=16,
        help="生成的 query 条数（传给 sample_query_with_budget）",
    )
    parser.add_argument(
        "--runs-per-query",
        type=int,
        default=1,
        help="每条 query 重复观测次数（用于估计均值与经验违反率）",
    )
    parser.add_argument(
        "--eta-c",
        type=float,
        default=0.15,
        dest="eta_c",
        help="η_C：成本侧允许的经验违反率上界，判定 viol_c ≤ η_C",
    )
    parser.add_argument(
        "--eta-t",
        type=float,
        default=0.15,
        dest="eta_t",
        help="η_T：延迟侧允许的经验违反率上界，判定 viol_t ≤ η_T",
    )
    parser.add_argument(
        "--search-runs",
        type=int,
        default=8,
        help="所提方法在搜索最优部署时，每条 query 的蒙特卡洛次数（越大越准、越慢）",
    )
    parser.add_argument(
        "--do-samples-per-query",
        type=int,
        default=8,
        dest="do_samples_per_query",
        help="DO 在逐 query 机会约束判别时，每条 query 的采样次数",
    )
    parser.add_argument(
        "--fair-eval-seed",
        type=int,
        default=42,
        help="测评阶段共同随机场景基种子：每条 (query, run) 在 sample_observation 前重置 RNG，"
        "使各 baseline 共享同一组动态环境实现",
    )
    parser.add_argument(
        "--show-selection",
        action="store_true",
        help="在表格下方打印每条结果对应的 NodeSelection",
    )
    args = parser.parse_args()

    if not (0.0 <= args.eta_c <= 1.0 and 0.0 <= args.eta_t <= 1.0):
        parser.error("η_C 与 η_T 须在 [0, 1] 内")
    if args.search_runs <= 0:
        parser.error("--search-runs 须为正整数")
    if args.do_samples_per_query <= 0:
        parser.error("--do-samples-per-query 须为正整数")

    fair_eval_seed = int(args.fair_eval_seed)

    queries = sample_query_with_budget(float(args.queries))
    params = build_distribution_parameters()
    workflow = Workflow(params=params)

    print(
        f"SLO 阈值（经验判定）: η_C={args.eta_c:g}, η_T={args.eta_t:g} "
        f"(viol_c ≤ η_C 且 viol_t ≤ η_T 则 SLO_ok=Y)"
    )
    print(
        "运行方法: LO, GL, SC(global), "
        + ", ".join(f"SC({p})" for p in CLOUD_PROVIDERS)
        + ", DO, Proposed"
    )
    eval_scenarios = sample_dynamic_eval_scenarios(
        queries, args.runs_per_query, fair_eval_seed
    )
    print(
        f"测评: CRN 场景数={len(eval_scenarios)}，基种子 fair_eval_seed={fair_eval_seed} "
        "（所有 baseline 在同一批场景上评估）"
    )

    labeled_selections = collect_baseline_selections(
        workflow,
        queries,
        eta_c=args.eta_c,
        eta_t=args.eta_t,
        search_runs=args.search_runs,
        do_samples_per_query=args.do_samples_per_query,
        fair_eval_seed_for_search=fair_eval_seed,
    )

    header = (
        f"{'baseline':<22}"
        f"  {'mean_U':>10}"
        f"  {'mean_cost':>12}"
        f"  {'mean_lat(s)':>12}"
        f"  {'viol_c':>8}"
        f"  {'viol_t':>8}"
        f"  {'n_obs':>6}"
        f"  {'SLO_ok':>5}"
    )
    print(header)
    print("-" * len(header))

    rows = evaluate_all_selections_on_scenarios(
        workflow,
        queries,
        labeled_selections,
        runs_per_query=args.runs_per_query,
        eval_scenarios=eval_scenarios,
    )

    for label, sel, m in rows:
        print(
            _fmt_row(
                label,
                m,
                eta_c=args.eta_c,
                eta_t=args.eta_t,
                show_selection=args.show_selection,
                sel=sel,
            )
        )

    _print_comparison_summary(rows, eta_c=args.eta_c, eta_t=args.eta_t)


if __name__ == "__main__":
    main()
