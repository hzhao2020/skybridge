"""
试运行：用 ``sample_query_with_budget`` 生成带预算的 queries，对 LO / GL / SC /
DO（Deterministic-Optimal）/ 所提方法（经验 chance 约束）跑实验并打印指标表。

与论文 chance constraint 对应：给定 η_C、η_T，若经验违反率 viol_c ≤ η_C 且 viol_t ≤ η_T，
则认为该 baseline 在该次仿真下满足 SLO 口径（经验版）。

用法示例::

    python simulation.py --queries 20 --runs-per-query 2
    python simulation.py --eta-c 0.1 --eta-t 0.05
"""

from __future__ import annotations

import argparse
from typing import Literal, cast

from param import CLOUD_PROVIDERS
from distribution import build_distribution_parameters, sample_query_with_budget
from workflow import Workflow
from algos import (
    BaselineMetrics,
    NodeSelection,
    run_baseline_do,
    run_baseline_gl,
    run_baseline_lo,
    run_baseline_sc,
    run_proposed,
)


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
        default=0.05,
        dest="eta_c",
        help="η_C：成本侧允许的经验违反率上界，判定 viol_c ≤ η_C",
    )
    parser.add_argument(
        "--eta-t",
        type=float,
        default=0.05,
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
        "--show-selection",
        action="store_true",
        help="在表格下方打印每条结果对应的 NodeSelection",
    )
    args = parser.parse_args()

    if not (0.0 <= args.eta_c <= 1.0 and 0.0 <= args.eta_t <= 1.0):
        parser.error("η_C 与 η_T 须在 [0, 1] 内")
    if args.search_runs <= 0:
        parser.error("--search-runs 须为正整数")

    queries = sample_query_with_budget(float(args.queries))
    params = build_distribution_parameters()
    workflow = Workflow(params=params)

    print(
        f"SLO 阈值（经验判定）: η_C={args.eta_c:g}, η_T={args.eta_t:g} "
        f"(viol_c ≤ η_C 且 viol_t ≤ η_T 则 SLO_ok=Y)"
    )
    print(
        "运行方法: LO, GL(mean), GL(max), SC(global), "
        + ", ".join(f"SC({p})" for p in CLOUD_PROVIDERS)
        + ", DO, Proposed"
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

    rows: list[tuple[str, NodeSelection, BaselineMetrics]] = []

    sel_lo, m_lo = run_baseline_lo(
        workflow, queries, runs_per_query=args.runs_per_query
    )
    rows.append(("LO", sel_lo, m_lo))

    for ref in ("mean", "max"):
        sel_gl, m_gl = run_baseline_gl(
            workflow,
            queries,
            runs_per_query=args.runs_per_query,
            ref_size=cast(Literal["mean", "max"], ref),
        )
        rows.append((f"GL(ref={ref})", sel_gl, m_gl))

    sel_g, m_g = run_baseline_sc(
        workflow, queries, runs_per_query=args.runs_per_query, provider=None
    )
    rows.append(("SC(global)", sel_g, m_g))
    for p in CLOUD_PROVIDERS:
        sel_p, m_p = run_baseline_sc(
            workflow,
            queries,
            runs_per_query=args.runs_per_query,
            provider=p,
        )
        rows.append((f"SC({p})", sel_p, m_p))

    sel_do, m_do = run_baseline_do(
        workflow, queries, runs_per_query=args.runs_per_query
    )
    rows.append(("DO", sel_do, m_do))

    sel_pr, m_pr = run_proposed(
        workflow,
        queries,
        args.eta_c,
        args.eta_t,
        runs_per_query=args.runs_per_query,
        search_runs_per_query=args.search_runs,
    )
    rows.append(("Proposed", sel_pr, m_pr))

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
