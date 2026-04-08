"""
实验脚本：生成 queries，对比 **你的方法** 与 baselines。

你的方法（proposed）
-------------------
由于当前仓库的工作流固定为链式 `local→segment→split→caption→query→local`，本脚本将你的方法实现为：

- 在训练集（train queries）上，对所有候选流水线组合做搜索；
- 用 Monte Carlo（多次 `Workflow.calculate`）估计每条流水线的
  `mean_utility / mean_cost / mean_latency / violation_rate`；
- 在满足 `cost_violation_rate <= eta_cost` 且 `latency_violation_rate <= eta_latency` 的可行解中，
  选取 `mean_utility` 最大的流水线；若无可行解，则用惩罚项选择：
  `score = mean_utility - penalty*(viol_cost + viol_lat)` 最大者。

这样能直接展示“部署耦合 + 约束下”相对 baselines 的优势（尤其相对 logical_optimal）。

输出指标
--------
- utility / cost / latency：测试集上的均值
- cost_violation_rate / latency_violation_rate：测试集上超预算比例
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import time

from distribution import Query, sample_query_with_budget
from workflow import Workflow
from nodes import (
    CaptionCloudNode,
    CaptionNonCloudNode,
    QueryCloudNode,
    QueryNonCloudNode,
    SegmentNode,
    SplitNode,
)

from baselines import (
    BaselineMetrics,
    evaluate_pipeline,
    format_metrics_table,
    run_deterministic,
    run_greedy,
    run_logical_optimal,
    run_single_cloud,
)


Pipeline = tuple[
    SegmentNode,
    SplitNode,
    CaptionCloudNode | CaptionNonCloudNode,
    QueryCloudNode | QueryNonCloudNode,
]


@dataclass(frozen=True, slots=True)
class ProposedSearchConfig:
    train_runs: int = 20
    test_runs: int = 40
    eta_cost: float = 0.10
    eta_latency: float = 0.10
    penalty: float = 1.0
    max_candidates_per_stage: int | None = 30
    """为避免组合爆炸：每个 stage 只保留 utility 最高的前 N 个（None 表示全保留）。"""
    show_progress: bool = True


def _topk_by_utility(values: list, k: int | None) -> list:
    if k is None:
        return values
    return sorted(values, key=lambda n: float(getattr(n, "utility", 0.0)), reverse=True)[:k]


def proposed_select_pipeline(
    wf: Workflow,
    train_queries: list[Query],
    cfg: ProposedSearchConfig,
) -> tuple[Pipeline, BaselineMetrics]:
    """
    在训练集上选择 pipeline，返回 (pipeline, 训练集表现)。
    """
    segs = list(wf.nodes.segment.values())
    spls = list(wf.nodes.split.values())
    caps = list(wf.nodes.caption.values())
    qrys = list(wf.nodes.query.values())

    best_pipeline: Pipeline | None = None
    best_metrics: BaselineMetrics | None = None
    best_score: float | None = None
    total = len(segs) * len(spls) * len(caps) * len(qrys)
    done = 0
    start_ts = time.time()

    def _render_progress(done_cnt: int) -> None:
        if not cfg.show_progress:
            return
        elapsed = time.time() - start_ts
        ratio = (done_cnt / total) if total > 0 else 1.0
        eta = (elapsed / done_cnt) * (total - done_cnt) if done_cnt > 0 else 0.0
        width = 28
        filled = int(round(width * max(0.0, min(1.0, ratio))))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[simulation] proposed搜索进度 [{bar}] {done_cnt}/{total} "
            f"({ratio*100:5.1f}%) | elapsed {elapsed:7.1f}s | eta {eta:7.1f}s",
            end="",
            flush=True,
        )

    for seg, spl, cap, qry in itertools.product(segs, spls, caps, qrys):
        mu, mc, ml, cr, lr, n = evaluate_pipeline(
            wf, seg, spl, cap, qry, train_queries, cfg.train_runs, deterministic=False
        )
        feasible = (cr <= cfg.eta_cost) and (lr <= cfg.eta_latency)
        score = mu if feasible else (mu - cfg.penalty * (cr + lr))

        if best_score is None or score > best_score:
            best_score = score
            best_pipeline = (seg, spl, cap, qry)
            best_metrics = BaselineMetrics(
                name="proposed(train)",
                mean_utility=mu,
                mean_cost=mc,
                mean_latency=ml,
                cost_violation_rate=cr,
                latency_violation_rate=lr,
                n_samples=n,
            )
        done += 1
        _render_progress(done)

    assert best_pipeline is not None and best_metrics is not None
    if cfg.show_progress:
        _render_progress(total)
        print()
    return best_pipeline, best_metrics


def proposed_evaluate(
    wf: Workflow,
    train_queries: list[Query],
    test_queries: list[Query],
    cfg: ProposedSearchConfig,
) -> tuple[BaselineMetrics, BaselineMetrics]:
    """
    返回 (train_metrics, test_metrics)。
    """
    pipeline, train_m = proposed_select_pipeline(wf, train_queries, cfg)
    seg, spl, cap, qry = pipeline
    mu, mc, ml, cr, lr, n = evaluate_pipeline(
        wf, seg, spl, cap, qry, test_queries, cfg.test_runs, deterministic=False
    )
    test_m = BaselineMetrics(
        name="proposed",
        mean_utility=mu,
        mean_cost=mc,
        mean_latency=ml,
        cost_violation_rate=cr,
        latency_violation_rate=lr,
        n_samples=n,
    )
    return train_m, test_m


def split_train_test(queries: list[Query], train_ratio: float = 0.5) -> tuple[list[Query], list[Query]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    n_train = max(1, int(round(len(queries) * train_ratio)))
    return queries[:n_train], queries[n_train:]


def run_experiment(
    num_queries: int = 30,
    train_ratio: float = 0.5,
    baseline_runs: int = 40,
    *,
    greedy_source_provider: str = "p1",
    greedy_source_region: str = "r1",
    greedy_delta: float = 0.05,
    proposed_cfg: ProposedSearchConfig | None = None,
) -> None:
    if proposed_cfg is None:
        proposed_cfg = ProposedSearchConfig(train_runs=20, test_runs=baseline_runs)

    wf = Workflow()
    queries = sample_query_with_budget(num_queries)
    train_q, test_q = split_train_test(queries, train_ratio=train_ratio)
    if not test_q:
        raise ValueError("test set is empty; increase num_queries or lower train_ratio")

    # proposed
    proposed_train, proposed_test = proposed_evaluate(wf, train_q, test_q, proposed_cfg)

    # baselines（在 test 上评估）
    rows = [
        run_single_cloud(wf, test_q, baseline_runs),
        run_logical_optimal(wf, test_q, baseline_runs),
        run_greedy(
            wf,
            test_q,
            baseline_runs,
            source_provider=greedy_source_provider,
            source_region=greedy_source_region,
            accuracy_improvement_threshold=greedy_delta,
        ),
        run_deterministic(wf, test_q),
        proposed_test,
    ]

    print("== Proposed selection on train ==")
    print(
        format_metrics_table(
            [
                BaselineMetrics(
                    name="proposed(train)",
                    mean_utility=proposed_train.mean_utility,
                    mean_cost=proposed_train.mean_cost,
                    mean_latency=proposed_train.mean_latency,
                    cost_violation_rate=proposed_train.cost_violation_rate,
                    latency_violation_rate=proposed_train.latency_violation_rate,
                    n_samples=proposed_train.n_samples,
                )
            ]
        )
    )
    print()

    print("== Test comparison ==")
    print(format_metrics_table(rows))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run simulation experiment comparing proposed method vs baselines.")
    ap.add_argument("--queries", type=int, default=100, help="number of generated queries")
    ap.add_argument("--train-ratio", type=float, default=0.5, help="fraction of queries used for training (0-1)")
    ap.add_argument("--runs", type=int, default=40, help="Monte Carlo runs per test query")
    ap.add_argument("--train-runs", type=int, default=20, help="proposed: Monte Carlo runs per train query")
    ap.add_argument("--greedy-src", default="p1", help="greedy baseline: assumed upload provider")
    ap.add_argument("--greedy-region", default="r1", help="greedy baseline: assumed upload region")
    ap.add_argument("--greedy-delta", type=float, default=0.05, help="greedy baseline: cross-cloud if utility gain exceeds this")
    ap.add_argument("--eta-cost", type=float, default=0.10, help="proposed: max allowed cost violation rate on train")
    ap.add_argument("--eta-lat", type=float, default=0.10, help="proposed: max allowed latency violation rate on train")
    ap.add_argument("--penalty", type=float, default=1.0, help="proposed: penalty weight when no feasible pipeline")
    ap.add_argument("--topk", type=int, default=30, help="proposed: keep top-k by utility per stage to limit search")
    ap.add_argument("--no-progress", action="store_true", help="disable proposed search progress bar")
    args = ap.parse_args()

    run_experiment(
        num_queries=args.queries,
        train_ratio=args.train_ratio,
        baseline_runs=args.runs,
        greedy_source_provider=args.greedy_src,
        greedy_source_region=args.greedy_region,
        greedy_delta=args.greedy_delta,
        proposed_cfg=ProposedSearchConfig(
            train_runs=args.train_runs,
            test_runs=args.runs,
            eta_cost=args.eta_cost,
            eta_latency=args.eta_lat,
            penalty=args.penalty,
            max_candidates_per_stage=args.topk,
            show_progress=not args.no_progress,
        ),
    )

