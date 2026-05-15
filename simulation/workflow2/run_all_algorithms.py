"""
一键运行 Workflow 2：**LO / SC / DO / Sky**（互斥路径 DAG 上的 CVaR–SAA MILP），并打印蒙特卡洛 KPI。

默认对 **四种 budget–α** 各跑一次（与 workflow1 一致，``workflow1.utils.BUDGET_ALPHA_SUITE_DEFAULT_WF1``）；
每条 query 的 ``Θ_C,Θ_T`` 由 **GCP/AWS/Aliyun 各一条「按云 SC」锚 + LO** 在 plug-in mean 下 mean-field 端到端
标量的 min–max 包络上 ``Θ=min+α(max−min)`` 插值得出（见 ``generate_realistic_queries_wf2``）。

在 ``simulation/`` 下（``--path`` 取值须为下列之一：
``video_caption`` \| ``ocr`` \| ``label_detection`` \| ``speech_transcription``）::

    python -m workflow2.run_all_algorithms --path video_caption --num-queries 20
    python -m workflow2.run_all_algorithms --path speech_transcription --skip-sky
    python -m workflow2.run_all_algorithms --budget-alpha 0.25 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from workflow1.utils import BUDGET_ALPHA_SUITE_DEFAULT_WF1

from . import baseline as baseline_runner
from . import sky as sky_runner
from . import utils as wf2_utils
from .evaluation import (
    EmpiricalDeploymentMetricsWf2,
    evaluate_deployment_empirical_wf2,
    print_metrics_report_wf2,
)


def _evaluate_and_print(
    path_id: sky_runner.WF2PathId,
    label: str,
    nodes: tuple[Any, ...],
    queries: list,
    *,
    weights: tuple[float, ...],
    samples_per_query: int,
    eval_seed: int,
    extra: str = "",
) -> EmpiricalDeploymentMetricsWf2:
    m = evaluate_deployment_empirical_wf2(
        path_id,
        nodes,
        queries,
        weights=weights,
        samples_per_query=samples_per_query,
        eval_seed=eval_seed,
    )
    tag = f"{label}{' | ' + extra if extra else ''}"
    print_metrics_report_wf2(algorithm_label=tag, metrics=m)
    print()
    return m


def _print_summary_table_wf2(results: list[tuple[str, EmpiricalDeploymentMetricsWf2 | None]]) -> None:
    print()
    print("=" * 70)
    print(
        "SUMMARY (empirical U, mean cost USD, mean latency path-sum s, "
        "mean display-max latency s, VR_cost, VR_lat on path-sum)"
    )
    print("=" * 70)
    hdr = (
        f"{'algo':<6} {'U':>10} {'mean_C':>11} {'mean_TΣ':>11} {'mean_Tmax':>11} "
        f"{'VR_C':>9} {'VR_TΣ':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, m in results:
        if m is None:
            print(
                f"{name:<6} {'(skipped)':>10} {'-':>11} {'-':>11} {'-':>11} {'-':>9} {'-':>9}"
            )
            continue
        print(
            f"{name:<6} {m.aggregate_utility_u:>10.6g} "
            f"{m.mean_cost_usd:>11.6g} {m.mean_latency_sec:>11.6g} "
            f"{m.mean_workflow_display_latency_sec:>11.6g} "
            f"{m.slo_cost_violation_rate:>9.6g} {m.slo_latency_violation_rate:>9.6g}"
        )
    print("=" * 70)


def _run_algorithms_for_queries_wf2(
    path_id: sky_runner.WF2PathId,
    queries: list,
    args: argparse.Namespace,
    *,
    weights: tuple[float, ...],
    eval_seed: int,
    cands: tuple[Any, ...],
) -> list[tuple[str, EmpiricalDeploymentMetricsWf2 | None]]:
    results: list[tuple[str, EmpiricalDeploymentMetricsWf2 | None]] = []

    t0 = time.perf_counter()
    lo = baseline_runner.logical_optimal_baseline_wf2(path_id, cands, weights=weights)
    print(f"[LO] closed-form U={lo.total_utility:.6g}  time={time.perf_counter() - t0:.2f}s")
    m_lo = _evaluate_and_print(
        path_id,
        "Logical-Optimal (LO)",
        lo.nodes,
        queries,
        weights=weights,
        samples_per_query=args.eval_samples_per_query,
        eval_seed=eval_seed,
        extra=f"closed_form_U={lo.total_utility:.6g}",
    )
    results.append(("LO", m_lo))

    t0 = time.perf_counter()
    sc = baseline_runner.single_cloud_baseline_wf2(
        path_id,
        cands,
        weights=weights,
        queries=queries,
        samples_per_query=args.eval_samples_per_query,
        violation_eval_seed=eval_seed,
    )
    provs = sorted({n.provider for n in sc.nodes})
    regs = sorted({n.region for n in sc.nodes})
    vc, vl = baseline_runner.mc_violation_counts_wf2(
        path_id,
        sc.nodes,
        queries,
        samples_per_query=args.eval_samples_per_query,
        violation_eval_seed=eval_seed,
    )
    print(
        f"[SC] selection MC violations cost={vc} latency={vl} total={vc + vl} | "
        f"closed-form U={sc.total_utility:.6g} providers={provs} regions={regs} "
        f"time={time.perf_counter() - t0:.2f}s"
    )
    m_sc = _evaluate_and_print(
        path_id,
        "Single-Cloud (SC)",
        sc.nodes,
        queries,
        weights=weights,
        samples_per_query=args.eval_samples_per_query,
        eval_seed=eval_seed,
        extra=f"closed_form_U={sc.total_utility:.6g}",
    )
    results.append(("SC", m_sc))

    t0 = time.perf_counter()
    do = baseline_runner.deterministic_optimal_baseline_wf2(
        path_id,
        queries,
        cands,
        weights=weights,
        token_seed=eval_seed,
    )
    print(
        f"[DO] Gurobi status={do.gurobi_status} MILP_obj_U={do.total_utility} "
        f"time={time.perf_counter() - t0:.2f}s"
    )
    print(f"[DO] deployment tuple: {do.nodes}")
    extra_do = f"gurobi_status={do.gurobi_status}"
    if do.gurobi_status == "Optimal":
        extra_do += f" MILP_U={do.total_utility:.6g}"
    m_do = _evaluate_and_print(
        path_id,
        "Deterministic-Optimal (DO)",
        do.nodes,
        queries,
        weights=weights,
        samples_per_query=args.eval_samples_per_query,
        eval_seed=eval_seed,
        extra=extra_do,
    )
    results.append(("DO", m_do))

    if args.skip_sky:
        print("[Sky] skipped (--skip-sky)")
        results.append(("Sky", None))
    else:
        sky_dec, sky_warm = sky_runner.sky_ablation_settings_wf2(args.sky_variant)
        t0 = time.perf_counter()
        rep = sky_runner.run_sky_deployment_wf2(
            path_id,
            queries=queries,
            s_per_query=args.sky_s_per_query,
            eta_c=args.eta_c,
            eta_t=args.eta_t,
            lamb_c=0.55,
            lamb_t=0.00118,
            weights=weights,
            batch_add_ratio=args.sky_batch_ratio,
            decomposition=sky_dec,
            use_warm_start=sky_warm,
            rng_seed=args.sky_rng,
        )
        elapsed_sky = time.perf_counter() - t0

        if isinstance(rep, sky_runner.DecompositionResultWf2):
            sol = rep.solution
            extra_sky = (
                f"decomposition iters={rep.iterations} "
                f"active_scenarios={len(rep.active_indices)} "
                f"elapsed_solver={elapsed_sky:.2f}s "
                f"gurobi_obj={sol.objective_value}"
            )
        else:
            sol = rep
            extra_sky = f"full_MILP elapsed={elapsed_sky:.2f}s gurobi_obj={sol.objective_value}"

        print(f"[Sky] {extra_sky}")
        print(f"[Sky] gurobi_status={sol.gurobi_status}")
        print(f"[Sky] deployment: {sol.nodes}")
        print()

        sky_labels = {
            "full": "Sky (CVaR-SAA MILP full: decomposition+warm-start)",
            "no_warm_start": "Sky (CVaR-SAA MILP ablation: decomposition, no warm-start)",
            "direct_milp": "Sky (CVaR-SAA MILP ablation: direct full MILP)",
        }
        m_sky = _evaluate_and_print(
            path_id,
            sky_labels[args.sky_variant],
            sol.nodes,
            queries,
            weights=weights,
            samples_per_query=args.eval_samples_per_query,
            eval_seed=eval_seed,
            extra=extra_sky,
        )
        results.append(("Sky", m_sky))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LO, SC, DO, Sky for workflow2 + KPIs.")
    parser.add_argument(
        "--path",
        choices=["video_caption", "ocr", "label_detection", "speech_transcription"],
        default="video_caption",
        help="exclusive DAG path id",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="per-layer weights (length = path depth); default uniform",
    )
    parser.add_argument("--eval-samples-per-query", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=9000)
    parser.add_argument("--sky-s-per-query", type=int, default=50)
    parser.add_argument(
        "--sky-batch-ratio",
        type=float,
        default=0.05,
        help="decomposition: each iteration adds max(1, ceil(ratio * total scenarios)) violators",
    )
    parser.add_argument("--sky-rng", type=int, default=0)
    parser.add_argument("--eta-c", type=float, default=0.1)
    parser.add_argument("--eta-t", type=float, default=0.1)
    parser.add_argument("--skip-sky", action="store_true")
    parser.add_argument(
        "--sky-variant",
        choices=["full", "no_warm_start", "direct_milp"],
        default="full",
    )
    parser.add_argument(
        "--budget-alpha",
        type=float,
        nargs="*",
        default=None,
        metavar="A",
        help="预算插值 α（可多选）；未指定值 → 0.25 0.5 0.75 1.0。",
    )
    args = parser.parse_args()

    path_id = args.path
    default_w = wf2_utils.default_weights_for_path(path_id)
    if args.weights is not None:
        if len(args.weights) != len(default_w):
            parser.error(
                f"--weights expects {len(default_w)} values for path {path_id!r}, "
                f"got {len(args.weights)}"
            )
        weights = tuple(args.weights)
    else:
        weights = default_w

    eval_seed = int(args.eval_seed)

    balpha = args.budget_alpha
    if balpha is not None and len(balpha) > 0:
        alphas_run = tuple(balpha)
    else:
        alphas_run = BUDGET_ALPHA_SUITE_DEFAULT_WF1
    run_plan = [(f"α={a:g}", a) for a in alphas_run]

    print("=" * 70)
    print(f"workflow2.run_all_algorithms | path={path_id}")
    print("=" * 70)

    print(
        f"[setup] budget_mode=alpha runs={len(run_plan)} query_seed={args.query_seed} "
        f"num_queries_each={args.num_queries} weights={weights} eval_seed={eval_seed}"
    )
    print()

    cands = sky_runner.enumerate_candidates_wf2(path_id)
    print(f"[setup] candidate layer sizes: {[len(c) for c in cands]}")
    lo_ch = wf2_utils.wf2_logical_optimal_chain(path_id, cands, weights)
    print("[setup] budget anchors: GCP/AWS/Aliyun 按云 SC + LO")

    mega: list[tuple[str, list[tuple[str, EmpiricalDeploymentMetricsWf2 | None]]]] = []

    for label, alpha in run_plan:
        assert alpha is not None
        queries = wf2_utils.generate_realistic_queries_wf2(
            args.num_queries,
            path_id,
            seed=args.query_seed,
            budget_alpha=float(alpha),
            lo_chain=lo_ch,
            weights=weights,
            cands=cands,
        )
        print("*" * 70)
        print(
            f"[budget α] {label}  "
            f"Θ=min+α(max−min)（GCP/AWS/Aliyun 按云 SC 锚 + LO；并行 DAG / speech 串行）"
        )
        print("*" * 70)

        results = _run_algorithms_for_queries_wf2(
            path_id,
            queries,
            args,
            weights=weights,
            eval_seed=eval_seed,
            cands=cands,
        )
        mega.append((label, results))
        _print_summary_table_wf2(results)
        print()

    if len(mega) > 1:
        print("=" * 70)
        print("ALL PRESETS SUMMARY")
        print("=" * 70)
        bh = (
            f"{'budget':<26} {'algo':<6} "
            f"{'U':>10} {'mean_C':>11} {'mean_TΣ':>11} {'mean_Tmax':>11} "
            f"{'VR_C':>9} {'VR_TΣ':>9}"
        )
        print(bh)
        print("-" * len(bh))
        for budget_label, rs in mega:
            for name, m in rs:
                if m is None:
                    print(
                        f"{budget_label:<26} {name:<6} "
                        f"{'(skip)':>10} {'-':>11} {'-':>11} {'-':>11} {'-':>9} {'-':>9}"
                    )
                    continue
                print(
                    f"{budget_label:<26} {name:<6} "
                    f"{m.aggregate_utility_u:>10.6g} {m.mean_cost_usd:>11.6g} "
                    f"{m.mean_latency_sec:>11.6g} {m.mean_workflow_display_latency_sec:>11.6g} "
                    f"{m.slo_cost_violation_rate:>9.6g} {m.slo_latency_violation_rate:>9.6g}"
                )
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
