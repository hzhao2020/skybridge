"""
一键顺序运行：**Logical-Optimal (LO)**、**Single-Cloud (SC)**、**Deterministic-Optimal (DO)**、**Sky (CVaR–SAA MILP)**，
并对同一批 ``queries`` 调用 ``evaluation.evaluate_deployment_empirical`` 打印 KPI：
Aggregate Utility / mean Cost / mean Latency / SVR。

默认对 **四种 budget–α 插值** 各跑一次（``workflow1.utils.BUDGET_ALPHA_SUITE_DEFAULT_WF1``，即 α∈{0.25,0.5,0.75,1}）：
每条 query 的 ``Θ_C, Θ_T`` 由三条锚链（LO + ``workflow1.budget`` 上 mean 最小的 cost / latency 链）
按 ``Θ = T_{\\min} + α (T_{LO} - T_{\\min})`` 形式在费用维与时延维分别插值（见 ``generate_realistic_queries``）。

在包含 ``workflow1`` 与 ``sim_env`` 的 ``simulation/`` 目录下执行::

    python -m workflow1.run_all_algorithms
    python -m workflow1.run_all_algorithms --num-queries 20
    python -m workflow1.run_all_algorithms --budget-alpha 0.25 0.5

大规模实验请调大 ``--num-queries`` / ``--sky-s-per-query``（Sky 会非常慢）。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from sim_env.utility import QueryProfile

from . import baseline as baseline_runner
from . import sky as sky_runner
from .budget import wf1_mean_min_anchor_chains
from .evaluation import EmpiricalDeploymentMetrics, evaluate_deployment_empirical, print_metrics_report
from .utils import (
    BUDGET_ALPHA_SUITE_DEFAULT_WF1,
    generate_realistic_queries,
    wf1_logical_optimal_chain,
)


def _evaluate_and_print(
    label: str,
    nodes: tuple[Any, Any, Any, Any],
    queries: list[QueryProfile],
    *,
    weights: tuple[float, float, float, float],
    samples_per_query: int,
    eval_seed: int,
    extra: str = "",
) -> EmpiricalDeploymentMetrics:
    m = evaluate_deployment_empirical(
        nodes,
        queries,
        weights=weights,
        samples_per_query=samples_per_query,
        eval_seed=eval_seed,
    )
    tag = f"{label}{' | ' + extra if extra else ''}"
    print_metrics_report(algorithm_label=tag, metrics=m)
    print()
    return m


def _run_algorithms_for_queries(
    queries: list[QueryProfile],
    args: argparse.Namespace,
    *,
    weights: tuple[float, float, float, float],
    eval_seed: int,
    cands: tuple[Any, ...],
) -> list[tuple[str, EmpiricalDeploymentMetrics | None]]:
    results: list[tuple[str, EmpiricalDeploymentMetrics | None]] = []

    # --- Fast baselines -------------------------------------------------
    t0 = time.perf_counter()
    lo = baseline_runner.logical_optimal_baseline(cands, weights=weights)
    print(f"[LO] closed-form U={lo.total_utility:.6g}  time={time.perf_counter() - t0:.2f}s")
    m_lo = _evaluate_and_print(
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
    sc = baseline_runner.single_cloud_baseline(
        cands,
        weights=weights,
        queries=queries,
        samples_per_query=args.eval_samples_per_query,
        violation_eval_seed=eval_seed,
    )
    provs = sorted({n.provider for n in sc.nodes})
    vc, vl = baseline_runner.mc_violation_counts_wf1(
        sc.nodes,
        queries,
        samples_per_query=args.eval_samples_per_query,
        violation_eval_seed=eval_seed,
    )
    print(
        f"[SC] selection MC violations cost={vc} latency={vl} total={vc + vl} | "
        f"closed-form U={sc.total_utility:.6g} providers={provs} "
        f"time={time.perf_counter() - t0:.2f}s"
    )
    m_sc = _evaluate_and_print(
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
    do = baseline_runner.deterministic_optimal_baseline(
        queries, cands, weights=weights, token_seed=eval_seed
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
        "Deterministic-Optimal (DO)",
        do.nodes,
        queries,
        weights=weights,
        samples_per_query=args.eval_samples_per_query,
        eval_seed=eval_seed,
        extra=extra_do,
    )
    results.append(("DO", m_do))

    # --- Sky -----------------------------------------------------------
    if args.skip_sky:
        print("[Sky] skipped (--skip-sky)")
        results.append(("Sky", None))
    else:
        sky_dec, sky_warm = sky_runner.sky_ablation_settings(args.sky_variant)
        t0 = time.perf_counter()
        rep = sky_runner.run_sky_deployment(
            queries=queries,
            s_per_query=args.sky_s_per_query,
            eta_c=args.eta_c,
            eta_t=args.eta_t,
            # WF1 (4-layer chain): soft penalties scaled so λ·ε ~ O(U); separate from workflow2.
            lamb_c=0.35,
            lamb_t=0.00112,
            weights=weights,
            batch_add_ratio=args.sky_batch_ratio,
            decomposition=sky_dec,
            use_warm_start=sky_warm,
            rng_seed=args.sky_rng,
        )
        elapsed_sky = time.perf_counter() - t0

        if isinstance(rep, sky_runner.DecompositionResult):
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


def _print_summary_table(results: list[tuple[str, EmpiricalDeploymentMetrics | None]]) -> None:
    print()
    print("=" * 70)
    print("SUMMARY (empirical U, mean cost USD, mean latency s, VR_cost, VR_lat)")
    print("=" * 70)
    hdr = f"{'algo':<6} {'U':>10} {'mean_C':>11} {'mean_T':>11} {'VR_C':>9} {'VR_T':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in results:
        if m is None:
            print(f"{name:<6} {'(skipped)':>10} {'-':>11} {'-':>11} {'-':>9} {'-':>9}")
            continue
        print(
            f"{name:<6} {m.aggregate_utility_u:>10.6g} "
            f"{m.mean_cost_usd:>11.6g} {m.mean_latency_sec:>11.6g} "
            f"{m.slo_cost_violation_rate:>9.6g} {m.slo_latency_violation_rate:>9.6g}"
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LO, SC, DO, Sky + print KPIs.")
    parser.add_argument("--num-queries", type=int, default=100, help="calibration queries Q")
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    parser.add_argument("--eval-samples-per-query", type=int, default=50, help="MC draws per query for KPIs")
    parser.add_argument(
        "--eval-seed",
        "--eval-seed-base",
        dest="eval_seed",
        type=int,
        default=9000,
        help="single RNG seed for empirical KPI evaluation (same MC draws for LO/SC/DO/Sky; alias: --eval-seed-base)",
    )
    parser.add_argument("--sky-s-per-query", type=int, default=50, help="SAA scenarios S per query for Sky")
    parser.add_argument(
        "--sky-batch-ratio",
        type=float,
        default=0.05,
        help="decomposition: each iteration adds max(1, ceil(ratio * total scenarios)) violators",
    )
    parser.add_argument("--sky-rng", type=int, default=0)
    parser.add_argument("--eta-c", type=float, default=0.1)
    parser.add_argument("--eta-t", type=float, default=0.1)
    parser.add_argument("--skip-sky", action="store_true", help="only run baselines (faster)")
    parser.add_argument(
        "--sky-variant",
        choices=["full", "no_warm_start", "direct_milp"],
        default="full",
        help="Sky ablation: full (decomposition+warm-start), no_warm_start, or direct_milp (single Q×S MILP)",
    )
    parser.add_argument(
        "--budget-alpha",
        type=float,
        nargs="*",
        default=None,
        metavar="A",
        help=(
            "预算插值系数 α（可多选）；未写值时默认 0.25 0.5 0.75 1.0。"
        ),
    )
    args = parser.parse_args()

    weights = tuple(args.weights)
    eval_seed = int(args.eval_seed)

    balpha = args.budget_alpha
    if balpha is not None and len(balpha) > 0:
        alphas_run = tuple(balpha)
    else:
        alphas_run = BUDGET_ALPHA_SUITE_DEFAULT_WF1
    run_plan = [(f"α={a:g}", a) for a in alphas_run]

    print("=" * 70)
    print("run_all_algorithms: shared calibration set + empirical KPIs")
    print("=" * 70)

    print(
        f"[setup] budget_mode=alpha runs={len(run_plan)} query_seed={args.query_seed} "
        f"num_queries_each={args.num_queries} weights={weights} eval_seed={eval_seed}"
    )
    print()

    cands = sky_runner.enumerate_candidates()
    print(f"[setup] candidate layer sizes: {[len(c) for c in cands]}")
    min_c_ch, min_l_ch = wf1_mean_min_anchor_chains(
        cands,
        num_queries=args.num_queries,
        query_sample_seed=args.query_seed,
    )
    lo_ch = wf1_logical_optimal_chain(cands, weights)
    print("[setup] budget anchors: LO + mean-min-cost chain + mean-min-latency chain (see workflow1.budget)")
    print()

    mega: list[tuple[str, list[tuple[str, EmpiricalDeploymentMetrics | None]]]] = []

    for label, alpha in run_plan:
        assert alpha is not None
        queries = generate_realistic_queries(
            args.num_queries,
            seed=args.query_seed,
            budget_alpha=float(alpha),
            lo_chain=lo_ch,
            min_mean_cost_chain=min_c_ch,
            min_mean_latency_chain=min_l_ch,
        )
        print("*" * 70)
        print(
            f"[budget α] {label}  "
            f"Θ_C = C_min + α (C_LO - C_min), Θ_T = T_min + α (T_LO - T_min) "
            f"(mean-min-cost / mean-min-lat / LO, plug-in mean ρ)"
        )
        print("*" * 70)

        results = _run_algorithms_for_queries(queries, args, weights=weights, eval_seed=eval_seed, cands=cands)
        mega.append((label, results))
        _print_summary_table(results)
        print()

    if len(mega) > 1:
        print("=" * 70)
        print("ALL PRESETS SUMMARY")
        print("=" * 70)
        bh = f"{'budget':<26} {'algo':<6} {'U':>10} {'mean_C':>11} {'mean_T':>11} {'VR_C':>9} {'VR_T':>9}"
        print(bh)
        print("-" * len(bh))
        for budget_label, rs in mega:
            for name, m in rs:
                if m is None:
                    print(
                        f"{budget_label:<26} {name:<6} "
                        f"{'(skip)':>10} {'-':>11} {'-':>11} {'-':>9} {'-':>9}"
                    )
                    continue
                print(
                    f"{budget_label:<26} {name:<6} "
                    f"{m.aggregate_utility_u:>10.6g} {m.mean_cost_usd:>11.6g} "
                    f"{m.mean_latency_sec:>11.6g} {m.slo_cost_violation_rate:>9.6g} "
                    f"{m.slo_latency_violation_rate:>9.6g}"
                )
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
