"""
一键顺序运行：**Logical-Optimal (LO)**、**Single-Cloud (SC)**、**Deterministic-Optimal (DO)**、**Sky (CVaR–SAA MILP)**，
并对同一批 ``queries`` 调用 ``evaluation.evaluate_deployment_empirical`` 打印 KPI：
Aggregate Utility / mean Cost / mean Latency / SVR。

默认对三种 budget 预设各跑一次（参见 ``workflow1.utils.BUDGET_PRESET_MULTIPLIERS_WF1``）：
cost_sensitivity → 1.5×min_cost、2×min_latency；latency_sensitivity → 2×、1.5×；balanced → 1.75× / 1.75×。

在包含 ``workflow1`` 与 ``sim_env`` 的 ``simulation/`` 目录下执行::

    python -m workflow1.run_all_algorithms
    python -m workflow1.run_all_algorithms --num-queries 20
    python -m workflow1.run_all_algorithms --budget-preset balanced

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
from .evaluation import EmpiricalDeploymentMetrics, evaluate_deployment_empirical, print_metrics_report
from .utils import (
    BUDGET_PRESET_MULTIPLIERS_WF1,
    BUDGET_PRESET_SUITE_ORDER_WF1,
    generate_realistic_queries,
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
            lamb_c=5.0,
            lamb_t=0.00125,
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
        "--budget-preset",
        nargs="+",
        default=None,
        choices=list(BUDGET_PRESET_MULTIPLIERS_WF1.keys()),
        metavar="PRESET",
        help=(
            "one or more budget multiplier presets (%(choices)s); "
            "default runs the full suite in order: "
            + ", ".join(BUDGET_PRESET_SUITE_ORDER_WF1)
        ),
    )
    args = parser.parse_args()

    weights = tuple(args.weights)
    eval_seed = int(args.eval_seed)
    presets = tuple(args.budget_preset) if args.budget_preset else BUDGET_PRESET_SUITE_ORDER_WF1

    print("=" * 70)
    print("run_all_algorithms: shared calibration set + empirical KPIs")
    print("=" * 70)

    print(
        f"[setup] budget_presets={list(presets)} query_seed={args.query_seed} "
        f"num_queries_each={args.num_queries} weights={weights} eval_seed={eval_seed}"
    )
    print()

    cands = sky_runner.enumerate_candidates()
    print(f"[setup] candidate layer sizes: {[len(c) for c in cands]}")
    print()

    mega: list[tuple[str, float, float, list[tuple[str, EmpiricalDeploymentMetrics | None]]]] = []

    for preset_name in presets:
        mc, ml = BUDGET_PRESET_MULTIPLIERS_WF1[preset_name]
        queries = generate_realistic_queries(
            args.num_queries,
            seed=args.query_seed,
            budget_cost_multiplier=mc,
            budget_latency_multiplier=ml,
        )
        print("*" * 70)
        print(
            f"[budget preset] {preset_name} "
            f"(Θ_cost = {mc} × min_meanfield_cost_ref, "
            f"Θ_latency = {ml} × min_meanfield_latency_ref)"
        )
        print("*" * 70)

        results = _run_algorithms_for_queries(queries, args, weights=weights, eval_seed=eval_seed, cands=cands)
        mega.append((preset_name, mc, ml, results))
        _print_summary_table(results)
        print()

    if len(mega) > 1:
        print("=" * 70)
        print("ALL PRESETS SUMMARY")
        print("=" * 70)
        bh = (
            f"{'budget':<20} {'mc':>5} {'ml':>5} {'algo':<6} "
            f"{'U':>10} {'mean_C':>11} {'mean_T':>11} {'VR_C':>9} {'VR_T':>9}"
        )
        print(bh)
        print("-" * len(bh))
        for preset_name, mc, ml, rs in mega:
            for name, m in rs:
                if m is None:
                    print(
                        f"{preset_name:<20} {mc:>5.3g} {ml:>5.3g} {name:<6} "
                        f"{'(skip)':>10} {'-':>11} {'-':>11} {'-':>9} {'-':>9}"
                    )
                    continue
                print(
                    f"{preset_name:<20} {mc:>5.3g} {ml:>5.3g} {name:<6} "
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
