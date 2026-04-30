"""
一键顺序运行：**Logical-Optimal (LO)**、**Single-Cloud (SC)**、**Deterministic-Optimal (DO)**、**Sky (CVaR–SAA MILP)**，
并对同一批 ``queries`` 调用 ``evaluation.evaluate_deployment_empirical`` 打印 KPI：
Aggregate Utility / mean Cost / mean Latency / SVR。

在 ``simulation/`` 目录下执行::

    python run_all_algorithms.py
    python run_all_algorithms.py --num-queries 20

大规模实验请调大 ``--num-queries`` / ``--sky-s-per-query``（Sky 会非常慢）。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import baseline as baseline_runner
import sky as sky_runner
from evaluation import EmpiricalDeploymentMetrics, evaluate_deployment_empirical, print_metrics_report
from sim_env.utility import QueryProfile
from utils import generate_realistic_queries


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
    parser.add_argument("--sky-batch-k", type=int, default=10)
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
    args = parser.parse_args()

    weights = tuple(args.weights)
    eval_seed = int(args.eval_seed)

    print("=" * 70)
    print("run_all_algorithms: shared calibration set + empirical KPIs")
    print("=" * 70)

    queries = generate_realistic_queries(args.num_queries, seed=args.query_seed)
    print(
        f"[setup] num_queries={len(queries)} query_seed={args.query_seed} "
        f"weights={weights} eval_seed={eval_seed}"
    )
    print()

    cands = sky_runner.enumerate_candidates()
    print(f"[setup] candidate layer sizes: {[len(c) for c in cands]}")

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
    sc = baseline_runner.single_cloud_baseline(cands, weights=weights)
    provs = sorted({n.provider for n in sc.nodes})
    print(
        f"[SC] closed-form U={sc.total_utility:.6g} providers={provs} "
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
        f"[DO] MILP status={do.pulp_status} MILP_obj_U={do.total_utility} "
        f"time={time.perf_counter() - t0:.2f}s"
    )
    print(f"[DO] deployment tuple: {do.nodes}")
    extra_do = f"MILP={do.pulp_status}"
    if do.pulp_status == "Optimal":
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
            lamb_c=1.0,
            lamb_t=1.0,
            weights=weights,
            batch_k=args.sky_batch_k,
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
                f"pulp_obj={sol.objective_value}"
            )
        else:
            sol = rep
            extra_sky = f"full_MILP elapsed={elapsed_sky:.2f}s pulp_obj={sol.objective_value}"

        print(f"[Sky] {extra_sky}")
        print(f"[Sky] pulp_status={sol.pulp_status}")
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

    # --- Compact summary table -----------------------------------------
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
