"""
Sky 消融实验：独立变量 × 三种变体，记录求解时间、迭代次数、峰值内存与目标值。

在 ``simulation/`` 目录下执行::

    python sky_ablation_experiment.py --out results/sky_ablation.csv
    python sky_ablation_experiment.py --s-list 10,20,30 --q-fixed 50 --q-list 10,30,50 --s-fixed 50

每次实验在 **独立子进程** 中运行，以便 ``resource.getrusage`` 反映该次求解的峰值驻留内存
（Linux: ru_maxrss 为 KB）。

依赖: PuLP、solve 前已满足的 sim_env/utils；可选 ``tqdm`` 显示进度。
"""

from __future__ import annotations

import argparse
import csv
import resource
import subprocess
import sys
import time
import json
import math
from pathlib import Path
from typing import Any, Literal

# Default sweeps (paper-style)
DEFAULT_S_LIST = list(range(10, 101, 10))  # 10,20,...,100
DEFAULT_Q_LIST = list(range(10, 101, 10))  # 10,20,...,100

VariantName = Literal["full", "no_warm_start", "direct_milp"]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ru_maxrss_to_bytes(rss: int) -> int:
    """``resource.getrusage`` max RSS: Linux = kilobytes; macOS = bytes."""
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def run_sky_single_in_process(payload: dict[str, Any]) -> dict[str, Any]:
    """
    在当前进程执行一次 ``run_sky_deployment``，用于子进程入口或单元测试。
    """
    import sky as sky_runner
    from utils import generate_realistic_queries

    variant: VariantName = payload["variant"]
    num_queries = int(payload["num_queries"])
    s_per_query = int(payload["s_per_query"])
    query_seed = int(payload["query_seed"])
    sky_rng_seed = int(payload["sky_rng_seed"])
    eta_c = float(payload.get("eta_c", 0.1))
    eta_t = float(payload.get("eta_t", 0.1))
    lamb_c = float(payload.get("lamb_c", 1.0))
    lamb_t = float(payload.get("lamb_t", 1.0))
    weights: tuple[float, float, float, float] = tuple(
        float(x) for x in payload.get("weights", (1.0, 1.0, 1.0, 1.0))
    )
    batch_k = int(payload.get("batch_k", 8))

    dec, warm = sky_runner.sky_ablation_settings(variant)
    queries = generate_realistic_queries(num_queries, seed=query_seed)

    t0 = time.perf_counter()
    rep = sky_runner.run_sky_deployment(
        queries=queries,
        s_per_query=s_per_query,
        eta_c=eta_c,
        eta_t=eta_t,
        lamb_c=lamb_c,
        lamb_t=lamb_t,
        weights=weights,
        batch_k=batch_k,
        decomposition=dec,
        use_warm_start=warm,
        rng_seed=sky_rng_seed,
    )
    wall_sec = time.perf_counter() - t0

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_bytes = _ru_maxrss_to_bytes(ru)

    if isinstance(rep, sky_runner.DecompositionResult):
        sol = rep.solution
        master_iters = rep.iterations
    else:
        sol = rep
        master_iters = None

    return {
        "ok": True,
        "wall_clock_sec": wall_sec,
        "master_iterations": master_iters,
        "peak_memory_bytes": peak_memory_bytes,
        "objective_value": sol.objective_value,
        "pulp_status": sol.pulp_status,
        "error": "",
    }


def _worker_main() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        out = run_sky_single_in_process(payload)
    except Exception as e:  # noqa: BLE001 — report back to parent
        out = {
            "ok": False,
            "wall_clock_sec": math.nan,
            "master_iterations": None,
            "peak_memory_bytes": None,
            "objective_value": None,
            "pulp_status": "",
            "error": f"{type(e).__name__}: {e}",
        }
    sys.stdout.write(json.dumps(out))
    sys.exit(0 if out.get("ok") else 1)


def run_isolated_subprocess(
    payload: dict[str, Any],
    *,
    timeout_sec: float | None,
) -> dict[str, Any]:
    """新解释器进程中运行单次实验，峰值内存为该子进程生命周期内最大值。"""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    inp = json.dumps(payload).encode("utf-8")
    try:
        proc = subprocess.run(
            cmd,
            input=inp,
            capture_output=True,
            timeout=timeout_sec,
            cwd=str(Path(__file__).resolve().parent),
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "wall_clock_sec": math.nan,
            "master_iterations": None,
            "peak_memory_bytes": None,
            "objective_value": None,
            "pulp_status": "",
            "error": f"timeout after {timeout_sec}s",
        }
    out_txt = proc.stdout.decode("utf-8", errors="replace").strip()
    if out_txt:
        try:
            return json.loads(out_txt)
        except json.JSONDecodeError:
            pass
    err = proc.stderr.decode("utf-8", errors="replace")[:2000]
    return {
        "ok": False,
        "wall_clock_sec": math.nan,
        "master_iterations": None,
        "peak_memory_bytes": None,
        "objective_value": None,
        "pulp_status": "",
        "error": err or f"exit {proc.returncode} (no JSON on stdout)",
    }


def _stable_sky_rng(
    sweep: str,
    q: int,
    s: int,
    variant: str,
    base: int,
) -> int:
    """可复现：同一 (sweep, Q, S, variant) 组合对应固定 RNG seed（与子进程无关）。"""
    sweep_code = 1 if sweep == "vary_S_fixed_Q" else 2
    vcode = {"full": 11, "no_warm_start": 22, "direct_milp": 33}.get(variant, 0)
    return int(base + sweep_code * 100_000 + q * 10_007 + s * 1301 + vcode)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants: list[VariantName] = list(args.variants)  # type: ignore[assignment]
    jobs: list[dict[str, Any]] = []

    # Sweep 1: fixed Q, vary S
    for s in args.s_list:
        q = args.q_fixed
        for variant in variants:
            jobs.append(
                {
                    "sweep": "vary_S_fixed_Q",
                    "fixed_dimension": "Q",
                    "fixed_value": q,
                    "independent_S": s,
                    "independent_Q": q,
                    "variant": variant,
                    "payload": {
                        "variant": variant,
                        "num_queries": q,
                        "s_per_query": s,
                        "query_seed": args.query_seed,
                        "sky_rng_seed": _stable_sky_rng(
                            "vary_S_fixed_Q", q, s, variant, args.sky_rng_base
                        ),
                        "eta_c": args.eta_c,
                        "eta_t": args.eta_t,
                        "lamb_c": args.lamb_c,
                        "lamb_t": args.lamb_t,
                        "weights": list(args.weights),
                        "batch_k": args.batch_k,
                    },
                }
            )

    # Sweep 2: fixed S, vary Q
    for q in args.q_list:
        s = args.s_fixed
        for variant in variants:
            jobs.append(
                {
                    "sweep": "vary_Q_fixed_S",
                    "fixed_dimension": "S",
                    "fixed_value": s,
                    "independent_S": s,
                    "independent_Q": q,
                    "variant": variant,
                    "payload": {
                        "variant": variant,
                        "num_queries": q,
                        "s_per_query": s,
                        "query_seed": args.query_seed,
                        "sky_rng_seed": _stable_sky_rng(
                            "vary_Q_fixed_S", q, s, variant, args.sky_rng_base
                        ),
                        "eta_c": args.eta_c,
                        "eta_t": args.eta_t,
                        "lamb_c": args.lamb_c,
                        "lamb_t": args.lamb_t,
                        "weights": list(args.weights),
                        "batch_k": args.batch_k,
                    },
                }
            )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sky ablation: S/Q sweeps × variants → CSV metrics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sky_ablation_results.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--s-list",
        type=_parse_int_list,
        default=DEFAULT_S_LIST,
        help="comma-separated S values for sweep vary_S_fixed_Q (default 10..100 step 10)",
    )
    parser.add_argument("--q-fixed", type=int, default=50, help="fixed Q when varying S")
    parser.add_argument(
        "--q-list",
        type=_parse_int_list,
        default=DEFAULT_Q_LIST,
        help="comma-separated Q values for sweep vary_Q_fixed_S",
    )
    parser.add_argument("--s-fixed", type=int, default=50, help="fixed S when varying Q")
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument("--sky-rng-base", type=int, default=0, help="base for derived sky RNG seeds")
    parser.add_argument("--batch-k", type=int, default=8)
    parser.add_argument("--eta-c", type=float, default=0.1)
    parser.add_argument("--eta-t", type=float, default=0.1)
    parser.add_argument("--lamb-c", type=float, default=1.0)
    parser.add_argument("--lamb-t", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.0],
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["full", "no_warm_start", "direct_milp"],
        choices=["full", "no_warm_start", "direct_milp"],
        help="subset of ablation variants",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="do not spawn subprocess (faster but peak_memory is less reliable)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="per-job time limit for subprocess solve (only with default subprocess mode)",
    )
    args = parser.parse_args()
    args.weights = tuple(args.weights)  # type: ignore[assignment]

    jobs = build_jobs(args)
    fieldnames = [
        "sweep",
        "fixed_dimension",
        "fixed_value",
        "Q",
        "S",
        "variant",
        "wall_clock_sec",
        "master_iterations",
        "peak_memory_bytes",
        "objective_value",
        "pulp_status",
        "error",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    iterator = tqdm(jobs, desc="ablation") if tqdm else jobs

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        f.flush()

        for job in iterator:
            payload = job["payload"]
            if args.in_process:
                r = run_sky_single_in_process(payload)
            else:
                r = run_isolated_subprocess(payload, timeout_sec=args.timeout_sec)

            row = {
                "sweep": job["sweep"],
                "fixed_dimension": job["fixed_dimension"],
                "fixed_value": job["fixed_value"],
                "Q": job["independent_Q"],
                "S": job["independent_S"],
                "variant": job["variant"],
                "wall_clock_sec": r.get("wall_clock_sec", math.nan),
                "master_iterations": r.get("master_iterations", ""),
                "peak_memory_bytes": r.get("peak_memory_bytes", ""),
                "objective_value": r.get("objective_value", ""),
                "pulp_status": r.get("pulp_status", ""),
                "error": r.get("error", ""),
            }
            if row["master_iterations"] is None:
                row["master_iterations"] = ""
            w.writerow(row)
            f.flush()

    print(f"Wrote {len(jobs)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main()
    else:
        main()
