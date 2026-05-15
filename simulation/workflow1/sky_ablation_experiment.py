"""
Sky 消融实验：独立变量 × 三种变体，记录求解时间、迭代次数、峰值内存与目标值。

在包含 ``workflow1`` 与 ``sim_env`` 的 ``simulation/`` 目录下执行::

    python -m workflow1.sky_ablation_experiment --out workflow1/results/sky_ablation.csv
    python -m workflow1.sky_ablation_experiment --s-list 10,20,30 --q-fixed 50 --q-list 10,30,50 --s-fixed 50

每次实验在 **独立子进程** 中运行，以便度量该次求解的进程峰值内存：
Unix/macOS 使用 ``resource.getrusage(RUSAGE_SELF).ru_maxrss``；
Windows 无 ``resource`` 时使用 ``psutil.Process().memory_full_info().peak_wset``（峰值工作集，字节）。

结果写入 ``--out`` CSV：若文件已存在且非空则 **续写**（不写表头）；新文件或空文件则先写表头。
需要整文件重写时使用 ``--overwrite``。列 ``batch_add_ratio``（每次迭代新增场景占 ``|Ω|`` 的比例）与命令行 ``--batch-add-ratio`` 一致；列 ``gurobi_status`` 为 Gurobi 求解状态字符串。

对同一网格点 (Q, S, sweep)，三种变体共用同一 ``rng_seed``（**不含 variant**），因而 ``build_joint_scenarios`` 的联合场景、每条 ω 上的 shot_detection/video_split 执行噪声、LLM token、链路的 ``sample_link`` 旋转均由此种子确定性导出（``sky.prepare_coefficients`` / ``utils.det_rng``），消融对比下随机环境一致。

单点消融（例如 Q=50、S=50）::

    python -m workflow1.sky_ablation_experiment --only-q 50 --only-s 50 --out workflow1/results/q50s50.csv

"""

from __future__ import annotations

import argparse
import csv
import subprocess

try:
    import resource
except ImportError:
    resource = None  # Windows: stdlib ``resource`` unavailable
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


def _peak_memory_bytes_self() -> int | None:
    """
    当前进程的峰值常驻/工作集内存（字节），尽力而为。

    - Unix/macOS：``resource.getrusage`` 的 ``ru_maxrss``（与原先逻辑一致）。
    - Windows：``psutil.memory_full_info().peak_wset``；若无 ``peak_wset`` 则退回当前 ``rss``。
    """
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return _ru_maxrss_to_bytes(ru)
    try:
        import psutil
    except ImportError:
        return None
    try:
        mi = psutil.Process().memory_full_info()
    except psutil.Error:
        return None
    peak_wset = getattr(mi, "peak_wset", None)
    if peak_wset is not None:
        return int(peak_wset)
    rss = getattr(mi, "rss", None)
    return int(rss) if rss is not None else None


def run_sky_single_in_process(payload: dict[str, Any]) -> dict[str, Any]:
    """
    在当前进程执行一次 ``run_sky_deployment``，用于子进程入口或单元测试。
    """
    from . import sky as sky_runner
    from .utils import generate_realistic_queries, wf1_logical_optimal_chain

    variant: VariantName = payload["variant"]
    num_queries = int(payload["num_queries"])
    s_per_query = int(payload["s_per_query"])
    query_seed = int(payload["query_seed"])
    sky_rng_seed = int(payload["sky_rng_seed"])
    eta_c = float(payload.get("eta_c", 0.1))
    eta_t = float(payload.get("eta_t", 0.1))
    lamb_c = float(payload.get("lamb_c", 0.35))
    lamb_t = float(payload.get("lamb_t", 0.00112))
    weights: tuple[float, float, float, float] = tuple(
        float(x) for x in payload.get("weights", (0.25, 0.25, 0.25, 0.25))
    )
    batch_add_ratio = float(payload.get("batch_add_ratio", 0.05))
    budget_alpha = float(payload.get("budget_alpha", 1.0))

    dec, warm = sky_runner.sky_ablation_settings(variant)
    cands = sky_runner.enumerate_candidates()
    lo_ch = wf1_logical_optimal_chain(cands, weights)
    queries = generate_realistic_queries(
        num_queries,
        seed=query_seed,
        budget_alpha=budget_alpha,
        lo_chain=lo_ch,
        weights=weights,
        cands=cands,
    )

    t0 = time.perf_counter()
    rep = sky_runner.run_sky_deployment(
        queries=queries,
        s_per_query=s_per_query,
        eta_c=eta_c,
        eta_t=eta_t,
        lamb_c=lamb_c,
        lamb_t=lamb_t,
        weights=weights,
        batch_add_ratio=batch_add_ratio,
        decomposition=dec,
        use_warm_start=warm,
        rng_seed=sky_rng_seed,
    )
    wall_sec = time.perf_counter() - t0

    peak_memory_bytes = _peak_memory_bytes_self()

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
        "gurobi_status": sol.gurobi_status,
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
            "peak_memory_bytes": _peak_memory_bytes_self(),
            "objective_value": None,
            "gurobi_status": "",
            "error": f"{type(e).__name__}: {e}",
        }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()
    sys.exit(0 if out.get("ok") else 1)


def _simulation_root() -> Path:
    """``simulation/``：上一级目录，内含 ``workflow1/`` 与 ``sim_env/``。"""
    return Path(__file__).resolve().parent.parent


def _decode_worker_stdout(proc_stdout: bytes) -> dict[str, Any] | None:
    """
    解析 worker 写入的 JSON。

    若某依赖在 import 阶段向 stdout 打了日志，整段 stdout 可能不是合法 JSON；此处从最后一行起向前尝试
    ``json.loads``，并额外接受 ``strip()`` 后的整段文本。
    """
    text = proc_stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for candidate in (*reversed(lines), text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def run_isolated_subprocess(
    payload: dict[str, Any],
    *,
    timeout_sec: float | None,
) -> dict[str, Any]:
    """新解释器进程中运行单次实验，峰值内存为该子进程生命周期内最大值。"""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "workflow1.sky_ablation_experiment",
        "--worker",
    ]
    inp = json.dumps(payload).encode("utf-8")
    try:
        proc = subprocess.run(
            cmd,
            input=inp,
            capture_output=True,
            timeout=timeout_sec,
            cwd=str(_simulation_root()),
            text=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "wall_clock_sec": math.nan,
            "master_iterations": None,
            "peak_memory_bytes": None,
            "objective_value": None,
            "gurobi_status": "",
            "error": f"timeout after {timeout_sec}s",
        }
    parsed = _decode_worker_stdout(proc.stdout)
    if parsed is not None:
        return parsed
    out_preview = proc.stdout.decode("utf-8", errors="replace").strip().replace("\n", "\\n")[:500]
    err = proc.stderr.decode("utf-8", errors="replace")[:2000]
    detail_parts = [
        err,
        f"exit {proc.returncode}",
        (f"stdout_preview={out_preview!r}" if out_preview else "stdout empty"),
    ]
    detail = " | ".join(p for p in detail_parts if p)
    return {
        "ok": False,
        "wall_clock_sec": math.nan,
        "master_iterations": None,
        "peak_memory_bytes": None,
        "objective_value": None,
        "gurobi_status": "",
        "error": detail or f"exit {proc.returncode} (no JSON on stdout)",
    }


def _stable_experiment_rng_seed(sweep: str, q: int, s: int, base: int) -> int:
    """
    固定 (sweep, Q, S) 下的主随机种子，**故意不包含 variant**。

    ``sky.run_sky_deployment(rng_seed=...)`` 用同一 RNG 先后驱动 ``build_joint_scenarios`` 与
    ``scenario_adaptive_decomposition`` 的初始 shuffle，保证消融对比时场景数据与分解随机序一致。
    """
    if sweep == "vary_S_fixed_Q":
        sweep_code = 1
    elif sweep == "vary_Q_fixed_S":
        sweep_code = 2
    else:
        sweep_code = 3  # fixed_Q_S
    return int(base + sweep_code * 100_000 + q * 10_007 + s * 1301)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants: list[VariantName] = list(args.variants)  # type: ignore[assignment]
    jobs: list[dict[str, Any]] = []

    if args.only_q is not None:
        q, s = args.only_q, args.only_s
        for variant in variants:
            jobs.append(
                {
                    "sweep": "fixed_Q_S",
                    "independent_S": s,
                    "independent_Q": q,
                    "variant": variant,
                    "payload": {
                        "variant": variant,
                        "num_queries": q,
                        "s_per_query": s,
                        "query_seed": args.query_seed,
                        "sky_rng_seed": _stable_experiment_rng_seed(
                            "fixed_Q_S", q, s, args.sky_rng_base
                        ),
                        "eta_c": args.eta_c,
                        "eta_t": args.eta_t,
                        "lamb_c": args.lamb_c,
                        "lamb_t": args.lamb_t,
                        "weights": list(args.weights),
                        "batch_add_ratio": args.batch_add_ratio,
                        "budget_alpha": args.budget_alpha,
                    },
                }
            )
        return jobs

    # Sweep 1: fixed Q, vary S
    for s in args.s_list:
        q = args.q_fixed
        for variant in variants:
            jobs.append(
                {
                    "sweep": "vary_S_fixed_Q",
                    "independent_S": s,
                    "independent_Q": q,
                    "variant": variant,
                    "payload": {
                        "variant": variant,
                        "num_queries": q,
                        "s_per_query": s,
                        "query_seed": args.query_seed,
                        "sky_rng_seed": _stable_experiment_rng_seed(
                            "vary_S_fixed_Q", q, s, args.sky_rng_base
                        ),
                        "eta_c": args.eta_c,
                        "eta_t": args.eta_t,
                        "lamb_c": args.lamb_c,
                        "lamb_t": args.lamb_t,
                        "weights": list(args.weights),
                        "batch_add_ratio": args.batch_add_ratio,
                        "budget_alpha": args.budget_alpha,
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
                    "independent_S": s,
                    "independent_Q": q,
                    "variant": variant,
                    "payload": {
                        "variant": variant,
                        "num_queries": q,
                        "s_per_query": s,
                        "query_seed": args.query_seed,
                        "sky_rng_seed": _stable_experiment_rng_seed(
                            "vary_Q_fixed_S", q, s, args.sky_rng_base
                        ),
                        "eta_c": args.eta_c,
                        "eta_t": args.eta_t,
                        "lamb_c": args.lamb_c,
                        "lamb_t": args.lamb_t,
                        "weights": list(args.weights),
                        "batch_add_ratio": args.batch_add_ratio,
                        "budget_alpha": args.budget_alpha,
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
        help="output CSV path (append if file exists and is non-empty, unless --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="truncate --out and write a fresh header before this run",
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
    parser.add_argument(
        "--only-q",
        type=int,
        default=None,
        metavar="Q",
        help="with --only-s: run exactly one (Q,S) cell × variants (ignores default sweeps)",
    )
    parser.add_argument(
        "--only-s",
        type=int,
        default=None,
        metavar="S",
        help="with --only-q: per-query scenario count S for that single cell",
    )
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--sky-rng-base",
        type=int,
        default=0,
        help="base for experiment RNG seed (same per Q,S,sweep for all variants; not per-variant)",
    )
    parser.add_argument(
        "--batch-add-ratio",
        type=float,
        default=0.05,
        metavar="R",
        help="decomposition: each iteration adds max(1, ceil(R * total joint scenarios))",
    )
    parser.add_argument(
        "--budget-alpha",
        type=float,
        default=1.0,
        metavar="A",
        help="query 预算 α：Θ=min+α(max−min)（四锚链 SC×3+LO；见 generate_realistic_queries）",
    )
    parser.add_argument("--eta-c", type=float, default=0.1)
    parser.add_argument("--eta-t", type=float, default=0.1)
    parser.add_argument("--lamb-c", type=float, default=0.35)
    parser.add_argument("--lamb-t", type=float, default=0.00112)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=[0.25, 0.25, 0.25, 0.25],
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

    if (args.only_q is None) ^ (args.only_s is None):
        parser.error("--only-q and --only-s must be used together")

    jobs = build_jobs(args)

    fieldnames = [
        "sweep",
        "Q",
        "S",
        "variant",
        "batch_add_ratio",
        "wall_clock_sec",
        "master_iterations",
        "peak_memory_bytes",
        "objective_value",
        "gurobi_status",
        "error",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    append_mode = (
        not args.overwrite
        and args.out.exists()
        and args.out.stat().st_size > 0
    )
    file_mode = "a" if append_mode else "w"
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    iterator = tqdm(jobs, desc="ablation") if tqdm else jobs

    with args.out.open(file_mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not append_mode:
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
                "Q": job["independent_Q"],
                "S": job["independent_S"],
                "variant": job["variant"],
                "batch_add_ratio": payload.get("batch_add_ratio", ""),
                "wall_clock_sec": r.get("wall_clock_sec", math.nan),
                "master_iterations": r.get("master_iterations", ""),
                "peak_memory_bytes": r.get("peak_memory_bytes", ""),
                "objective_value": r.get("objective_value", ""),
                "gurobi_status": r.get("gurobi_status") or r.get("pulp_status", ""),
                "error": r.get("error", ""),
            }
            if row["master_iterations"] is None:
                row["master_iterations"] = ""
            if row["peak_memory_bytes"] is None:
                row["peak_memory_bytes"] = ""
            w.writerow(row)
            f.flush()

    action = "Appended" if append_mode else "Wrote"
    print(f"{action} {len(jobs)} rows to {args.out.resolve()}")


if __name__ == "__main__":
    # 勿仅用 argv[1]：某些启动器/包装会在模块名前插入额外标志位。
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
