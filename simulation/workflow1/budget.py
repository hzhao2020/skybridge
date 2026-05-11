"""
在 plug-in mean ρ 下，用 N 条随机抽样的源大小队列，枚举候选层笛卡尔积中的**每一条部署链**，
计算在该链上端到端 mean-field cost / latency 的样本均值，找出**平均 cost 最小**与**平均 latency 最小**
的链并打印。

运行（在 ``simulation/`` 目录）::

    python -m workflow1.budget
"""

from __future__ import annotations

import itertools
import math
import random
import sys
import time
from typing import Iterable, Sequence

from sim_env import config as cfg
from sim_env.utility import PhysicalNode, QueryProfile

from . import utils as wf_utils
from .sky import enumerate_candidates

N_QUERIES = 100
SEED = 42
N_CALIBRATION_SAMPLES = 4096


def sample_source_sizes_gb(*, num_queries: int, seed: int) -> list[float]:
    """与 ``generate_realistic_queries`` 相同：时长 U[60,3600]s → ``s_src_gb``。"""
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        out.append(s_src_mb / 1000.0)
    return out


def mean_plugin_rho_wf1(*, seed: int) -> tuple[float, float, float, float]:
    return cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=N_CALIBRATION_SAMPLES,
        rng=random.Random(seed + 100),
        operations=("segment", "split", "caption", "query"),
    )


def iter_all_chains(
    cands: tuple[tuple[PhysicalNode, ...], ...],
) -> Iterable[tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode]]:
    return itertools.product(cands[0], cands[1], cands[2], cands[3])  # type: ignore[misc]


def chain_mean_cost_latency(
    chain: Sequence[PhysicalNode],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, float, float, float],
    *,
    eval_seed: int,
) -> tuple[float, float]:
    """对 sizes 上逐条 query 累加；每条 query 用独立但可复现的 ``workflow_rng``。"""
    tot_c = 0.0
    tot_l = 0.0
    n = len(sizes_gb)
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf_utils.det_rng(eval_seed, "budget_combo_eval", q_idx)
        c, ell = wf_utils.end_to_end_cost_and_latency(
            (chain[0], chain[1], chain[2], chain[3]),
            float(s_gb),
            mean_rho,
            workflow_rng=wf,
        )
        if not (math.isfinite(c) and math.isfinite(ell)):
            return float("inf"), float("inf")
        tot_c += float(c)
        tot_l += float(ell)
    inv = 1.0 / float(n)
    return tot_c * inv, tot_l * inv


def query_profiles_from_chain(
    chain: Sequence[PhysicalNode],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, float, float, float],
    *,
    eval_seed: int,
) -> list[QueryProfile]:
    """用**同一条**固定链与 mean ρ，为每个 ``s_src_gb`` 生成 ``QueryProfile(Θ_C,Θ_T)``。"""
    out: list[QueryProfile] = []
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf_utils.det_rng(eval_seed, "budget_profile", q_idx)
        c, ell = wf_utils.end_to_end_cost_and_latency(
            (chain[0], chain[1], chain[2], chain[3]),
            float(s_gb),
            mean_rho,
            workflow_rng=wf,
        )
        out.append(
            QueryProfile(
                s_src_gb=float(s_gb),
                theta_cost=float(c),
                theta_latency_sec=float(ell),
            )
        )
    return out


def run_search(*, verbose_every: int = 5000) -> None:
    sizes = sample_source_sizes_gb(num_queries=N_QUERIES, seed=SEED)
    mean_rho = mean_plugin_rho_wf1(seed=SEED)
    cands = enumerate_candidates()
    layer_sizes = [len(cands[i]) for i in range(4)]
    n_chains = math.prod(layer_sizes)

    print("workflow1.budget | plug-in mean ρ (calibration samples)", N_CALIBRATION_SAMPLES)
    print(f"  queries={N_QUERIES}  eval_seed={SEED}  chains={n_chains}  layers={layer_sizes}")

    best_c_mean = float("inf")
    best_l_mean = float("inf")
    best_c_chain: tuple[PhysicalNode, ...] | None = None
    best_l_chain: tuple[PhysicalNode, ...] | None = None

    t0 = time.perf_counter()
    for idx, chain_t in enumerate(iter_all_chains(cands), start=1):
        if verbose_every and idx % verbose_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"  ... scanned {idx}/{n_chains} chains ({elapsed:.1f}s)", file=sys.stderr)
        chain = tuple(chain_t)
        m_c, m_l = chain_mean_cost_latency(chain, sizes, mean_rho, eval_seed=SEED)
        if m_c < best_c_mean - 1e-15:
            best_c_mean = m_c
            best_c_chain = chain
        if m_l < best_l_mean - 1e-15:
            best_l_mean = m_l
            best_l_chain = chain

    elapsed = time.perf_counter() - t0
    print(f"\n=== workflow1：mean cost 最小的部署链（mean ρ，{N_QUERIES} 条随机源大小）===")
    print(f"  mean cost (USD): {best_c_mean:.6g}")
    print(f"  链: {best_c_chain}")

    print(f"\n=== workflow1：mean latency 最小的部署链（同上）===")
    print(f"  mean latency (s): {best_l_mean:.6g}")
    print(f"  链: {best_l_chain}")

    if best_l_chain is not None:
        _mc, _ = chain_mean_cost_latency(
            best_l_chain, sizes, mean_rho, eval_seed=SEED
        )
        print(f"  （该链的 mean cost 为 {_mc:.6g} USD，供对照）")
    if best_c_chain is not None:
        mc_on_best_c, _ml = chain_mean_cost_latency(
            best_c_chain, sizes, mean_rho, eval_seed=SEED
        )
        print(f"  （该链的 mean latency 为 {_ml:.6g} s，供对照）")

    print(f"\n  总耗时 {elapsed:.1f}s")


def main() -> None:
    run_search()


if __name__ == "__main__":
    main()
