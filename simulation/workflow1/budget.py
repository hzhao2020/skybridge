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
    """与 ``generate_realistic_queries`` 相同：时长 U[60,3600]s → ``s_src_gb``（不含 Θ 预算；预算由插值 API 单独构造）。"""
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
        operations=("shot_detection", "video_split", "video_caption", "query"),
    )


def iter_all_chains(
    cands: tuple[tuple[PhysicalNode, ...], ...],
) -> Iterable[tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode]]:
    return itertools.product(cands[0], cands[1], cands[2], cands[3])  # type: ignore[misc]


def find_wf1_mean_min_cost_and_latency_chains(
    cands: tuple[tuple[PhysicalNode, ...], ...],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, float, float, float],
    *,
    eval_seed: int,
) -> tuple[
    tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
]:
    """
    与 ``run_search`` 相同：在 plug-in mean ρ 与给定源大小队列上枚举 ``cands`` 全链，
    返回 **样本平均 cost 最小** 与 **样本平均 latency 最小** 的两条部署链（可为不同链）。
    """
    best_c_mean = float("inf")
    best_l_mean = float("inf")
    best_c_chain: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode] | None = None
    best_l_chain: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode] | None = None
    for chain_t in iter_all_chains(cands):
        chain = tuple(chain_t)
        m_c, m_l = chain_mean_cost_latency(chain, sizes_gb, mean_rho, eval_seed=eval_seed)
        if m_c < best_c_mean - 1e-15:
            best_c_mean = m_c
            best_c_chain = chain  # type: ignore[assignment]
        if m_l < best_l_mean - 1e-15:
            best_l_mean = m_l
            best_l_chain = chain  # type: ignore[assignment]
    if best_c_chain is None or best_l_chain is None:
        raise RuntimeError("failed to find mean-min cost/latency chains (empty candidates?)")
    return best_c_chain, best_l_chain


def wf1_mean_min_anchor_chains(
    cands: tuple[tuple[PhysicalNode, ...], ...],
    *,
    num_queries: int,
    query_sample_seed: int,
    n_calibration_samples: int = N_CALIBRATION_SAMPLES,
    chain_eval_seed: int | None = None,
) -> tuple[
    tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
]:
    """
    构造与 ``generate_realistic_queries`` 一致的 plug-in mean ρ（``query_sample_seed+100``），
    在给定 ``num_queries`` 条随机源大小上枚举全链，返回 (mean-cost-min 链, mean-latency-min 链)。
    ``chain_eval_seed`` 默认等于 ``query_sample_seed``（与 ``chain_mean_cost_latency`` / budget 脚本口径一致）。
    """
    sizes = sample_source_sizes_gb(num_queries=num_queries, seed=query_sample_seed)
    mean_rho = cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=n_calibration_samples,
        rng=random.Random(query_sample_seed + 100),
        operations=("shot_detection", "video_split", "video_caption", "query"),
    )
    ev = int(chain_eval_seed) if chain_eval_seed is not None else int(query_sample_seed)
    return find_wf1_mean_min_cost_and_latency_chains(cands, sizes, mean_rho, eval_seed=ev)


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

    t0 = time.perf_counter()
    best_c_chain, best_l_chain = find_wf1_mean_min_cost_and_latency_chains(
        cands, sizes, mean_rho, eval_seed=SEED
    )
    best_c_mean, _ = chain_mean_cost_latency(best_c_chain, sizes, mean_rho, eval_seed=SEED)
    _, best_l_mean = chain_mean_cost_latency(best_l_chain, sizes, mean_rho, eval_seed=SEED)

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
