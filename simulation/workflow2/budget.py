"""
在 plug-in mean ρ 下，用 N 条随机源大小，枚举部署链集合，对每条链计算 **mean-field** cost / latency
的样本均值，输出 **mean cost 最小**与 **mean latency 最小** 的链。

对 ``video_caption`` / ``ocr`` / ``label_detection`` 主干，费用与 workflow DAG 一致：
**各并行 multimodal 分支费用相加**；时延为 **shot→split + max(并行支路→db) + db→qa + 下行**
（见 ``end_to_end_cost_parallel_shot_modalities`` /
``end_to_end_latency_parallel_shot_modalities``）。并行分支上 ``video_caption`` 的物理节点固定为
workflow1 budget（mean cost）参考链上的 caption 节点；``ocr``/``label_detection`` 与 WF1 的
``video_split`` 同 region 对齐。

``video_caption`` 路径若做「全候选层笛卡尔积」链数约 4e6，在 100 条 query 上全量枚举耗时过长；
默认采用 **单云全组合**：对 GCP / AWS / Aliyun 分别做各层 ``provider`` 过滤后的笛卡尔积，再合并
（与「每朵云内跨 region/model 的全组合」一致，仍覆盖多 region）。

全量跨云枚举可将 ``workflow2.utils.WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT = True``

运行::

    python -m workflow2.budget              # 先跑 shot 并行 DAG（端到端），再跑 speech
    python -m workflow2.budget all            # 同上
    python -m workflow2.budget video_caption  # 仅 shot 并行 DAG（与 ocr/label 同一枚举）
    python -m workflow2.budget speech_transcription  # 仅 speech 三节点链
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import Sequence

from sim_env import config as cfg
from sim_env.utility import QueryProfile

from workflow1 import utils as wf1_utils

from .candidates import enumerate_candidates_wf2
from .utils import (
    WF2PathId,
    WF2PhysicalNode,
    WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT,
    count_chains_wf2,
    count_chains_wf2_end_to_end_shot_modalities,
    iter_chains_wf2,
    iter_chains_wf2_end_to_end_shot_modalities,
    plugin_mean_data_conversion_ratios_wf2,
    wf2_budget_meanfield_cost_latency,
    wf2_modality_nodes_from_wf1_physical_ref,
    wf2_parallel_budget_mean_rhos,
)

N_QUERIES = 100
SEED = 42
N_CALIBRATION_SAMPLES = 4096
# ``video_caption`` 全笛卡尔积 ~4234032；可调 ``workflow2.utils.WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT``
WF2_FULL_CROSS_PRODUCT = WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT


def sample_source_sizes_gb(*, num_queries: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        out.append(s_src_mb / 1000.0)
    return out


def chain_mean_cost_latency_wf2(
    path_id: WF2PathId,
    chain: Sequence[WF2PhysicalNode],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, ...],
    parallel_rho: tuple[float, float, dict[str, float], float, float] | None,
    modality_nodes: dict[str, WF2PhysicalNode] | None,
    *,
    eval_seed: int,
) -> tuple[float, float]:
    ch = tuple(chain)
    tot_c = 0.0
    tot_l = 0.0
    n = len(sizes_gb)
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf1_utils.det_rng(eval_seed, "wf2_budget_combo_eval", q_idx)
        c, ell = wf2_budget_meanfield_cost_latency(
            path_id,
            ch,
            float(s_gb),
            mean_rho_exclusive=mean_rho,
            parallel_rho=parallel_rho,
            modality_nodes=modality_nodes,
            workflow_rng=wf,
        )
        if not (math.isfinite(c) and math.isfinite(ell)):
            return float("inf"), float("inf")
        tot_c += float(c)
        tot_l += float(ell)
    inv = 1.0 / float(n)
    return tot_c * inv, tot_l * inv


def query_profiles_from_chain_wf2(
    path_id: WF2PathId,
    chain: Sequence[WF2PhysicalNode],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, ...],
    parallel_rho: tuple[float, float, dict[str, float], float, float] | None,
    modality_nodes: dict[str, WF2PhysicalNode] | None,
    *,
    eval_seed: int,
) -> list[QueryProfile]:
    ch = tuple(chain)
    out: list[QueryProfile] = []
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf1_utils.det_rng(eval_seed, "wf2_budget_profile", q_idx)
        c, ell = wf2_budget_meanfield_cost_latency(
            path_id,
            ch,
            float(s_gb),
            mean_rho_exclusive=mean_rho,
            parallel_rho=parallel_rho,
            modality_nodes=modality_nodes,
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


def run_search_shot_modalities_end_to_end(
    *,
    path_label: WF2PathId,
    full_cross: bool = WF2_FULL_CROSS_PRODUCT,
    verbose_every: int = 2000,
) -> None:
    """在 **并行多模态 DAG**（固定三模态支路 + WF1 参考）下仅枚举 shot/split/db/qa。``path_label`` 用于 ρ 插件与打印。"""
    if path_label == "speech_transcription":
        raise ValueError("use run_search_speech for speech path")
    sizes = sample_source_sizes_gb(num_queries=N_QUERIES, seed=SEED)
    mean_rho = plugin_mean_data_conversion_ratios_wf2(
        path_label,
        n_calibration_samples=N_CALIBRATION_SAMPLES,
        rng=random.Random(SEED + 100),
    )
    parallel_rho = wf2_parallel_budget_mean_rhos(
        n_calibration_samples=N_CALIBRATION_SAMPLES,
        rng=random.Random(SEED + 100),
    )
    modality_nodes = wf2_modality_nodes_from_wf1_physical_ref(
        wf1_utils._BUDGET_REF_CHAIN_COST_WF1
    )
    cands_vc = enumerate_candidates_wf2("video_caption")
    layer_sizes = [len(cands_vc[i]) for i in range(5)]
    full_n = math.prod(len(cands_vc[i]) for i in (0, 1, 3, 4))
    n_chains = count_chains_wf2_end_to_end_shot_modalities(
        cands_vc, full_cross_product=full_cross
    )

    print(f"workflow2.budget | end-to-end parallel DAG | path_label={path_label!r}")
    print(f"  plug-in mean ρ samples={N_CALIBRATION_SAMPLES}")
    print(f"  queries={N_QUERIES}  eval_seed={SEED}")
    print(f"  video_caption.layers={layer_sizes}  enumerating_layers=[0,1,3,4] product={full_n}")
    print(
        f"  enumeration: {'FULL cross-product' if full_cross else 'single-cloud union (dedup)'} "
        f"| distinct (shot,split,db,qa) combos={n_chains}"
    )

    best_c_mean = float("inf")
    best_l_mean = float("inf")
    best_c_chain: tuple[WF2PhysicalNode, ...] | None = None
    best_l_chain: tuple[WF2PhysicalNode, ...] | None = None

    t0 = time.perf_counter()
    for idx, chain in enumerate(
        iter_chains_wf2_end_to_end_shot_modalities(
            cands_vc, full_cross_product=full_cross
        ),
        start=1,
    ):
        if verbose_every and idx % verbose_every == 0:
            print(
                f"  ... scanned {idx}/{n_chains} combos ({time.perf_counter() - t0:.1f}s)",
                file=sys.stderr,
            )
        m_c, m_l = chain_mean_cost_latency_wf2(
            path_label,
            chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        if m_c < best_c_mean - 1e-15:
            best_c_mean = m_c
            best_c_chain = tuple(chain)
        if m_l < best_l_mean - 1e-15:
            best_l_mean = m_l
            best_l_chain = tuple(chain)

    elapsed = time.perf_counter() - t0
    print(f"\n=== workflow2 端到端并行 DAG（path_ref={path_label}）：mean cost 最小的 (shot,split,db,qa) ===")
    print(f"  mean cost (USD): {best_c_mean:.6g}")
    print(f"  链: {best_c_chain}")

    print(f"\n=== workflow2 端到端并行 DAG（path_ref={path_label}）：mean latency 最小的 (shot,split,db,qa) ===")
    print(f"  mean latency (s): {best_l_mean:.6g}")
    print(f"  链: {best_l_chain}")

    if best_l_chain is not None:
        mc, _ = chain_mean_cost_latency_wf2(
            path_label,
            best_l_chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        print(f"  （latency 最优组合的 mean cost = {mc:.6g} USD）")
    if best_c_chain is not None:
        _, ml = chain_mean_cost_latency_wf2(
            path_label,
            best_c_chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        print(f"  （cost 最优组合的 mean latency = {ml:.6g} s）")

    print(f"\n  总耗时 {elapsed:.1f}s")


def run_search_speech(
    *,
    full_cross: bool = WF2_FULL_CROSS_PRODUCT,
    verbose_every: int = 2000,
) -> None:
    sizes = sample_source_sizes_gb(num_queries=N_QUERIES, seed=SEED)
    mean_rho = plugin_mean_data_conversion_ratios_wf2(
        "speech_transcription",
        n_calibration_samples=N_CALIBRATION_SAMPLES,
        rng=random.Random(SEED + 100),
    )
    parallel_rho: tuple[float, float, dict[str, float], float, float] | None = None
    modality_nodes: dict[str, WF2PhysicalNode] | None = None
    path_id: WF2PathId = "speech_transcription"
    cands = enumerate_candidates_wf2(path_id)
    layer_sizes = [len(cands[i]) for i in range(len(cands))]
    full_n = math.prod(layer_sizes)
    n_chains = count_chains_wf2(cands, full_cross_product=full_cross)

    print(f"workflow2.budget | path={path_id!r}")
    print(f"  plug-in mean ρ samples={N_CALIBRATION_SAMPLES}")
    print(f"  queries={N_QUERIES}  eval_seed={SEED}")
    print(f"  layers={layer_sizes}  full_cross_product_size={full_n}")
    print(
        f"  enumeration: {'FULL cross-product' if full_cross else 'single-cloud union (dedup)'} "
        f"| distinct chains={n_chains}"
    )

    best_c_mean = float("inf")
    best_l_mean = float("inf")
    best_c_chain: tuple[WF2PhysicalNode, ...] | None = None
    best_l_chain: tuple[WF2PhysicalNode, ...] | None = None

    t0 = time.perf_counter()
    for idx, chain in enumerate(
        iter_chains_wf2(cands, path_id=path_id, full_cross_product=full_cross), start=1
    ):
        if verbose_every and idx % verbose_every == 0:
            print(
                f"  ... scanned {idx}/{n_chains} chains ({time.perf_counter() - t0:.1f}s)",
                file=sys.stderr,
            )
        m_c, m_l = chain_mean_cost_latency_wf2(
            path_id,
            chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        if m_c < best_c_mean - 1e-15:
            best_c_mean = m_c
            best_c_chain = tuple(chain)
        if m_l < best_l_mean - 1e-15:
            best_l_mean = m_l
            best_l_chain = tuple(chain)

    elapsed = time.perf_counter() - t0
    print(f"\n=== workflow2 ({path_id})：mean cost 最小的部署链 ===")
    print(f"  mean cost (USD): {best_c_mean:.6g}")
    print(f"  链: {best_c_chain}")

    print(f"\n=== workflow2 ({path_id})：mean latency 最小的部署链 ===")
    print(f"  mean latency (s): {best_l_mean:.6g}")
    print(f"  链: {best_l_chain}")

    if best_l_chain is not None:
        mc, _ = chain_mean_cost_latency_wf2(
            path_id,
            best_l_chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        print(f"  （latency 最优链的 mean cost = {mc:.6g} USD）")
    if best_c_chain is not None:
        _, ml = chain_mean_cost_latency_wf2(
            path_id,
            best_c_chain,
            sizes,
            mean_rho,
            parallel_rho,
            modality_nodes,
            eval_seed=SEED,
        )
        print(f"  （cost 最优链的 mean latency = {ml:.6g} s）")

    print(f"\n  总耗时 {elapsed:.1f}s")


def main() -> None:
    cand = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else "all")
    if cand in ("all", "end-to-end", "e2e"):
        run_search_shot_modalities_end_to_end(path_label="video_caption")
        print()
        run_search_speech()
        return
    if cand == "speech_transcription":
        run_search_speech()
        return
    if cand in ("video_caption", "ocr", "label_detection"):
        run_search_shot_modalities_end_to_end(path_label=cand)  # type: ignore[arg-type]
        return
    print(
        f"unknown arg {cand!r}; use all | video_caption | ocr | label_detection | speech_transcription",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
