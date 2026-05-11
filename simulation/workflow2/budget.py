"""
在 plug-in mean ρ 下，用 N 条随机源大小，枚举部署链集合，对每条链计算端到端 mean-field cost / latency
的样本均值，输出 **mean cost 最小**与 **mean latency 最小** 的链。

``caption`` 路径若做「全候选层笛卡尔积」链数约 4e6，在 100 条 query 上全量枚举耗时过长；
默认采用 **单云全组合**：对 GCP / AWS / Aliyun 分别做各层 ``provider`` 过滤后的笛卡尔积，再合并
（与「每朵云内跨 region/model 的全组合」一致，仍覆盖多 region）。

全量跨云枚举可将 ``WF2_FULL_CROSS_PRODUCT = True``（仅建议在路径很短或能接受长时间运行时开启）。

运行::

    python -m workflow2.budget
    python -m workflow2.budget ocr
"""

from __future__ import annotations

import itertools
import math
import random
import sys
import time
from typing import Iterable, Sequence

from sim_env import config as cfg
from sim_env.utility import QueryProfile

from workflow1 import utils as wf1_utils

from .candidates import enumerate_candidates_wf2
from .utils import (
    WF2PathId,
    WF2PhysicalNode,
    end_to_end_cost_exclusive_path,
    end_to_end_latency_exclusive_path,
    plugin_mean_data_conversion_ratios_wf2,
    validate_exclusive_path_nodes,
)

N_QUERIES = 100
SEED = 42
N_CALIBRATION_SAMPLES = 4096
# ``caption`` 全笛卡尔积 ~4234032；设为 True 则枚举全部（极慢）
WF2_FULL_CROSS_PRODUCT = False

_SINGLE_CLOUD_PROVIDERS = ("GCP", "AWS", "Aliyun")


def sample_source_sizes_gb(*, num_queries: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        out.append(s_src_mb / 1000.0)
    return out


def iter_chains_wf2(
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    path_id: WF2PathId,
    full_cross_product: bool,
) -> Iterable[tuple[WF2PhysicalNode, ...]]:
    L = len(cands)
    layers = [list(cands[i]) for i in range(L)]
    if full_cross_product:
        for tup in itertools.product(*layers):
            yield tuple(tup)
        return

    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS:
        prov_layers: list[list[WF2PhysicalNode]] = []
        ok = True
        for i in range(L):
            layer = [n for n in layers[i] if n.provider == prov]
            if not layer:
                ok = False
                break
            prov_layers.append(layer)
        if not ok:
            continue
        for tup in itertools.product(*prov_layers):
            ch = tuple(tup)
            seen.add(ch)
    for ch in sorted(seen, key=lambda c: tuple((n.operation, n.provider, n.region, n.model or "") for n in c)):
        validate_exclusive_path_nodes(path_id, ch)
        yield ch


def count_chains_wf2(
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    full_cross_product: bool,
) -> int:
    if full_cross_product:
        return math.prod(len(cands[i]) for i in range(len(cands)))
    L = len(cands)
    layers = [list(cands[i]) for i in range(L)]
    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS:
        prov_layers: list[list[WF2PhysicalNode]] = []
        ok = True
        for i in range(L):
            layer = [n for n in layers[i] if n.provider == prov]
            if not layer:
                ok = False
                break
            prov_layers.append(layer)
        if not ok:
            continue
        for tup in itertools.product(*prov_layers):
            seen.add(tuple(tup))
    return len(seen)


def chain_mean_cost_latency_wf2(
    path_id: WF2PathId,
    chain: Sequence[WF2PhysicalNode],
    sizes_gb: Sequence[float],
    mean_rho: tuple[float, ...],
    *,
    eval_seed: int,
) -> tuple[float, float]:
    ch = tuple(chain)
    tot_c = 0.0
    tot_l = 0.0
    n = len(sizes_gb)
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf1_utils.det_rng(eval_seed, "wf2_budget_combo_eval", q_idx)
        llm_seed = wf.randrange(0, 2**31)
        c = end_to_end_cost_exclusive_path(
            path_id,
            ch,
            float(s_gb),
            mean_rho,
            llm_token_rng_seed=llm_seed,
        )
        ell = end_to_end_latency_exclusive_path(
            path_id,
            ch,
            float(s_gb),
            mean_rho,
            llm_token_rng_seed=llm_seed,
            env_rng=wf,
            execution_scale_scope=str(llm_seed),
            execution_scale_seed=llm_seed,
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
    *,
    eval_seed: int,
) -> list[QueryProfile]:
    ch = tuple(chain)
    out: list[QueryProfile] = []
    for q_idx, s_gb in enumerate(sizes_gb):
        wf = wf1_utils.det_rng(eval_seed, "wf2_budget_profile", q_idx)
        llm_seed = wf.randrange(0, 2**31)
        c = end_to_end_cost_exclusive_path(
            path_id,
            ch,
            float(s_gb),
            mean_rho,
            llm_token_rng_seed=llm_seed,
        )
        ell = end_to_end_latency_exclusive_path(
            path_id,
            ch,
            float(s_gb),
            mean_rho,
            llm_token_rng_seed=llm_seed,
            env_rng=wf,
            execution_scale_scope=str(llm_seed),
            execution_scale_seed=llm_seed,
        )
        out.append(
            QueryProfile(
                s_src_gb=float(s_gb),
                theta_cost=float(c),
                theta_latency_sec=float(ell),
            )
        )
    return out


def run_search(
    path_id: WF2PathId,
    *,
    full_cross: bool = WF2_FULL_CROSS_PRODUCT,
    verbose_every: int = 2000,
) -> None:
    sizes = sample_source_sizes_gb(num_queries=N_QUERIES, seed=SEED)
    mean_rho = plugin_mean_data_conversion_ratios_wf2(
        path_id,
        n_calibration_samples=N_CALIBRATION_SAMPLES,
        rng=random.Random(SEED + 100),
    )
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
            path_id, chain, sizes, mean_rho, eval_seed=SEED
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
            path_id, best_l_chain, sizes, mean_rho, eval_seed=SEED
        )
        print(f"  （latency 最优链的 mean cost = {mc:.6g} USD）")
    if best_c_chain is not None:
        _, ml = chain_mean_cost_latency_wf2(
            path_id, best_c_chain, sizes, mean_rho, eval_seed=SEED
        )
        print(f"  （cost 最优链的 mean latency = {ml:.6g} s）")

    print(f"\n  总耗时 {elapsed:.1f}s")


def main() -> None:
    path: WF2PathId = "caption"
    if len(sys.argv) > 1:
        cand = sys.argv[1]
        if cand not in ("caption", "ocr", "label", "speech"):
            print(f"unknown path {cand!r}, use caption|ocr|label|speech", file=sys.stderr)
            sys.exit(2)
        path = cand  # type: ignore[assignment]
    run_search(path)


if __name__ == "__main__":
    main()
