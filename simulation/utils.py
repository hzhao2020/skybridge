"""
End-to-end workflow metrics (cost, latency, utility) for four logical operations
(segment → split → caption → query) with one physical node per step.

Implements the aggregation in the SkyXXX paper sketch:
  - Cost: execution + storage(S_in + S_out) + Σ edge transfer volume × unit egress
  - Latency: critical path = sum of execution times + per-edge (data/B + RTT/2)
  - Utility: U = Σ_i w_i μ_i (default weights all 1 per user request)

Data volumes use the chain form of Eq. (data propagation): inputs S_i and
stochastic ratios ρ_i; S_out,i = S_i · ρ_i, and S_{i+1} = S_out,i for a pipeline.
"""

from __future__ import annotations

import math
import random
from typing import Sequence, Tuple

from sim_env import config as cfg
from sim_env.config import video_duration_sec_from_megabytes
from sim_env.cost import ProviderRegion, egress_cost_usd, llm_token_cost_usd, split_cost_usd, storage_cost_usd, video_service_cost_usd
from sim_env.execution_latency import llm_decode_duration_sec
from sim_env.llm import caption_visual_input_tokens, sample_caption_output_tokens, sample_query_output_tokens
from sim_env.network import NetworkSample, sample_link
from sim_env.utility import OperationName, PhysicalNode, QueryProfile, physical_node_utility

_OPS_ORDER: tuple[OperationName, OperationName, OperationName, OperationName] = (
    "segment",
    "split",
    "caption",
    "query",
)


def _endpoint(node: PhysicalNode) -> ProviderRegion:
    return (node.provider, node.region)


def _propagate_sizes_gb(
    s_src_gb: float,
    rho: tuple[float, float, float, float],
) -> tuple[list[float], list[float]]:
    """Per-node input sizes (GB) and per-edge transfer payloads (GB): output of upstream."""
    if s_src_gb <= 0:
        raise ValueError("s_src_gb must be positive")
    s_in: list[float] = [0.0] * 4
    s_in[0] = float(s_src_gb)
    for i in range(1, 4):
        s_in[i] = s_in[i - 1] * float(rho[i - 1])
    transfer_gb = [s_in[i] * float(rho[i]) for i in range(4)]  # includes final output; edges use first 3
    return s_in, transfer_gb


def _effective_bytes_per_sec(sample: NetworkSample) -> float:
    """Convert measured Mbit/s bottleneck to bytes/s (paper uses data/B)."""
    mbps = float(sample.bandwidth_effective_mbits_per_sec)
    if math.isinf(mbps) or mbps <= 0:
        return math.inf
    return (mbps * 1_000_000.0) / 8.0


def _transfer_seconds(payload_gb: float, sample: NetworkSample) -> float:
    if payload_gb <= 0:
        return 0.0
    bps = _effective_bytes_per_sec(sample)
    if math.isinf(bps):
        return 0.0
    nbytes = payload_gb * 1_000_000_000.0
    return nbytes / bps


def _segment_video_minutes(s_in_gb: float) -> float:
    megabytes = float(s_in_gb) * 1000.0
    duration_sec = video_duration_sec_from_megabytes(megabytes)
    return duration_sec / 60.0


def _resolve_llm_tokens(
    segment_video_minutes: float,
    *,
    caption_tokens: tuple[float, float] | None,
    query_tokens: tuple[float, float] | None,
    rng_seed: int | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    duration_sec = max(segment_video_minutes * 60.0, 1e-6)

    if caption_tokens is not None:
        ct = caption_tokens
    else:
        rng = rng_seed
        cin = caption_visual_input_tokens(duration_sec)
        cout = sample_caption_output_tokens(duration_sec, rng=rng)
        ct = (cin, cout)

    if query_tokens is not None:
        qt = query_tokens
    else:
        import numpy as np

        g = np.random.default_rng(int(rng_seed) + 1 if rng_seed is not None else None)
        qout = sample_query_output_tokens(rng=g)
        qt = (ct[1], qout)

    return ct, qt


def _segment_exec_seconds(segment_video_minutes: float, segment_exe_sec: float | None) -> float:
    if segment_exe_sec is not None:
        return float(segment_exe_sec)
    try:
        from execution_latency import sample_segment_execute_sec

        duration_sec = max(segment_video_minutes * 60.0, 1e-6)
        return sample_segment_execute_sec(duration_sec)
    except Exception:
        return float(segment_video_minutes * 60.0) * 0.02


def _split_exec_seconds(segment_video_minutes: float, split_exe_sec: float | None) -> float:
    if split_exe_sec is not None:
        return float(split_exe_sec)
    try:
        from execution_latency import sample_split_execute_sec

        duration_sec = max(segment_video_minutes * 60.0, 1e-6)
        return sample_split_execute_sec(duration_sec)
    except Exception:
        return 30.0


def end_to_end_cost(
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    s_src_gb: float,
    rho: tuple[float, float, float, float],
    *,
    caption_tokens: tuple[float, float] | None = None,
    query_tokens: tuple[float, float] | None = None,
    llm_token_rng_seed: int | None = 42,
) -> float:
    """
    Paper-style total cost: Σ execution + Σ storage(S_i + S_i ρ_i) + Σ egress over edges.

    Uses cloud price tables in ``sim_env.cost``; LLM billable tokens default to
    ``sim_env.llm`` estimators when ``caption_tokens`` / ``query_tokens`` are omitted.
    """
    for i, n in enumerate(nodes):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")

    s_in, transfer_gb = _propagate_sizes_gb(s_src_gb, rho)
    seg_min = _segment_video_minutes(s_in[0])
    (cap_in, cap_out), (q_in, q_out) = _resolve_llm_tokens(
        seg_min,
        caption_tokens=caption_tokens,
        query_tokens=query_tokens,
        rng_seed=llm_token_rng_seed,
    )

    exe = 0.0
    exe += video_service_cost_usd(nodes[0].provider, nodes[0].region, "segment", seg_min)
    exe += split_cost_usd(nodes[1].provider, nodes[1].region, minutes=1.0)

    cap_model = nodes[2].model
    q_model = nodes[3].model
    if not cap_model or not q_model:
        raise ValueError("caption and query nodes require model names")
    exe += llm_token_cost_usd(nodes[2].provider, nodes[2].region, cap_model, cap_in, cap_out)
    exe += llm_token_cost_usd(nodes[3].provider, nodes[3].region, q_model, q_in, q_out)

    stor = 0.0
    for i in range(4):
        gb = s_in[i] * (1.0 + float(rho[i]))
        stor += storage_cost_usd(nodes[i].provider, nodes[i].region, gb, days=1.0)

    net = 0.0
    for i in range(3):
        src, dst = _endpoint(nodes[i]), _endpoint(nodes[i + 1])
        net += egress_cost_usd(src, dst, transfer_gb[i])

    return exe + stor + net


def end_to_end_latency(
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    s_src_gb: float,
    rho: tuple[float, float, float, float],
    *,
    t_exe: tuple[float | None, float | None, float | None, float | None] | None = None,
    caption_output_tokens: float | None = None,
    query_output_tokens: float | None = None,
    network_samples: tuple[NetworkSample, NetworkSample, NetworkSample] | None = None,
    llm_token_rng_seed: int | None = 42,
) -> float:
    """
    Paper-style end-to-end latency for a linear DAG = single path (max over paths = path sum):

      Σ T_k^exe + Σ ( S_i ρ_i / B + RTT/2 )

    Same-endpoint links use zero network penalty (``sample_link`` / infinite B, 0 RTT).
    Execution times default to sampled segment/split and LLM decode from token counts.
    """
    for i, n in enumerate(nodes):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")

    s_in, transfer_gb = _propagate_sizes_gb(s_src_gb, rho)
    seg_min = _segment_video_minutes(s_in[0])

    te = t_exe if t_exe is not None else (None, None, None, None)
    t0 = _segment_exec_seconds(seg_min, te[0])
    t1 = _split_exec_seconds(seg_min, te[1])

    cap_pair, q_pair = _resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_token_rng_seed,
    )
    cap_out = float(caption_output_tokens) if caption_output_tokens is not None else cap_pair[1]
    q_out = float(query_output_tokens) if query_output_tokens is not None else q_pair[1]

    cap_model = nodes[2].model
    q_model = nodes[3].model
    if not cap_model or not q_model:
        raise ValueError("caption and query nodes require model names")
    t2 = te[2] if te[2] is not None else llm_decode_duration_sec(cap_model, cap_out)
    t3 = te[3] if te[3] is not None else llm_decode_duration_sec(q_model, q_out)

    lat = t0 + t1 + t2 + t3

    if network_samples is None:
        samples = tuple(
            sample_link(_endpoint(nodes[i]), _endpoint(nodes[i + 1])) for i in range(3)
        )
    else:
        samples = network_samples

    for i in range(3):
        pay = transfer_gb[i]
        rtt_s = samples[i].rtt_ms / 1000.0
        lat += _transfer_seconds(pay, samples[i]) + 0.5 * rtt_s

    return lat


def end_to_end_utility(
    nodes: Sequence[PhysicalNode],
    *,
    weights: Tuple[float, float, float, float] | None = None,
) -> float:
    """
    U(x) = Σ_i w_i · μ_i with μ from ``physical_node_utility``. Default weights are all 1
    (paper uses normalized Σ w_i = 1; set ``weights=(0.25,)*4`` if you need that).
    """
    ns = tuple(nodes)
    if len(ns) != 4:
        raise ValueError(f"expected 4 PhysicalNode, got {len(ns)}")
    for i, n in enumerate(ns):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")
    w = weights if weights is not None else (1.0, 1.0, 1.0, 1.0)
    if len(w) != 4:
        raise ValueError("weights must have length 4")
    return sum(w[i] * physical_node_utility(ns[i]) for i in range(4))


def generate_realistic_queries(num_queries: int, seed: int = 42) -> list[QueryProfile]:
    """
    Generate test queries whose budgets ``(theta_cost, theta_latency_sec)`` are scaled
    from reference cost/latency on a fixed **cross-cloud, cross-region** baseline pipeline
    (long-haul edges → higher ref latency) plus random tightness factors.

    生成符合物理规律的测试 Query：预算锚定在跨云跨区基准链路，再按比例扰动。
    """
    queries: list[QueryProfile] = []

    # Long-haul baseline: each hop crosses provider and/or geography so reference
    # cost/latency (and thus query SLO scales) reflects expensive cross-region edges.
    baseline_nodes = (
        PhysicalNode("segment", "GCP", "europe-west1"),
        PhysicalNode("split", "AWS", "ap-southeast-1"),
        PhysicalNode("caption", "Aliyun", "cn-beijing", "Qwen3-VL-Plus"),
        PhysicalNode("query", "GCP", "us-east1", "Gemini 2.5 Pro"),
    )

    calib_rng = random.Random(seed)
    mean_rho = cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=4096,
        rng=calib_rng,
        operations=("segment", "split", "caption", "query"),
    )

    rng = random.Random(seed)

    for _ in range(num_queries):
        duration_sec = rng.uniform(60.0, 1800.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        s_src_gb = s_src_mb / 1000.0

        ref_cost = end_to_end_cost(
            baseline_nodes, s_src_gb, mean_rho, llm_token_rng_seed=42
        )
        ref_lat = end_to_end_latency(
            baseline_nodes, s_src_gb, mean_rho, llm_token_rng_seed=42
        )

        factor_c = rng.uniform(0.8, 1.5)
        factor_t = rng.uniform(0.8, 1.5)
        if rng.random() < 0.1:
            factor_c = rng.uniform(0.6, 0.8)
            factor_t = rng.uniform(0.6, 0.8)

        queries.append(
            QueryProfile(
                s_src_gb=s_src_gb,
                theta_cost=ref_cost * factor_c,
                theta_latency_sec=ref_lat * factor_t,
            )
        )

    return queries
