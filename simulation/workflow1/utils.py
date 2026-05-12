"""
End-to-end workflow metrics (cost, latency, utility) for four logical operations
(segment → split → caption → query) with one physical node per step.

Implements the aggregation in the SkyXXX paper sketch:
  - Cost: execution + storage(S_in + S_out) + Σ edge transfer volume × unit egress
  - Latency: critical path = sum of execution times + per-edge (data/B + RTT/2)
  - Utility: U = Σ_i w_i μ_i (default weights sum to 1, each 0.25)

Data volumes use the chain form of Eq. (data propagation): inputs S_i and
stochastic ratios ρ_i; S_out,i = S_i · ρ_i, and S_{i+1} = S_out,i for a pipeline.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Literal, Sequence, Tuple

from sim_env import config as cfg
from sim_env.config import video_duration_sec_from_megabytes
from sim_env.cost import ProviderRegion, egress_cost_usd, llm_token_cost_usd, split_cost_usd, storage_cost_usd, video_service_cost_usd
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    sample_segment_execute_sec,
    sample_split_execute_sec,
)
from sim_env.llm import caption_visual_input_tokens, llm_tokens_from_data_payload_gb
from sim_env.network import (
    LOCAL_PROVIDER,
    LOCAL_REGION,
    NetworkSample,
    reset_link_counters,
    sample_link,
)
from sim_env.utility import OperationName, PhysicalNode, QueryProfile, physical_node_utility

_OPS_ORDER: tuple[OperationName, OperationName, OperationName, OperationName] = (
    "segment",
    "split",
    "caption",
    "query",
)

_DEFAULT_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

# 由 ``python -m workflow1.budget`` 得到（N_QUERIES=100, SEED=42，plug-in mean ρ，全链枚举）；
# mean cost 最小链与 mean latency 最小链在此配置下相同。
_BUDGET_REF_CHAIN_COST_WF1: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode] = (
    PhysicalNode("segment", "Aliyun", "cn-shanghai", None),
    PhysicalNode("split", "Aliyun", "cn-shanghai", None),
    PhysicalNode("caption", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    PhysicalNode("query", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
)
_BUDGET_REF_CHAIN_LATENCY_WF1 = _BUDGET_REF_CHAIN_COST_WF1
# 每条 query 的 Θ 相对参考链上 mean-field 指标放大该倍数（与 budget 脚本口径一致）。
QUERY_BUDGET_REFERENCE_MULTIPLIER = 1.5


def det_rng(master: int, *parts: int | str) -> random.Random:
    """
    Deterministic ``random.Random`` (stable across processes; not Python's salted ``hash()``).
    Used for network rotations and other keyed draws tied to a scenario seed.
    """
    buf = hashlib.sha256(str(master).encode("utf-8"))
    for p in parts:
        buf.update(b"|")
        buf.update(str(p).encode("utf-8"))
        buf.update(b"\x1e")
    return random.Random(int.from_bytes(buf.digest()[:8], "big"))


def _endpoint(node: PhysicalNode) -> ProviderRegion:
    return (node.provider, node.region)


_LOCAL_ENDPOINT: ProviderRegion = (LOCAL_PROVIDER, LOCAL_REGION)


def local_edge_cost_latency(
    cloud_endpoint: ProviderRegion,
    payload_gb: float,
    *,
    direction: Literal["upload", "download"],
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """
    本地与云上首/末跳：``upload`` = Local→cloud，``download`` = cloud→Local。
    返回 (出口流量费 USD, 传输+RTT/2 秒)。
    """
    if direction == "upload":
        ep_s, ep_d = _LOCAL_ENDPOINT, cloud_endpoint
    else:
        ep_s, ep_d = cloud_endpoint, _LOCAL_ENDPOINT
    cents = egress_cost_usd(ep_s, ep_d, payload_gb)
    reset_link_counters([(ep_s, ep_d)])
    sm = sample_link(ep_s, ep_d, rng=rng)
    lat = _transfer_seconds(payload_gb, sm) + 0.5 * (sm.rtt_ms / 1000.0)
    return cents, lat


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
    caption_output_payload_gb: float | None = None,
    query_output_payload_gb: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    duration_sec = max(segment_video_minutes * 60.0, 1e-6)

    if caption_tokens is not None:
        ct = caption_tokens
    else:
        if caption_output_payload_gb is None:
            raise ValueError(
                "caption_output_payload_gb is required when caption_tokens is omitted"
            )
        cin = caption_visual_input_tokens(duration_sec)
        cout = llm_tokens_from_data_payload_gb(caption_output_payload_gb)
        ct = (cin, cout)

    if query_tokens is not None:
        qt = query_tokens
    else:
        if query_output_payload_gb is None:
            raise ValueError(
                "query_output_payload_gb is required when query_tokens is omitted"
            )
        if caption_output_payload_gb is not None:
            qin = llm_tokens_from_data_payload_gb(caption_output_payload_gb)
        else:
            qin = ct[1]
        qout = llm_tokens_from_data_payload_gb(query_output_payload_gb)
        qt = (qin, qout)

    return ct, qt


def _segment_exec_seconds(
    segment_video_minutes: float,
    segment_exe_sec: float | None,
    *,
    rng: random.Random | None = None,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    if segment_exe_sec is not None:
        return float(segment_exe_sec)
    duration_sec = max(segment_video_minutes * 60.0, 1e-6)
    return sample_segment_execute_sec(
        duration_sec,
        rng=rng,
        node=node,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )


def _split_exec_seconds(
    segment_video_minutes: float,
    split_exe_sec: float | None,
    *,
    rng: random.Random | None = None,
    node: PhysicalNode | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    if split_exe_sec is not None:
        return float(split_exe_sec)
    duration_sec = max(segment_video_minutes * 60.0, 1e-6)
    return sample_split_execute_sec(
        duration_sec,
        rng=rng,
        node=node,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )


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
    Paper-style total cost: Σ execution + Σ storage(S_i + S_i ρ_i) + Σ egress over edges,
    plus **Local→首云** 上传与 **末云→Local** 回传最终输出。

    Uses cloud price tables in ``sim_env.cost``. When tokenizer overrides are omitted,
    caption/query text tokens follow ``llm_tokens_from_data_payload_gb`` from chain payloads.
    """
    for i, n in enumerate(nodes):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")

    s_in, transfer_gb = _propagate_sizes_gb(s_src_gb, rho)
    seg_min = _segment_video_minutes(s_in[0])
    cap_gb = float(s_in[2]) * float(rho[2])
    q_out_gb = float(s_in[3]) * float(rho[3])
    (cap_in, cap_out), (q_in, q_out) = _resolve_llm_tokens(
        seg_min,
        caption_tokens=caption_tokens,
        query_tokens=query_tokens,
        rng_seed=llm_token_rng_seed,
        caption_output_payload_gb=cap_gb,
        query_output_payload_gb=q_out_gb,
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
        stor += storage_cost_usd(nodes[i].provider, nodes[i].region, gb, hours=1.0)

    net = 0.0
    net += egress_cost_usd(_LOCAL_ENDPOINT, _endpoint(nodes[0]), float(s_in[0]))
    for i in range(3):
        src, dst = _endpoint(nodes[i]), _endpoint(nodes[i + 1])
        net += egress_cost_usd(src, dst, transfer_gb[i])
    net += egress_cost_usd(_endpoint(nodes[3]), _LOCAL_ENDPOINT, float(transfer_gb[3]))

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
    env_rng: random.Random | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> float:
    """
    Paper-style end-to-end latency for a linear DAG = single path (max over paths = path sum):

      Σ T_k^exe + Σ ( S_i ρ_i / B + RTT/2 )

    Same-endpoint links use zero network penalty (``sample_link`` / infinite B, 0 RTT).

    ``execution_scale_scope`` + ``execution_scale_seed``: deterministic segment/split factors
    ``k`` via ``node_execution_scale_k`` (omit seed to use RNG+cache path inside ``sample_*``).

    Execution times default to sampled segment/split and LLM decode from token counts.
    """
    for i, n in enumerate(nodes):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")

    s_in, transfer_gb = _propagate_sizes_gb(s_src_gb, rho)
    seg_min = _segment_video_minutes(s_in[0])

    te = t_exe if t_exe is not None else (None, None, None, None)
    t0 = _segment_exec_seconds(
        seg_min,
        te[0],
        rng=env_rng,
        node=nodes[0],
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )
    t1 = _split_exec_seconds(
        seg_min,
        te[1],
        rng=env_rng,
        node=nodes[1],
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )

    if caption_output_tokens is not None and query_output_tokens is not None:
        cap_out = float(caption_output_tokens)
        q_out = float(query_output_tokens)
    else:
        cap_gb = float(s_in[2]) * float(rho[2])
        q_out_gb = float(s_in[3]) * float(rho[3])
        cap_pair, q_pair = _resolve_llm_tokens(
            seg_min,
            caption_tokens=None,
            query_tokens=None,
            rng_seed=llm_token_rng_seed,
            caption_output_payload_gb=cap_gb,
            query_output_payload_gb=q_out_gb,
        )
        cap_out = cap_pair[1]
        q_out = q_pair[1]

    cap_model = nodes[2].model
    q_model = nodes[3].model
    if not cap_model or not q_model:
        raise ValueError("caption and query nodes require model names")
    t2 = (
        te[2]
        if te[2] is not None
        else llm_decode_duration_sec(cap_model, cap_out, rng=env_rng)
    )
    t3 = (
        te[3]
        if te[3] is not None
        else llm_decode_duration_sec(q_model, q_out, rng=env_rng)
    )

    up_rng = (
        det_rng(env_rng.randrange(0, 2**31), "lat_upload")
        if env_rng is not None
        else None
    )
    dn_rng = (
        det_rng(env_rng.randrange(0, 2**31), "lat_download")
        if env_rng is not None
        else None
    )
    _, lat_up = local_edge_cost_latency(
        _endpoint(nodes[0]), float(s_in[0]), direction="upload", rng=up_rng
    )
    _, lat_dn = local_edge_cost_latency(
        _endpoint(nodes[3]), float(transfer_gb[3]), direction="download", rng=dn_rng
    )
    lat = lat_up + t0 + t1 + t2 + t3 + lat_dn

    if network_samples is None:
        if env_rng is not None:
            samples = tuple(
                sample_link(
                    _endpoint(nodes[i]),
                    _endpoint(nodes[i + 1]),
                    rng=det_rng(env_rng.randrange(0, 2**31), "lat_edge", i),
                )
                for i in range(3)
            )
        else:
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


def end_to_end_cost_and_latency(
    nodes: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
    s_src_gb: float,
    rho: tuple[float, float, float, float],
    *,
    workflow_rng: random.Random,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> tuple[float, float]:
    """
    One Monte Carlo draw: shared ρ, LLM tokens, segment/split noise, and network draws.

    ``execution_scale_scope``: optional string mixed into deterministic ``k`` (with
    ``execution_scale_seed``). When omitted, uses ``str(llm_seed)`` per draw.

    ``execution_scale_seed``: integer master seed for reproducible node scale factors ``k``;
    when omitted, defaults to ``llm_seed`` for this draw so end-to-end runs stay repeatable.
    """
    s_in, transfer_gb = _propagate_sizes_gb(s_src_gb, rho)
    seg_min = _segment_video_minutes(s_in[0])
    dur_sec = max(seg_min * 60.0, 1e-6)
    llm_seed = workflow_rng.randrange(0, 2**31)
    trial_scope = execution_scale_scope if execution_scale_scope is not None else str(llm_seed)
    trial_exec_scale_seed = (
        execution_scale_seed if execution_scale_seed is not None else llm_seed
    )
    cap_gb = float(s_in[2]) * float(rho[2])
    q_out_gb = float(s_in[3]) * float(rho[3])
    (cap_in, cap_out), (q_in, q_out) = _resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_seed,
        caption_output_payload_gb=cap_gb,
        query_output_payload_gb=q_out_gb,
    )
    c = end_to_end_cost(
        nodes,
        s_src_gb,
        rho,
        caption_tokens=(cap_in, cap_out),
        query_tokens=(q_in, q_out),
        llm_token_rng_seed=llm_seed,
    )
    seg_exe = sample_segment_execute_sec(
        dur_sec,
        rng=workflow_rng,
        node=nodes[0],
        execution_scale_scope=trial_scope,
        execution_scale_seed=trial_exec_scale_seed,
    )
    spl_exe = sample_split_execute_sec(
        dur_sec,
        rng=workflow_rng,
        node=nodes[1],
        execution_scale_scope=trial_scope,
        execution_scale_seed=trial_exec_scale_seed,
    )
    reset_link_counters(None)
    samples = tuple(
        sample_link(
            _endpoint(nodes[i]),
            _endpoint(nodes[i + 1]),
            rng=det_rng(workflow_rng.randrange(0, 2**31), "e2e_net", i),
        )
        for i in range(3)
    )
    ell = end_to_end_latency(
        nodes,
        s_src_gb,
        rho,
        t_exe=(seg_exe, spl_exe, None, None),
        caption_output_tokens=cap_out,
        query_output_tokens=q_out,
        network_samples=samples,
        llm_token_rng_seed=llm_seed,
        env_rng=workflow_rng,
        execution_scale_scope=trial_scope,
        execution_scale_seed=trial_exec_scale_seed,
    )
    return c, ell


def end_to_end_utility(
    nodes: Sequence[PhysicalNode],
    *,
    weights: Tuple[float, float, float, float] | None = None,
) -> float:
    """
    U(x) = Σ_i w_i · μ_i with μ from ``physical_node_utility``. Default ``weights`` sum to 1
    (four equal weights 0.25 each).
    """
    ns = tuple(nodes)
    if len(ns) != 4:
        raise ValueError(f"expected 4 PhysicalNode, got {len(ns)}")
    for i, n in enumerate(ns):
        if n.operation != _OPS_ORDER[i]:
            raise ValueError(f"nodes[{i}] must be operation {_OPS_ORDER[i]!r}, got {n.operation!r}")
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    if len(w) != 4:
        raise ValueError("weights must have length 4")
    return sum(w[i] * physical_node_utility(ns[i]) for i in range(4))


def generate_realistic_queries(num_queries: int, seed: int = 42) -> list[QueryProfile]:
    """
    每条 query：plug-in mean ρ 下，在 **budget 搜索得到的参考链**上计算 mean-field
    端到端 cost / latency（cost 与 latency 可来自不同参考链），再乘以
    ``QUERY_BUDGET_REFERENCE_MULTIPLIER`` 得到 ``Θ_C``、``Θ_T``。
    """
    import random

    from sim_env.utility import QueryProfile

    rng = random.Random(seed)
    calib_rng = random.Random(seed + 100)

    mean_rho = cfg.plugin_mean_data_conversion_ratios(
        n_calibration_samples=4096,
        rng=calib_rng,
        operations=("segment", "split", "caption", "query"),
    )

    chain_c = _BUDGET_REF_CHAIN_COST_WF1
    chain_l = _BUDGET_REF_CHAIN_LATENCY_WF1
    mult = QUERY_BUDGET_REFERENCE_MULTIPLIER

    queries: list[QueryProfile] = []

    for q_idx in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        s_src_gb = s_src_mb / 1000.0

        wf_c = det_rng(seed, "query_budget_ref_cost", q_idx)
        ref_c, _ = end_to_end_cost_and_latency(
            chain_c, s_src_gb, mean_rho, workflow_rng=wf_c
        )
        wf_l = det_rng(seed, "query_budget_ref_latency", q_idx)
        _, ref_l = end_to_end_cost_and_latency(
            chain_l, s_src_gb, mean_rho, workflow_rng=wf_l
        )

        queries.append(
            QueryProfile(
                s_src_gb=s_src_gb,
                theta_cost=ref_c * mult,
                theta_latency_sec=ref_l * mult,
            )
        )

    return queries