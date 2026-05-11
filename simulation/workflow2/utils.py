"""
Workflow 2 — 多路径 + 并行分支（Database / QA / Answer）仿真骨架。

结构来自 ``workflow.pdf`` 第二条 workflow：

**四条互斥主干路径**（每次请求选其一）::

    video_segment → shot_detection → video_caption → database → qa → answer
    video_segment → shot_detection → ocr → database → qa → answer
    video_segment → shot_detection → label_detection → database → qa → answer
    video_segment → speech_transcription → database → qa → answer

**并行（可选）**：在 ``shot_detection`` 之后，可同时执行
``{video_caption, ocr, label_detection}`` 中任意子集；检索库前融合多条分支输出。
费用为各分支之和；-wall-clock 近似为 ``max`` 分支耗时 + 前后缀串行部分。

命名约定：``video_segment`` 对应云上 segment API；``shot_detection`` 复用 workflow1 的
split 计费与分割执行延迟模型（占位语义：镜头检测 / 切分）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from sim_env import config as cfg
from sim_env.cost import (
    ProviderRegion,
    database_invocation_cost_usd,
    egress_cost_usd,
    llm_token_cost_usd,
    split_cost_usd,
    storage_cost_usd,
    video_service_cost_usd,
)
from sim_env.execution_latency import (
    llm_decode_duration_sec,
    sample_database_query_execute_sec,
    sample_label_detection_execute_sec,
    sample_ocr_execute_sec,
    sample_segment_execute_sec,
    sample_speech_transcription_execute_sec,
    sample_split_execute_sec,
)
from sim_env.llm import sample_query_output_tokens
from sim_env.network import NetworkSample, reset_link_counters, sample_link
from sim_env.utility import PhysicalNode, QueryProfile, physical_node_utility

from workflow1 import utils as wf1_utils

# ---------------------------------------------------------------------------
# 路径定义（逻辑算子名）
# ---------------------------------------------------------------------------

WF2_PATH_CAPTION: tuple[str, ...] = (
    "video_segment",
    "shot_detection",
    "video_caption",
    "database",
    "qa",
    "answer",
)
WF2_PATH_OCR: tuple[str, ...] = (
    "video_segment",
    "shot_detection",
    "ocr",
    "database",
    "qa",
    "answer",
)
WF2_PATH_LABEL: tuple[str, ...] = (
    "video_segment",
    "shot_detection",
    "label_detection",
    "database",
    "qa",
    "answer",
)
WF2_PATH_SPEECH: tuple[str, ...] = (
    "video_segment",
    "speech_transcription",
    "database",
    "qa",
    "answer",
)

WF2PathId = Literal["caption", "ocr", "label", "speech"]

WF2ParallelModality = Literal["video_caption", "ocr", "label_detection"]

WF2LogicalOp = Literal[
    "video_segment",
    "shot_detection",
    "video_caption",
    "ocr",
    "label_detection",
    "speech_transcription",
    "database",
    "qa",
    "answer",
]


def path_logical_ops(path_id: WF2PathId) -> tuple[str, ...]:
    if path_id == "caption":
        return WF2_PATH_CAPTION
    if path_id == "ocr":
        return WF2_PATH_OCR
    if path_id == "label":
        return WF2_PATH_LABEL
    if path_id == "speech":
        return WF2_PATH_SPEECH
    raise ValueError(f"unknown path_id: {path_id!r}")


# Placeholder USD / 延迟（qa）；database 使用 ``cost.database_invocation_cost_usd`` 与
# ``execution_latency.sample_database_query_execute_sec``。
WF2_PLACEHOLDER_QA_FIXED_COST_USD = 0.001
WF2_PLACEHOLDER_QA_LATENCY_SEC = 0.03

WF2_PLACEHOLDER_ANSWER_INPUT_TOKENS = 12_000.0


def _lognormal_variance(mu: float, sigma: float) -> float:
    """``Var(R)``，其中 ``ln(R) ~ N(mu, sigma^2)``（``sigma`` 为对数尺度标准差）。"""
    return math.exp(2.0 * mu + sigma * sigma) * (math.exp(sigma * sigma) - 1.0)


def _lognormal_mu_sigma_from_mean_variance(mean: float, variance: float) -> tuple[float, float]:
    """由目标均值与方差反推 LogNormal 的 ``(mu, sigma)``（与 ``ln(R)~N(mu,sigma^2)`` 记号一致）。"""
    if mean <= 0:
        raise ValueError("mean must be positive")
    if variance < 0:
        raise ValueError("variance must be non-negative")
    sigma_sq = math.log(1.0 + variance / (mean * mean))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - 0.5 * sigma_sq
    return mu, sigma


# database：线性均值 0.2，线性方差与 caption 的 Var(R) 相同（caption 参数取自 ``config``）
_mu_cap, _sig_cap = cfg.data_conversion_ratio_lognormal_params("caption")
_CAPTION_RATIO_VARIANCE = _lognormal_variance(_mu_cap, _sig_cap)
WF2_DATABASE_RATIO_TARGET_MEAN = 0.2
WF2_DATABASE_RATIO_LOGNORMAL_MU, WF2_DATABASE_RATIO_LOGNORMAL_SIGMA = _lognormal_mu_sigma_from_mean_variance(
    WF2_DATABASE_RATIO_TARGET_MEAN,
    _CAPTION_RATIO_VARIANCE,
)

# ln(R)~N(mu,sigma^2)，用于仍无 config 条目的逻辑算子（qa / answer）
WF2_PLACEHOLDER_RATIO_LOGNORMAL: dict[str, tuple[float, float]] = {
    "qa": (-3.0, 0.4),
    "answer": (-1.72, 0.47),
}

WF2_PLACEHOLDER_VI_UTILITY = 0.88
WF2_PLACEHOLDER_DB_UTILITY = 0.82


def _endpoint(provider: str, region: str) -> ProviderRegion:
    return (provider, region)


def _carrier_segment(provider: str, region: str) -> PhysicalNode:
    """仅用于 VI / split 采样：底层只依赖 provider/region。"""
    return PhysicalNode("segment", provider, region)


def _ratio_model_key(logical_op: str) -> str:
    if logical_op == "video_segment":
        return "segment"
    if logical_op == "shot_detection":
        return "split"
    # ocr / label / speech 与 caption 共用同一 LogNormal（``config`` 中 ``caption`` 条目）
    if logical_op in ("video_caption", "ocr", "label_detection", "speech_transcription"):
        return "caption"
    if logical_op in ("database", "qa", "answer"):
        return logical_op
    raise KeyError(f"unknown logical_op for ratio: {logical_op!r}")


def sample_wf2_logical_ratio(logical_op: str, rng: random.Random | None = None) -> float:
    r = rng or random.Random()
    if logical_op == "database":
        mu, sigma = WF2_DATABASE_RATIO_LOGNORMAL_MU, WF2_DATABASE_RATIO_LOGNORMAL_SIGMA
        return math.exp(mu + sigma * r.gauss(0.0, 1.0))
    key = _ratio_model_key(logical_op)
    if key in cfg.DATA_CONVERSION_RATIO_LOGNORMAL:
        return cfg.sample_data_conversion_ratio(key, r)
    try:
        mu, sigma = WF2_PLACEHOLDER_RATIO_LOGNORMAL[key]
    except KeyError as e:
        raise KeyError(f"No ratio model for logical_op={logical_op!r} (key={key!r})") from e
    z = r.gauss(0.0, 1.0)
    return math.exp(mu + sigma * z)


def sample_wf2_path_rho(path_id: WF2PathId, rng: random.Random | None = None) -> tuple[float, ...]:
    """与 ``path_logical_ops`` 等长的 ρ 抽样。"""
    r = rng or random.Random()
    ops = path_logical_ops(path_id)
    return tuple(sample_wf2_logical_ratio(op, r) for op in ops)


def propagate_path_sizes(
    s_src_gb: float,
    rho: Sequence[float],
) -> tuple[list[float], list[float]]:
    """线性链：``s_in[i+1]=s_in[i]*rho[i]``；边 ``i→i+1`` 载荷 ``s_in[i]*rho[i]``."""
    if s_src_gb <= 0:
        raise ValueError("s_src_gb must be positive")
    n = len(rho)
    s_in = [0.0] * n
    s_in[0] = float(s_src_gb)
    for i in range(1, n):
        s_in[i] = s_in[i - 1] * float(rho[i - 1])
    xfer = [s_in[i] * float(rho[i]) for i in range(n)]
    return s_in, xfer


def _segment_video_minutes_from_source(s_src_gb: float) -> float:
    megabytes = float(s_src_gb) * 1000.0
    duration_sec = cfg.video_duration_sec_from_megabytes(megabytes)
    return duration_sec / 60.0


@dataclass(frozen=True)
class WF2PhysicalNode:
    operation: WF2LogicalOp
    provider: str
    region: str
    model: str | None = None


def validate_exclusive_path_nodes(path_id: WF2PathId, nodes: Sequence[WF2PhysicalNode]) -> None:
    ops = path_logical_ops(path_id)
    if len(nodes) != len(ops):
        raise ValueError(f"path {path_id!r} expects {len(ops)} nodes, got {len(nodes)}")
    for i, (n, op) in enumerate(zip(nodes, ops)):
        if n.operation != op:
            raise ValueError(f"nodes[{i}] must be {op!r}, got {n.operation!r}")


def wf2_node_utility(node: WF2PhysicalNode) -> float:
    op = node.operation
    if op == "video_segment":
        return physical_node_utility(PhysicalNode("segment", node.provider, node.region))
    if op == "shot_detection":
        return physical_node_utility(PhysicalNode("split", node.provider, node.region))
    if op == "video_caption":
        if not node.model:
            raise ValueError("video_caption requires model")
        return physical_node_utility(PhysicalNode("caption", node.provider, node.region, node.model))
    if op in ("ocr", "label_detection", "speech_transcription"):
        return WF2_PLACEHOLDER_VI_UTILITY
    if op in ("database", "qa"):
        return WF2_PLACEHOLDER_DB_UTILITY
    if op == "answer":
        if not node.model:
            raise ValueError("answer requires model")
        return physical_node_utility(PhysicalNode("query", node.provider, node.region, node.model))
    raise ValueError(f"unknown operation: {op!r}")


def end_to_end_utility_exclusive_path(
    path_id: WF2PathId,
    nodes: Sequence[WF2PhysicalNode],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    validate_exclusive_path_nodes(path_id, nodes)
    k = len(nodes)
    if weights is None:
        w = tuple(1.0 / k for _ in range(k))
    else:
        w = tuple(weights)
        if len(w) != k:
            raise ValueError("weights length must match path length")
    return sum(w[i] * wf2_node_utility(nodes[i]) for i in range(k))


def _storage_cost_nodes(nodes: Sequence[WF2PhysicalNode], s_in: Sequence[float], rho: Sequence[float]) -> float:
    t = 0.0
    for i, n in enumerate(nodes):
        gb = float(s_in[i]) * (1.0 + float(rho[i]))
        t += storage_cost_usd(n.provider, n.region, gb, hours=1.0)
    return t


def _edge_cost_latency(
    src: WF2PhysicalNode,
    dst: WF2PhysicalNode,
    xfer_gb: float,
    *,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    ep_s, ep_d = _endpoint(src.provider, src.region), _endpoint(dst.provider, dst.region)
    cents = egress_cost_usd(ep_s, ep_d, xfer_gb)
    sm = sample_link(ep_s, ep_d, rng=rng)
    lat = wf1_utils._transfer_seconds(xfer_gb, sm) + 0.5 * (sm.rtt_ms / 1000.0)
    return cents, lat


def _exe_cost_logical_step(
    logical_op: str,
    node: WF2PhysicalNode,
    *,
    seg_minutes_source_video: float,
    llm_tokens_bundle: tuple[tuple[float, float], tuple[float, float]] | None,
    answer_tokens_override: tuple[float, float] | None,
    database_storage_gb: float | None = None,
) -> float:
    """单步执行费（不含存储/网络）。"""
    p, r = node.provider, node.region
    if logical_op == "video_segment":
        return video_service_cost_usd(p, r, "segment", seg_minutes_source_video)
    if logical_op == "shot_detection":
        return split_cost_usd(p, r, minutes=1.0)
    if logical_op == "video_caption":
        if not node.model or llm_tokens_bundle is None:
            raise ValueError("caption requires model and llm_tokens_bundle")
        cin, cout = llm_tokens_bundle[0]
        return llm_token_cost_usd(p, r, node.model, cin, cout)
    if logical_op == "ocr":
        return video_service_cost_usd(p, r, "ocr", seg_minutes_source_video)
    if logical_op == "label_detection":
        return video_service_cost_usd(p, r, "label_detection", seg_minutes_source_video)
    if logical_op == "speech_transcription":
        return video_service_cost_usd(p, r, "speech_transcription", seg_minutes_source_video)
    if logical_op == "database":
        gb = float(database_storage_gb) if database_storage_gb is not None else 0.0
        return database_invocation_cost_usd(p, r, gb)
    if logical_op == "qa":
        return WF2_PLACEHOLDER_QA_FIXED_COST_USD
    if logical_op == "answer":
        if not node.model:
            raise ValueError("answer requires model")
        if answer_tokens_override is not None:
            ain, aout = answer_tokens_override
            return llm_token_cost_usd(p, r, node.model, ain, aout)
        if llm_tokens_bundle is None:
            raise ValueError("answer needs tokens bundle or answer_tokens_override")
        _, (_, qout) = llm_tokens_bundle
        ain = WF2_PLACEHOLDER_ANSWER_INPUT_TOKENS
        return llm_token_cost_usd(p, r, node.model, ain, qout)
    raise ValueError(f"unknown logical_op: {logical_op!r}")


def _sample_vi_execute_sec(
    logical_op: str,
    dur_sec: float,
    node: WF2PhysicalNode,
    *,
    rng: random.Random,
    execution_scale_scope: str | None,
    execution_scale_seed: int | None,
) -> float:
    car = _carrier_segment(node.provider, node.region)
    if logical_op == "video_segment":
        return sample_segment_execute_sec(
            dur_sec,
            rng=rng,
            node=car,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    if logical_op == "shot_detection":
        return sample_split_execute_sec(
            dur_sec,
            rng=rng,
            node=PhysicalNode("split", node.provider, node.region),
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    if logical_op == "ocr":
        return sample_ocr_execute_sec(
            dur_sec,
            rng=rng,
            node=car,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    if logical_op == "label_detection":
        return sample_label_detection_execute_sec(
            dur_sec,
            rng=rng,
            node=car,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    if logical_op == "speech_transcription":
        return sample_speech_transcription_execute_sec(
            dur_sec,
            rng=rng,
            node=car,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    raise ValueError(f"not a VI step: {logical_op!r}")


def end_to_end_cost_exclusive_path(
    path_id: WF2PathId,
    nodes: tuple[WF2PhysicalNode, ...],
    s_src_gb: float,
    rho: tuple[float, ...],
    *,
    llm_token_rng_seed: int | None = 42,
    answer_tokens: tuple[float, float] | None = None,
) -> float:
    validate_exclusive_path_nodes(path_id, nodes)
    if len(rho) != len(nodes):
        raise ValueError("rho length must match number of nodes")

    seg_min = _segment_video_minutes_from_source(s_src_gb)
    ops = path_logical_ops(path_id)
    llm_bundle = wf1_utils._resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_token_rng_seed,
    )

    s_in, xfer = propagate_path_sizes(s_src_gb, rho)
    exe = 0.0
    for i, op in enumerate(ops):
        db_gb = float(s_in[i]) * (1.0 + float(rho[i])) if op == "database" else None
        exe += _exe_cost_logical_step(
            op,
            nodes[i],
            seg_minutes_source_video=seg_min,
            llm_tokens_bundle=llm_bundle,
            answer_tokens_override=answer_tokens if op == "answer" else None,
            database_storage_gb=db_gb,
        )
    stor = _storage_cost_nodes(nodes, s_in, rho)
    net = 0.0
    for i in range(len(nodes) - 1):
        net += egress_cost_usd(
            _endpoint(nodes[i].provider, nodes[i].region),
            _endpoint(nodes[i + 1].provider, nodes[i + 1].region),
            xfer[i],
        )
    return exe + stor + net


def end_to_end_latency_exclusive_path(
    path_id: WF2PathId,
    nodes: tuple[WF2PhysicalNode, ...],
    s_src_gb: float,
    rho: tuple[float, ...],
    *,
    llm_token_rng_seed: int | None = 42,
    env_rng: random.Random | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
    vi_exe_override: dict[str, float] | None = None,
    network_samples: tuple[NetworkSample, ...] | None = None,
    answer_output_tokens: float | None = None,
) -> float:
    validate_exclusive_path_nodes(path_id, nodes)
    if len(rho) != len(nodes):
        raise ValueError("rho length must match number of nodes")

    er = env_rng if env_rng is not None else random.Random()

    seg_min = _segment_video_minutes_from_source(s_src_gb)
    dur_sec = max(seg_min * 60.0, 1e-6)
    ops = path_logical_ops(path_id)

    cap_pair, q_pair = wf1_utils._resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_token_rng_seed,
    )
    cap_out = cap_pair[1]
    q_out = float(answer_output_tokens) if answer_output_tokens is not None else q_pair[1]

    vo = vi_exe_override or {}

    lat = 0.0
    for i, op in enumerate(ops):
        n = nodes[i]
        if op == "video_segment":
            lat += vo.get(
                op,
                _sample_vi_execute_sec(
                    op,
                    dur_sec,
                    n,
                    rng=er,
                    execution_scale_scope=execution_scale_scope,
                    execution_scale_seed=execution_scale_seed,
                ),
            )
        elif op == "shot_detection":
            lat += vo.get(
                op,
                _sample_vi_execute_sec(
                    op,
                    dur_sec,
                    n,
                    rng=er,
                    execution_scale_scope=execution_scale_scope,
                    execution_scale_seed=execution_scale_seed,
                ),
            )
        elif op in ("ocr", "label_detection", "speech_transcription"):
            lat += vo.get(
                op,
                _sample_vi_execute_sec(
                    op,
                    dur_sec,
                    n,
                    rng=er,
                    execution_scale_scope=execution_scale_scope,
                    execution_scale_seed=execution_scale_seed,
                ),
            )
        elif op == "video_caption":
            if not n.model:
                raise ValueError("caption requires model")
            lat += llm_decode_duration_sec(n.model, cap_out, rng=er)
        elif op == "database":
            lat += sample_database_query_execute_sec(
                rng=er,
                node=_carrier_segment(n.provider, n.region),
                execution_scale_scope=execution_scale_scope,
                execution_scale_seed=execution_scale_seed,
            )
        elif op == "qa":
            lat += WF2_PLACEHOLDER_QA_LATENCY_SEC
        elif op == "answer":
            if not n.model:
                raise ValueError("answer requires model")
            lat += llm_decode_duration_sec(n.model, q_out, rng=er)
        else:
            raise RuntimeError(op)

    _, xfer = propagate_path_sizes(s_src_gb, rho)
    if network_samples is None:
        samples = tuple(
            sample_link(
                _endpoint(nodes[i].provider, nodes[i].region),
                _endpoint(nodes[i + 1].provider, nodes[i + 1].region),
                rng=wf1_utils.det_rng(er.randrange(0, 2**31), "wf2_exc_net", path_id, i),
            )
            for i in range(len(nodes) - 1)
        )
    else:
        samples = network_samples

    for i in range(len(nodes) - 1):
        lat += wf1_utils._transfer_seconds(xfer[i], samples[i]) + 0.5 * (samples[i].rtt_ms / 1000.0)

    return lat


def end_to_end_cost_parallel_shot_modalities(
    *,
    node_video: WF2PhysicalNode,
    node_shot: WF2PhysicalNode,
    modality_nodes: dict[str, WF2PhysicalNode],
    node_database: WF2PhysicalNode,
    node_qa: WF2PhysicalNode,
    node_answer: WF2PhysicalNode,
    active_modalities: frozenset[str],
    s_src_gb: float,
    rho_video: float,
    rho_shot: float,
    rho_modalities: dict[str, float],
    rho_database: float,
    rho_qa: float,
    rho_answer: float,
    llm_token_rng_seed: int | None = 42,
    answer_tokens: tuple[float, float] | None = None,
) -> float:
    """
    shot 之后并行跑多种 VI/字幕 分支时的费用（分支费用相加；写入 DB 的载荷按分支输出之和占位）。
    ``active_modalities`` 元素须为 ``video_caption`` / ``ocr`` / ``label_detection``。
    """
    if not active_modalities:
        raise ValueError("active_modalities must be non-empty")

    seg_min = _segment_video_minutes_from_source(s_src_gb)
    llm_bundle = wf1_utils._resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_token_rng_seed,
    )

    # --- prefix: video → shot
    exe = _exe_cost_logical_step(
        "video_segment",
        node_video,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        answer_tokens_override=None,
    )
    exe += _exe_cost_logical_step(
        "shot_detection",
        node_shot,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        answer_tokens_override=None,
    )

    s_after_video = s_src_gb * rho_video
    s_after_shot = s_after_video * rho_shot
    xfer_v_s = s_src_gb * rho_video

    net = egress_cost_usd(
        _endpoint(node_video.provider, node_video.region),
        _endpoint(node_shot.provider, node_shot.region),
        xfer_v_s,
    )

    stor = storage_cost_usd(
        node_video.provider,
        node_video.region,
        s_src_gb * (1.0 + rho_video),
        hours=1.0,
    )
    stor += storage_cost_usd(
        node_shot.provider,
        node_shot.region,
        s_after_video * (1.0 + rho_shot),
        hours=1.0,
    )

    branch_outputs_gb: list[float] = []
    for m in sorted(active_modalities):
        if m not in modality_nodes:
            raise KeyError(f"missing modality node for {m!r}")
        if m not in rho_modalities:
            raise KeyError(f"missing rho for modality {m!r}")
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        s_out_m = s_after_shot * rho_m
        branch_outputs_gb.append(s_out_m)

        exe += _exe_cost_logical_step(
            m,
            mn,
            seg_minutes_source_video=seg_min,
            llm_tokens_bundle=llm_bundle,
            answer_tokens_override=None,
        )
        xfer_s_m = s_after_shot * rho_m
        net += egress_cost_usd(
            _endpoint(node_shot.provider, node_shot.region),
            _endpoint(mn.provider, mn.region),
            xfer_s_m,
        )
        stor += storage_cost_usd(
            mn.provider,
            mn.region,
            s_after_shot * (1.0 + rho_m),
            hours=1.0,
        )

    s_db_in = sum(branch_outputs_gb)

    for m in sorted(active_modalities):
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        s_out_m = s_after_shot * rho_m
        net += egress_cost_usd(
            _endpoint(mn.provider, mn.region),
            _endpoint(node_database.provider, node_database.region),
            s_out_m,
        )

    exe += _exe_cost_logical_step(
        "database",
        node_database,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        answer_tokens_override=None,
        database_storage_gb=s_db_in * (1.0 + rho_database),
    )
    exe += _exe_cost_logical_step(
        "qa",
        node_qa,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        answer_tokens_override=None,
    )
    exe += _exe_cost_logical_step(
        "answer",
        node_answer,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        answer_tokens_override=answer_tokens,
    )

    stor += storage_cost_usd(
        node_database.provider,
        node_database.region,
        s_db_in * (1.0 + rho_database),
        hours=1.0,
    )
    s_after_db = s_db_in * rho_database
    xfer_db_q = s_db_in * rho_database
    net += egress_cost_usd(
        _endpoint(node_database.provider, node_database.region),
        _endpoint(node_qa.provider, node_qa.region),
        xfer_db_q,
    )

    stor += storage_cost_usd(
        node_qa.provider,
        node_qa.region,
        s_after_db * (1.0 + rho_qa),
        hours=1.0,
    )
    s_after_qa = s_after_db * rho_qa
    xfer_q_a = s_after_db * rho_qa
    net += egress_cost_usd(
        _endpoint(node_qa.provider, node_qa.region),
        _endpoint(node_answer.provider, node_answer.region),
        xfer_q_a,
    )

    stor += storage_cost_usd(
        node_answer.provider,
        node_answer.region,
        s_after_qa * (1.0 + rho_answer),
        hours=1.0,
    )

    return exe + stor + net


def end_to_end_latency_parallel_shot_modalities(
    *,
    node_video: WF2PhysicalNode,
    node_shot: WF2PhysicalNode,
    modality_nodes: dict[str, WF2PhysicalNode],
    node_database: WF2PhysicalNode,
    node_qa: WF2PhysicalNode,
    node_answer: WF2PhysicalNode,
    active_modalities: frozenset[str],
    s_src_gb: float,
    rho_video: float,
    rho_shot: float,
    rho_modalities: dict[str, float],
    rho_database: float,
    rho_qa: float,
    rho_answer: float,
    llm_token_rng_seed: int | None = 42,
    env_rng: random.Random | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
    answer_output_tokens: float | None = None,
) -> float:
    """并行分支 wall-clock：``max`` 各分支（shot→mod→db 段）+ 前缀 video→shot + 后缀。"""
    if not active_modalities:
        raise ValueError("active_modalities must be non-empty")

    er = env_rng or random.Random()
    seg_min = _segment_video_minutes_from_source(s_src_gb)
    dur_sec = max(seg_min * 60.0, 1e-6)
    cap_pair, q_pair = wf1_utils._resolve_llm_tokens(
        seg_min,
        caption_tokens=None,
        query_tokens=None,
        rng_seed=llm_token_rng_seed,
    )
    cap_out = cap_pair[1]
    q_out = float(answer_output_tokens) if answer_output_tokens is not None else q_pair[1]

    t_vid = _sample_vi_execute_sec(
        "video_segment",
        dur_sec,
        node_video,
        rng=er,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )
    t_shot = _sample_vi_execute_sec(
        "shot_detection",
        dur_sec,
        node_shot,
        rng=er,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )
    s_after_video = s_src_gb * rho_video
    xfer_vs = s_src_gb * rho_video
    _, el_vs = _edge_cost_latency(node_video, node_shot, xfer_vs, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_vs"))

    branch_times: list[float] = []
    s_after_shot = s_after_video * rho_shot
    for m in sorted(active_modalities):
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        xfer_sm = s_after_shot * rho_m
        _, el_sm = _edge_cost_latency(node_shot, mn, xfer_sm, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_sm", m))
        if m == "video_caption":
            if not mn.model:
                raise ValueError("video_caption requires model")
            t_m = llm_decode_duration_sec(mn.model, cap_out, rng=er)
        elif m in ("ocr", "label_detection"):
            t_m = _sample_vi_execute_sec(
                m,
                dur_sec,
                mn,
                rng=er,
                execution_scale_scope=execution_scale_scope,
                execution_scale_seed=execution_scale_seed,
            )
        else:
            raise ValueError(f"unsupported modality {m!r}")

        s_out_m = s_after_shot * rho_m
        _, el_md = _edge_cost_latency(mn, node_database, s_out_m, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_md", m))
        branch_times.append(el_sm + t_m + el_md)

    t_parallel = max(branch_times)

    s_db_in = sum(s_after_shot * rho_modalities[m] for m in active_modalities)
    xfer_suffix = s_db_in * rho_database
    _, el_dq = _edge_cost_latency(node_database, node_qa, xfer_suffix, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_dq"))
    s_after_db = s_db_in * rho_database
    xfer_qa = s_after_db * rho_qa
    _, el_qa = _edge_cost_latency(node_qa, node_answer, xfer_qa, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_qa"))

    if not node_answer.model:
        raise ValueError("answer requires model")
    t_ans = llm_decode_duration_sec(node_answer.model, q_out, rng=er)

    t_db = sample_database_query_execute_sec(
        rng=er,
        node=_carrier_segment(node_database.provider, node_database.region),
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )

    return (
        t_vid
        + el_vs
        + t_shot
        + t_parallel
        + t_db
        + el_dq
        + WF2_PLACEHOLDER_QA_LATENCY_SEC
        + el_qa
        + t_ans
    )


def reference_deployment_exclusive_path(path_id: WF2PathId) -> tuple[WF2PhysicalNode, ...]:
    p, r = "GCP", "us-east1"
    cap_m = "Gemini 2.5 Pro"
    ans_m = "Gemini 2.5 Flash"
    if path_id == "speech":
        return (
            WF2PhysicalNode("video_segment", p, r),
            WF2PhysicalNode("speech_transcription", p, r),
            WF2PhysicalNode("database", p, r),
            WF2PhysicalNode("qa", p, r),
            WF2PhysicalNode("answer", p, r, ans_m),
        )
    mid: WF2LogicalOp
    if path_id == "caption":
        mid = "video_caption"
    elif path_id == "ocr":
        mid = "ocr"
    elif path_id == "label":
        mid = "label_detection"
    else:
        raise ValueError(path_id)
    return (
        WF2PhysicalNode("video_segment", p, r),
        WF2PhysicalNode("shot_detection", p, r),
        WF2PhysicalNode(mid, p, r, cap_m if mid == "video_caption" else None),
        WF2PhysicalNode("database", p, r),
        WF2PhysicalNode("qa", p, r),
        WF2PhysicalNode("answer", p, r, ans_m),
    )


def default_weights_for_path(path_id: WF2PathId) -> tuple[float, ...]:
    """均匀权重，长度与路径逻辑层数一致。"""
    n = len(path_logical_ops(path_id))
    return tuple(1.0 / float(n) for _ in range(n))


def plugin_mean_data_conversion_ratios_wf2(
    path_id: WF2PathId,
    *,
    n_calibration_samples: int,
    rng: random.Random,
) -> tuple[float, ...]:
    """与 ``workflow1.config.plugin_mean_data_conversion_ratios`` 同 spirit 的 plug-in 均值（沿路径各逻辑算子）。"""
    ops = path_logical_ops(path_id)
    if n_calibration_samples <= 0:
        raise ValueError("n_calibration_samples must be positive")
    sums = [0.0] * len(ops)
    inv = 1.0 / float(n_calibration_samples)
    for _ in range(n_calibration_samples):
        for i, op in enumerate(ops):
            sums[i] += sample_wf2_logical_ratio(op, rng)
    return tuple(s * inv for s in sums)


def generate_realistic_queries_wf2(
    num_queries: int,
    path_id: WF2PathId,
    *,
    seed: int = 42,
    n_calibration_samples: int = 4096,
) -> list[QueryProfile]:
    """
    参照 ``workflow1.utils.generate_realistic_queries``：锚点部署 + plug-in mean ρ，
    单次路径费用/时延定预算 Θ_C、Θ_T。
    """
    rng = random.Random(seed)
    calib_rng = random.Random(seed + 100)
    mean_rho = plugin_mean_data_conversion_ratios_wf2(
        path_id,
        n_calibration_samples=n_calibration_samples,
        rng=calib_rng,
    )
    anchor = reference_deployment_exclusive_path(path_id)
    validate_exclusive_path_nodes(path_id, anchor)

    queries: list[QueryProfile] = []
    for q_idx in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        s_src_gb = s_src_mb / 1000.0

        wf = wf1_utils.det_rng(seed, "wf2_query_budget", q_idx)
        llm_seed = wf.randrange(0, 2**31)

        ref_cost = end_to_end_cost_exclusive_path(
            path_id,
            anchor,
            s_src_gb,
            mean_rho,
            llm_token_rng_seed=llm_seed,
        )
        ref_lat = end_to_end_latency_exclusive_path(
            path_id,
            anchor,
            s_src_gb,
            mean_rho,
            llm_token_rng_seed=llm_seed,
            env_rng=wf,
            execution_scale_scope=str(llm_seed),
            execution_scale_seed=llm_seed,
        )

        factor_c = rng.uniform(1.05, 1.15)
        factor_t = rng.uniform(1.30, 1.60)

        queries.append(
            QueryProfile(
                s_src_gb=s_src_gb,
                theta_cost=ref_cost * factor_c,
                theta_latency_sec=ref_lat * factor_t,
            )
        )

    return queries


def exclusive_path_cost_and_latency_mc(
    path_id: WF2PathId,
    nodes: tuple[WF2PhysicalNode, ...],
    s_src_gb: float,
    *,
    workflow_rng: random.Random,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
) -> tuple[float, float, float]:
    """单次蒙特卡洛：费用、**算法路径延迟（沿所选路径求和）**、**展示用 workflow 延迟（岔路 max）**。"""
    from .display_latency import workflow_display_latency_max_fork

    rho = sample_wf2_path_rho(path_id, workflow_rng)
    llm_seed = workflow_rng.randrange(0, 2**31)
    g = np.random.default_rng(wf1_utils.det_rng(llm_seed, "wf2_exc_aout").randrange(0, 2**31))
    q_out = float(sample_query_output_tokens(rng=g))

    c = end_to_end_cost_exclusive_path(
        path_id,
        nodes,
        s_src_gb,
        rho,
        llm_token_rng_seed=llm_seed,
        answer_tokens=(WF2_PLACEHOLDER_ANSWER_INPUT_TOKENS, q_out),
    )

    trial_scope = execution_scale_scope if execution_scale_scope is not None else str(llm_seed)
    trial_seed = execution_scale_seed if execution_scale_seed is not None else llm_seed

    reset_link_counters(None)
    ell_sum = end_to_end_latency_exclusive_path(
        path_id,
        nodes,
        s_src_gb,
        rho,
        llm_token_rng_seed=llm_seed,
        env_rng=workflow_rng,
        execution_scale_scope=trial_scope,
        execution_scale_seed=trial_seed,
        answer_output_tokens=q_out,
    )
    ell_disp = workflow_display_latency_max_fork(
        path_id,
        nodes,
        s_src_gb,
        rho_deployed=rho,
        q_out=q_out,
        workflow_rng=workflow_rng,
        llm_seed=llm_seed,
        execution_scale_scope=trial_scope,
        execution_scale_seed=trial_seed,
        latency_sum_optimized_path=ell_sum,
    )
    return c, ell_sum, ell_disp


if __name__ == "__main__":  # pragma: no cover
    ri = random.Random(0)
    for pid in ("caption", "ocr", "label", "speech"):
        chain = reference_deployment_exclusive_path(pid)
        rho = sample_wf2_path_rho(pid, ri)
        print(pid, "rho_len", len(rho))
        print(
            "  cost",
            end_to_end_cost_exclusive_path(pid, chain, 0.5, rho, llm_token_rng_seed=7),
        )
        print(
            "  latency",
            end_to_end_latency_exclusive_path(pid, chain, 0.5, rho, llm_token_rng_seed=7),
        )

    nodes_mod = {
        "video_caption": WF2PhysicalNode("video_caption", "GCP", "us-east1", "Gemini 2.5 Pro"),
        "ocr": WF2PhysicalNode("ocr", "GCP", "us-east1"),
        "label_detection": WF2PhysicalNode("label_detection", "GCP", "us-east1"),
    }
    rho_m = {k: sample_wf2_logical_ratio(k, ri) for k in nodes_mod}
    print(
        "parallel_cost",
        end_to_end_cost_parallel_shot_modalities(
            node_video=WF2PhysicalNode("video_segment", "GCP", "us-east1"),
            node_shot=WF2PhysicalNode("shot_detection", "GCP", "us-east1"),
            modality_nodes=nodes_mod,
            node_database=WF2PhysicalNode("database", "GCP", "us-east1"),
            node_qa=WF2PhysicalNode("qa", "GCP", "us-east1"),
            node_answer=WF2PhysicalNode("answer", "GCP", "us-east1", "Gemini 2.5 Flash"),
            active_modalities=frozenset({"video_caption", "ocr"}),
            s_src_gb=0.5,
            rho_video=sample_wf2_logical_ratio("video_segment", ri),
            rho_shot=sample_wf2_logical_ratio("shot_detection", ri),
            rho_modalities=rho_m,
            rho_database=sample_wf2_logical_ratio("database", ri),
            rho_qa=sample_wf2_logical_ratio("qa", ri),
            rho_answer=sample_wf2_logical_ratio("answer", ri),
            llm_token_rng_seed=11,
        ),
    )
