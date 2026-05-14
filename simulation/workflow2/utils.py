"""
Workflow 2 — 与论文图一致的 **Database / Q/A** 视频工作流骨架。

**四条互斥主干路径**（每次请求选其一）::

    shot_detection → video_split → video_caption → database → qa
    shot_detection → video_split → ocr → database → qa
    shot_detection → video_split → label_detection → database → qa
    speech_transcription → database → qa

与示意图一致：``shot_detection`` 为镜头检测；``video_split`` 为视频切段（split 计费/ρ）；
``speech_transcription`` 支线自 **Video** 直通（不经 shot/split）。优化链在 **Q/A** 结束；
``answer`` 仅为最终输出语义，不作为单独逻辑算子（终局 LLM 费用/时延并入 ``qa``）。

**并行（展示用）**：在 ``video_split`` 之后可并行 ``video_caption`` / ``ocr`` / ``label_detection``；
费用为各分支之和；wall-clock 近似 ``max`` 分支 + 串行前/后缀。
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from sim_env import config as cfg
from sim_env.cost import (
    ProviderRegion,
    database_instance_cost_usd,
    database_storage_cost_usd,
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
    sample_shot_detection_execute_sec,
    sample_speech_transcription_execute_sec,
    sample_video_split_execute_sec,
)
from sim_env.llm import caption_visual_input_tokens, llm_tokens_from_data_payload_gb
from sim_env.network import (
    LOCAL_PROVIDER,
    LOCAL_REGION,
    NetworkSample,
    reset_link_counters,
    sample_link,
)
from sim_env.utility import PhysicalNode, QueryProfile, physical_node_utility, visual_intelligence_utility

from workflow1 import utils as wf1_utils


_WF2_LOCAL_EP: ProviderRegion = (LOCAL_PROVIDER, LOCAL_REGION)
# 路径定义（逻辑算子名）
# ---------------------------------------------------------------------------

WF2_PATH_CAPTION: tuple[str, ...] = (
    "shot_detection",
    "video_split",
    "video_caption",
    "database",
    "qa",
)
WF2_PATH_OCR: tuple[str, ...] = (
    "shot_detection",
    "video_split",
    "ocr",
    "database",
    "qa",
)
WF2_PATH_LABEL: tuple[str, ...] = (
    "shot_detection",
    "video_split",
    "label_detection",
    "database",
    "qa",
)
WF2_PATH_SPEECH: tuple[str, ...] = (
    "speech_transcription",
    "database",
    "qa",
)

WF2PathId = Literal["video_caption", "ocr", "label_detection", "speech_transcription"]

WF2ParallelModality = Literal["video_caption", "ocr", "label_detection"]

WF2LogicalOp = Literal[
    "shot_detection",
    "video_split",
    "video_caption",
    "ocr",
    "label_detection",
    "speech_transcription",
    "database",
    "qa",
]


def path_logical_ops(path_id: WF2PathId) -> tuple[str, ...]:
    if path_id == "video_caption":
        return WF2_PATH_CAPTION
    if path_id == "ocr":
        return WF2_PATH_OCR
    if path_id == "label_detection":
        return WF2_PATH_LABEL
    if path_id == "speech_transcription":
        return WF2_PATH_SPEECH
    raise ValueError(f"unknown path_id: {path_id!r}")


# Placeholder USD / 延迟（qa）；database 使用 ``sim_env.cost`` 中托管库实例价 + DB 存储单价 + 下表占位执行延迟
WF2_PLACEHOLDER_QA_FIXED_COST_USD = 0.001
WF2_PLACEHOLDER_QA_LATENCY_SEC = 0.03


def sample_database_conversion_ratio(s_in_gb: float, rng: random.Random) -> float:
    """Database row: uniform file size in [4 KiB, 6 KiB]; ratio = payload_gb / s_in."""
    if s_in_gb <= 0:
        raise ValueError("s_in_gb must be positive")
    out_b = cfg.sample_database_output_file_bytes(rng)
    out_gb = float(out_b) / 1_000_000_000.0
    return out_gb / s_in_gb


def wf2_llm_token_bundle(
    path_id: WF2PathId,
    s_src_gb: float,
    rho: tuple[float, ...],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Caption **vision** input unchanged; text-side tokens from payload GB via 1 KiB≈250 tok.
    ``rho`` length may be a prefix of the full path (prefix aggregates).
    """
    ops_full = path_logical_ops(path_id)
    n = len(rho)
    s_in, _ = propagate_path_sizes(s_src_gb, rho)
    seg_min = _segment_video_minutes_from_source(s_src_gb)
    dur = max(seg_min * 60.0, 1e-6)

    if "video_caption" in ops_full:
        ci = ops_full.index("video_caption")
        if ci < n:
            cap_out_gb = float(s_in[ci]) * float(rho[ci])
            cin = caption_visual_input_tokens(dur)
            cout = llm_tokens_from_data_payload_gb(cap_out_gb)
            cap_pair: tuple[float, float] = (cin, cout)
        else:
            cap_pair = (0.0, 1.0)
    else:
        cap_pair = (0.0, 1.0)

    qa_i = ops_full.index("qa")
    if qa_i < n:
        qa_in_gb = float(s_in[qa_i])
        qa_out_gb = qa_in_gb * float(rho[qa_i])
    else:
        qa_in_gb, qa_out_gb = 0.0, 0.0
    q_pair = (
        llm_tokens_from_data_payload_gb(qa_in_gb),
        llm_tokens_from_data_payload_gb(qa_out_gb),
    )
    return cap_pair, q_pair


def _endpoint(provider: str, region: str) -> ProviderRegion:
    return (provider, region)


def _carrier_shot_detection(provider: str, region: str) -> PhysicalNode:
    """database / VI 采样：底层 segment 延迟表与 shot_detection 共享 provider/region。"""
    return PhysicalNode("shot_detection", provider, region)


def wf2_ratio_model_key_for_layer(path_ops: Sequence[str], layer_index: int) -> str:
    """数据转换比 ρ 在 ``config`` 中的键名（与论文章节算子命名一致）。"""
    op = path_ops[layer_index]
    if op in (
        "shot_detection",
        "video_split",
        "video_caption",
        "speech_transcription",
        "ocr",
        "label_detection",
        "qa",
    ):
        return op
    raise KeyError(f"unknown logical op for ratio: {op!r}")


def sample_wf2_logical_ratio(
    logical_op: str,
    rng: random.Random | None = None,
    *,
    path_ops: tuple[str, ...] | None = None,
    layer_index: int | None = None,
) -> float:
    """Per-step stochastic ρ（LogNormal）；``database`` 须用 ``sample_database_conversion_ratio``。"""
    if logical_op == "database":
        raise ValueError(
            "database ρ depends on upstream size; use sample_database_conversion_ratio "
            "or sample_wf2_path_rho(path_id, rng, s_src_gb)"
        )
    r = rng or random.Random()
    if path_ops is not None and layer_index is not None:
        key = wf2_ratio_model_key_for_layer(path_ops, layer_index)
    elif logical_op in (
        "shot_detection",
        "video_split",
        "video_caption",
        "speech_transcription",
        "ocr",
        "label_detection",
        "qa",
    ):
        key = logical_op
    else:
        raise KeyError(f"unknown logical_op for ratio: {logical_op!r}")
    return cfg.sample_data_conversion_ratio(key, r)


def sample_wf2_path_rho(
    path_id: WF2PathId,
    rng: random.Random,
    s_src_gb: float,
) -> tuple[float, ...]:
    """沿路径顺序抽样；database 步为 4–6 KiB 均匀payload。"""
    ops = path_logical_ops(path_id)
    rhos: list[float] = []
    s = float(s_src_gb)
    ops_t = tuple(ops)
    for idx, op in enumerate(ops):
        if op == "database":
            rho = sample_database_conversion_ratio(s, rng)
        else:
            rho = sample_wf2_logical_ratio(op, rng, path_ops=ops_t, layer_index=idx)
        rhos.append(rho)
        s *= rho
    return tuple(rhos)


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


# 来自 ``python -m workflow2.budget``（N_QUERIES=100, SEED=42；shot 路径为端到端并行 DAG 枚举 layer 0,1,3,4）。
_BUDGET_REF_CHAINS_WF2_COST: dict[WF2PathId, tuple[WF2PhysicalNode, ...]] = {
    "video_caption": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_caption", "GCP", "us-east1", "Gemini 2.5 Pro"),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "ocr": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("ocr", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "label_detection": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("label_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "speech_transcription": (
        WF2PhysicalNode("speech_transcription", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
}

_BUDGET_REF_CHAINS_WF2_LATENCY: dict[WF2PathId, tuple[WF2PhysicalNode, ...]] = {
    "video_caption": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_caption", "GCP", "us-east1", "Gemini 2.5 Pro"),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "ocr": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("ocr", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "label_detection": (
        WF2PhysicalNode("shot_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("video_split", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("label_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "cn-beijing", "Qwen3-VL-Flash"),
    ),
    "speech_transcription": (
        WF2PhysicalNode("speech_transcription", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("qa", "Aliyun", "ap-southeast-1", "Qwen3-VL-Flash"),
    ),
}


def validate_exclusive_path_nodes(path_id: WF2PathId, nodes: Sequence[WF2PhysicalNode]) -> None:
    ops = path_logical_ops(path_id)
    if len(nodes) != len(ops):
        raise ValueError(f"path {path_id!r} expects {len(ops)} nodes, got {len(nodes)}")
    for i, (n, op) in enumerate(zip(nodes, ops)):
        if n.operation != op:
            raise ValueError(f"nodes[{i}] must be {op!r}, got {n.operation!r}")
        if op in ("qa", "video_caption") and not n.model:
            raise ValueError(f"nodes[{i}] ({op!r}) requires a model name")


for _pid in _BUDGET_REF_CHAINS_WF2_COST:
    validate_exclusive_path_nodes(_pid, _BUDGET_REF_CHAINS_WF2_COST[_pid])
    validate_exclusive_path_nodes(_pid, _BUDGET_REF_CHAINS_WF2_LATENCY[_pid])

# 与 ``workflow2.budget`` 一致的部署链枚举（默认单云并集）；全量笛卡尔积见 ``WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT``。
WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT = False
_SINGLE_CLOUD_PROVIDERS_WF2_ENUM = ("GCP", "AWS", "Aliyun")


def iter_chains_wf2(
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    path_id: WF2PathId,
    full_cross_product: bool = False,
) -> Iterable[tuple[WF2PhysicalNode, ...]]:
    L = len(cands)
    layers = [list(cands[i]) for i in range(L)]
    if full_cross_product:
        for tup in itertools.product(*layers):
            yield tuple(tup)
        return

    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS_WF2_ENUM:
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
    for ch in sorted(seen, key=lambda c: tuple((n.operation, n.provider, n.region, n.model or "") for n in c)):
        validate_exclusive_path_nodes(path_id, ch)
        yield ch


def count_chains_wf2(
    cands: tuple[tuple[WF2PhysicalNode, ...], ...], *, full_cross_product: bool = False
) -> int:
    if full_cross_product:
        return math.prod(len(cands[i]) for i in range(len(cands)))
    L = len(cands)
    layers = [list(cands[i]) for i in range(L)]
    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS_WF2_ENUM:
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


# 均衡权重下各 path 的 LO（由候选枚举独立 argmax，与 ``enumerate_candidates_wf2`` 口径一致）。
WF2_LOGICAL_OPTIMAL_NODES: dict[WF2PathId, tuple[WF2PhysicalNode, ...]] = {
    "video_caption": (
        WF2PhysicalNode("shot_detection", "GCP", "us-east1", None),
        WF2PhysicalNode("video_split", "GCP", "us-east1", None),
        WF2PhysicalNode("video_caption", "AWS", "us-west-2", "Amazon Nova Pro"),
        WF2PhysicalNode("database", "GCP", "us-east1", None),
        WF2PhysicalNode("qa", "GCP", "us-east1", "Gemini 2.5 Pro"),
    ),
    "ocr": (
        WF2PhysicalNode("shot_detection", "GCP", "us-east1", None),
        WF2PhysicalNode("video_split", "GCP", "us-east1", None),
        WF2PhysicalNode("ocr", "GCP", "us-east1", None),
        WF2PhysicalNode("database", "GCP", "us-east1", None),
        WF2PhysicalNode("qa", "GCP", "us-east1", "Gemini 2.5 Pro"),
    ),
    "label_detection": (
        WF2PhysicalNode("shot_detection", "GCP", "us-east1", None),
        WF2PhysicalNode("video_split", "GCP", "us-east1", None),
        WF2PhysicalNode("label_detection", "Aliyun", "cn-shanghai", None),
        WF2PhysicalNode("database", "GCP", "us-east1", None),
        WF2PhysicalNode("qa", "GCP", "us-east1", "Gemini 2.5 Pro"),
    ),
    "speech_transcription": (
        WF2PhysicalNode("speech_transcription", "GCP", "us-east1", None),
        WF2PhysicalNode("database", "GCP", "us-east1", None),
        WF2PhysicalNode("qa", "GCP", "us-east1", "Gemini 2.5 Pro"),
    ),
}


def iter_chains_wf2_end_to_end_shot_modalities(
    cands_vc: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    full_cross_product: bool = False,
    mid_layer_index: int = 2,
) -> Iterable[tuple[WF2PhysicalNode, ...]]:
    """
    **端到端并行主视频 DAG** 的 budget 枚举：只优化 shot / split / database / qa；
    中间层用 ``cands_vc[mid_layer_index][0]`` 占位（并行 mean-field 度量不依赖该占位）。

    ``cands_vc`` 须为 ``enumerate_candidates_wf2(\"video_caption\")``（长度 5）。
    """
    if len(cands_vc) != 5:
        raise ValueError("expected video_caption 5-layer candidates")
    mid = cands_vc[mid_layer_index][0]
    layers_four = [list(cands_vc[0]), list(cands_vc[1]), list(cands_vc[3]), list(cands_vc[4])]
    if full_cross_product:
        for a, b, d, q in itertools.product(*layers_four):
            ch = (a, b, mid, d, q)
            validate_exclusive_path_nodes("video_caption", ch)
            yield ch
        return

    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS_WF2_ENUM:
        prov_layers: list[list[WF2PhysicalNode]] = []
        ok = True
        for layer_list in layers_four:
            layer = [n for n in layer_list if n.provider == prov]
            if not layer:
                ok = False
                break
            prov_layers.append(layer)
        if not ok:
            continue
        for tup in itertools.product(*prov_layers):
            seen.add((tup[0], tup[1], mid, tup[2], tup[3]))
    for ch in sorted(seen, key=lambda c: tuple((n.operation, n.provider, n.region, n.model or "") for n in c)):
        validate_exclusive_path_nodes("video_caption", ch)
        yield ch


def count_chains_wf2_end_to_end_shot_modalities(
    cands_vc: tuple[tuple[WF2PhysicalNode, ...], ...],
    *,
    full_cross_product: bool = False,
) -> int:
    if len(cands_vc) != 5:
        raise ValueError("expected video_caption 5-layer candidates")
    layers_four = [list(cands_vc[0]), list(cands_vc[1]), list(cands_vc[3]), list(cands_vc[4])]
    if full_cross_product:
        return math.prod(len(x) for x in layers_four)
    seen: set[tuple[WF2PhysicalNode, ...]] = set()
    for prov in _SINGLE_CLOUD_PROVIDERS_WF2_ENUM:
        prov_layers: list[list[WF2PhysicalNode]] = []
        ok = True
        for layer_list in layers_four:
            layer = [n for n in layer_list if n.provider == prov]
            if not layer:
                ok = False
                break
            prov_layers.append(layer)
        if not ok:
            continue
        for tup in itertools.product(*prov_layers):
            seen.add(tup)
    return len(seen)


def wf2_node_utility(node: WF2PhysicalNode) -> float:
    op = node.operation
    # video_split / database：无独立质量表，默认归一化效用 1（与 Evaluation.md 口径一致）。
    if op in ("database", "video_split"):
        return 1.0
    # shot_detection：表值均为 0.95 → 除以全局 max 后恒为 1；经 physical_node_utility 与 workflow1 同源。
    if op == "shot_detection":
        return physical_node_utility(
            PhysicalNode("shot_detection", node.provider, node.region, None)
        )
    if op == "video_caption":
        if not node.model:
            raise ValueError("video_caption requires model")
        return physical_node_utility(PhysicalNode("video_caption", node.provider, node.region, node.model))
    if op == "ocr":
        return visual_intelligence_utility(node.provider, "ocr")
    if op == "label_detection":
        return visual_intelligence_utility(node.provider, "label_detection")
    if op == "speech_transcription":
        return visual_intelligence_utility(node.provider, "speech_transcription")
    if op == "qa":
        if not node.model:
            raise ValueError("qa requires model (same LLM catalogue as workflow1 query)")
        return physical_node_utility(PhysicalNode("query", node.provider, node.region, node.model))
    raise ValueError(f"unknown operation: {op!r}")


for _pid_lo, _ch_lo in WF2_LOGICAL_OPTIMAL_NODES.items():
    validate_exclusive_path_nodes(_pid_lo, _ch_lo)


def wf2_logical_optimal_chain(
    path_id: WF2PathId,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...],
    weights: Sequence[float],
) -> tuple[WF2PhysicalNode, ...]:
    """LO：各层独立 argmax w_i μ_i；均衡权重且写死 LO 落在当前候选中时直接返回。"""
    L = len(cands)
    if len(weights) != L:
        raise ValueError("weights length must match layers")
    w_t = tuple(weights)
    inv_l = 1.0 / float(L)
    if all(abs(w_t[i] - inv_l) < 1e-15 for i in range(L)):
        fixed = WF2_LOGICAL_OPTIMAL_NODES[path_id]
        if all(fixed[i] in cands[i] for i in range(L)):
            return fixed
    picks: list[WF2PhysicalNode] = []
    for i in range(L):
        best_j = max(
            range(len(cands[i])),
            key=lambda j: weights[i] * wf2_node_utility(cands[i][j]),
        )
        picks.append(cands[i][best_j])
    return tuple(picks)


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
        if n.operation == "database":
            t += database_storage_cost_usd(n.provider, n.region, gb, days=1.0)
        else:
            t += storage_cost_usd(n.provider, n.region, gb, days=1.0)
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
    qa_llm_tokens_override: tuple[float, float] | None = None,
) -> float:
    """单步执行费（不含存储/网络）。"""
    p, r = node.provider, node.region
    if logical_op == "shot_detection":
        return video_service_cost_usd(p, r, "shot_detection", seg_minutes_source_video)
    if logical_op == "video_split":
        return split_cost_usd(p, r, minutes=1.0)
    if logical_op == "video_caption":
        if not node.model or llm_tokens_bundle is None:
            raise ValueError("video_caption requires model and llm_tokens_bundle")
        cin, cout = llm_tokens_bundle[0]
        return llm_token_cost_usd(p, r, node.model, cin, cout)
    if logical_op == "ocr":
        return video_service_cost_usd(p, r, "ocr", seg_minutes_source_video)
    if logical_op == "label_detection":
        return video_service_cost_usd(p, r, "label_detection", seg_minutes_source_video)
    if logical_op == "speech_transcription":
        return video_service_cost_usd(p, r, "speech_transcription", seg_minutes_source_video)
    if logical_op == "database":
        return database_instance_cost_usd(p, r, days=1.0)
    if logical_op == "qa":
        if not node.model or llm_tokens_bundle is None:
            raise ValueError("qa requires model and llm bundle for LLM costing")
        qin, qout = (
            qa_llm_tokens_override
            if qa_llm_tokens_override is not None
            else llm_tokens_bundle[1]
        )
        return llm_token_cost_usd(p, r, node.model, qin, qout)
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
    car = _carrier_shot_detection(node.provider, node.region)
    if logical_op == "shot_detection":
        return sample_shot_detection_execute_sec(
            dur_sec,
            rng=rng,
            node=car,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
        )
    if logical_op == "video_split":
        return sample_video_split_execute_sec(
            dur_sec,
            rng=rng,
            node=PhysicalNode("video_split", node.provider, node.region),
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
    llm_bundle = wf2_llm_token_bundle(path_id, s_src_gb, rho)
    qa_tok = answer_tokens if answer_tokens is not None else llm_bundle[1]

    s_in, xfer = propagate_path_sizes(s_src_gb, rho)
    ops = path_logical_ops(path_id)
    exe = 0.0
    for i, op in enumerate(ops):
        exe += _exe_cost_logical_step(
            op,
            nodes[i],
            seg_minutes_source_video=seg_min,
            llm_tokens_bundle=llm_bundle,
            qa_llm_tokens_override=qa_tok if op == "qa" else None,
        )
    stor = _storage_cost_nodes(nodes, s_in, rho)
    net = egress_cost_usd(
        _WF2_LOCAL_EP,
        _endpoint(nodes[0].provider, nodes[0].region),
        float(s_in[0]),
    )
    for i in range(len(nodes) - 1):
        net += egress_cost_usd(
            _endpoint(nodes[i].provider, nodes[i].region),
            _endpoint(nodes[i + 1].provider, nodes[i + 1].region),
            xfer[i],
        )
    net += egress_cost_usd(
        _endpoint(nodes[-1].provider, nodes[-1].region),
        _WF2_LOCAL_EP,
        float(xfer[-1]),
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

    cap_pair, q_pair = wf2_llm_token_bundle(path_id, s_src_gb, rho)
    cap_out = cap_pair[1]
    q_out = float(answer_output_tokens) if answer_output_tokens is not None else q_pair[1]

    vo = vi_exe_override or {}

    s_in, xfer = propagate_path_sizes(s_src_gb, rho)
    _, lat_up = wf1_utils.local_edge_cost_latency(
        _endpoint(nodes[0].provider, nodes[0].region),
        float(s_in[0]),
        direction="upload",
        rng=wf1_utils.det_rng(er.randrange(0, 2**31), "wf2_exc_up", path_id),
    )
    lat = lat_up
    for i, op in enumerate(ops):
        n = nodes[i]
        if op == "shot_detection":
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
        elif op == "video_split":
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
                node=_carrier_shot_detection(n.provider, n.region),
                execution_scale_scope=execution_scale_scope,
                execution_scale_seed=execution_scale_seed,
            )
        elif op == "qa":
            if not n.model:
                raise ValueError("qa requires model")
            lat += llm_decode_duration_sec(n.model, q_out, rng=er)
        else:
            raise RuntimeError(op)

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

    _, lat_dn = wf1_utils.local_edge_cost_latency(
        _endpoint(nodes[-1].provider, nodes[-1].region),
        float(xfer[-1]),
        direction="download",
        rng=wf1_utils.det_rng(er.randrange(0, 2**31), "wf2_exc_dn", path_id),
    )
    lat += lat_dn

    return lat


def end_to_end_cost_parallel_shot_modalities(
    *,
    node_video: WF2PhysicalNode,
    node_shot: WF2PhysicalNode,
    modality_nodes: dict[str, WF2PhysicalNode],
    node_database: WF2PhysicalNode,
    node_qa: WF2PhysicalNode,
    active_modalities: frozenset[str],
    s_src_gb: float,
    rho_video: float,
    rho_shot: float,
    rho_modalities: dict[str, float],
    rho_database: float,
    rho_qa: float,
    llm_token_rng_seed: int | None = 42,
    answer_tokens: tuple[float, float] | None = None,
) -> float:
    """
    ``video_split`` 之后并行跑多种 VI/字幕 分支时的费用（分支费用相加；写入 DB 的载荷按分支输出之和占位）。
    ``active_modalities`` 元素须为 ``video_caption`` / ``ocr`` / ``label_detection``。
    """
    if not active_modalities:
        raise ValueError("active_modalities must be non-empty")

    seg_min = _segment_video_minutes_from_source(s_src_gb)

    s_after_video = s_src_gb * rho_video
    s_after_split = s_after_video * rho_shot
    rho_cap = float(rho_modalities.get("video_caption", 0.0))
    rho_syn = (rho_video, rho_shot, rho_cap, rho_database, rho_qa)
    llm_bundle = wf2_llm_token_bundle("video_caption", s_src_gb, rho_syn)

    qa_pair = answer_tokens if answer_tokens is not None else llm_bundle[1]

    exe = _exe_cost_logical_step(
        "shot_detection",
        node_video,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
    )
    exe += _exe_cost_logical_step(
        "video_split",
        node_shot,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
    )

    xfer_v_s = s_src_gb * rho_video

    net = egress_cost_usd(
        _WF2_LOCAL_EP,
        _endpoint(node_video.provider, node_video.region),
        float(s_src_gb),
    )
    net += egress_cost_usd(
        _endpoint(node_video.provider, node_video.region),
        _endpoint(node_shot.provider, node_shot.region),
        xfer_v_s,
    )

    stor = storage_cost_usd(
        node_video.provider,
        node_video.region,
        s_src_gb * (1.0 + rho_video),
        days=1.0,
    )
    stor += storage_cost_usd(
        node_shot.provider,
        node_shot.region,
        s_after_video * (1.0 + rho_shot),
        days=1.0,
    )

    branch_outputs_gb: list[float] = []
    for m in sorted(active_modalities):
        if m not in modality_nodes:
            raise KeyError(f"missing modality node for {m!r}")
        if m not in rho_modalities:
            raise KeyError(f"missing rho for modality {m!r}")
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        s_out_m = s_after_split * rho_m
        branch_outputs_gb.append(s_out_m)

        exe += _exe_cost_logical_step(
            m,
            mn,
            seg_minutes_source_video=seg_min,
            llm_tokens_bundle=llm_bundle,
        )
        xfer_s_m = s_after_split * rho_m
        net += egress_cost_usd(
            _endpoint(node_shot.provider, node_shot.region),
            _endpoint(mn.provider, mn.region),
            xfer_s_m,
        )
        stor += storage_cost_usd(
            mn.provider,
            mn.region,
            s_after_split * (1.0 + rho_m),
            days=1.0,
        )

    s_db_in = sum(branch_outputs_gb)

    for m in sorted(active_modalities):
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        s_out_m = s_after_split * rho_m
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
    )
    exe += _exe_cost_logical_step(
        "qa",
        node_qa,
        seg_minutes_source_video=seg_min,
        llm_tokens_bundle=llm_bundle,
        qa_llm_tokens_override=qa_pair,
    )

    stor += database_storage_cost_usd(
        node_database.provider,
        node_database.region,
        s_db_in * (1.0 + rho_database),
        days=1.0,
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
        days=1.0,
    )
    s_qa_out = float(s_after_db) * float(rho_qa)
    net += egress_cost_usd(
        _endpoint(node_qa.provider, node_qa.region),
        _WF2_LOCAL_EP,
        s_qa_out,
    )

    return exe + stor + net


def end_to_end_latency_parallel_shot_modalities(
    *,
    node_video: WF2PhysicalNode,
    node_shot: WF2PhysicalNode,
    modality_nodes: dict[str, WF2PhysicalNode],
    node_database: WF2PhysicalNode,
    node_qa: WF2PhysicalNode,
    active_modalities: frozenset[str],
    s_src_gb: float,
    rho_video: float,
    rho_shot: float,
    rho_modalities: dict[str, float],
    rho_database: float,
    rho_qa: float,
    llm_token_rng_seed: int | None = 42,
    env_rng: random.Random | None = None,
    execution_scale_scope: str | None = None,
    execution_scale_seed: int | None = None,
    answer_output_tokens: float | None = None,
) -> float:
    """并行分支 wall-clock（示意）：前缀 shot→video_split + ``max`` 并行支路 + database→qa + 下行。"""
    if not active_modalities:
        raise ValueError("active_modalities must be non-empty")

    er = env_rng or random.Random()
    seg_min = _segment_video_minutes_from_source(s_src_gb)
    dur_sec = max(seg_min * 60.0, 1e-6)
    s_after_video = s_src_gb * rho_video
    s_after_split = s_after_video * rho_shot
    rho_cap = float(rho_modalities.get("video_caption", 0.0))
    rho_syn = (rho_video, rho_shot, rho_cap, rho_database, rho_qa)
    cap_pair, q_pair = wf2_llm_token_bundle("video_caption", s_src_gb, rho_syn)
    cap_out = cap_pair[1]
    q_out = float(answer_output_tokens) if answer_output_tokens is not None else q_pair[1]

    _, el_up = wf1_utils.local_edge_cost_latency(
        _endpoint(node_video.provider, node_video.region),
        float(s_src_gb),
        direction="upload",
        rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_up"),
    )

    t_vid = _sample_vi_execute_sec(
        "shot_detection",
        dur_sec,
        node_video,
        rng=er,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )
    t_vspl = _sample_vi_execute_sec(
        "video_split",
        dur_sec,
        node_shot,
        rng=er,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )
    xfer_vs = s_src_gb * rho_video
    _, el_vs = _edge_cost_latency(node_video, node_shot, xfer_vs, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_vs"))

    branch_times: list[float] = []
    for m in sorted(active_modalities):
        mn = modality_nodes[m]
        rho_m = rho_modalities[m]
        xfer_sm = s_after_split * rho_m
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

        s_out_m = s_after_split * rho_m
        _, el_md = _edge_cost_latency(mn, node_database, s_out_m, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_md", m))
        branch_times.append(el_sm + t_m + el_md)

    t_parallel = max(branch_times)

    s_db_in = sum(s_after_split * rho_modalities[m] for m in active_modalities)
    xfer_suffix = s_db_in * rho_database
    _, el_dq = _edge_cost_latency(node_database, node_qa, xfer_suffix, rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_dq"))
    s_after_db = s_db_in * rho_database

    if not node_qa.model:
        raise ValueError("qa requires model")
    t_qa = llm_decode_duration_sec(node_qa.model, q_out, rng=er)

    t_db = sample_database_query_execute_sec(
        rng=er,
        node=_carrier_shot_detection(node_database.provider, node_database.region),
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
    )

    s_qa_out = float(s_after_db) * float(rho_qa)
    _, el_dn = wf1_utils.local_edge_cost_latency(
        _endpoint(node_qa.provider, node_qa.region),
        s_qa_out,
        direction="download",
        rng=wf1_utils.det_rng(er.randrange(0, 2**31), "par_dn"),
    )

    return el_up + t_vid + el_vs + t_vspl + t_parallel + t_db + el_dq + t_qa + el_dn


def reference_deployment_exclusive_path(path_id: WF2PathId) -> tuple[WF2PhysicalNode, ...]:
    p, r = "GCP", "us-east1"
    cap_m = "Gemini 2.5 Pro"
    ans_m = "Gemini 2.5 Flash"
    if path_id == "speech_transcription":
        return (
            WF2PhysicalNode("speech_transcription", p, r),
            WF2PhysicalNode("database", p, r),
            WF2PhysicalNode("qa", p, r, ans_m),
        )
    mid: WF2LogicalOp
    if path_id == "video_caption":
        mid = "video_caption"
    elif path_id == "ocr":
        mid = "ocr"
    elif path_id == "label_detection":
        mid = "label_detection"
    else:
        raise ValueError(path_id)
    return (
        WF2PhysicalNode("shot_detection", p, r),
        WF2PhysicalNode("video_split", p, r),
        WF2PhysicalNode(mid, p, r, cap_m if mid == "video_caption" else None),
        WF2PhysicalNode("database", p, r),
        WF2PhysicalNode("qa", p, r, ans_m),
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
    s_src_gb_ref: float = 1.0,
) -> tuple[float, ...]:
    """Plug-in mean ρ：每次蒙特卡洛抽取整条路径（含 sequential database）再按位取平均。"""
    ops = path_logical_ops(path_id)
    if n_calibration_samples <= 0:
        raise ValueError("n_calibration_samples must be positive")
    sums = [0.0] * len(ops)
    inv = 1.0 / float(n_calibration_samples)
    for _ in range(n_calibration_samples):
        rho = sample_wf2_path_rho(path_id, rng, s_src_gb_ref)
        for i, r in enumerate(rho):
            sums[i] += r
    return tuple(s * inv for s in sums)


_WF2_BUDGET_PARALLEL_MODALITIES = frozenset({"video_caption", "ocr", "label_detection"})


def _pick_wf2_same_region(cands: tuple[WF2PhysicalNode, ...], prov: str, reg: str) -> WF2PhysicalNode:
    for n in cands:
        if n.provider == prov and n.region == reg:
            return n
    return cands[0]


def wf2_modality_nodes_from_wf1_physical_ref(
    wf1_chain: tuple[PhysicalNode, PhysicalNode, PhysicalNode, PhysicalNode],
) -> dict[str, WF2PhysicalNode]:
    """
    并行 multimodal 段：``video_caption`` 采用 workflow1 budget 的 caption 节点；
    ``ocr`` / ``label_detection`` 在候选中与 WF1 ``video_split`` 同 region 对齐（与展示 DAG 一致）。
    """
    from .candidates import candidates_for_logical_op

    _sd, split_wf1, cap_wf1, _q = wf1_chain
    cap_m = cap_wf1.model
    if not cap_m:
        raise ValueError("WF1 reference caption requires model")
    cap = WF2PhysicalNode("video_caption", cap_wf1.provider, cap_wf1.region, cap_m)
    ocr = _pick_wf2_same_region(
        candidates_for_logical_op("ocr"), split_wf1.provider, split_wf1.region
    )
    lab = _pick_wf2_same_region(
        candidates_for_logical_op("label_detection"), split_wf1.provider, split_wf1.region
    )
    return {"video_caption": cap, "ocr": ocr, "label_detection": lab}


def wf2_parallel_budget_mean_rhos(
    *,
    n_calibration_samples: int,
    rng: random.Random,
) -> tuple[float, float, dict[str, float], float, float]:
    """
    shot→split→(并行 VI)→db→qa 的 plug-in mean：video/split/db/qa 与 ``video_caption`` 主干一致，
    各并行分支的压缩比取各自 path 上该算子位的平均。
    """
    r_vc = plugin_mean_data_conversion_ratios_wf2(
        "video_caption",
        n_calibration_samples=n_calibration_samples,
        rng=random.Random(rng.randrange(0, 2**31)),
    )
    r_ocr = plugin_mean_data_conversion_ratios_wf2(
        "ocr",
        n_calibration_samples=n_calibration_samples,
        rng=random.Random(rng.randrange(0, 2**31)),
    )
    r_lab = plugin_mean_data_conversion_ratios_wf2(
        "label_detection",
        n_calibration_samples=n_calibration_samples,
        rng=random.Random(rng.randrange(0, 2**31)),
    )
    rho_modalities = {
        "video_caption": float(r_vc[2]),
        "ocr": float(r_ocr[2]),
        "label_detection": float(r_lab[2]),
    }
    return float(r_vc[0]), float(r_vc[1]), rho_modalities, float(r_vc[3]), float(r_vc[4])


def wf2_budget_meanfield_cost_latency(
    path_id: WF2PathId,
    chain: tuple[WF2PhysicalNode, ...],
    s_src_gb: float,
    *,
    mean_rho_exclusive: tuple[float, ...],
    parallel_rho: tuple[float, float, dict[str, float], float, float] | None,
    modality_nodes: dict[str, WF2PhysicalNode] | None,
    workflow_rng: random.Random,
) -> tuple[float, float]:
    """
    ``workflow2.budget`` / 校准 query 共用：**非 speech** 路径按完整 DAG（并行支路费用求和、
    wall-clock≈前缀+max(支路)+db+qa）；``speech_transcription`` 仍为单路径串行。
    """
    ch = tuple(chain)
    if path_id == "speech_transcription":
        llm_seed = workflow_rng.randrange(0, 2**31)
        c = end_to_end_cost_exclusive_path(
            path_id,
            ch,
            float(s_src_gb),
            mean_rho_exclusive,
            llm_token_rng_seed=llm_seed,
        )
        ell = end_to_end_latency_exclusive_path(
            path_id,
            ch,
            float(s_src_gb),
            mean_rho_exclusive,
            llm_token_rng_seed=llm_seed,
            env_rng=workflow_rng,
            execution_scale_scope=str(llm_seed),
            execution_scale_seed=llm_seed,
        )
        return float(c), float(ell)

    if parallel_rho is None or modality_nodes is None:
        raise ValueError("parallel_rho and modality_nodes required for shot-modalities paths")
    rho_video, rho_shot, rho_modalities, rho_db, rho_qa = parallel_rho
    llm_seed = workflow_rng.randrange(0, 2**31)
    c = end_to_end_cost_parallel_shot_modalities(
        node_video=ch[0],
        node_shot=ch[1],
        modality_nodes=modality_nodes,
        node_database=ch[3],
        node_qa=ch[4],
        active_modalities=_WF2_BUDGET_PARALLEL_MODALITIES,
        s_src_gb=float(s_src_gb),
        rho_video=rho_video,
        rho_shot=rho_shot,
        rho_modalities=rho_modalities,
        rho_database=rho_db,
        rho_qa=rho_qa,
        llm_token_rng_seed=llm_seed,
    )
    ell = end_to_end_latency_parallel_shot_modalities(
        node_video=ch[0],
        node_shot=ch[1],
        modality_nodes=modality_nodes,
        node_database=ch[3],
        node_qa=ch[4],
        active_modalities=_WF2_BUDGET_PARALLEL_MODALITIES,
        s_src_gb=float(s_src_gb),
        rho_video=rho_video,
        rho_shot=rho_shot,
        rho_modalities=rho_modalities,
        rho_database=rho_db,
        rho_qa=rho_qa,
        llm_token_rng_seed=llm_seed,
        env_rng=workflow_rng,
        execution_scale_scope=str(llm_seed),
        execution_scale_seed=llm_seed,
    )
    return float(c), float(ell)


def generate_realistic_queries_wf2(
    num_queries: int,
    path_id: WF2PathId,
    *,
    seed: int = 42,
    n_calibration_samples: int = 4096,
    budget_cost_multiplier: float | None = None,
    budget_latency_multiplier: float | None = None,
    budget_alpha: float | None = None,
    cands: tuple[tuple[WF2PhysicalNode, ...], ...] | None = None,
    weights: Sequence[float] | None = None,
    wf2_enum_full_cross_product: bool = WF2_BUDGET_ENUM_FULL_CROSS_PRODUCT,
) -> list[QueryProfile]:
    """
    生成 workflow2 校准 query 及 ``(Θ_C, Θ_T)``。

    * **插值模式**（``budget_alpha`` 非空）：在 ``iter_chains_wf2`` 枚举的部署链上，以
      ``det_rng(seed, "wf2_budget_combo_eval", q_idx)`` 与 ``workflow2.budget`` 相同口径计算
      各链 mean-field cost/latency，取 ``C_min,L_min``；在 LO 链上取 ``C_max,L_max``
      （``det_rng(seed, "wf2_budget_alpha_lo", q_idx)``），再
      ``Θ_C=C_min+α(C_max-C_min)``、``Θ_T=L_min+α(L_max-L_min)``。
      需提供 ``cands`` 与 ``weights``。

    * **参考链乘子模式**：使用 ``_BUDGET_REF_CHAINS_WF2_*`` 与乘子（默认 1.75）。
      video 类 path 上 ``Θ`` 与 ``workflow2.budget`` 相同：**并行多模态费用求和**、
      wall-clock 时延按 ``max(分支)`` 聚合（见 ``wf2_budget_meanfield_cost_latency``）。
    """
    if budget_alpha is not None and (
        budget_cost_multiplier is not None or budget_latency_multiplier is not None
    ):
        raise ValueError("budget_alpha 与 budget_*_multiplier 不可同时指定")

    if budget_alpha is not None:
        if cands is None or weights is None:
            raise ValueError("budget_alpha 模式下必须提供 cands 与 weights")
        lo_chain = wf2_logical_optimal_chain(path_id, cands, weights)
    else:
        lo_chain = None

    rng = random.Random(seed)
    calib_rng = random.Random(seed + 100)
    mean_rho = plugin_mean_data_conversion_ratios_wf2(
        path_id,
        n_calibration_samples=n_calibration_samples,
        rng=calib_rng,
    )

    parallel_rho: tuple[float, float, dict[str, float], float, float] | None = None
    modality_nodes: dict[str, WF2PhysicalNode] | None = None
    if path_id != "speech_transcription":
        parallel_rho = wf2_parallel_budget_mean_rhos(
            n_calibration_samples=n_calibration_samples,
            rng=random.Random(seed + 101),
        )
        modality_nodes = wf2_modality_nodes_from_wf1_physical_ref(
            wf1_utils._BUDGET_REF_CHAIN_COST_WF1
        )

    chain_c = _BUDGET_REF_CHAINS_WF2_COST[path_id]
    chain_l = _BUDGET_REF_CHAINS_WF2_LATENCY[path_id]
    mult_c = (
        budget_cost_multiplier
        if budget_cost_multiplier is not None
        else wf1_utils.QUERY_BUDGET_REFERENCE_MULTIPLIER
    )
    mult_l = (
        budget_latency_multiplier
        if budget_latency_multiplier is not None
        else wf1_utils.QUERY_BUDGET_REFERENCE_MULTIPLIER
    )

    queries: list[QueryProfile] = []
    for q_idx in range(num_queries):
        duration_sec = rng.uniform(60.0, 3600.0)
        s_src_mb = cfg.video_megabytes_from_duration_sec(duration_sec)
        s_src_gb = s_src_mb / 1000.0

        if budget_alpha is not None:
            assert cands is not None and lo_chain is not None
            c_min = math.inf
            l_min = math.inf
            if path_id == "speech_transcription":
                chain_iter = iter_chains_wf2(
                    cands,
                    path_id=path_id,
                    full_cross_product=wf2_enum_full_cross_product,
                )
            else:
                from .candidates import enumerate_candidates_wf2 as _enum_vc

                cands_vc = _enum_vc("video_caption")
                chain_iter = iter_chains_wf2_end_to_end_shot_modalities(
                    cands_vc,
                    full_cross_product=wf2_enum_full_cross_product,
                )
            for chain in chain_iter:
                wf = wf1_utils.det_rng(seed, "wf2_budget_combo_eval", q_idx)
                c_i, ell_i = wf2_budget_meanfield_cost_latency(
                    path_id,
                    tuple(chain),
                    s_src_gb,
                    mean_rho_exclusive=mean_rho,
                    parallel_rho=parallel_rho,
                    modality_nodes=modality_nodes,
                    workflow_rng=wf,
                )
                if math.isfinite(c_i):
                    c_min = min(c_min, float(c_i))
                if math.isfinite(ell_i):
                    l_min = min(l_min, float(ell_i))
            wf_lo = wf1_utils.det_rng(seed, "wf2_budget_alpha_lo", q_idx)
            c_lo, l_lo = wf2_budget_meanfield_cost_latency(
                path_id,
                lo_chain,
                s_src_gb,
                mean_rho_exclusive=mean_rho,
                parallel_rho=parallel_rho,
                modality_nodes=modality_nodes,
                workflow_rng=wf_lo,
            )
            span_c = max(c_lo, c_min) - min(c_lo, c_min)
            span_l = max(l_lo, l_min) - min(l_lo, l_min)
            theta_c = min(c_lo, c_min) + float(budget_alpha) * span_c
            theta_l = min(l_lo, l_min) + float(budget_alpha) * span_l
            queries.append(
                QueryProfile(
                    s_src_gb=s_src_gb,
                    theta_cost=theta_c,
                    theta_latency_sec=theta_l,
                )
            )
            continue

        wf_c = wf1_utils.det_rng(seed, "wf2_query_budget_ref_cost", q_idx)
        ref_c, _ = wf2_budget_meanfield_cost_latency(
            path_id,
            chain_c,
            s_src_gb,
            mean_rho_exclusive=mean_rho,
            parallel_rho=parallel_rho,
            modality_nodes=modality_nodes,
            workflow_rng=wf_c,
        )
        wf_l = wf1_utils.det_rng(seed, "wf2_query_budget_ref_latency", q_idx)
        _, ref_l = wf2_budget_meanfield_cost_latency(
            path_id,
            chain_l,
            s_src_gb,
            mean_rho_exclusive=mean_rho,
            parallel_rho=parallel_rho,
            modality_nodes=modality_nodes,
            workflow_rng=wf_l,
        )

        queries.append(
            QueryProfile(
                s_src_gb=s_src_gb,
                theta_cost=ref_c * mult_c,
                theta_latency_sec=ref_l * mult_l,
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

    rho = sample_wf2_path_rho(path_id, workflow_rng, s_src_gb)
    llm_seed = workflow_rng.randrange(0, 2**31)
    _cap, q_pair = wf2_llm_token_bundle(path_id, s_src_gb, rho)
    q_out = float(q_pair[1])

    c = end_to_end_cost_exclusive_path(
        path_id,
        nodes,
        s_src_gb,
        rho,
        llm_token_rng_seed=llm_seed,
        answer_tokens=q_pair,
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
    for pid in ("video_caption", "ocr", "label_detection", "speech_transcription"):
        chain = reference_deployment_exclusive_path(pid)
        rho = sample_wf2_path_rho(pid, ri, 0.5)
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
    cap_ops = path_logical_ops("video_caption")
    rv = sample_wf2_logical_ratio("shot_detection", ri, path_ops=cap_ops, layer_index=0)
    rs = sample_wf2_logical_ratio("video_split", ri, path_ops=cap_ops, layer_index=1)
    rho_m = {k: sample_wf2_logical_ratio(k, ri) for k in nodes_mod}
    s_db_in = sum(0.5 * rv * rs * rho_m[k] for k in sorted(rho_m))
    rho_db = sample_database_conversion_ratio(s_db_in, ri)
    print(
        "parallel_cost",
        end_to_end_cost_parallel_shot_modalities(
            node_video=WF2PhysicalNode("shot_detection", "GCP", "us-east1"),
            node_shot=WF2PhysicalNode("video_split", "GCP", "us-east1"),
            modality_nodes=nodes_mod,
            node_database=WF2PhysicalNode("database", "GCP", "us-east1"),
            node_qa=WF2PhysicalNode("qa", "GCP", "us-east1", "Gemini 2.5 Flash"),
            active_modalities=frozenset({"video_caption", "ocr"}),
            s_src_gb=0.5,
            rho_video=rv,
            rho_shot=rs,
            rho_modalities=rho_m,
            rho_database=rho_db,
            rho_qa=sample_wf2_logical_ratio("qa", ri, path_ops=cap_ops, layer_index=4),
            llm_token_rng_seed=11,
        ),
    )
