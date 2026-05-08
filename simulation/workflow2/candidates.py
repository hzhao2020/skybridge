"""
每层逻辑算子的物理候选枚举（与 ``sim_env.config`` / ``sim_env.cost`` 对齐）。

database / qa： footprint 与 segment 一致以控制 MILP 规模。
"""

from __future__ import annotations

from sim_env import config as cfg
from sim_env.cost import VIDEO_SERVICE_USD_PER_MINUTE

from .utils import WF2LogicalOp, WF2PathId, WF2PhysicalNode, path_logical_ops


def _segment_like_candidates(op: WF2LogicalOp) -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["segment"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            out.append(WF2PhysicalNode(op, prov, reg))
    if not out:
        raise RuntimeError(f"No candidates for {op!r} (segment footprint)")
    return out


def _split_like_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["split"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            out.append(WF2PhysicalNode("shot_detection", prov, reg))
    if not out:
        raise RuntimeError("No shot_detection candidates")
    return out


def _caption_like_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["caption"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            for m in pdata["models"]:
                out.append(WF2PhysicalNode("video_caption", prov, reg, m))
    if not out:
        raise RuntimeError("No video_caption candidates")
    return out


def _query_like_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["query"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            for m in pdata["models"]:
                out.append(WF2PhysicalNode("answer", prov, reg, m))
    if not out:
        raise RuntimeError("No answer candidates")
    return out


def _video_intelligence_candidates(service_key: str, op: WF2LogicalOp) -> list[WF2PhysicalNode]:
    out: list[WF2PhysicalNode] = []
    for prov, regmap in VIDEO_SERVICE_USD_PER_MINUTE.items():
        for reg, svc in regmap.items():
            if service_key in svc:
                out.append(WF2PhysicalNode(op, prov, reg))
    if not out:
        raise RuntimeError(f"No {op!r} candidates for service {service_key!r}")
    return out


def candidates_for_logical_op(op: WF2LogicalOp) -> tuple[WF2PhysicalNode, ...]:
    if op == "video_segment":
        return tuple(_segment_like_candidates("video_segment"))
    if op == "shot_detection":
        return tuple(_split_like_candidates())
    if op == "video_caption":
        return tuple(_caption_like_candidates())
    if op == "ocr":
        return tuple(_video_intelligence_candidates("ocr", "ocr"))
    if op == "label_detection":
        return tuple(_video_intelligence_candidates("label_detection", "label_detection"))
    if op == "speech_transcription":
        return tuple(_video_intelligence_candidates("speech_transcription", "speech_transcription"))
    if op == "database":
        return tuple(_segment_like_candidates("database"))
    if op == "qa":
        return tuple(_segment_like_candidates("qa"))
    if op == "answer":
        return tuple(_query_like_candidates())
    raise ValueError(f"unknown logical op: {op!r}")


def enumerate_candidates_wf2(path_id: WF2PathId) -> tuple[tuple[WF2PhysicalNode, ...], ...]:
    """与 ``path_logical_ops(path_id)`` 等长的候选元组。"""
    ops = path_logical_ops(path_id)
    return tuple(candidates_for_logical_op(op) for op in ops)
