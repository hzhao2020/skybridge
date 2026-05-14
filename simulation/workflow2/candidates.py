"""
每层逻辑算子的物理候选枚举（与 ``sim_env.config`` / ``sim_env.cost`` 对齐）。

database：footprint 与 shot_detection 一致。``qa`` 的 **(provider, region, model)** 与 workflow1
``query`` 相同枚举规模；终局 LLM 费用/解码时延在仿真中计入 ``qa``（无单独 ``answer`` 算子）。

拓扑：``shot_detection``（镜头检测云服务）→ ``video_split``（切分服务）→ …
"""

from __future__ import annotations

from sim_env import config as cfg
from sim_env.cost import VIDEO_SERVICE_USD_PER_MINUTE

from .utils import WF2LogicalOp, WF2PathId, WF2PhysicalNode, path_logical_ops


def _shot_detection_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["shot_detection"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            out.append(WF2PhysicalNode("shot_detection", prov, reg))
    if not out:
        raise RuntimeError("No shot_detection candidates")
    return out


def _video_split_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["video_split"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            out.append(WF2PhysicalNode("video_split", prov, reg))
    if not out:
        raise RuntimeError("No video_split candidates")
    return out


def _shot_detection_footprint_candidates(op: WF2LogicalOp) -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["shot_detection"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            out.append(WF2PhysicalNode(op, prov, reg))
    if not out:
        raise RuntimeError(f"No candidates for {op!r} (shot_detection footprint)")
    return out


def _caption_like_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["video_caption"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            for m in pdata["models"]:
                out.append(WF2PhysicalNode("video_caption", prov, reg, m))
    if not out:
        raise RuntimeError("No video_caption candidates")
    return out


def _qa_like_candidates() -> list[WF2PhysicalNode]:
    spec = cfg.WORKFLOW_OPERATIONS["query"]
    out: list[WF2PhysicalNode] = []
    for prov, pdata in spec.items():
        for reg in pdata["supported_regions"]:
            for m in pdata["models"]:
                out.append(WF2PhysicalNode("qa", prov, reg, m))
    if not out:
        raise RuntimeError("No qa candidates")
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


def candidates_for_logical_op(
    op: WF2LogicalOp,
    *,
    path_ops: tuple[str, ...] | None = None,
    layer_index: int | None = None,
) -> tuple[WF2PhysicalNode, ...]:
    """单上下文候选。``path_ops`` / ``layer_index`` 可为将来扩展预留，当前不参与分派。"""
    _ = path_ops, layer_index
    if op == "shot_detection":
        return tuple(_shot_detection_candidates())
    if op == "video_split":
        return tuple(_video_split_candidates())
    if op == "video_caption":
        return tuple(_caption_like_candidates())
    if op == "ocr":
        return tuple(_video_intelligence_candidates("ocr", "ocr"))
    if op == "label_detection":
        return tuple(_video_intelligence_candidates("label_detection", "label_detection"))
    if op == "speech_transcription":
        return tuple(_video_intelligence_candidates("speech_transcription", "speech_transcription"))
    if op == "database":
        return tuple(_shot_detection_footprint_candidates("database"))
    if op == "qa":
        return tuple(_qa_like_candidates())
    raise ValueError(f"unknown logical op: {op!r}")


def enumerate_candidates_wf2(path_id: WF2PathId) -> tuple[tuple[WF2PhysicalNode, ...], ...]:
    """与 ``path_logical_ops(path_id)`` 等长的候选元组。"""
    ops = path_logical_ops(path_id)
    ot = tuple(ops)
    return tuple(candidates_for_logical_op(ot[i], path_ops=ot, layer_index=i) for i in range(len(ops)))
