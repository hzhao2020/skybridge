"""
结果展示用端到端时延：论文 DAG 中 video 后存在 speech 与 shot→多模态 两条岔路，
展示延迟取 ``max(T_parallel_shot_modalities, T_speech_chain)``。

注意：**不参与** Sky MILP / CVaR 系数中的延迟聚合（那里仍为所选单路径上的求和）。
"""

from __future__ import annotations

from workflow1 import utils as wf1_utils

from .candidates import candidates_for_logical_op
from .utils import (
    WF2PathId,
    WF2PhysicalNode,
    end_to_end_latency_exclusive_path,
    end_to_end_latency_parallel_shot_modalities,
    path_logical_ops,
    sample_wf2_logical_ratio,
    validate_exclusive_path_nodes,
)


def _pick_same_region(
    cands: tuple[WF2PhysicalNode, ...],
    prov: str,
    reg: str,
) -> WF2PhysicalNode:
    for n in cands:
        if n.provider == prov and n.region == reg:
            return n
    return cands[0]


def workflow_display_latency_max_fork(
    path_id: WF2PathId,
    deployed: tuple[WF2PhysicalNode, ...],
    s_src_gb: float,
    *,
    rho_deployed: tuple[float, ...],
    q_out: float,
    workflow_rng,
    llm_seed: int,
    execution_scale_scope: str,
    execution_scale_seed: int,
    latency_sum_optimized_path: float,
) -> float:
    """
    展示用 workflow 端到端时延：``max(T_parallel, T_speech)``。

    - ``T_parallel``：``video→shot`` 后 ``caption/ocr/label`` 并行（分支段取 max）+ ``database→qa→answer``。
    - ``T_speech``：若当前优化路径为 ``speech``，则等于算法路径求和时延；否则为同区域 speech 支路 + 共享尾部。
    """
    validate_exclusive_path_nodes(path_id, deployed)

    if path_id == "speech":
        vid, _spk, db, qa, ans = deployed
    else:
        vid, shot_deployed, _mid, db, qa, ans = deployed

    prov, reg = vid.provider, vid.region

    if path_id == "speech":
        node_shot = _pick_same_region(candidates_for_logical_op("shot_detection"), prov, reg)
    else:
        node_shot = shot_deployed

    modality_nodes = {
        "video_caption": _pick_same_region(candidates_for_logical_op("video_caption"), prov, reg),
        "ocr": _pick_same_region(candidates_for_logical_op("ocr"), prov, reg),
        "label_detection": _pick_same_region(candidates_for_logical_op("label_detection"), prov, reg),
    }

    if path_id == "speech":
        rho_vid = rho_deployed[0]
        rho_shot = sample_wf2_logical_ratio(
            "shot_detection",
            wf1_utils.det_rng(llm_seed, "disp_fork_shot_rho"),
        )
        rho_db, rho_qa, rho_ans = rho_deployed[2], rho_deployed[3], rho_deployed[4]
    else:
        rho_vid, rho_shot = rho_deployed[0], rho_deployed[1]
        rho_db, rho_qa, rho_ans = rho_deployed[3], rho_deployed[4], rho_deployed[5]

    rho_modalities = {
        "video_caption": sample_wf2_logical_ratio(
            "video_caption", wf1_utils.det_rng(llm_seed, "disp_rm_cap")
        ),
        "ocr": sample_wf2_logical_ratio("ocr", wf1_utils.det_rng(llm_seed, "disp_rm_ocr")),
        "label_detection": sample_wf2_logical_ratio(
            "label_detection", wf1_utils.det_rng(llm_seed, "disp_rm_lab")
        ),
    }

    lat_parallel = end_to_end_latency_parallel_shot_modalities(
        node_video=vid,
        node_shot=node_shot,
        modality_nodes=modality_nodes,
        node_database=db,
        node_qa=qa,
        node_answer=ans,
        active_modalities=frozenset({"video_caption", "ocr", "label_detection"}),
        s_src_gb=s_src_gb,
        rho_video=rho_vid,
        rho_shot=rho_shot,
        rho_modalities=rho_modalities,
        rho_database=rho_db,
        rho_qa=rho_qa,
        rho_answer=rho_ans,
        llm_token_rng_seed=llm_seed,
        env_rng=workflow_rng,
        execution_scale_scope=execution_scale_scope,
        execution_scale_seed=execution_scale_seed,
        answer_output_tokens=q_out,
    )

    if path_id == "speech":
        lat_speech = latency_sum_optimized_path
    else:
        sp_cands = candidates_for_logical_op("speech_transcription")
        node_sp = _pick_same_region(sp_cands, prov, reg)
        speech_nodes = (vid, node_sp, db, qa, ans)
        rho_speech = tuple(
            sample_wf2_logical_ratio(
                op,
                wf1_utils.det_rng(llm_seed, "disp_rho_speech", i),
            )
            for i, op in enumerate(path_logical_ops("speech"))
        )
        sp_llm = wf1_utils.det_rng(llm_seed, "disp_sp_llm").randrange(0, 2**31)
        lat_speech = end_to_end_latency_exclusive_path(
            "speech",
            speech_nodes,
            s_src_gb,
            rho_speech,
            llm_token_rng_seed=sp_llm,
            env_rng=workflow_rng,
            execution_scale_scope=execution_scale_scope,
            execution_scale_seed=execution_scale_seed,
            answer_output_tokens=q_out,
        )

    return max(lat_parallel, lat_speech)
