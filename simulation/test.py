#!/usr/bin/env python3
"""
根据一组 operation 节点选择，计算端到端 latency（秒）、cost（USD）、accuracy（utility 加权和，作为质量指标）。

用法示例:
  python simulation/test.py --video-size 120 \\
    --segment p1_r1_segment --split p2_r1_split \\
    --caption p4_m3_caption --query p5_m6_query

  加 --verbose 或 -v 可输出 latency/cost/accuracy 各组成部分，便于 debug:
  python simulation/test.py --video-size 120 -v

或在代码中（在 ``simulation/`` 目录下或已把该目录加入 PYTHONPATH）:
  from test import evaluate_path
  r = evaluate_path(video_size_mb=120.0, segment_name="p1_r1_segment", ...)
  r = evaluate_path(..., return_breakdown=True)  # 返回各组成部分
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from distribution import SimulationParams

from distribution import (
    Node,
    compute_end_to_end_cost_usd,
    compute_end_to_end_cost_usd_breakdown,
    compute_end_to_end_latency_s,
    compute_end_to_end_latency_s_breakdown,
    compute_end_to_end_utility_score,
    compute_end_to_end_utility_score_breakdown,
    get_fixed_simulation_params,
    sample_data_conversion_ratios_all,
)


def _find_node(nodes: Iterable[Node], name: str) -> Node:
    for n in nodes:
        if n.name == name:
            return n
    raise KeyError(f"未找到节点: {name!r}。可选名称示例: {[x.name for x in list(nodes)[:3]]}...")


def evaluate_path(
    *,
    video_size_mb: float,
    segment_name: str,
    split_name: str,
    caption_name: str,
    query_name: str,
    return_breakdown: bool = False,
    params: "SimulationParams | None" = None,
) -> dict[str, float] | dict[str, float | dict[str, float]]:
    """
    输入四个 operation 的节点名称，返回 latency_s、cost_usd、accuracy。

    - accuracy: 与 ``compute_end_to_end_utility_score`` 相同，为 segment/caption/query
      节点 utility 的加权和（split 权重为 0），取值约在 [0, 1]。

    - return_breakdown: 若为 True，则额外返回 latency_breakdown、cost_breakdown、accuracy_breakdown，
      便于 debug 时查看各组成部分。

    - params: 仿真参数，None 则用 get_fixed_simulation_params()（首次调用时采样一次）。
    """
    p = params or get_fixed_simulation_params()
    seg = _find_node(p.build_nodes("segment"), segment_name)
    spl = _find_node(p.build_nodes("split"), split_name)
    cap = _find_node(p.build_nodes("caption"), caption_name)
    qry = _find_node(p.build_nodes("query"), query_name)

    # 一次采样 data conversion ratios，确保 Latency 与 Cost 使用同一套数据量（Pareto 分析自洽）
    data_ratios = sample_data_conversion_ratios_all()

    if return_breakdown:
        latency_bd = compute_end_to_end_latency_s_breakdown(
            video_size_mb=video_size_mb,
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
            data_conversion_ratios=data_ratios,
        )
        cost_bd = compute_end_to_end_cost_usd_breakdown(
            video_size_mb=video_size_mb,
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
            data_conversion_ratios=data_ratios,
        )
        accuracy_bd = compute_end_to_end_utility_score_breakdown(
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
        )
        return {
            "latency_s": latency_bd["total_s"],
            "cost_usd": cost_bd["total_usd"],
            "accuracy": accuracy_bd["total"],
            "latency_breakdown": latency_bd,
            "cost_breakdown": cost_bd,
            "accuracy_breakdown": accuracy_bd,
        }
    else:
        latency_s = compute_end_to_end_latency_s(
            video_size_mb=video_size_mb,
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
            data_conversion_ratios=data_ratios,
        )
        cost_usd = compute_end_to_end_cost_usd(
            video_size_mb=video_size_mb,
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
            data_conversion_ratios=data_ratios,
        )
        accuracy = compute_end_to_end_utility_score(
            segment_node=seg,
            split_node=spl,
            caption_node=cap,
            query_node=qry,
        )
        return {
            "latency_s": latency_s,
            "cost_usd": cost_usd,
            "accuracy": accuracy,
        }


@dataclass(frozen=True)
class PathResult:
    latency_s: float
    cost_usd: float
    accuracy: float


def evaluate_path_dataclass(
    *,
    video_size_mb: float,
    segment_name: str,
    split_name: str,
    caption_name: str,
    query_name: str,
) -> PathResult:
    d = evaluate_path(
        video_size_mb=video_size_mb,
        segment_name=segment_name,
        split_name=split_name,
        caption_name=caption_name,
        query_name=query_name,
    )
    return PathResult(
        latency_s=d["latency_s"],
        cost_usd=d["cost_usd"],
        accuracy=d["accuracy"],
    )


def _list_example_node_names() -> None:
    for wf in ("segment", "split", "caption", "query"):
        nodes = get_fixed_simulation_params().build_nodes(wf)  # type: ignore[arg-type]
        print(f"[{wf}] 共 {len(nodes)} 个节点，示例: {[n.name for n in nodes[:4]]}")


def _print_breakdown(r: dict) -> None:
    """打印 latency / cost / accuracy 各组成部分，便于 debug。"""
    for key in ("latency_breakdown", "cost_breakdown", "accuracy_breakdown"):
        if key not in r:
            continue
        bd = r[key]
        label = {"latency_breakdown": "Latency (s)", "cost_breakdown": "Cost (USD)", "accuracy_breakdown": "Accuracy"}[key]
        print(f"\n--- {label} breakdown ---")
        for k, v in bd.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端 latency / cost / accuracy（utility 分数）")
    parser.add_argument("--video-size", type=float, default=100.0, help="视频大小 (MB)")
    parser.add_argument("--segment", type=str, default="p1_r1_segment", help="segment 节点 name")
    parser.add_argument("--split", type=str, default="p2_r1_split", help="split 节点 name")
    parser.add_argument("--caption", type=str, default="p4_m1_caption", help="caption 节点 name")
    parser.add_argument("--query", type=str, default="p4_m1_query", help="query 节点 name")
    parser.add_argument("--list-names", action="store_true", help="仅列出各 operation 节点名示例后退出")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 latency/cost/accuracy 各组成部分，便于 debug")
    args = parser.parse_args()

    if args.list_names:
        _list_example_node_names()
        return

    r = evaluate_path(
        video_size_mb=args.video_size,
        segment_name=args.segment,
        split_name=args.split,
        caption_name=args.caption,
        query_name=args.query,
        return_breakdown=args.verbose,
    )
    print(f"video_size_mb = {args.video_size}")
    print(f"segment={args.segment}, split={args.split}, caption={args.caption}, query={args.query}")
    print(f"latency_s   = {r['latency_s']:.6f}")
    print(f"cost_usd    = {r['cost_usd']:.6f}")
    print(f"accuracy    = {r['accuracy']:.6f}  # utility 加权和 [0,1] 量级")

    if args.verbose:
        _print_breakdown(r)


if __name__ == "__main__":
    main()
