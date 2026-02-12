"""
Operation 计算延迟测试工具

使用三个数据集（EgoSchema, NExTQA, ActivityNetQA）的 train 集合作为测试数据，
对当前注册的每一个 operation 进行延迟测试，并将结果保存到
`results/node_latency/` 目录下的 JSON 文件中。
"""

import os
import sys
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal

# 将项目根目录添加到 sys.path，方便从命令行独立运行
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.dataset import build_dataset
from ops.registry import REGISTRY
from ops.base import VideoSegmenter, VisualCaptioner, VideoSplitter, LLMQuery


OperationCategory = Literal["segment", "split", "caption", "llm"]


@dataclass
class OperationLatencyRecord:
    """单次 operation 调用的延迟记录"""

    pid: str
    category: OperationCategory
    provider: str
    region: str
    model_name: Optional[str]

    dataset: str
    sample_index: int
    qid: Optional[str]
    video_name: Optional[str]

    latency_seconds: float
    success: bool
    error_message: Optional[str] = None


class NodeLatencyTester:
    """基于三个数据集 train 集合的 operation 延迟测试器"""

    DATASETS = ["EgoSchema", "NExTQA", "ActivityNetQA"]

    def __init__(
        self,
        datasets_root: str | Path = "datasets",
        max_samples_per_dataset: Optional[int] = None,
        categories: Optional[List[OperationCategory]] = None,
    ):
        """
        Args:
            datasets_root: 数据集根目录（默认为项目下的 `datasets`）
            max_samples_per_dataset: 每个数据集用于测试的样本数上限；
                为 None 或 <=0 时表示“使用该数据集 train 集的所有样本”
            categories: 需要测试的 operation 类别，默认为全部
        """
        self.datasets_root = Path(datasets_root)
        if max_samples_per_dataset is None or int(max_samples_per_dataset) <= 0:
            # 默认：不做限制，使用整个训练集
            self.max_samples_per_dataset = None
        else:
            self.max_samples_per_dataset = int(max_samples_per_dataset)
        self.categories = set(categories) if categories else {"segment", "split", "caption", "llm"}

        self.records: List[OperationLatencyRecord] = []

        # 预加载三个数据集的 train 集合
        self._datasets: Dict[str, object] = {}
        for name in self.DATASETS:
            ds = build_dataset(name, "train", self.datasets_root)
            self._datasets[name] = ds

    @staticmethod
    def _categorize_operation(op) -> Optional[OperationCategory]:
        """根据 operation 类型判断所属类别"""
        if isinstance(op, VideoSegmenter):
            return "segment"
        if isinstance(op, VideoSplitter):
            return "split"
        if isinstance(op, VisualCaptioner):
            return "caption"
        if isinstance(op, LLMQuery):
            return "llm"
        return None

    def _iter_dataset_samples(self):
        """遍历三个数据集的 train 样本（受 max_samples_per_dataset 限制）"""
        for dataset_name, dataset in self._datasets.items():
            total = len(dataset)
            # None 表示不做限制，使用整个训练集
            limit = total if self.max_samples_per_dataset is None else min(total, self.max_samples_per_dataset)
            for idx in range(limit):
                sample = dataset[idx]
                yield dataset_name, idx, sample

    def _test_segment_op(self, pid: str, op, dataset_name: str, idx: int, sample: dict):
        video_path = sample.get("video_path")
        if not video_path:
            return

        qid = sample.get("qid")
        video_name = sample.get("video_name")

        # 将数据上传到各自云端时，用一个统一的前缀，便于后续清理
        target_path = f"videos/node_latency/{dataset_name.lower()}"

        start = time.time()
        try:
            # segment operation 统一接口：execute(video_uri, **kwargs)
            op.execute(video_path, target_path=target_path)
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="segment",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=True,
                )
            )
        except Exception as e:
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="segment",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=False,
                    error_message=str(e),
                )
            )

    def _test_split_op(self, pid: str, op, dataset_name: str, idx: int, sample: dict):
        video_path = sample.get("video_path")
        if not video_path:
            return

        qid = sample.get("qid")
        video_name = sample.get("video_name")

        # 为了简单起见，对每个视频只切一个短片段 [0, 5] 秒
        segments = [{"start": 0.0, "end": 5.0}]
        target_path = f"videos/node_latency/{dataset_name.lower()}"

        start = time.time()
        try:
            # VideoSplitter 接口：execute(video_uri, segments, **kwargs)
            op.execute(video_path, segments=segments, target_path=target_path)
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="split",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=True,
                )
            )
        except Exception as e:
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="split",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=False,
                    error_message=str(e),
                )
            )

    def _test_caption_op(self, pid: str, op, dataset_name: str, idx: int, sample: dict):
        video_path = sample.get("video_path")
        if not video_path:
            return

        qid = sample.get("qid")
        video_name = sample.get("video_name")

        target_path = f"videos/node_latency/{dataset_name.lower()}"

        start = time.time()
        try:
            # caption 接口：execute(video_uri, **kwargs)
            op.execute(video_path, target_path=target_path)
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="caption",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=True,
                )
            )
        except Exception as e:
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="caption",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=False,
                    error_message=str(e),
                )
            )

    def _test_llm_op(self, pid: str, op, dataset_name: str, idx: int, sample: dict):
        # 使用样本中的问题作为 prompt；若不存在则跳过
        question = sample.get("question")
        if not question:
            return

        qid = sample.get("qid")
        video_name = sample.get("video_name")

        # 控制输出长度，避免消耗过多 token
        llm_kwargs: Dict[str, object] = {}
        if hasattr(op, "provider") and getattr(op, "provider") == "openai":
            # OpenAILLMImpl 使用 max_tokens
            llm_kwargs["max_tokens"] = 256
        else:
            # 其他 LLM 实现普遍使用 max_output_tokens / max_tokens
            llm_kwargs["max_output_tokens"] = 256

        start = time.time()
        try:
            op.execute(question, **llm_kwargs)
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="llm",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=True,
                )
            )
        except Exception as e:
            end = time.time()
            self.records.append(
                OperationLatencyRecord(
                    pid=pid,
                    category="llm",
                    provider=getattr(op, "provider", ""),
                    region=getattr(op, "region", ""),
                    model_name=getattr(op, "model_name", None),
                    dataset=dataset_name,
                    sample_index=idx,
                    qid=qid,
                    video_name=video_name,
                    latency_seconds=end - start,
                    success=False,
                    error_message=str(e),
                )
            )

    def run(self):
        """对所有注册的 operation 进行延迟测试"""
        print("=" * 80)
        print("Operation 计算延迟测试")
        print("=" * 80)
        print(f"数据集根目录: {self.datasets_root}")
        print(f"使用数据集: {', '.join(self.DATASETS)} (train)")
        if self.max_samples_per_dataset is None:
            print("每个数据集样本数上限: 全部训练样本")
        else:
            print(f"每个数据集样本数上限: {self.max_samples_per_dataset}")
        print(f"测试类别: {', '.join(sorted(self.categories))}")
        print("=" * 80)

        # 遍历所有已注册的 operations
        for pid, op in REGISTRY.items():
            category = self._categorize_operation(op)
            if category is None or category not in self.categories:
                continue

            print(f"\n--- 测试 Operation: {pid} "
                  f"(provider={getattr(op, 'provider', '')}, "
                  f"region={getattr(op, 'region', '')}, "
                  f"model={getattr(op, 'model_name', None)}) "
                  f"[category={category}] ---")

            for dataset_name, idx, sample in self._iter_dataset_samples():
                print(f"  - 数据集: {dataset_name}, 样本索引: {idx}")
                if category == "segment":
                    self._test_segment_op(pid, op, dataset_name, idx, sample)
                elif category == "split":
                    self._test_split_op(pid, op, dataset_name, idx, sample)
                elif category == "caption":
                    self._test_caption_op(pid, op, dataset_name, idx, sample)
                elif category == "llm":
                    self._test_llm_op(pid, op, dataset_name, idx, sample)

        print("\n全部 operation 延迟测试完成。")

    def _build_summary(self) -> Dict[str, object]:
        """根据 records 生成按 operation 聚合的统计信息"""
        summary: Dict[str, Dict[str, object]] = {}

        for rec in self.records:
            op_key = rec.pid
            if op_key not in summary:
                summary[op_key] = {
                    "pid": rec.pid,
                    "category": rec.category,
                    "provider": rec.provider,
                    "region": rec.region,
                    "model_name": rec.model_name,
                    "total_calls": 0,
                    "success_calls": 0,
                    "latencies": [],
                    "per_dataset": {},
                }

            s = summary[op_key]
            s["total_calls"] += 1
            if rec.success:
                s["success_calls"] += 1
                s["latencies"].append(rec.latency_seconds)

            ds_stats = s["per_dataset"].setdefault(
                rec.dataset,
                {
                    "dataset": rec.dataset,
                    "total_calls": 0,
                    "success_calls": 0,
                    "latencies": [],
                },
            )
            ds_stats["total_calls"] += 1
            if rec.success:
                ds_stats["success_calls"] += 1
                ds_stats["latencies"].append(rec.latency_seconds)

        # 计算统计量（均值 / 最小值 / 最大值）
        for op_key, s in summary.items():
            latencies: List[float] = s["latencies"]
            if latencies:
                s["latency_stats"] = {
                    "count": len(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                }
            else:
                s["latency_stats"] = {
                    "count": 0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
            # 移除原始列表，避免 JSON 体积过大
            del s["latencies"]

            for ds_name, ds_stats in s["per_dataset"].items():
                ds_latencies: List[float] = ds_stats["latencies"]
                if ds_latencies:
                    ds_stats["latency_stats"] = {
                        "count": len(ds_latencies),
                        "mean": sum(ds_latencies) / len(ds_latencies),
                        "min": min(ds_latencies),
                        "max": max(ds_latencies),
                    }
                else:
                    ds_stats["latency_stats"] = {
                        "count": 0,
                        "mean": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }
                del ds_stats["latencies"]

        return summary

    def save_results(self, output_dir: str | Path = "results/node_latency", filename: Optional[str] = None):
        """保存测试结果到 JSON 文件

        Args:
            output_dir: 输出目录，将在项目根目录下创建 `results/node_latency/`
            filename: 输出文件名（可选），若为 None 则使用带时间戳的默认文件名
        """
        output_dir = Path(output_dir)
        # 确保目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"node_latency_{timestamp}.json"

        output_path = output_dir / filename

        summary = self._build_summary()

        result_obj = {
            "test_time": datetime.now().isoformat(),
            "datasets_root": str(self.datasets_root),
            "datasets": self.DATASETS,
            "max_samples_per_dataset": self.max_samples_per_dataset,
            "categories": sorted(self.categories),
            "summary": summary,
            "records": [asdict(r) for r in self.records],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2, ensure_ascii=False)

        print(f"\nOperation 延迟测试结果已保存到: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Operation 计算延迟测试工具")
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="datasets",
        help="数据集根目录（默认: datasets）",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=0,
        help="每个数据集用于测试的样本数上限；<=0 表示使用该数据集全部训练样本（默认: 0）",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        choices=["segment", "split", "caption", "llm"],
        help="需要测试的 operation 类别（不指定则测试全部）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件名（可选，默认: node_latency_时间戳.json）",
    )

    args = parser.parse_args()

    tester = NodeLatencyTester(
        datasets_root=args.datasets_root,
        max_samples_per_dataset=args.max_samples_per_dataset,
        categories=args.categories,
    )
    try:
        tester.run()
        tester.save_results(output_dir="results/node_latency", filename=args.output)
    except KeyboardInterrupt:
        print("\n测试被用户中断，保存当前已完成的结果。")
        tester.save_results(output_dir="results/node_latency", filename=args.output)
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        tester.save_results(output_dir="results/node_latency", filename=args.output)


if __name__ == "__main__":
    main()

