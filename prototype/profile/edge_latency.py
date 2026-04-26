"""
边缘节点（存储桶）带宽 & RTT 长时间监测工具

功能：
- 周期性地在各个存储节点（bucket）之间发送测试数据，测量：
  - RTT（Round-Trip Time / 单次传输时延）
  - 带宽（Bandwidth）
- 持续运行一段时间（默认 7 天），定期记录测量结果
- 所有结果以 JSON 形式保存在 `results/edge_latency/` 目录中，便于后续分析

说明：
- 底层单次测试逻辑复用 `profile/test_bucket_transmission.py` 中的
  `BucketTransmissionTester` 与 `TransmissionTestResult`。
- 本脚本只负责“长期定时调度 + 结果落盘”，不改变单次测试行为。
"""

import sys
import time
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# 将项目根目录加入 sys.path，确保可以导入 ops / utils 等模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _load_bucket_tester_module():
    """
    通过文件路径动态加载 `test_bucket_transmission.py` 模块，避免与 Python 内置 `profile` 模块冲突。
    返回加载后的模块对象。
    """
    import importlib.util

    module_path = Path(__file__).parent / "test_bucket_transmission.py"
    spec = importlib.util.spec_from_file_location("bucket_transmission_profile", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EdgeLatencyMonitor:
    """
    边缘节点（存储桶）之间网络性能长时间监控器。

    通过周期性调用 `BucketTransmissionTester.run_all_tests` 来完成实际测试。
    """

    def __init__(
        self,
        duration_days: float = 7.0,
        interval_seconds: int = 600,
        rtt_interval_seconds: Optional[int] = None,
        bandwidth_interval_seconds: Optional[int] = None,
        test_rtt: bool = True,
        test_bandwidth: bool = True,
        bucket_pairs: Optional[List[Tuple[str, str]]] = None,
        results_dir: str | Path = "results/edge_latency",
    ):
        """
        Args:
            duration_days: 总测试时长（天），默认 7 天。
            interval_seconds: 默认测试间隔（秒），当未单独指定 rtt/bandwidth 间隔时使用。
            rtt_interval_seconds: RTT 测试间隔（秒），为 None 时使用 interval_seconds。
            bandwidth_interval_seconds: 带宽测试间隔（秒），为 None 时使用 interval_seconds。
            test_rtt: 是否测试 RTT。
            test_bandwidth: 是否测试带宽。
            bucket_pairs: 指定需要测试的 bucket 对列表，如 [("gcp_us", "aws_us"), ...]。
                          为 None 时会测试所有组合（与 BucketTransmissionTester.run_all_tests 一致）。
            results_dir: 结果输出目录（默认 `results/edge_latency`）。
        """
        if duration_days <= 0:
            raise ValueError("duration_days 必须大于 0")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds 必须大于 0")

        self.duration_days = float(duration_days)
        self.interval_seconds = int(interval_seconds)
        self.rtt_interval_seconds = int(rtt_interval_seconds or interval_seconds)
        self.bandwidth_interval_seconds = int(bandwidth_interval_seconds or interval_seconds)
        if self.rtt_interval_seconds <= 0 or self.bandwidth_interval_seconds <= 0:
            raise ValueError("rtt_interval_seconds 和 bandwidth_interval_seconds 必须大于 0")
        self.test_rtt = bool(test_rtt)
        self.test_bandwidth = bool(test_bandwidth)
        self.bucket_pairs = bucket_pairs
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 动态加载底层测试模块和类
        bt_module = _load_bucket_tester_module()
        self.BucketTransmissionTester = bt_module.BucketTransmissionTester
        self.TransmissionTestResult = bt_module.TransmissionTestResult

    def _run_single_round(
        self,
        round_index: int,
        run_rtt: bool,
        run_bandwidth: bool,
    ) -> Optional[Path]:
        """
        执行一次 bucket 间测试（可指定只测 RTT 或只测带宽或两者都测），并将结果写入 JSON 文件。

        Args:
            round_index: 轮次编号。
            run_rtt: 本轮是否执行 RTT 测试。
            run_bandwidth: 本轮是否执行带宽测试。

        Returns:
            结果文件路径；若两者都不测则返回 None。
        """
        if not run_rtt and not run_bandwidth:
            return None

        test_types = []
        if run_rtt:
            test_types.append("RTT")
        if run_bandwidth:
            test_types.append("Bandwidth")
        print("\n" + "=" * 80)
        print(f"[EdgeLatency] 开始第 {round_index} 轮测试 ({'+'.join(test_types)})")
        print("=" * 80)

        start_time = datetime.now()
        tester = self.BucketTransmissionTester()

        tester.run_all_tests(
            test_rtt=run_rtt,
            test_bandwidth=run_bandwidth,
            bucket_pairs=self.bucket_pairs,
        )

        end_time = datetime.now()

        round_results = {
            "round_index": round_index,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "test_rtt": run_rtt,
            "test_bandwidth": run_bandwidth,
            "bucket_pairs": self.bucket_pairs,
            "results": [asdict(r) for r in tester.results],
        }

        timestamp = end_time.strftime("%Y%m%d_%H%M%S")
        filename = f"edge_latency_round{round_index:04d}_{timestamp}.json"
        output_path = self.results_dir / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(round_results, f, indent=2, ensure_ascii=False)

        print(f"[EdgeLatency] 第 {round_index} 轮测试完成，结果已保存到: {output_path}")
        return output_path

    def run(self):
        """按照给定的时长和间隔，持续执行测试，直到达到总测试时长。"""
        total_duration = timedelta(days=self.duration_days)
        start_time = datetime.now()
        end_time = start_time + total_duration

        # 使用较短的间隔作为主循环的 tick 间隔
        tick_interval = min(self.rtt_interval_seconds, self.bandwidth_interval_seconds)

        print("=" * 80)
        print("边缘节点带宽 & RTT 长时间监测启动")
        print("=" * 80)
        print(f"开始时间: {start_time.isoformat()}")
        print(f"计划结束时间: {end_time.isoformat()} (约 {self.duration_days} 天)")
        print(f"RTT 测试间隔: {self.rtt_interval_seconds} 秒")
        print(f"带宽测试间隔: {self.bandwidth_interval_seconds} 秒")
        print(f"结果目录: {self.results_dir}")
        print("=" * 80)

        round_index = 0
        last_rtt_time = 0.0
        last_bandwidth_time = 0.0

        try:
            while datetime.now() < end_time:
                round_index += 1
                round_start_wall = time.time()
                now_ts = time.time()

                # 判断本轮是否需要执行 RTT 和带宽测试
                run_rtt = (
                    self.test_rtt
                    and (now_ts - last_rtt_time >= self.rtt_interval_seconds or last_rtt_time == 0)
                )
                run_bandwidth = (
                    self.test_bandwidth
                    and (
                        now_ts - last_bandwidth_time >= self.bandwidth_interval_seconds
                        or last_bandwidth_time == 0
                    )
                )

                if run_rtt or run_bandwidth:
                    self._run_single_round(round_index, run_rtt, run_bandwidth)
                    if run_rtt:
                        last_rtt_time = round_start_wall
                    if run_bandwidth:
                        last_bandwidth_time = round_start_wall

                # 计算到下次 tick 的剩余时间
                next_tick_time = round_start_wall + tick_interval
                sleep_seconds = max(0.0, next_tick_time - time.time())

                if sleep_seconds > 0 and datetime.now() < end_time:
                    print(f"[EdgeLatency] 休眠 {sleep_seconds:.1f} 秒后检查下一轮测试...")
                    time.sleep(sleep_seconds)

            print("\n" + "=" * 80)
            print("边缘节点带宽 & RTT 长时间监测已完成预定时长。")
            print(f"结束时间: {datetime.now().isoformat()}")
            print("=" * 80)

        except KeyboardInterrupt:
            print("\n[EdgeLatency] 收到中断信号，提前结束监测。")
            print(f"当前时间: {datetime.now().isoformat()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="边缘节点（存储桶）带宽 & RTT 长时间监测工具")
    parser.add_argument(
        "--duration-days",
        type=float,
        default=7.0,
        help="总测试时长（天），默认 7 天",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=600,
        help="默认测试间隔（秒），当未单独指定 RTT/带宽间隔时使用；默认 600 秒",
    )
    parser.add_argument(
        "--rtt-interval",
        type=int,
        default=None,
        help="RTT 测试间隔（秒），默认 600 秒（10 分钟）",
    )
    parser.add_argument(
        "--bandwidth-interval",
        type=int,
        default=None,
        help="带宽测试间隔（秒），默认 3600 秒（1 小时）",
    )
    parser.add_argument(
        "--rtt-only",
        action="store_true",
        help="仅测试 RTT（不测试带宽）",
    )
    parser.add_argument(
        "--bandwidth-only",
        action="store_true",
        help="仅测试带宽（不测试 RTT）",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="指定要测试的 bucket 对，格式: src1:dst1 src2:dst2；不指定则测试所有组合",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/edge_latency",
        help="结果保存目录（默认: results/edge_latency）",
    )

    args = parser.parse_args()

    # 确定测试内容
    test_rtt = not args.bandwidth_only
    test_bandwidth = not args.rtt_only

    # 解析 bucket 对
    bucket_pairs: Optional[List[Tuple[str, str]]] = None
    if args.pairs:
        bucket_pairs = []
        for pair_str in args.pairs:
            parts = pair_str.split(":")
            if len(parts) != 2:
                print(f"Error: 无效的 bucket 对格式: {pair_str}，应为 src:dst")
                return
            bucket_pairs.append((parts[0], parts[1]))

    rtt_interval = args.rtt_interval if args.rtt_interval is not None else args.interval_seconds
    bandwidth_interval = (
        args.bandwidth_interval if args.bandwidth_interval is not None else 3600
    )

    monitor = EdgeLatencyMonitor(
        duration_days=args.duration_days,
        interval_seconds=args.interval_seconds,
        rtt_interval_seconds=rtt_interval,
        bandwidth_interval_seconds=bandwidth_interval,
        test_rtt=test_rtt,
        test_bandwidth=test_bandwidth,
        bucket_pairs=bucket_pairs,
        results_dir=args.results_dir,
    )
    monitor.run()


if __name__ == "__main__":
    main()

