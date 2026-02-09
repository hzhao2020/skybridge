"""
存储桶传输性能测试工具

测试不同 storage bucket 之间的传输性能，包括：
- RTT (Round-Trip Time): 往返时间测试
- Bandwidth: 带宽测试

支持的传输类型：
- 同云内跨 region 传输（S3->S3, GCS->GCS）
- 跨云传输（S3<->GCS）
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 将项目根目录添加到 sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ops.utils import DataTransmission
from ops.registry import BUCKETS
import config


@dataclass
class TransmissionTestResult:
    """传输测试结果"""
    source_bucket: str
    target_bucket: str
    source_provider: str
    target_provider: str
    test_type: str  # 'rtt' or 'bandwidth'
    file_size_bytes: int
    duration_seconds: float
    bandwidth_mbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class BucketTransmissionTester:
    """存储桶传输性能测试器"""
    
    # Bucket配置映射
    BUCKET_CONFIG = {
        "gcp_us": {"provider": "google", "bucket": "video_us", "region": "us-west1"},
        "gcp_tw": {"provider": "google", "bucket": "video_tw", "region": "asia-east1"},
        "gcp_sg": {"provider": "google", "bucket": "video_sg", "region": "asia-southeast1"},
        "aws_us": {"provider": "amazon", "bucket": "sky-video-us", "region": "us-west-2"},
        "aws_sg": {"provider": "amazon", "bucket": "sky-video-sg", "region": "ap-southeast-1"},
    }
    
    def __init__(self):
        """初始化测试器"""
        self.transmitter = DataTransmission()
        self.results: List[TransmissionTestResult] = []
        
    def _create_test_file(self, size_bytes: int) -> str:
        """创建测试文件
        
        Args:
            size_bytes: 文件大小（字节）
            
        Returns:
            临时文件路径
        """
        fd, path = tempfile.mkstemp(suffix='.bin', prefix='test_')
        try:
            with os.fdopen(fd, 'wb') as f:
                # 写入随机数据
                chunk_size = 1024 * 1024  # 1MB chunks
                remaining = size_bytes
                while remaining > 0:
                    chunk = os.urandom(min(chunk_size, remaining))
                    f.write(chunk)
                    remaining -= len(chunk)
            return path
        except Exception as e:
            os.unlink(path)
            raise e
    
    def _cleanup_test_file(self, file_path: str):
        """清理测试文件"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass
    
    def _get_bucket_uri(self, bucket_key: str, file_path: str) -> str:
        """获取bucket URI
        
        Args:
            bucket_key: bucket配置键（如 'gcp_us'）
            file_path: 文件路径
            
        Returns:
            URI字符串（如 'gs://video_us/test.bin'）
        """
        config = self.BUCKET_CONFIG[bucket_key]
        provider = config["provider"]
        bucket = config["bucket"]
        filename = os.path.basename(file_path)
        
        if provider == "google":
            return f"gs://{bucket}/{filename}"
        elif provider == "amazon":
            return f"s3://{bucket}/{filename}"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _upload_file(self, local_path: str, bucket_key: str) -> str:
        """上传文件到bucket
        
        Args:
            local_path: 本地文件路径
            bucket_key: 目标bucket配置键
            
        Returns:
            上传后的URI
        """
        config = self.BUCKET_CONFIG[bucket_key]
        provider = config["provider"]
        bucket = config["bucket"]
        
        return self.transmitter.upload_local_to_cloud(local_path, provider, bucket)
    
    def _transfer_file(self, source_uri: str, source_bucket_key: str, target_bucket_key: str) -> str:
        """传输文件（跨bucket）
        
        Args:
            source_uri: 源文件URI
            source_bucket_key: 源bucket配置键
            target_bucket_key: 目标bucket配置键
            
        Returns:
            目标URI
        """
        source_config = self.BUCKET_CONFIG[source_bucket_key]
        target_config = self.BUCKET_CONFIG[target_bucket_key]
        
        target_provider = target_config["provider"]
        target_bucket = target_config["bucket"]
        
        return self.transmitter.smart_move(
            source_uri,
            target_provider=target_provider,
            target_bucket=target_bucket
        )
    
    def _delete_file(self, uri: str, bucket_key: str):
        """删除bucket中的文件
        
        Args:
            uri: 文件URI
            bucket_key: bucket配置键
        """
        try:
            config = self.BUCKET_CONFIG[bucket_key]
            provider = config["provider"]
            bucket = config["bucket"]
            
            parsed = self.transmitter._parse_uri(uri)
            if len(parsed) >= 3:
                key = parsed[2]
                
                if provider == "google":
                    blob = self.transmitter.gcs_client.bucket(bucket).blob(key)
                    blob.delete()
                elif provider == "amazon":
                    self.transmitter.s3_client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            print(f"Warning: Failed to delete {uri}: {e}")
    
    def test_rtt(self, source_bucket_key: str, target_bucket_key: str, 
                 num_runs: int = 3) -> TransmissionTestResult:
        """测试RTT（往返时间）
        
        Args:
            source_bucket_key: 源bucket配置键
            target_bucket_key: 目标bucket配置键
            num_runs: 测试运行次数（取平均值）
            
        Returns:
            测试结果
        """
        print(f"\n{'='*60}")
        print(f"RTT测试: {source_bucket_key} -> {target_bucket_key}")
        print(f"{'='*60}")
        
        source_config = self.BUCKET_CONFIG[source_bucket_key]
        target_config = self.BUCKET_CONFIG[target_bucket_key]
        
        # 创建小测试文件（1KB用于RTT测试）
        test_file_size = 1024  # 1KB
        test_file_path = None
        source_uri = None
        
        try:
            # 创建测试文件
            test_file_path = self._create_test_file(test_file_size)
            print(f"创建测试文件: {test_file_path} ({test_file_size} bytes)")
            
            # 先上传一次到源bucket
            print(f"\n上传测试文件到源bucket: {source_bucket_key}...")
            upload_start = time.time()
            source_uri = self._upload_file(test_file_path, source_bucket_key)
            upload_time = time.time() - upload_start
            print(f"上传完成 ({upload_time:.2f}s)")
            
            # 多次运行传输测试取平均值（只测试传输时间，不包括上传时间）
            transfer_durations = []
            for run in range(num_runs):
                print(f"\n运行 {run + 1}/{num_runs}...")
                
                # 传输到目标bucket
                print(f"  传输到目标bucket: {target_bucket_key}...")
                transfer_start = time.time()
                target_uri = self._transfer_file(source_uri, source_bucket_key, target_bucket_key)
                transfer_time = time.time() - transfer_start
                print(f"  传输完成 ({transfer_time:.2f}s)")
                
                transfer_durations.append(transfer_time)
                
                # 清理目标文件
                self._delete_file(target_uri, target_bucket_key)
            
            # 计算平均值（只计算传输时间）
            avg_transfer_time = sum(transfer_durations) / len(transfer_durations)
            rtt_ms = avg_transfer_time * 1000  # 转换为毫秒
            
            print(f"\n平均传输时间: {avg_transfer_time:.2f}s")
            print(f"平均RTT: {rtt_ms:.2f}ms")
            
            # 清理源文件
            if source_uri:
                self._delete_file(source_uri, source_bucket_key)
            
            return TransmissionTestResult(
                source_bucket=source_bucket_key,
                target_bucket=target_bucket_key,
                source_provider=source_config["provider"],
                target_provider=target_config["provider"],
                test_type="rtt",
                file_size_bytes=test_file_size,
                duration_seconds=avg_transfer_time,
                rtt_ms=rtt_ms,
                success=True
            )
            
        except Exception as e:
            print(f"RTT测试失败: {e}")
            return TransmissionTestResult(
                source_bucket=source_bucket_key,
                target_bucket=target_bucket_key,
                source_provider=source_config["provider"],
                target_provider=target_config["provider"],
                test_type="rtt",
                file_size_bytes=test_file_size,
                duration_seconds=0,
                success=False,
                error_message=str(e)
            )
        finally:
            if test_file_path:
                self._cleanup_test_file(test_file_path)
    
    def test_bandwidth(self, source_bucket_key: str, target_bucket_key: str,
                      file_sizes_mb: List[int] = [1, 10, 100]) -> List[TransmissionTestResult]:
        """测试带宽
        
        Args:
            source_bucket_key: 源bucket配置键
            target_bucket_key: 目标bucket配置键
            file_sizes_mb: 测试文件大小列表（MB）
            
        Returns:
            测试结果列表
        """
        print(f"\n{'='*60}")
        print(f"带宽测试: {source_bucket_key} -> {target_bucket_key}")
        print(f"{'='*60}")
        
        source_config = self.BUCKET_CONFIG[source_bucket_key]
        target_config = self.BUCKET_CONFIG[target_bucket_key]
        
        results = []
        
        for size_mb in file_sizes_mb:
            print(f"\n测试文件大小: {size_mb}MB")
            size_bytes = size_mb * 1024 * 1024
            test_file_path = None
            source_uri = None
            
            try:
                # 创建测试文件
                print(f"创建测试文件 ({size_mb}MB)...")
                test_file_path = self._create_test_file(size_bytes)
                
                # 上传到源bucket
                print(f"上传到源bucket: {source_bucket_key}...")
                upload_start = time.time()
                source_uri = self._upload_file(test_file_path, source_bucket_key)
                upload_time = time.time() - upload_start
                print(f"上传完成 ({upload_time:.2f}s)")
                
                # 传输到目标bucket
                print(f"传输到目标bucket: {target_bucket_key}...")
                transfer_start = time.time()
                target_uri = self._transfer_file(source_uri, source_bucket_key, target_bucket_key)
                transfer_time = time.time() - transfer_start
                print(f"传输完成 ({transfer_time:.2f}s)")
                
                # 计算带宽（只计算传输时间，不包括上传时间）
                total_time = upload_time + transfer_time
                bandwidth_mbps = (size_bytes * 8) / (transfer_time * 1_000_000)  # Mbps
                
                print(f"总耗时: {total_time:.2f}s")
                print(f"传输带宽: {bandwidth_mbps:.2f} Mbps")
                
                results.append(TransmissionTestResult(
                    source_bucket=source_bucket_key,
                    target_bucket=target_bucket_key,
                    source_provider=source_config["provider"],
                    target_provider=target_config["provider"],
                    test_type="bandwidth",
                    file_size_bytes=size_bytes,
                    duration_seconds=transfer_time,
                    bandwidth_mbps=bandwidth_mbps,
                    success=True
                ))
                
                # 清理文件
                self._delete_file(target_uri, target_bucket_key)
                self._delete_file(source_uri, source_bucket_key)
                
            except Exception as e:
                print(f"带宽测试失败 ({size_mb}MB): {e}")
                results.append(TransmissionTestResult(
                    source_bucket=source_bucket_key,
                    target_bucket=target_bucket_key,
                    source_provider=source_config["provider"],
                    target_provider=target_config["provider"],
                    test_type="bandwidth",
                    file_size_bytes=size_bytes,
                    duration_seconds=0,
                    success=False,
                    error_message=str(e)
                ))
            finally:
                if test_file_path:
                    self._cleanup_test_file(test_file_path)
        
        return results
    
    def run_all_tests(self, test_rtt: bool = True, test_bandwidth: bool = True,
                     bucket_pairs: Optional[List[Tuple[str, str]]] = None):
        """运行所有测试
        
        Args:
            test_rtt: 是否测试RTT
            test_bandwidth: 是否测试带宽
            bucket_pairs: 指定要测试的bucket对列表，如果为None则测试所有组合
        """
        print("="*60)
        print("存储桶传输性能测试")
        print("="*60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 获取所有bucket键
        all_buckets = list(self.BUCKET_CONFIG.keys())
        
        # 确定要测试的bucket对
        if bucket_pairs is None:
            # 测试所有组合
            bucket_pairs = [(src, dst) for src in all_buckets for dst in all_buckets if src != dst]
        else:
            # 验证bucket键
            for src, dst in bucket_pairs:
                if src not in self.BUCKET_CONFIG:
                    raise ValueError(f"Unknown source bucket: {src}")
                if dst not in self.BUCKET_CONFIG:
                    raise ValueError(f"Unknown target bucket: {dst}")
        
        print(f"\n将测试 {len(bucket_pairs)} 个bucket对")
        
        # 运行测试
        for idx, (source_bucket, target_bucket) in enumerate(bucket_pairs, 1):
            print(f"\n\n进度: {idx}/{len(bucket_pairs)}")
            
            if test_rtt:
                rtt_result = self.test_rtt(source_bucket, target_bucket)
                self.results.append(rtt_result)
            
            if test_bandwidth:
                bandwidth_results = self.test_bandwidth(source_bucket, target_bucket)
                self.results.extend(bandwidth_results)
        
        print(f"\n\n{'='*60}")
        print("所有测试完成")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
    
    def save_results(self, output_file: str = "transmission_test_results.json"):
        """保存测试结果到JSON文件
        
        Args:
            output_file: 输出文件名
        """
        output_path = Path(__file__).parent / output_file
        
        # 转换为字典格式
        results_dict = {
            "test_time": datetime.now().isoformat(),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: {output_path}")
    
    def print_summary(self):
        """打印测试结果摘要"""
        print("\n" + "="*60)
        print("测试结果摘要")
        print("="*60)
        
        # RTT结果
        rtt_results = [r for r in self.results if r.test_type == "rtt" and r.success]
        if rtt_results:
            print("\nRTT测试结果 (ms):")
            print("-" * 60)
            for result in rtt_results:
                print(f"{result.source_bucket:12} -> {result.target_bucket:12}: "
                      f"{result.rtt_ms:8.2f}ms")
        
        # 带宽结果
        bandwidth_results = [r for r in self.results if r.test_type == "bandwidth" and r.success]
        if bandwidth_results:
            print("\n带宽测试结果 (Mbps):")
            print("-" * 60)
            # 按bucket对和文件大小分组
            grouped = {}
            for result in bandwidth_results:
                key = (result.source_bucket, result.target_bucket, result.file_size_bytes)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(result)
            
            for (src, dst, size), results in sorted(grouped.items()):
                size_mb = size / (1024 * 1024)
                avg_bandwidth = sum(r.bandwidth_mbps for r in results) / len(results)
                print(f"{src:12} -> {dst:12} ({size_mb:4.0f}MB): "
                      f"{avg_bandwidth:8.2f} Mbps")
        
        # 失败统计
        failed = [r for r in self.results if not r.success]
        if failed:
            print(f"\n失败测试数: {len(failed)}")
            for result in failed:
                print(f"  {result.source_bucket} -> {result.target_bucket} "
                      f"({result.test_type}): {result.error_message}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="存储桶传输性能测试工具")
    parser.add_argument("--rtt-only", action="store_true", help="仅测试RTT")
    parser.add_argument("--bandwidth-only", action="store_true", help="仅测试带宽")
    parser.add_argument("--pairs", nargs="+", help="指定要测试的bucket对，格式: src1:dst1 src2:dst2")
    parser.add_argument("--output", default="transmission_test_results.json", help="输出文件名")
    
    args = parser.parse_args()
    
    # 确定测试类型
    test_rtt = not args.bandwidth_only
    test_bandwidth = not args.rtt_only
    
    # 解析bucket对
    bucket_pairs = None
    if args.pairs:
        bucket_pairs = []
        for pair_str in args.pairs:
            parts = pair_str.split(":")
            if len(parts) != 2:
                print(f"Error: Invalid bucket pair format: {pair_str}. Use src:dst")
                return
            bucket_pairs.append((parts[0], parts[1]))
    
    # 运行测试
    tester = BucketTransmissionTester()
    try:
        tester.run_all_tests(
            test_rtt=test_rtt,
            test_bandwidth=test_bandwidth,
            bucket_pairs=bucket_pairs
        )
        tester.print_summary()
        tester.save_results(args.output)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        tester.print_summary()
        tester.save_results(args.output)
    except Exception as e:
        print(f"\n\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
