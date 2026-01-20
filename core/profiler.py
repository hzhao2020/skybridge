"""
视频信息统计工具
统计三个数据集（EgoSchema, NExTQA, ActivityNetQA）中 train 和 test 的视频长度信息
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Install with: pip install opencv-python")

from utils.dataset import build_dataset


def get_video_duration(video_path: str) -> Optional[float]:
    """
    获取视频长度（秒）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频长度（秒），如果文件不存在或读取失败返回 None
    """
    if not CV2_AVAILABLE:
        return None
        
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # 获取帧率和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return duration
        else:
            return None
    except Exception as e:
        print(f"    Error reading {video_path}: {e}")
        return None


def calculate_statistics(durations: List[float]) -> Dict[str, float]:
    """
    计算统计信息
    
    Args:
        durations: 视频长度列表
        
    Returns:
        包含统计信息的字典
    """
    if not durations:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "total": 0.0
        }
    
    durations_sorted = sorted(durations)
    
    stats = {
        "count": len(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "min": min(durations),
        "max": max(durations),
        "total": sum(durations)
    }
    
    # 计算标准差
    if len(durations) > 1:
        stats["std"] = statistics.stdev(durations)
    else:
        stats["std"] = 0.0
    
    return stats


def format_duration(seconds: float) -> str:
    """
    格式化时长显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的字符串，如 "1m 23.45s" 或 "45.67s"
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def profile_dataset(dataset_name: str, split: str, datasets_root: str = "datasets") -> Dict[str, any]:
    """
    统计单个数据集单个 split 的视频信息
    
    Args:
        dataset_name: 数据集名称 ('EgoSchema', 'NExTQA', 'ActivityNetQA')
        split: 'train' 或 'test'
        datasets_root: 数据集根目录
        
    Returns:
        统计结果字典
    """
    print(f"\n{'='*60}")
    print(f"正在统计: {dataset_name} - {split}")
    print(f"{'='*60}")
    
    try:
        dataset = build_dataset(dataset_name, split, datasets_root)
    except Exception as e:
        print(f"错误: 无法加载数据集 {dataset_name} ({split}): {e}")
        return {
            "dataset": dataset_name,
            "split": split,
            "error": str(e),
            "stats": None
        }
    
    total_samples = len(dataset)
    print(f"总样本数: {total_samples}")
    
    durations = []
    missing_files = []
    failed_reads = []
    
    # 尝试导入 tqdm 显示进度条
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # 遍历所有样本，获取视频长度
    iterator = range(total_samples)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"  处理视频", unit="个")
    
    for idx in iterator:
        try:
            sample = dataset[idx]
            video_path = sample.get("video_path")
            
            if not video_path:
                missing_files.append(idx)
                continue
            
            duration = get_video_duration(video_path)
            
            if duration is not None:
                durations.append(duration)
            else:
                failed_reads.append(video_path)
                
        except Exception as e:
            if not use_tqdm:
                print(f"  警告: 处理样本 {idx} 时出错: {e}")
            failed_reads.append(f"sample_{idx}")
    
    # 计算统计信息
    stats = calculate_statistics(durations)
    
    # 打印统计结果
    print(f"\n统计结果:")
    print(f"  成功读取: {stats['count']} / {total_samples}")
    if missing_files:
        print(f"  缺失文件: {len(missing_files)}")
    if failed_reads:
        print(f"  读取失败: {len(failed_reads)}")
    
    if stats['count'] > 0:
        print(f"\n视频长度统计:")
        print(f"  总数: {stats['count']}")
        print(f"  平均长度: {format_duration(stats['mean'])} ({stats['mean']:.2f}s)")
        print(f"  中位数: {format_duration(stats['median'])} ({stats['median']:.2f}s)")
        print(f"  最短: {format_duration(stats['min'])} ({stats['min']:.2f}s)")
        print(f"  最长: {format_duration(stats['max'])} ({stats['max']:.2f}s)")
        print(f"  标准差: {stats['std']:.2f}s")
        print(f"  总时长: {format_duration(stats['total'])} ({stats['total']:.2f}s)")
    
    return {
        "dataset": dataset_name,
        "split": split,
        "total_samples": total_samples,
        "successful_reads": stats['count'],
        "missing_files": len(missing_files),
        "failed_reads": len(failed_reads),
        "stats": stats if stats['count'] > 0 else None
    }


def profile_all_datasets(datasets_root: str = "datasets"):
    """
    统计所有三个数据集的 train 和 test 视频信息
    
    Args:
        datasets_root: 数据集根目录
    """
    if not CV2_AVAILABLE:
        print("错误: opencv-python 未安装，无法获取视频长度")
        print("请安装: pip install opencv-python")
        return
    
    datasets = ["EgoSchema", "NExTQA", "ActivityNetQA"]
    splits = ["train", "test"]
    
    all_results = []
    
    # 统计每个数据集和每个 split
    for dataset_name in datasets:
        for split in splits:
            result = profile_dataset(dataset_name, split, datasets_root)
            all_results.append(result)
    
    # 打印汇总报告
    print(f"\n\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}\n")
    
    # 创建汇总表格
    print(f"{'数据集':<20} {'Split':<10} {'样本数':<10} {'成功':<10} {'平均长度':<15} {'最短':<15} {'最长':<15}")
    print("-" * 100)
    
    for result in all_results:
        dataset = result["dataset"]
        split = result["split"]
        total = result["total_samples"]
        success = result["successful_reads"]
        
        if result["stats"]:
            stats = result["stats"]
            avg = format_duration(stats["mean"])
            min_dur = format_duration(stats["min"])
            max_dur = format_duration(stats["max"])
        else:
            avg = "N/A"
            min_dur = "N/A"
            max_dur = "N/A"
        
        print(f"{dataset:<20} {split:<10} {total:<10} {success:<10} {avg:<15} {min_dur:<15} {max_dur:<15}")
    
    # 按数据集汇总
    print(f"\n\n{'='*80}")
    print("按数据集汇总")
    print(f"{'='*80}\n")
    
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        train_result = next((r for r in all_results if r["dataset"] == dataset_name and r["split"] == "train"), None)
        test_result = next((r for r in all_results if r["dataset"] == dataset_name and r["split"] == "test"), None)
        
        if train_result and train_result["stats"]:
            train_stats = train_result["stats"]
            print(f"  Train: {train_result['successful_reads']} 个视频, "
                  f"平均 {format_duration(train_stats['mean'])}, "
                  f"总时长 {format_duration(train_stats['total'])}")
        
        if test_result and test_result["stats"]:
            test_stats = test_result["stats"]
            print(f"  Test:  {test_result['successful_reads']} 个视频, "
                  f"平均 {format_duration(test_stats['mean'])}, "
                  f"总时长 {format_duration(test_stats['total'])}")
        
        if train_result and train_result["stats"] and test_result and test_result["stats"]:
            combined_count = train_stats['count'] + test_stats['count']
            combined_total = train_stats['total'] + test_stats['total']
            combined_avg = combined_total / combined_count if combined_count > 0 else 0
            print(f"  总计: {combined_count} 个视频, "
                  f"平均 {format_duration(combined_avg)}, "
                  f"总时长 {format_duration(combined_total)}")


if __name__ == "__main__":
    # 设置数据集根目录
    datasets_root = "datasets"
    
    # 如果提供了命令行参数，使用该参数作为数据集根目录
    if len(sys.argv) > 1:
        datasets_root = sys.argv[1]
    
    print("视频信息统计工具")
    print(f"数据集根目录: {datasets_root}")
    
    profile_all_datasets(datasets_root)
