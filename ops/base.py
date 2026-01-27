# ops/base.py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from ops.utils import DataTransmission


class Operation(ABC):
    """
    所有逻辑操作的基类
    
    注意：
    - transmitter: 数据传输辅助类（向后兼容，供内部使用）
    - storage_helper: 数据存储辅助类（可选，供需要存储操作的 Operation 使用）
    """

    def __init__(self, provider: str, region: str, storage_bucket: Optional[str], model_name: Optional[str] = None):
        self.provider = provider
        self.region = region
        self.storage_bucket = storage_bucket
        self.model_name = model_name

        # 初始化数据传输器 (不需要参数，懒加载)
        # 注意：保持向后兼容，现有代码可以继续使用 self.transmitter
        self.transmitter = DataTransmission()
        
        # 可选的存储辅助类（懒加载，不自动初始化）
        # 子类可以按需创建：self._storage_helper = DataStorageHelper(...)
        # self._storage_helper = None

    def _record_operation_timing(self, operation_name: str, operation_pid: str, start_time: float, end_time: float):
        """记录operation执行时间（排除传输时间）
        
        这个方法应该在operation的execute方法中调用，在传输完成后、实际处理开始前记录start_time，
        在实际处理完成后记录end_time。
        
        Args:
            operation_name: operation名称
            operation_pid: operation的pid
            start_time: operation实际开始时间（传输后）
            end_time: operation结束时间
        """
        try:
            from utils.timing import TimingRecorder
            recorder = TimingRecorder()
            recorder.record_operation(operation_name, operation_pid, start_time, end_time)
        except Exception:
            pass  # 如果记录失败，不影响主流程
    
    def _build_result_path(self, video_uri: str, operation_name: str, filename: str, target_path: Optional[str] = None) -> str:
        """
        构建操作结果的存储路径
        
        路径格式：results/[operation_name]/[video_name]/[filename]
        
        Args:
            video_uri: 视频 URI（可以是本地路径或云存储 URI）
            operation_name: 操作名称（如 "segment", "object_detection"）
            filename: 结果文件名（如 "result.json"）
            target_path: 可选的原始目标路径（用于提取 dataset 信息，如果视频路径中没有）
            
        Returns:
            结果路径（相对路径，不含 bucket）
        """
        # 从视频 URI 中提取视频名称
        if video_uri.startswith('s3://') or video_uri.startswith('gs://'):
            # 云存储 URI
            parsed = urlparse(video_uri)
            video_path = parsed.path.lstrip('/')
        else:
            # 本地路径
            video_path = video_uri
        
        # 获取视频文件名（不含扩展名）
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]  # 移除扩展名
        
        # 如果 video_name 为空，尝试从 target_path 中提取
        if not video_name and target_path:
            # target_path 格式可能是 "videos/[dataset]/[video_name.mp4]"
            parts = target_path.strip('/').split('/')
            if len(parts) >= 2 and parts[0] == 'videos':
                # 从 target_path 中提取 dataset 和 video_name
                dataset = parts[1] if len(parts) > 1 else None
                if len(parts) > 2:
                    # 有具体的视频文件名
                    video_filename_from_path = parts[-1]
                    video_name = os.path.splitext(video_filename_from_path)[0]
        
        # 如果仍然没有 video_name，使用默认值
        if not video_name:
            video_name = "unknown_video"
        
        # 构建结果路径：results/[operation_name]/[video_name]/[filename]
        result_path = f"results/{operation_name}/{video_name}/{filename}"
        
        return result_path

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """执行逻辑"""
        pass


# --- 具体的逻辑操作接口 ---

class VideoSegmenter(Operation):
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        pass


class VisualCaptioner(Operation):
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        pass


class LLMQuery(Operation):
    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass


class VideoSplitter(Operation):
    """
    视频分割操作：将视频切割成多个片段
    
    与 VideoSegmenter 的区别：
    - VideoSegmenter: 检测视频中的 shot changes（场景切换），返回时间段
    - VideoSplitter: 根据指定的时间段或规则，将视频物理切割成多个文件
    """
    @abstractmethod
    def execute(self, video_uri: str, segments: list, **kwargs) -> Dict[str, Any]:
        """
        执行视频分割
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            segments: 片段列表，每个片段包含 start_time 和 end_time
            **kwargs: 其他参数，如 target_path, output_format 等
            
        Returns:
            包含分割结果的字典，通常包含 output_uris（输出文件URI列表）
        """
        pass


class VisualEncoder(Operation):
    """
    视觉编码操作：将视频编码成向量进行存储
    """
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        执行视觉编码
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            **kwargs: 其他参数，如 target_path, save_embedding 等
            
        Returns:
            包含编码结果的字典，通常包含 embedding（向量）和 metadata
        """
        pass


class TextEncoder(Operation):
    """
    文本编码操作：将文本编码成向量进行存储
    """
    @abstractmethod
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        执行文本编码
        
        Args:
            text: 要编码的文本
            **kwargs: 其他参数，如 save_embedding, dimensions 等
            
        Returns:
            包含编码结果的字典，通常包含 embedding（向量）和 metadata
        """
        pass


class ObjectDetector(Operation):
    """
    物体检测操作：在视频中检测和跟踪物体
    """
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        执行物体检测
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            **kwargs: 其他参数，如 target_path, save_results 等
            
        Returns:
            包含检测结果的字典，通常包含 detected_objects（检测到的物体列表）和 metadata
        """
        pass