# ops/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
        self._storage_helper = None

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