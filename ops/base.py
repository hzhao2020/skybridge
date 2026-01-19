# ops/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.transmission import DataTransmission


class Operation(ABC):
    """
    所有逻辑操作的基类
    """

    def __init__(self, provider: str, region: str, storage_bucket: Optional[str], model_name: Optional[str] = None):
        self.provider = provider
        self.region = region
        self.storage_bucket = storage_bucket
        self.model_name = model_name

        # 初始化数据传输器 (不需要参数，懒加载)
        self.transmitter = DataTransmission()

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