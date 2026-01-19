from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
# 假设 core/transmission.py 已经存在
from core.transmission import DataTransmission


class Operation(ABC):
    """所有操作的基类"""

    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: Optional[str] = None):
        self.provider = provider
        self.region = region
        self.storage_bucket = storage_bucket
        self.model_name = model_name

        # 初始化传输器 (负责数据搬运)
        self.transmitter = DataTransmission()

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        pass


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