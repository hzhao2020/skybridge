# ops/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Operation(ABC):
    """所有操作的基类"""
    def __init__(self, provider: str, region: str, model_name: Optional[str] = None):
        self.provider = provider
        self.region = region
        self.model_name = model_name

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        pass

class VideoSegmenter(Operation):
    """Logical Operation: video segment"""
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        :param video_uri: Cloud Storage URI (s3:// or gs://)
        :return: {'segments': [...], 'meta': ...}
        """
        pass

class VisualCaptioner(Operation):
    """Logical Operation: visual caption"""
    @abstractmethod
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        :param video_uri: Cloud Storage URI
        :return: {'caption': 'A cat jumping...', 'meta': ...}
        """
        pass

class LLMQuery(Operation):
    """Logical Operation: LLM querying"""
    @abstractmethod
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        :param prompt: Text prompt
        :return: {'response': 'The answer is...', 'meta': ...}
        """
        pass