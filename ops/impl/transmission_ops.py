"""
数据传输 Operation 实现

将跨云/跨区域数据传输抽象为独立的 Operation。
"""

import os
import tempfile
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from ops.base import Operation


class DataTransmissionImpl(Operation):
    """数据传输 Operation - 支持跨云传输"""
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        # 传输操作不需要特定的 provider，可以跨云传输
        from core.transmission import DataTransmission
        self.transmission_helper = DataTransmission()
    
    def execute(self, source_uri: str, target_provider: str, target_bucket: Optional[str] = None, target_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        执行数据传输
        
        Args:
            source_uri: 源文件 URI（本地路径、s3:// 或 gs://）
            target_provider: 目标云服务提供商 ('google' 或 'amazon')
            target_bucket: 目标存储桶（如果为 None，使用 self.storage_bucket）
            target_path: 目标路径
        """
        if target_bucket is None:
            target_bucket = self.storage_bucket
        
        print(f"--- [Data Transmission] {source_uri} -> {target_provider} ---")
        
        # 使用 smart_move 进行智能传输
        target_uri = self.transmission_helper.smart_move(
            source_uri, target_provider, target_bucket, target_path
        )
        
        return {
            "provider": "transmission",
            "source_uri": source_uri,
            "target_provider": target_provider,
            "target_uri": target_uri,
            "transferred": source_uri != target_uri  # 是否实际发生了传输
        }


class S3ToGCSImpl(Operation):
    """S3 到 GCS 的专用传输 Operation"""
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        from core.transmission import DataTransmission
        self.transmission_helper = DataTransmission()
    
    def execute(self, source_uri: str, target_bucket: Optional[str] = None, target_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """从 S3 传输到 GCS"""
        if not source_uri.startswith('s3://'):
            raise ValueError(f"Source URI must be S3 URI, got: {source_uri}")
        
        if target_bucket is None:
            target_bucket = self.storage_bucket
        
        print(f"--- [S3 -> GCS Transmission] {source_uri} -> gs://{target_bucket} ---")
        
        target_uri = self.transmission_helper.transfer_s3_to_gcs(
            source_uri, target_bucket, target_path
        )
        
        return {
            "provider": "transmission",
            "source_uri": source_uri,
            "target_uri": target_uri,
            "direction": "s3_to_gcs"
        }


class GCSToS3Impl(Operation):
    """GCS 到 S3 的专用传输 Operation"""
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        from core.transmission import DataTransmission
        self.transmission_helper = DataTransmission()
    
    def execute(self, source_uri: str, target_bucket: Optional[str] = None, target_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """从 GCS 传输到 S3"""
        if not source_uri.startswith('gs://'):
            raise ValueError(f"Source URI must be GCS URI, got: {source_uri}")
        
        if target_bucket is None:
            target_bucket = self.storage_bucket
        
        print(f"--- [GCS -> S3 Transmission] {source_uri} -> s3://{target_bucket} ---")
        
        target_uri = self.transmission_helper.transfer_gcs_to_s3(
            source_uri, target_bucket, target_path
        )
        
        return {
            "provider": "transmission",
            "source_uri": source_uri,
            "target_uri": target_uri,
            "direction": "gcs_to_s3"
        }
