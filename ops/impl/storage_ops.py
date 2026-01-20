"""
数据存储 Operation 实现

将数据存储抽象为独立的 Operation，可以作为独立的操作使用。
"""

import os
import tempfile
from typing import Dict, Any, List, Optional
from ops.base import Operation


class GoogleStorageImpl(Operation):
    """Google Cloud Storage Operation"""
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        from core.storage import DataStorageHelper
        self._storage_helper = DataStorageHelper(gcp_project=None)
    
    @property
    def storage_helper(self):
        """存储辅助类"""
        return self._storage_helper
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        执行存储操作
        
        Args:
            operation: 操作类型 ('upload', 'download', 'delete', 'list')
            **kwargs: 操作参数
                - upload: local_path, target_path
                - download: cloud_uri, local_path
                - delete: cloud_uri
                - list: cloud_uri, prefix
        """
        if operation == 'upload':
            local_path = kwargs.get('local_path')
            target_path = kwargs.get('target_path')
            if not local_path:
                raise ValueError("local_path is required for upload operation")
            
            cloud_uri = self.storage_helper.upload(
                local_path, 'google', self.storage_bucket, target_path
            )
            return {
                "provider": "google",
                "region": self.region,
                "operation": "upload",
                "local_path": local_path,
                "cloud_uri": cloud_uri
            }
        
        elif operation == 'download':
            cloud_uri = kwargs.get('cloud_uri')
            local_path = kwargs.get('local_path')
            if not cloud_uri:
                raise ValueError("cloud_uri is required for download operation")
            if not local_path:
                # 创建临时文件
                filename = os.path.basename(cloud_uri)
                local_path = tempfile.mktemp(suffix=f"_{filename}")
            
            self.storage_helper.download(cloud_uri, local_path)
            return {
                "provider": "google",
                "region": self.region,
                "operation": "download",
                "cloud_uri": cloud_uri,
                "local_path": local_path
            }
        
        elif operation == 'delete':
            cloud_uri = kwargs.get('cloud_uri')
            if not cloud_uri:
                raise ValueError("cloud_uri is required for delete operation")
            
            self.storage_helper.delete(cloud_uri)
            return {
                "provider": "google",
                "region": self.region,
                "operation": "delete",
                "cloud_uri": cloud_uri,
                "success": True
            }
        
        elif operation == 'list':
            cloud_uri = kwargs.get('cloud_uri', f"gs://{self.storage_bucket}/")
            prefix = kwargs.get('prefix')
            
            files = self.storage_helper.list_files(cloud_uri, prefix)
            return {
                "provider": "google",
                "region": self.region,
                "operation": "list",
                "cloud_uri": cloud_uri,
                "files": files,
                "count": len(files)
            }
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")


class AmazonStorageImpl(Operation):
    """Amazon S3 Storage Operation"""
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        from core.storage import DataStorageHelper
        self._storage_helper = DataStorageHelper(aws_region=region)
    
    @property
    def storage_helper(self):
        """存储辅助类"""
        return self._storage_helper
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """执行存储操作（与 GoogleStorageImpl 相同接口）"""
        if operation == 'upload':
            local_path = kwargs.get('local_path')
            target_path = kwargs.get('target_path')
            if not local_path:
                raise ValueError("local_path is required for upload operation")
            
            cloud_uri = self.storage_helper.upload(
                local_path, 'amazon', self.storage_bucket, target_path
            )
            return {
                "provider": "amazon",
                "region": self.region,
                "operation": "upload",
                "local_path": local_path,
                "cloud_uri": cloud_uri
            }
        
        elif operation == 'download':
            cloud_uri = kwargs.get('cloud_uri')
            local_path = kwargs.get('local_path')
            if not cloud_uri:
                raise ValueError("cloud_uri is required for download operation")
            if not local_path:
                filename = os.path.basename(cloud_uri)
                local_path = tempfile.mktemp(suffix=f"_{filename}")
            
            self.storage_helper.download(cloud_uri, local_path)
            return {
                "provider": "amazon",
                "region": self.region,
                "operation": "download",
                "cloud_uri": cloud_uri,
                "local_path": local_path
            }
        
        elif operation == 'delete':
            cloud_uri = kwargs.get('cloud_uri')
            if not cloud_uri:
                raise ValueError("cloud_uri is required for delete operation")
            
            self.storage_helper.delete(cloud_uri)
            return {
                "provider": "amazon",
                "region": self.region,
                "operation": "delete",
                "cloud_uri": cloud_uri,
                "success": True
            }
        
        elif operation == 'list':
            cloud_uri = kwargs.get('cloud_uri', f"s3://{self.storage_bucket}/")
            prefix = kwargs.get('prefix')
            
            files = self.storage_helper.list_files(cloud_uri, prefix)
            return {
                "provider": "amazon",
                "region": self.region,
                "operation": "list",
                "cloud_uri": cloud_uri,
                "files": files,
                "count": len(files)
            }
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
