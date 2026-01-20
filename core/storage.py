"""
数据存储辅助类：负责文件的存储和检索操作

注意：这是一个辅助类，供其他 Operation 内部使用。
如果需要作为独立的 Operation 使用，请使用 ops.impl.storage_ops 中的实现。
"""

import os
import logging
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any

logger = logging.getLogger("DataStorage")


class DataStorageHelper:
    """
    数据存储辅助类：负责文件的存储和检索
    
    职责：
    - 上传文件到云存储
    - 从云存储下载文件
    - 删除云存储文件
    - 列出云存储文件
    """
    
    def __init__(self, aws_region=None, gcp_project=None):
        self.aws_region = aws_region
        self.gcp_project = gcp_project
        self._s3_client = None
        self._gcs_client = None
    
    @property
    def s3_client(self):
        """懒加载 AWS S3 客户端"""
        if self._s3_client is None:
            import boto3
            try:
                if self.aws_region:
                    self._s3_client = boto3.client('s3', region_name=self.aws_region)
                else:
                    self._s3_client = boto3.client('s3')
                logger.info("AWS S3 Client initialized.")
            except Exception as e:
                logger.error(f"Failed to init AWS S3: {e}")
                raise e
        return self._s3_client
    
    @property
    def gcs_client(self):
        """懒加载 Google GCS 客户端"""
        if self._gcs_client is None:
            from google.cloud import storage
            try:
                self._gcs_client = storage.Client(project=self.gcp_project)
                logger.info("Google GCS Client initialized.")
            except Exception as e:
                logger.error(f"Failed to init Google GCS: {e}")
                raise e
        return self._gcs_client
    
    def _parse_uri(self, uri: str):
        """解析 URI"""
        parsed = urlparse(uri)
        return parsed.scheme, parsed.netloc, parsed.path.lstrip('/')
    
    def upload(self, local_path: str, provider: str, target_bucket: str, target_path: Optional[str] = None) -> str:
        """上传本地文件到云存储"""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        filename = os.path.basename(local_path)
        if target_path:
            target_path = target_path.strip('/')
            cloud_key = f"{target_path}/{filename}" if target_path else filename
        else:
            cloud_key = filename
        
        logger.info(f"Uploading {filename} to {provider} bucket {target_bucket}/{cloud_key}...")
        
        if provider == 'google':
            bucket = self.gcs_client.bucket(target_bucket)
            blob = bucket.blob(cloud_key)
            blob.upload_from_filename(local_path)
            return f"gs://{target_bucket}/{cloud_key}"
        elif provider == 'amazon':
            self.s3_client.upload_file(local_path, target_bucket, cloud_key)
            return f"s3://{target_bucket}/{cloud_key}"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def download(self, cloud_uri: str, local_path: str) -> str:
        """从云存储下载文件到本地"""
        scheme, bucket, key = self._parse_uri(cloud_uri)
        
        if scheme == 'gs':
            bucket_obj = self.gcs_client.bucket(bucket)
            blob = bucket_obj.blob(key)
            blob.download_to_filename(local_path)
        elif scheme == 's3':
            self.s3_client.download_file(bucket, key, local_path)
        else:
            raise ValueError(f"Unsupported URI scheme: {scheme}")
        
        return local_path
    
    def delete(self, cloud_uri: str) -> bool:
        """删除云存储文件"""
        scheme, bucket, key = self._parse_uri(cloud_uri)
        
        if scheme == 'gs':
            bucket_obj = self.gcs_client.bucket(bucket)
            blob = bucket_obj.blob(key)
            blob.delete()
        elif scheme == 's3':
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        else:
            raise ValueError(f"Unsupported URI scheme: {scheme}")
        
        return True
    
    def list_files(self, cloud_uri: str, prefix: Optional[str] = None) -> List[str]:
        """列出云存储文件"""
        scheme, bucket, key = self._parse_uri(cloud_uri)
        files = []
        
        if scheme == 'gs':
            bucket_obj = self.gcs_client.bucket(bucket)
            blobs = bucket_obj.list_blobs(prefix=prefix or key)
            files = [f"gs://{bucket}/{blob.name}" for blob in blobs]
        elif scheme == 's3':
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix or key
            )
            files = [f"s3://{bucket}/{obj['Key']}" for obj in response.get('Contents', [])]
        else:
            raise ValueError(f"Unsupported URI scheme: {scheme}")
        
        return files
