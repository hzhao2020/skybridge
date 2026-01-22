"""
Operation 辅助工具类

提供数据传输和存储的辅助功能，供 Operation 内部使用。
"""

import os
import logging
import time
from urllib.parse import urlparse
from typing import Optional, List

# 配置日志
logger = logging.getLogger("DataTransmission")


class DataTransmission:
    """
    数据传输中心：负责本地上传以及跨云搬运。
    采用 Lazy Loading 机制，只有在真正需要时才初始化云客户端。
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
                    self._s3_client = boto3.client('s3')  # 读取默认配置
                logger.info("AWS S3 Client initialized.")
            except Exception as e:
                # botocore 在某些 AWS 凭证模式（如 SSO/Identity Center 的 login provider）下
                # 需要额外依赖 `botocore[crt]`，否则会抛出 MissingDependencyException。
                try:
                    from botocore.exceptions import MissingDependencyException
                except Exception:
                    MissingDependencyException = None  # type: ignore

                if MissingDependencyException is not None and isinstance(e, MissingDependencyException):
                    msg = (
                        "初始化 AWS S3 客户端失败：缺少依赖 `botocore[crt]`。\n"
                        "你的 AWS 凭证配置触发了 login credential provider（常见于 AWS SSO/Identity Center）。\n"
                        "请执行：pip install \"botocore[crt]\"  或  pip install -r requirements.txt\n"
                    )
                    logger.error(msg.strip())
                    raise RuntimeError(msg) from e

                logger.error(f"Failed to init AWS S3: {e}")
                raise
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
        parsed = urlparse(uri)
        return parsed.scheme, parsed.netloc, parsed.path.lstrip('/')

    def upload_local_to_cloud(self, local_path: str, provider: str, target_bucket: str, target_path: Optional[str] = None) -> str:
        """将本地文件上传到指定云存储桶
        
        Args:
            local_path: 本地文件路径
            provider: 云服务提供商 ('google' 或 'amazon')
            target_bucket: 目标存储桶名称
            target_path: 目标路径（目录），如果为 None 则上传到根目录
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        filename = os.path.basename(local_path)
        
        # 构建目标路径
        if target_path:
            # 确保路径格式正确（移除开头的斜杠，添加结尾的斜杠）
            target_path = target_path.strip('/')
            if target_path:
                cloud_key = f"{target_path}/{filename}"
            else:
                cloud_key = filename
        else:
            cloud_key = filename
        
        logger.info(f"Uploading local file {filename} to {provider} bucket {target_bucket}/{cloud_key}...")

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

    def transfer_s3_to_gcs(self, s3_uri: str, target_gcs_bucket: str, target_path: Optional[str] = None) -> str:
        """AWS S3 -> Google GCS (需经过本地中转)
        
        Args:
            s3_uri: S3 源文件 URI
            target_gcs_bucket: 目标 GCS 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用 transfer_cache/
        """
        scheme, s3_bucket, s3_key = self._parse_uri(s3_uri)
        filename = os.path.basename(s3_key)
        local_tmp = f"tmp_{int(time.time())}_{filename}"

        try:
            logger.info(f"[Bridge] Downloading from S3: {s3_uri}")
            self.s3_client.download_file(s3_bucket, s3_key, local_tmp)

            # 构建目标路径
            if target_path:
                target_path = target_path.strip('/')
                if target_path:
                    gcs_key = f"{target_path}/{filename}"
                else:
                    gcs_key = filename
            else:
                gcs_key = f"transfer_cache/{filename}"

            logger.info(f"[Bridge] Uploading to GCS: {target_gcs_bucket}/{gcs_key}")
            bucket = self.gcs_client.bucket(target_gcs_bucket)
            blob = bucket.blob(gcs_key)
            blob.upload_from_filename(local_tmp)

            return f"gs://{target_gcs_bucket}/{gcs_key}"
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def transfer_gcs_to_s3(self, gcs_uri: str, target_s3_bucket: str, target_path: Optional[str] = None) -> str:
        """Google GCS -> AWS S3 (需经过本地中转)
        
        Args:
            gcs_uri: GCS 源文件 URI
            target_s3_bucket: 目标 S3 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用 transfer_cache/
        """
        scheme, gcs_bucket, gcs_blob = self._parse_uri(gcs_uri)
        filename = os.path.basename(gcs_blob)
        local_tmp = f"tmp_{int(time.time())}_{filename}"

        try:
            logger.info(f"[Bridge] Downloading from GCS: {gcs_uri}")
            bucket = self.gcs_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob)
            blob.download_to_filename(local_tmp)

            # 构建目标路径
            if target_path:
                target_path = target_path.strip('/')
                if target_path:
                    s3_key = f"{target_path}/{filename}"
                else:
                    s3_key = filename
            else:
                s3_key = f"transfer_cache/{filename}"

            logger.info(f"[Bridge] Uploading to S3: {target_s3_bucket}/{s3_key}")
            self.s3_client.upload_file(local_tmp, target_s3_bucket, s3_key)

            return f"s3://{target_s3_bucket}/{s3_key}"
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def smart_move(self, source_uri: str, target_provider: str, target_bucket: str, target_path: Optional[str] = None) -> str:
        """
        [智能路由入口]
        根据 source_uri 的类型（本地路径/S3/GCS）和目标 Provider，
        自动决定是直接上传、跨云搬运、还是保持原样。
        
        Args:
            source_uri: 源文件 URI（本地路径、s3:// 或 gs://）
            target_provider: 目标云服务提供商 ('google' 或 'amazon')
            target_bucket: 目标存储桶名称
            target_path: 目标路径（目录），如果为 None 则上传到根目录或使用默认路径
        """
        # 1. 本地文件 -> 上传
        if not (source_uri.startswith('s3://') or source_uri.startswith('gs://')):
            return self.upload_local_to_cloud(source_uri, target_provider, target_bucket, target_path)

        scheme, _, _ = self._parse_uri(source_uri)

        # 2. 已经在目标云了 -> 不动
        if target_provider == 'google' and scheme == 'gs': return source_uri
        if target_provider == 'amazon' and scheme == 's3': return source_uri

        # 3. 跨云搬运
        if target_provider == 'google' and scheme == 's3':
            return self.transfer_s3_to_gcs(source_uri, target_bucket, target_path)
        if target_provider == 'amazon' and scheme == 'gs':
            return self.transfer_gcs_to_s3(source_uri, target_bucket, target_path)

        return source_uri


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
