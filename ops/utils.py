"""
Operation 辅助工具类

提供数据传输和存储的辅助功能，供 Operation 内部使用。
跨云传输（S3 ↔ GCS ↔ Azure）采用流式直传，不经本地落盘，降低延迟。
"""

import io
import os
import logging
import time
from urllib.parse import urlparse
from typing import Optional, List, Dict

# 配置日志
logger = logging.getLogger("DataTransmission")


class DataTransmission:
    """
    数据传输中心：负责本地上传以及跨云搬运。
    跨云传输（S3 ↔ GCS ↔ Azure）使用流式直传，不经本地落盘，降低延迟。
    采用 Lazy Loading 机制，只有在真正需要时才初始化云客户端。
    """

    def __init__(self, aws_region=None, gcp_project=None, azure_account_name=None):
        self.aws_region = aws_region
        self.gcp_project = gcp_project
        self.azure_account_name = azure_account_name  # 默认使用的 Azure 账户名称
        self._s3_client = None
        self._gcs_client = None
        self._azure_clients = {}  # 字典，key 为账户名称，value 为 BlobServiceClient
        self._azure_config = None  # 缓存 Azure 配置

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

    def _get_azure_config(self):
        """获取 Azure 配置（从 config 模块）"""
        if self._azure_config is None:
            try:
                import config
                if hasattr(config, 'AZURE_STORAGE_ACCOUNTS'):
                    self._azure_config = config.AZURE_STORAGE_ACCOUNTS
                else:
                    self._azure_config = {}
                    logger.warning("Azure storage accounts not configured in config.py")
            except ImportError:
                self._azure_config = {}
                logger.warning("config.py not found, Azure storage accounts not available")
        return self._azure_config

    def get_azure_client(self, account_name: Optional[str] = None):
        """懒加载 Azure Blob Storage 客户端
        
        Args:
            account_name: Azure 账户名称，如果为 None 则使用 self.azure_account_name 或第一个可用账户
        """
        config = self._get_azure_config()
        if not config:
            raise RuntimeError("Azure storage accounts not configured. Please configure AZURE_STORAGE_ACCOUNTS in config.py")
        
        # 确定要使用的账户名称
        if account_name is None:
            account_name = self.azure_account_name or list(config.keys())[0]
        
        if account_name not in config:
            raise ValueError(f"Azure account '{account_name}' not found in configuration. Available accounts: {list(config.keys())}")
        
        # 懒加载客户端
        if account_name not in self._azure_clients:
            from azure.storage.blob import BlobServiceClient
            try:
                connection_string = config[account_name]["connection_string"]
                self._azure_clients[account_name] = BlobServiceClient.from_connection_string(connection_string)
                logger.info(f"Azure Blob Storage Client initialized for account: {account_name}")
            except Exception as e:
                logger.error(f"Failed to init Azure Blob Storage for account {account_name}: {e}")
                raise
        return self._azure_clients[account_name]

    def _parse_uri(self, uri: str):
        """解析 URI，返回 (scheme, bucket/container, key/blob)
        
        支持的 URI 格式：
        - s3://bucket/key
        - gs://bucket/key
        - azure://container/blob
        - https://account.blob.core.windows.net/container/blob
        
        Raises:
            ValueError: 如果 URI 格式无效
        """
        # 处理 Azure HTTPS URL
        if uri.startswith('https://') and '.blob.core.windows.net' in uri:
            # 解析 Azure HTTPS URL: https://account.blob.core.windows.net/container/blob
            parsed = urlparse(uri)
            path_parts = parsed.path.lstrip('/').split('/', 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid Azure HTTPS URI format: {uri}. Expected: https://account.blob.core.windows.net/container/blob")
            container = path_parts[0]
            blob = path_parts[1]
            # 从 hostname 提取账户名
            account_name = parsed.netloc.split('.')[0]
            return 'azure', container, blob, account_name
        
        # 处理标准 URI 格式 (s3://, gs://, azure://)
        parsed = urlparse(uri)
        scheme = parsed.scheme
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        if not scheme or not bucket:
            raise ValueError(f"Invalid URI format: {uri}. Expected format: s3://bucket/key, gs://bucket/key, or azure://container/blob")
        
        # 对于 Azure，返回额外的账户名（如果 URI 中没有指定，返回 None）
        if scheme == 'azure':
            return scheme, bucket, key, None
        
        return scheme, bucket, key
    
    def transfer_s3_to_s3(self, s3_uri: str, target_s3_bucket: str, target_path: Optional[str] = None) -> str:
        """S3 -> S3 跨 region 传输（流式直传，不落盘）
        
        Args:
            s3_uri: S3 源文件 URI
            target_s3_bucket: 目标 S3 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
        """
        parsed = self._parse_uri(s3_uri)
        scheme, s3_bucket, s3_key = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 's3':
            raise ValueError(f"Expected s3:// URI, got {scheme}:// in {s3_uri}")
        
        filename = os.path.basename(s3_key)
        
        if target_path:
            target_path = target_path.strip('/')
            target_key = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            target_key = s3_key
        
        target_uri = f"s3://{target_s3_bucket}/{target_key}"
        logger.info(f"[Bridge] S3 -> S3 跨 region 传输: {s3_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] S3 -> S3 跨 region 传输")
        print(f"  原视频位置: {s3_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            # 使用 copy_object 进行跨 region 复制（更高效）
            copy_source = {'Bucket': s3_bucket, 'Key': s3_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=target_s3_bucket, Key=target_key)
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间（记录传输发生的operation）
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    s3_uri, target_uri, "transfer_s3_to_s3", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass  # 如果记录失败，不影响主流程
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer S3 to S3: {s3_uri} -> {target_uri}: {e}")
            raise
    
    def transfer_gcs_to_gcs(self, gcs_uri: str, target_gcs_bucket: str, target_path: Optional[str] = None) -> str:
        """GCS -> GCS 跨 region 传输（流式直传，不落盘）
        
        Args:
            gcs_uri: GCS 源文件 URI
            target_gcs_bucket: 目标 GCS 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
        """
        parsed = self._parse_uri(gcs_uri)
        scheme, gcs_bucket, gcs_blob = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 'gs':
            raise ValueError(f"Expected gs:// URI, got {scheme}:// in {gcs_uri}")
        
        filename = os.path.basename(gcs_blob)
        
        if target_path:
            target_path = target_path.strip('/')
            target_blob = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            target_blob = gcs_blob
        
        target_uri = f"gs://{target_gcs_bucket}/{target_blob}"
        logger.info(f"[Bridge] GCS -> GCS 跨 region 传输: {gcs_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] GCS -> GCS 跨 region 传输")
        print(f"  原视频位置: {gcs_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            # 使用 copy_blob 进行跨 region 复制（更高效）
            source_bucket = self.gcs_client.bucket(gcs_bucket)
            source_blob = source_bucket.blob(gcs_blob)
            target_bucket = self.gcs_client.bucket(target_gcs_bucket)
            target_blob_obj = target_bucket.blob(target_blob)
            target_blob_obj.rewrite(source_blob)
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间（记录传输发生的operation）
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    gcs_uri, target_uri, "transfer_gcs_to_gcs", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass  # 如果记录失败，不影响主流程
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer GCS to GCS: {gcs_uri} -> {target_uri}: {e}")
            raise

    def transfer_s3_to_gcs(self, s3_uri: str, target_gcs_bucket: str, target_path: Optional[str] = None) -> str:
        """AWS S3 -> Google GCS 流式直传（不落盘，降低延迟）
        
        Args:
            s3_uri: S3 源文件 URI
            target_gcs_bucket: 目标 GCS 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
        """
        parsed = self._parse_uri(s3_uri)
        scheme, s3_bucket, s3_key = parsed[0], parsed[1], parsed[2]
        filename = os.path.basename(s3_key)

        if target_path:
            target_path = target_path.strip('/')
            gcs_key = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            gcs_key = s3_key

        # 验证 scheme
        if scheme != 's3':
            raise ValueError(f"Expected s3:// URI, got {scheme}:// in {s3_uri}")
        
        target_uri = f"gs://{target_gcs_bucket}/{gcs_key}"
        logger.info(f"[Bridge] S3 -> GCS 流式直传: {s3_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] S3 -> GCS 跨云传输")
        print(f"  原视频位置: {s3_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            resp = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            body = resp["Body"]

            bucket = self.gcs_client.bucket(target_gcs_bucket)
            blob = bucket.blob(gcs_key)
            
            # 使用 blob.open("wb") 进行流式写入，避免 upload_from_file 对不可寻址流的限制
            # 这与 transfer_gcs_to_s3 中的 blob.open("rb") 方法对应
            with blob.open("wb") as gcs_stream:
                # 分块读取 S3 数据并写入 GCS，避免一次性加载到内存
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        break
                    gcs_stream.write(chunk)
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间（记录传输发生的operation）
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    s3_uri, target_uri, "transfer_s3_to_gcs", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass  # 如果记录失败，不影响主流程
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer S3 to GCS: {s3_uri} -> {target_uri}: {e}")
            raise

    def transfer_gcs_to_s3(self, gcs_uri: str, target_s3_bucket: str, target_path: Optional[str] = None) -> str:
        """Google GCS -> AWS S3 流式直传（不落盘，降低延迟）
        
        Args:
            gcs_uri: GCS 源文件 URI
            target_s3_bucket: 目标 S3 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
        """
        parsed = self._parse_uri(gcs_uri)
        scheme, gcs_bucket, gcs_blob = parsed[0], parsed[1], parsed[2]
        filename = os.path.basename(gcs_blob)

        if target_path:
            target_path = target_path.strip('/')
            s3_key = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            s3_key = gcs_blob

        # 验证 scheme
        if scheme != 'gs':
            raise ValueError(f"Expected gs:// URI, got {scheme}:// in {gcs_uri}")
        
        target_uri = f"s3://{target_s3_bucket}/{s3_key}"
        logger.info(f"[Bridge] GCS -> S3 流式直传: {gcs_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] GCS -> S3 跨云传输")
        print(f"  原视频位置: {gcs_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            bucket = self.gcs_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob)
            with blob.open("rb") as stream:
                self.s3_client.upload_fileobj(stream, target_s3_bucket, s3_key)
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间（记录传输发生的operation）
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    gcs_uri, target_uri, "transfer_gcs_to_s3", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass  # 如果记录失败，不影响主流程
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer GCS to S3: {gcs_uri} -> {target_uri}: {e}")
            raise

    def transfer_azure_to_azure(self, azure_uri: str, target_container: str, target_path: Optional[str] = None, source_account_name: Optional[str] = None, target_account_name: Optional[str] = None) -> str:
        """Azure -> Azure 跨容器/跨账户传输（流式直传，不落盘）
        
        Args:
            azure_uri: Azure 源文件 URI (azure://container/blob 或 https://account.blob.core.windows.net/container/blob)
            target_container: 目标容器名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            source_account_name: 源 Azure 账户名称（如果 URI 中没有指定）
            target_account_name: 目标 Azure 账户名称（如果为 None 则使用源账户）
        """
        parsed = self._parse_uri(azure_uri)
        scheme = parsed[0]
        source_container = parsed[1]
        source_blob = parsed[2]
        parsed_account_name = parsed[3] if len(parsed) > 3 else None
        
        # 验证 scheme
        if scheme != 'azure':
            raise ValueError(f"Expected azure:// URI or Azure HTTPS URL, got {scheme}:// in {azure_uri}")
        
        # 确定账户名称
        source_account = parsed_account_name or source_account_name
        target_account = target_account_name or source_account
        
        filename = os.path.basename(source_blob)
        
        if target_path:
            target_path = target_path.strip('/')
            target_blob = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是container不同）
            target_blob = source_blob
        
        target_uri = f"azure://{target_container}/{target_blob}"
        logger.info(f"[Bridge] Azure -> Azure 传输: {azure_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] Azure -> Azure 传输")
        print(f"  原视频位置: {azure_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            source_client = self.get_azure_client(source_account)
            source_container_client = source_client.get_container_client(source_container)
            source_blob_client = source_container_client.get_blob_client(source_blob)
            
            target_client = self.get_azure_client(target_account)
            target_container_client = target_client.get_container_client(target_container)
            # 确保目标容器存在
            if not target_container_client.exists():
                target_container_client.create_container()
            target_blob_client = target_container_client.get_blob_client(target_blob)
            
            # 流式复制 - 使用 chunks() 迭代器进行流式传输
            download_stream = source_blob_client.download_blob()
            # 创建一个可读的流对象用于上传
            import io
            stream_data = io.BytesIO()
            for chunk in download_stream.chunks():
                stream_data.write(chunk)
            stream_data.seek(0)
            target_blob_client.upload_blob(stream_data, overwrite=True)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    azure_uri, target_uri, "transfer_azure_to_azure", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer Azure to Azure: {azure_uri} -> {target_uri}: {e}")
            raise

    def transfer_azure_to_gcs(self, azure_uri: str, target_gcs_bucket: str, target_path: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """Azure -> Google GCS 流式直传（不落盘，降低延迟）
        
        Args:
            azure_uri: Azure 源文件 URI
            target_gcs_bucket: 目标 GCS 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            azure_account_name: Azure 账户名称（如果 URI 中没有指定）
        """
        parsed = self._parse_uri(azure_uri)
        scheme = parsed[0]
        source_container = parsed[1]
        source_blob = parsed[2]
        parsed_account_name = parsed[3] if len(parsed) > 3 else None
        
        # 验证 scheme
        if scheme != 'azure':
            raise ValueError(f"Expected azure:// URI or Azure HTTPS URL, got {scheme}:// in {azure_uri}")
        
        account_name = parsed_account_name or azure_account_name
        filename = os.path.basename(source_blob)
        
        if target_path:
            target_path = target_path.strip('/')
            gcs_key = f"{target_path}/{filename}" if target_path else filename
        else:
            gcs_key = source_blob
        
        target_uri = f"gs://{target_gcs_bucket}/{gcs_key}"
        logger.info(f"[Bridge] Azure -> GCS 流式直传: {azure_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] Azure -> GCS 跨云传输")
        print(f"  原视频位置: {azure_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            blob_service_client = self.get_azure_client(account_name)
            container_client = blob_service_client.get_container_client(source_container)
            blob_client = container_client.get_blob_client(source_blob)
            
            bucket = self.gcs_client.bucket(target_gcs_bucket)
            gcs_blob = bucket.blob(gcs_key)
            
            # 流式传输 - Azure download_blob() 返回 StorageStreamDownloader，使用 chunks() 迭代器
            download_stream = blob_client.download_blob()
            with gcs_blob.open("wb") as gcs_stream:
                # 使用 chunks() 方法进行流式读取
                for chunk in download_stream.chunks():
                    gcs_stream.write(chunk)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    azure_uri, target_uri, "transfer_azure_to_gcs", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer Azure to GCS: {azure_uri} -> {target_uri}: {e}")
            raise

    def transfer_gcs_to_azure(self, gcs_uri: str, target_container: str, target_path: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """Google GCS -> Azure 流式直传（不落盘，降低延迟）
        
        Args:
            gcs_uri: GCS 源文件 URI
            target_container: 目标 Azure 容器名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            azure_account_name: Azure 账户名称
        """
        parsed = self._parse_uri(gcs_uri)
        scheme, gcs_bucket, gcs_blob = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 'gs':
            raise ValueError(f"Expected gs:// URI, got {scheme}:// in {gcs_uri}")
        
        filename = os.path.basename(gcs_blob)
        
        if target_path:
            target_path = target_path.strip('/')
            target_blob = f"{target_path}/{filename}" if target_path else filename
        else:
            target_blob = gcs_blob
        
        target_uri = f"azure://{target_container}/{target_blob}"
        logger.info(f"[Bridge] GCS -> Azure 流式直传: {gcs_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] GCS -> Azure 跨云传输")
        print(f"  原视频位置: {gcs_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            bucket = self.gcs_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob)
            
            blob_service_client = self.get_azure_client(azure_account_name)
            container_client = blob_service_client.get_container_client(target_container)
            # 确保容器存在
            if not container_client.exists():
                container_client.create_container()
            target_blob_client = container_client.get_blob_client(target_blob)
            
            # 流式传输
            with blob.open("rb") as gcs_stream:
                target_blob_client.upload_blob(gcs_stream, overwrite=True)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    gcs_uri, target_uri, "transfer_gcs_to_azure", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer GCS to Azure: {gcs_uri} -> {target_uri}: {e}")
            raise

    def transfer_azure_to_s3(self, azure_uri: str, target_s3_bucket: str, target_path: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """Azure -> AWS S3 流式直传（不落盘，降低延迟）
        
        Args:
            azure_uri: Azure 源文件 URI
            target_s3_bucket: 目标 S3 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            azure_account_name: Azure 账户名称（如果 URI 中没有指定）
        """
        parsed = self._parse_uri(azure_uri)
        scheme = parsed[0]
        source_container = parsed[1]
        source_blob = parsed[2]
        parsed_account_name = parsed[3] if len(parsed) > 3 else None
        
        # 验证 scheme
        if scheme != 'azure':
            raise ValueError(f"Expected azure:// URI or Azure HTTPS URL, got {scheme}:// in {azure_uri}")
        
        account_name = parsed_account_name or azure_account_name
        filename = os.path.basename(source_blob)
        
        if target_path:
            target_path = target_path.strip('/')
            s3_key = f"{target_path}/{filename}" if target_path else filename
        else:
            s3_key = source_blob
        
        target_uri = f"s3://{target_s3_bucket}/{s3_key}"
        logger.info(f"[Bridge] Azure -> S3 流式直传: {azure_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] Azure -> S3 跨云传输")
        print(f"  原视频位置: {azure_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            blob_service_client = self.get_azure_client(account_name)
            container_client = blob_service_client.get_container_client(source_container)
            blob_client = container_client.get_blob_client(source_blob)
            
            # 流式传输 - 使用 chunks() 迭代器
            download_stream = blob_client.download_blob()
            import io
            stream_data = io.BytesIO()
            for chunk in download_stream.chunks():
                stream_data.write(chunk)
            stream_data.seek(0)
            self.s3_client.upload_fileobj(stream_data, target_s3_bucket, s3_key)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    azure_uri, target_uri, "transfer_azure_to_s3", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer Azure to S3: {azure_uri} -> {target_uri}: {e}")
            raise

    def transfer_s3_to_azure(self, s3_uri: str, target_container: str, target_path: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """AWS S3 -> Azure 流式直传（不落盘，降低延迟）
        
        Args:
            s3_uri: S3 源文件 URI
            target_container: 目标 Azure 容器名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            azure_account_name: Azure 账户名称
        """
        parsed = self._parse_uri(s3_uri)
        scheme, s3_bucket, s3_key = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 's3':
            raise ValueError(f"Expected s3:// URI, got {scheme}:// in {s3_uri}")
        
        filename = os.path.basename(s3_key)
        
        if target_path:
            target_path = target_path.strip('/')
            target_blob = f"{target_path}/{filename}" if target_path else filename
        else:
            target_blob = s3_key
        
        target_uri = f"azure://{target_container}/{target_blob}"
        logger.info(f"[Bridge] S3 -> Azure 流式直传: {s3_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] S3 -> Azure 跨云传输")
        print(f"  原视频位置: {s3_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            resp = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            body = resp["Body"]
            
            blob_service_client = self.get_azure_client(azure_account_name)
            container_client = blob_service_client.get_container_client(target_container)
            # 确保容器存在
            if not container_client.exists():
                container_client.create_container()
            target_blob_client = container_client.get_blob_client(target_blob)
            
            # 流式传输 - S3的body已经是流对象，可以直接使用
            target_blob_client.upload_blob(body, overwrite=True)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    s3_uri, target_uri, "transfer_s3_to_azure", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer S3 to Azure: {s3_uri} -> {target_uri}: {e}")
            raise

    def upload_local_to_cloud(self, local_path: str, provider: str, target_bucket: str, target_path: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """将本地文件上传到指定云存储桶
        
        Args:
            local_path: 本地文件路径
            provider: 云服务提供商 ('google', 'amazon', 或 'azure')
            target_bucket: 目标存储桶/容器名称
            target_path: 目标路径（目录），如果为 None 则上传到根目录
            azure_account_name: Azure 账户名称（仅当 provider='azure' 时使用）
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

        # 构建目标URI用于打印
        if provider == 'google':
            target_uri = f"gs://{target_bucket}/{cloud_key}"
        elif provider == 'amazon':
            target_uri = f"s3://{target_bucket}/{cloud_key}"
        elif provider == 'azure':
            target_uri = f"azure://{target_bucket}/{cloud_key}"
        else:
            raise ValueError(f"Unknown provider: {provider}. Expected 'google', 'amazon', or 'azure'")
        
        # 打印传输信息
        print(f"\n[数据传输] 本地文件上传到 {provider.upper()}")
        print(f"  原视频位置: {local_path}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")

        # 记录传输时间
        start_time = time.time()
        try:
            if provider == 'google':
                bucket = self.gcs_client.bucket(target_bucket)
                blob = bucket.blob(cloud_key)
                blob.upload_from_filename(local_path)
            elif provider == 'amazon':
                self.s3_client.upload_file(local_path, target_bucket, cloud_key)
            elif provider == 'azure':
                blob_service_client = self.get_azure_client(azure_account_name)
                container_client = blob_service_client.get_container_client(target_bucket)
                # 确保容器存在
                if not container_client.exists():
                    container_client.create_container()
                blob_client = container_client.get_blob_client(cloud_key)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
            else:
                raise ValueError(f"Unknown provider: {provider}. Expected 'google', 'amazon', or 'azure'")
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间（记录传输发生的operation）
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    local_path, target_uri, f"upload_local_to_{provider}", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass  # 如果记录失败，不影响主流程
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to upload local file {local_path} to {provider} bucket {target_bucket}/{cloud_key}: {e}")
            raise

    def smart_move(self, source_uri: str, target_provider: str, target_bucket: str, target_path: Optional[str] = None, target_region: Optional[str] = None, azure_account_name: Optional[str] = None) -> str:
        """
        [智能路由入口]
        根据 source_uri 的类型（本地路径/S3/GCS/Azure）和目标 Provider，
        自动决定是直接上传、跨云流式直传、跨 region 传输、还是保持原样。
        跨云传输和跨 region 传输不落盘，降低延迟。
        
        Args:
            source_uri: 源文件 URI（本地路径、s3://、gs:// 或 azure://）
            target_provider: 目标云服务提供商 ('google', 'amazon', 或 'azure')
            target_bucket: 目标存储桶/容器名称
            target_path: 目标路径（目录），如果为 None：
                - 对于本地文件上传：上传到根目录
                - 对于跨bucket传输：使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            target_region: 目标 region（可选），如果提供则检查是否需要跨 region 传输
            azure_account_name: Azure 账户名称（仅当 target_provider='azure' 时使用）
        """
        # 1. 本地文件 -> 上传
        if not (source_uri.startswith('s3://') or source_uri.startswith('gs://') or source_uri.startswith('azure://') or source_uri.startswith('https://')):
            return self.upload_local_to_cloud(source_uri, target_provider, target_bucket, target_path, azure_account_name)

        parsed = self._parse_uri(source_uri)
        scheme = parsed[0]
        source_bucket = parsed[1]
        # 对于 Azure URI，可能有账户名（第4个元素）
        parsed_account = parsed[3] if len(parsed) > 3 else None

        # 2. 如果源 bucket 和目标 bucket 相同，直接返回（不需要移动）
        if source_bucket == target_bucket:
            print(f"\n[数据传输] 跳过传输（源和目标相同）")
            print(f"  视频位置: {source_uri}")
            print(f"  无需传输\n")
            return source_uri

        # 3. bucket 不同，需要传输
        # 判断是跨云还是同云跨 region
        if target_provider == 'google' and scheme == 'gs':
            # GCS -> GCS 跨 region 传输
            return self.transfer_gcs_to_gcs(source_uri, target_bucket, target_path)
        elif target_provider == 'amazon' and scheme == 's3':
            # S3 -> S3 跨 region 传输
            return self.transfer_s3_to_s3(source_uri, target_bucket, target_path)
        elif target_provider == 'azure' and scheme == 'azure':
            # Azure -> Azure 跨容器/跨账户传输
            return self.transfer_azure_to_azure(source_uri, target_bucket, target_path, parsed_account, azure_account_name)
        elif target_provider == 'google' and scheme == 's3':
            # S3 -> GCS 跨云传输
            return self.transfer_s3_to_gcs(source_uri, target_bucket, target_path)
        elif target_provider == 'amazon' and scheme == 'gs':
            # GCS -> S3 跨云传输
            return self.transfer_gcs_to_s3(source_uri, target_bucket, target_path)
        elif target_provider == 'google' and scheme == 'azure':
            # Azure -> GCS 跨云传输
            return self.transfer_azure_to_gcs(source_uri, target_bucket, target_path, parsed_account)
        elif target_provider == 'azure' and scheme == 'gs':
            # GCS -> Azure 跨云传输
            return self.transfer_gcs_to_azure(source_uri, target_bucket, target_path, azure_account_name)
        elif target_provider == 'amazon' and scheme == 'azure':
            # Azure -> S3 跨云传输
            return self.transfer_azure_to_s3(source_uri, target_bucket, target_path, parsed_account)
        elif target_provider == 'azure' and scheme == 's3':
            # S3 -> Azure 跨云传输
            return self.transfer_s3_to_azure(source_uri, target_bucket, target_path, azure_account_name)
        else:
            # 未匹配的情况：可能是无效的 provider 或 scheme 组合
            logger.warning(
                f"smart_move: No transfer method matched for source_uri={source_uri} "
                f"(scheme={scheme}), target_provider={target_provider}. Returning source_uri unchanged."
            )
            return source_uri


# class DataStorageHelper:
#     """
#     数据存储辅助类：负责文件的存储和检索
    
#     职责：
#     - 上传文件到云存储
#     - 从云存储下载文件
#     - 删除云存储文件
#     - 列出云存储文件
#     """
    
#     def __init__(self, aws_region=None, gcp_project=None):
#         self.aws_region = aws_region
#         self.gcp_project = gcp_project
#         self._s3_client = None
#         self._gcs_client = None
    
#     @property
#     def s3_client(self):
#         """懒加载 AWS S3 客户端"""
#         if self._s3_client is None:
#             import boto3
#             try:
#                 if self.aws_region:
#                     self._s3_client = boto3.client('s3', region_name=self.aws_region)
#                 else:
#                     self._s3_client = boto3.client('s3')
#                 logger.info("AWS S3 Client initialized.")
#             except Exception as e:
#                 logger.error(f"Failed to init AWS S3: {e}")
#                 raise e
#         return self._s3_client
    
#     @property
#     def gcs_client(self):
#         """懒加载 Google GCS 客户端"""
#         if self._gcs_client is None:
#             from google.cloud import storage
#             try:
#                 self._gcs_client = storage.Client(project=self.gcp_project)
#                 logger.info("Google GCS Client initialized.")
#             except Exception as e:
#                 logger.error(f"Failed to init Google GCS: {e}")
#                 raise e
#         return self._gcs_client
    
#     def _parse_uri(self, uri: str):
#         """解析 URI"""
#         parsed = urlparse(uri)
#         return parsed.scheme, parsed.netloc, parsed.path.lstrip('/')
    
#     def upload(self, local_path: str, provider: str, target_bucket: str, target_path: Optional[str] = None) -> str:
#         """上传本地文件到云存储"""
#         if not os.path.exists(local_path):
#             raise FileNotFoundError(f"Local file not found: {local_path}")
        
#         filename = os.path.basename(local_path)
#         if target_path:
#             target_path = target_path.strip('/')
#             cloud_key = f"{target_path}/{filename}" if target_path else filename
#         else:
#             cloud_key = filename
        
#         logger.info(f"Uploading {filename} to {provider} bucket {target_bucket}/{cloud_key}...")
        
#         if provider == 'google':
#             bucket = self.gcs_client.bucket(target_bucket)
#             blob = bucket.blob(cloud_key)
#             blob.upload_from_filename(local_path)
#             return f"gs://{target_bucket}/{cloud_key}"
#         elif provider == 'amazon':
#             self.s3_client.upload_file(local_path, target_bucket, cloud_key)
#             return f"s3://{target_bucket}/{cloud_key}"
#         else:
#             raise ValueError(f"Unknown provider: {provider}")
    
#     def download(self, cloud_uri: str, local_path: str) -> str:
#         """从云存储下载文件到本地"""
#         scheme, bucket, key = self._parse_uri(cloud_uri)
        
#         if scheme == 'gs':
#             bucket_obj = self.gcs_client.bucket(bucket)
#             blob = bucket_obj.blob(key)
#             blob.download_to_filename(local_path)
#         elif scheme == 's3':
#             self.s3_client.download_file(bucket, key, local_path)
#         else:
#             raise ValueError(f"Unsupported URI scheme: {scheme}")
        
#         return local_path
    
#     def delete(self, cloud_uri: str) -> bool:
#         """删除云存储文件"""
#         scheme, bucket, key = self._parse_uri(cloud_uri)
        
#         if scheme == 'gs':
#             bucket_obj = self.gcs_client.bucket(bucket)
#             blob = bucket_obj.blob(key)
#             blob.delete()
#         elif scheme == 's3':
#             self.s3_client.delete_object(Bucket=bucket, Key=key)
#         else:
#             raise ValueError(f"Unsupported URI scheme: {scheme}")
        
#         return True
    
#     def list_files(self, cloud_uri: str, prefix: Optional[str] = None) -> List[str]:
#         """列出云存储文件"""
#         scheme, bucket, key = self._parse_uri(cloud_uri)
#         files = []
        
#         if scheme == 'gs':
#             bucket_obj = self.gcs_client.bucket(bucket)
#             blobs = bucket_obj.list_blobs(prefix=prefix or key)
#             files = [f"gs://{bucket}/{blob.name}" for blob in blobs]
#         elif scheme == 's3':
#             response = self.s3_client.list_objects_v2(
#                 Bucket=bucket,
#                 Prefix=prefix or key
#             )
#             files = [f"s3://{bucket}/{obj['Key']}" for obj in response.get('Contents', [])]
#         else:
#             raise ValueError(f"Unsupported URI scheme: {scheme}")
        
#         return files
