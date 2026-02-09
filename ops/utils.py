"""
Operation 辅助工具类

提供数据传输和存储的辅助功能，供 Operation 内部使用。
跨云传输（S3 ↔ GCS ↔ OSS）采用流式直传，不经本地落盘，降低延迟。
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
    跨云传输（S3 ↔ GCS ↔ OSS）使用流式直传，不经本地落盘，降低延迟。
    采用 Lazy Loading 机制，只有在真正需要时才初始化云客户端。
    """

    def __init__(self, aws_region=None, gcp_project=None, aliyun_region=None):
        self.aws_region = aws_region
        self.gcp_project = gcp_project
        self.aliyun_region = aliyun_region
        self._s3_client = None
        self._gcs_client = None
        self._aliyun_buckets = {}  # 字典，key 为region，value 为 Bucket 对象
        self._aliyun_config = None  # 缓存阿里云配置

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

    def _get_aliyun_config(self):
        """获取阿里云配置（从 config 模块）"""
        if self._aliyun_config is None:
            try:
                import config
                if hasattr(config, 'ALIYUN_CONFIG') and hasattr(config, 'ALIYUN_STORAGE_CONFIG'):
                    self._aliyun_config = {
                        'auth': config.ALIYUN_CONFIG,
                        'storage': config.ALIYUN_STORAGE_CONFIG
                    }
                else:
                    self._aliyun_config = {}
                    logger.warning("Aliyun config not found in config.py")
            except ImportError:
                self._aliyun_config = {}
                logger.warning("config.py not found, Aliyun config not available")
        return self._aliyun_config

    def get_aliyun_bucket(self, region: Optional[str] = None):
        """懒加载阿里云 OSS Bucket 客户端
        
        Args:
            region: 阿里云区域，如果为 None 则使用 self.aliyun_region 或第一个可用区域
        """
        config = self._get_aliyun_config()
        if not config or 'auth' not in config or 'storage' not in config:
            raise RuntimeError("Aliyun config not configured. Please configure ALIYUN_CONFIG and ALIYUN_STORAGE_CONFIG in config.py")
        
        # 确定要使用的区域
        if region is None:
            region = self.aliyun_region or list(config['storage'].keys())[0]
        
        if region not in config['storage']:
            raise ValueError(f"Aliyun region '{region}' not found in configuration. Available regions: {list(config['storage'].keys())}")
        
        # 懒加载bucket客户端
        if region not in self._aliyun_buckets:
            import oss2
            try:
                auth_config = config['auth']
                storage_config = config['storage'][region]
                
                # 使用 AuthV4 进行认证
                auth = oss2.AuthV4(
                    auth_config['AccessKeyID'],
                    auth_config['AccessKeySecret']
                )
                
                # 创建Bucket对象
                bucket = oss2.Bucket(
                    auth,
                    storage_config['endpoint'],
                    storage_config['bucket'],
                    region=storage_config['region']
                )
                
                self._aliyun_buckets[region] = bucket
                logger.info(f"Aliyun OSS Bucket initialized for region: {region}")
            except Exception as e:
                logger.error(f"Failed to init Aliyun OSS Bucket for region {region}: {e}")
                raise
        return self._aliyun_buckets[region]

    def _parse_uri(self, uri: str):
        """解析 URI，返回 (scheme, bucket, key)
        
        支持的 URI 格式：
        - s3://bucket/key
        - gs://bucket/key
        - oss://bucket/key
        
        Raises:
            ValueError: 如果 URI 格式无效
        """
        parsed = urlparse(uri)
        scheme = parsed.scheme
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        if not scheme or not bucket:
            raise ValueError(f"Invalid URI format: {uri}. Expected format: s3://bucket/key, gs://bucket/key, or oss://bucket/key")
        
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

    def transfer_oss_to_oss(self, oss_uri: str, target_oss_bucket: str, target_region: str, target_path: Optional[str] = None, source_region: Optional[str] = None) -> str:
        """OSS -> OSS 跨 region 传输（流式直传，不落盘）
        
        Args:
            oss_uri: OSS 源文件 URI
            target_oss_bucket: 目标 OSS 存储桶名称
            target_region: 目标区域
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            source_region: 源区域（如果 URI 中没有指定）
        """
        parsed = self._parse_uri(oss_uri)
        scheme, source_bucket, source_key = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 'oss':
            raise ValueError(f"Expected oss:// URI, got {scheme}:// in {oss_uri}")
        
        filename = os.path.basename(source_key)
        
        if target_path:
            target_path = target_path.strip('/')
            target_key = f"{target_path}/{filename}" if target_path else filename
        else:
            # 如果没有指定target_path，使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            target_key = source_key
        
        target_uri = f"oss://{target_oss_bucket}/{target_key}"
        logger.info(f"[Bridge] OSS -> OSS 跨 region 传输: {oss_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] OSS -> OSS 跨 region 传输")
        print(f"  原视频位置: {oss_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            # 确定源区域
            if source_region is None:
                # 尝试从配置中找到源bucket对应的region
                config = self._get_aliyun_config()
                if config and 'storage' in config:
                    for reg, storage_config in config['storage'].items():
                        if storage_config['bucket'] == source_bucket:
                            source_region = reg
                            break
                    if source_region is None:
                        source_region = self.aliyun_region or list(config['storage'].keys())[0]
                else:
                    raise ValueError(f"Cannot determine source region for bucket {source_bucket}")
            
            # 获取源和目标bucket客户端
            source_bucket_client = self.get_aliyun_bucket(source_region)
            target_bucket_client = self.get_aliyun_bucket(target_region)
            
            # 流式传输
            source_obj = source_bucket_client.get_object(source_key)
            target_bucket_client.put_object(target_key, source_obj)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    oss_uri, target_uri, "transfer_oss_to_oss", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer OSS to OSS: {oss_uri} -> {target_uri}: {e}")
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

    def transfer_oss_to_s3(self, oss_uri: str, target_s3_bucket: str, target_path: Optional[str] = None, oss_region: Optional[str] = None) -> str:
        """OSS -> AWS S3 流式直传（不落盘，降低延迟）
        
        Args:
            oss_uri: OSS 源文件 URI
            target_s3_bucket: 目标 S3 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            oss_region: OSS 区域（如果 URI 中没有指定）
        """
        parsed = self._parse_uri(oss_uri)
        scheme, oss_bucket, oss_key = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 'oss':
            raise ValueError(f"Expected oss:// URI, got {scheme}:// in {oss_uri}")
        
        if oss_region is None:
            config = self._get_aliyun_config()
            if config and 'storage' in config:
                for reg, storage_config in config['storage'].items():
                    if storage_config['bucket'] == oss_bucket:
                        oss_region = reg
                        break
            if oss_region is None:
                oss_region = self.aliyun_region or list(config['storage'].keys())[0]
        
        filename = os.path.basename(oss_key)
        
        if target_path:
            target_path = target_path.strip('/')
            s3_key = f"{target_path}/{filename}" if target_path else filename
        else:
            s3_key = oss_key
        
        target_uri = f"s3://{target_s3_bucket}/{s3_key}"
        logger.info(f"[Bridge] OSS -> S3 流式直传: {oss_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] OSS -> S3 跨云传输")
        print(f"  原视频位置: {oss_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            oss_bucket_client = self.get_aliyun_bucket(oss_region)
            oss_obj = oss_bucket_client.get_object(oss_key)
            
            # 流式传输
            self.s3_client.upload_fileobj(oss_obj, target_s3_bucket, s3_key)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    oss_uri, target_uri, "transfer_oss_to_s3", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer OSS to S3: {oss_uri} -> {target_uri}: {e}")
            raise

    def transfer_s3_to_oss(self, s3_uri: str, target_oss_bucket: str, target_region: str, target_path: Optional[str] = None) -> str:
        """AWS S3 -> OSS 流式直传（不落盘，降低延迟）
        
        Args:
            s3_uri: S3 源文件 URI
            target_oss_bucket: 目标 OSS 存储桶名称
            target_region: 目标区域
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
            oss_key = f"{target_path}/{filename}" if target_path else filename
        else:
            oss_key = s3_key
        
        target_uri = f"oss://{target_oss_bucket}/{oss_key}"
        logger.info(f"[Bridge] S3 -> OSS 流式直传: {s3_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] S3 -> OSS 跨云传输")
        print(f"  原视频位置: {s3_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            resp = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            body = resp["Body"]
            
            oss_bucket_client = self.get_aliyun_bucket(target_region)
            
            # S3的body流不支持seek，需要分块读取并写入OSS
            # 使用BytesIO作为中间缓冲区
            import io
            buffer = io.BytesIO()
            chunk_size = 8192  # 8KB chunks
            
            # 从S3流读取数据并写入缓冲区
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
            
            # 重置缓冲区位置到开头
            buffer.seek(0)
            
            # 上传到OSS
            oss_bucket_client.put_object(oss_key, buffer)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    s3_uri, target_uri, "transfer_s3_to_oss", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer S3 to OSS: {s3_uri} -> {target_uri}: {e}")
            raise

    def transfer_oss_to_gcs(self, oss_uri: str, target_gcs_bucket: str, target_path: Optional[str] = None, oss_region: Optional[str] = None) -> str:
        """OSS -> Google GCS 流式直传（不落盘，降低延迟）
        
        Args:
            oss_uri: OSS 源文件 URI
            target_gcs_bucket: 目标 GCS 存储桶名称
            target_path: 目标路径（目录），如果为 None 则使用源文件的完整路径（保持相同路径结构）
            oss_region: OSS 区域（如果 URI 中没有指定）
        """
        parsed = self._parse_uri(oss_uri)
        scheme, oss_bucket, oss_key = parsed[0], parsed[1], parsed[2]
        
        # 验证 scheme
        if scheme != 'oss':
            raise ValueError(f"Expected oss:// URI, got {scheme}:// in {oss_uri}")
        
        if oss_region is None:
            config = self._get_aliyun_config()
            if config and 'storage' in config:
                for reg, storage_config in config['storage'].items():
                    if storage_config['bucket'] == oss_bucket:
                        oss_region = reg
                        break
            if oss_region is None:
                oss_region = self.aliyun_region or list(config['storage'].keys())[0]
        
        filename = os.path.basename(oss_key)
        
        if target_path:
            target_path = target_path.strip('/')
            gcs_key = f"{target_path}/{filename}" if target_path else filename
        else:
            gcs_key = oss_key
        
        target_uri = f"gs://{target_gcs_bucket}/{gcs_key}"
        logger.info(f"[Bridge] OSS -> GCS 流式直传: {oss_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] OSS -> GCS 跨云传输")
        print(f"  原视频位置: {oss_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            oss_bucket_client = self.get_aliyun_bucket(oss_region)
            oss_obj = oss_bucket_client.get_object(oss_key)
            
            bucket = self.gcs_client.bucket(target_gcs_bucket)
            gcs_blob = bucket.blob(gcs_key)
            
            # 流式传输
            with gcs_blob.open("wb") as gcs_stream:
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = oss_obj.read(chunk_size)
                    if not chunk:
                        break
                    gcs_stream.write(chunk)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    oss_uri, target_uri, "transfer_oss_to_gcs", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer OSS to GCS: {oss_uri} -> {target_uri}: {e}")
            raise

    def transfer_gcs_to_oss(self, gcs_uri: str, target_oss_bucket: str, target_region: str, target_path: Optional[str] = None) -> str:
        """Google GCS -> OSS 流式直传（不落盘，降低延迟）
        
        Args:
            gcs_uri: GCS 源文件 URI
            target_oss_bucket: 目标 OSS 存储桶名称
            target_region: 目标区域
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
            oss_key = f"{target_path}/{filename}" if target_path else filename
        else:
            oss_key = gcs_blob
        
        target_uri = f"oss://{target_oss_bucket}/{oss_key}"
        logger.info(f"[Bridge] GCS -> OSS 流式直传: {gcs_uri} -> {target_uri}")
        
        # 打印传输信息
        print(f"\n[数据传输] GCS -> OSS 跨云传输")
        print(f"  原视频位置: {gcs_uri}")
        print(f"  目标位置: {target_uri}")
        print(f"  传输中...")
        
        # 记录传输时间
        start_time = time.time()
        try:
            bucket = self.gcs_client.bucket(gcs_bucket)
            blob = bucket.blob(gcs_blob)
            
            oss_bucket_client = self.get_aliyun_bucket(target_region)
            
            # 流式传输
            with blob.open("rb") as gcs_stream:
                oss_bucket_client.put_object(oss_key, gcs_stream)
            
            end_time = time.time()
            print(f"  ✓ 传输完成 (耗时: {end_time - start_time:.2f} 秒)\n")
            
            # 记录传输时间
            try:
                from utils.timing import TimingRecorder
                recorder = TimingRecorder()
                recorder.record_transmission(
                    gcs_uri, target_uri, "transfer_gcs_to_oss", start_time, end_time,
                    operation=recorder._current_operation
                )
            except Exception:
                pass
            
            return target_uri
        except Exception as e:
            logger.error(f"Failed to transfer GCS to OSS: {gcs_uri} -> {target_uri}: {e}")
            raise

    def upload_local_to_cloud(self, local_path: str, provider: str, target_bucket: str, target_path: Optional[str] = None, target_region: Optional[str] = None) -> str:
        """将本地文件上传到指定云存储桶
        
        Args:
            local_path: 本地文件路径
            provider: 云服务提供商 ('google', 'amazon', 或 'aliyun')
            target_bucket: 目标存储桶名称
            target_path: 目标路径（目录），如果为 None 则上传到根目录
            target_region: 目标区域（仅当 provider='aliyun' 时使用）
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
        elif provider == 'aliyun':
            target_uri = f"oss://{target_bucket}/{cloud_key}"
        else:
            raise ValueError(f"Unknown provider: {provider}. Expected 'google', 'amazon', or 'aliyun'")
        
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
            elif provider == 'aliyun':
                if target_region is None:
                    raise ValueError("target_region is required when provider is 'aliyun'")
                oss_bucket_client = self.get_aliyun_bucket(target_region)
                oss_bucket_client.put_object_from_file(cloud_key, local_path)
            else:
                raise ValueError(f"Unknown provider: {provider}. Expected 'google', 'amazon', or 'aliyun'")
            
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

    def smart_move(self, source_uri: str, target_provider: str, target_bucket: str, target_path: Optional[str] = None, target_region: Optional[str] = None) -> str:
        """
        [智能路由入口]
        根据 source_uri 的类型（本地路径/S3/GCS/OSS）和目标 Provider，
        自动决定是直接上传、跨云流式直传、跨 region 传输、还是保持原样。
        跨云传输和跨 region 传输不落盘，降低延迟。
        
        Args:
            source_uri: 源文件 URI（本地路径、s3://、gs:// 或 oss://）
            target_provider: 目标云服务提供商 ('google', 'amazon', 或 'aliyun')
            target_bucket: 目标存储桶名称
            target_path: 目标路径（目录），如果为 None：
                - 对于本地文件上传：上传到根目录
                - 对于跨bucket传输：使用源文件的完整路径（保持相同路径结构，只是bucket不同）
            target_region: 目标 region（可选），对于阿里云是必需的
        """
        # 1. 本地文件 -> 上传
        if not (source_uri.startswith('s3://') or source_uri.startswith('gs://') or source_uri.startswith('oss://') or source_uri.startswith('https://')):
            return self.upload_local_to_cloud(source_uri, target_provider, target_bucket, target_path, target_region)

        parsed = self._parse_uri(source_uri)
        scheme = parsed[0]
        source_bucket = parsed[1]

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
        elif target_provider == 'aliyun' and scheme == 'oss':
            # OSS -> OSS 跨 region 传输
            if target_region is None:
                raise ValueError("target_region is required when transferring to Aliyun OSS")
            return self.transfer_oss_to_oss(source_uri, target_bucket, target_region, target_path)
        elif target_provider == 'google' and scheme == 's3':
            # S3 -> GCS 跨云传输
            return self.transfer_s3_to_gcs(source_uri, target_bucket, target_path)
        elif target_provider == 'amazon' and scheme == 'gs':
            # GCS -> S3 跨云传输
            return self.transfer_gcs_to_s3(source_uri, target_bucket, target_path)
        elif target_provider == 'aliyun' and scheme == 's3':
            # S3 -> OSS 跨云传输
            if target_region is None:
                raise ValueError("target_region is required when transferring to Aliyun OSS")
            return self.transfer_s3_to_oss(source_uri, target_bucket, target_region, target_path)
        elif target_provider == 'amazon' and scheme == 'oss':
            # OSS -> S3 跨云传输
            return self.transfer_oss_to_s3(source_uri, target_bucket, target_path)
        elif target_provider == 'google' and scheme == 'oss':
            # OSS -> GCS 跨云传输
            return self.transfer_oss_to_gcs(source_uri, target_bucket, target_path)
        elif target_provider == 'aliyun' and scheme == 'gs':
            # GCS -> OSS 跨云传输
            if target_region is None:
                raise ValueError("target_region is required when transferring to Aliyun OSS")
            return self.transfer_gcs_to_oss(source_uri, target_bucket, target_region, target_path)
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
