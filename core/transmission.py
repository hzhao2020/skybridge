import os
import boto3
from urllib.parse import urlparse
from google.cloud import storage
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataTransmission")


class DataTransmission:
    """
    数据传输中心：负责本地上传以及跨云搬运 (The Bridge)
    """

    def __init__(self, aws_region='us-west-2'):
        self.aws_region = aws_region

        # 初始化客户端 (Lazy loading 也可以，这里为了清晰直接初始化)
        try:
            self.s3_client = boto3.client('s3', region_name=aws_region)
            self.gcs_client = storage.Client()
        except Exception as e:
            logger.warning(f"Clients init failed (check credentials): {e}")

    def _parse_uri(self, uri: str):
        """解析 URI，返回 (scheme, bucket, key/blob_name)"""
        parsed = urlparse(uri)
        scheme = parsed.scheme  # 's3' or 'gs'
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return scheme, bucket, key

    def upload_local_to_cloud(self, local_path: str, provider: str, target_bucket: str) -> str:
        """
        将本地文件上传到指定云
        :param provider: 'google' or 'amazon'
        """
        filename = os.path.basename(local_path)

        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if provider == 'google':
            # Upload to GCS
            bucket = self.gcs_client.bucket(target_bucket)
            blob = bucket.blob(filename)  # 默认存根目录，也可加前缀
            logger.info(f"Uploading {local_path} to GCS bucket {target_bucket}...")
            blob.upload_from_filename(local_path)
            return f"gs://{target_bucket}/{filename}"

        elif provider == 'amazon':
            # Upload to S3
            logger.info(f"Uploading {local_path} to S3 bucket {target_bucket}...")
            self.s3_client.upload_file(local_path, target_bucket, filename)
            return f"s3://{target_bucket}/{filename}"

        else:
            raise ValueError(f"Unknown provider for upload: {provider}")

    def transfer_s3_to_gcs(self, s3_uri: str, target_gcs_bucket: str) -> str:
        """
        AWS S3 -> Google GCS
        逻辑：S3 -> 本地临时文件 -> GCS -> 删除临时文件
        """
        scheme, s3_bucket, s3_key = self._parse_uri(s3_uri)
        if scheme != 's3':
            raise ValueError("Source URI must start with s3://")

        filename = os.path.basename(s3_key)
        local_tmp = f"tmp_{filename}"  # 避免重名冲突

        try:
            # 1. 下载
            logger.info(f"[Bridge] Downloading from S3: {s3_uri}")
            self.s3_client.download_file(s3_bucket, s3_key, local_tmp)

            # 2. 上传
            logger.info(f"[Bridge] Uploading to GCS: {target_gcs_bucket}")
            bucket = self.gcs_client.bucket(target_gcs_bucket)
            blob_name = f"transfer_cache/{filename}"  # 建议放在专用目录
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_tmp)

            final_uri = f"gs://{target_gcs_bucket}/{blob_name}"
            return final_uri

        finally:
            # 3. 清理
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def transfer_gcs_to_s3(self, gcs_uri: str, target_s3_bucket: str) -> str:
        """
        Google GCS -> AWS S3
        逻辑：GCS -> 本地临时文件 -> S3 -> 删除临时文件
        """
        scheme, gcs_bucket_name, gcs_blob_name = self._parse_uri(gcs_uri)
        if scheme != 'gs':
            raise ValueError("Source URI must start with gs://")

        filename = os.path.basename(gcs_blob_name)
        local_tmp = f"tmp_{filename}"

        try:
            # 1. 下载
            logger.info(f"[Bridge] Downloading from GCS: {gcs_uri}")
            bucket = self.gcs_client.bucket(gcs_bucket_name)
            blob = bucket.blob(gcs_blob_name)
            blob.download_to_filename(local_tmp)

            # 2. 上传
            logger.info(f"[Bridge] Uploading to S3: {target_s3_bucket}")
            s3_key = f"transfer_cache/{filename}"
            self.s3_client.upload_file(local_tmp, target_s3_bucket, s3_key)

            final_uri = f"s3://{target_s3_bucket}/{s3_key}"
            return final_uri

        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def smart_move(self, source_uri: str, target_provider: str, target_bucket: str) -> str:
        """
        智能搬运：根据源地址和目标，自动决定怎么传
        这是外部调用的主要入口。
        """
        # 情况 1: 本地文件上传
        if not (source_uri.startswith('s3://') or source_uri.startswith('gs://')):
            return self.upload_local_to_cloud(source_uri, target_provider, target_bucket)

        scheme, _, _ = self._parse_uri(source_uri)

        # 情况 2: 云间传输
        if target_provider == 'google':
            if scheme == 'gs': return source_uri  # 已经在 Google 了，不用动
            if scheme == 's3': return self.transfer_s3_to_gcs(source_uri, target_bucket)

        elif target_provider == 'amazon':
            if scheme == 's3': return source_uri  # 已经在 AWS 了，不用动
            if scheme == 'gs': return self.transfer_gcs_to_s3(source_uri, target_bucket)

        return source_uri