import io
import os
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

# --- 依赖库 ---
try:
    import boto3
    from google.cloud import videointelligence, storage
except ImportError:
    print("请安装: pip install boto3 google-cloud-videointelligence google-cloud-storage")


class Operation:
    pass


class SegmenterBackend(ABC):
    """抽象策略基类：定义通用的搬运和分析接口"""

    def __init__(self, region: str, storage_bucket: str):
        self.region = region
        self.storage_bucket = storage_bucket  # "本地" Bucket，用于存放搬运的视频和结果

    @abstractmethod
    def process_segmentation(self, video_uri: str) -> Dict[str, Any]:
        pass

    def _get_filename_from_uri(self, uri: str) -> str:
        """从 URI 中提取文件名, e.g., s3://bucket/vid.mp4 -> vid.mp4"""
        path = urlparse(uri).path
        return os.path.basename(path)


class GoogleVideoBackend(SegmenterBackend):
    """
    Google 后端
    1. 如果视频在 S3，搬运到 GCS。
    2. 分析结果直接存入 GCS。
    """

    VALID_REGIONS = {
        'us-west1': 'us-west1-videointelligence.googleapis.com',
        'europe-west1': 'europe-west1-videointelligence.googleapis.com',
        'asia-east1': 'asia-east1-videointelligence.googleapis.com'
    }

    def __init__(self, region: str, storage_bucket: str):
        super().__init__(region, storage_bucket)
        if region not in self.VALID_REGIONS:
            raise ValueError(f"Google Region error. Choose: {list(self.VALID_REGIONS.keys())}")

        options = {"api_endpoint": self.VALID_REGIONS[region]}
        self.client = videointelligence.VideoIntelligenceServiceClient(client_options=options)
        self.gcs_client = storage.Client()

    def _transfer_s3_to_gcs(self, s3_uri: str) -> str:
        """S3 -> GCS 流式直传（不落盘），写入本项目的 GCS Bucket video_cache/"""
        print(f"[Transfer] S3 -> GCS 流式直传: {s3_uri} -> gs://{self.storage_bucket}/video_cache/...")

        filename = self._get_filename_from_uri(s3_uri)
        parsed = urlparse(s3_uri)
        s3_bucket, s3_key = parsed.netloc, parsed.path.lstrip('/')
        s3 = boto3.client('s3')
        resp = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        body, size = resp["Body"], resp.get("ContentLength")

        class _S3StreamAdapter(io.RawIOBase):
            def __init__(self, b):
                self._b = b
            def read(self, amt=-1):
                return self._b.read(amt) if amt != -1 else self._b.read()
            def readable(self):
                return True
            def seekable(self):
                return False

        bucket = self.gcs_client.bucket(self.storage_bucket)
        blob = bucket.blob(f"video_cache/{filename}")
        blob.upload_from_file(_S3StreamAdapter(body), rewind=False, size=size)
        new_uri = f"gs://{self.storage_bucket}/video_cache/{filename}"
        print(f"[Transfer] Done. New URI: {new_uri}")
        return new_uri

    def process_segmentation(self, video_uri: str) -> Dict[str, Any]:
        # Step 1: 检查并搬运视频
        target_video_uri = video_uri
        if video_uri.startswith("s3://"):
            target_video_uri = self._transfer_s3_to_gcs(video_uri)
        elif not video_uri.startswith("gs://"):
            raise ValueError("Google backend requires gs:// or s3:// URI")

        # Step 2: 准备结果存储路径 (跟随节点存储)
        filename = self._get_filename_from_uri(target_video_uri)
        result_storage_uri = f"gs://{self.storage_bucket}/results/{filename}.json"

        print(f"[Google {self.region}] Processing {target_video_uri}...")
        print(f"[Google] Results will be saved to {result_storage_uri}")

        # Step 3: 调用 API (使用 output_uri 让 Google 自动存储结果)
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
        operation = self.client.annotate_video(
            request={
                "features": features,
                "input_uri": target_video_uri,
                "output_uri": result_storage_uri  # <--- 关键：结果自动存云
            }
        )

        # 这里为了演示阻塞等待，实际生产中可异步
        result = operation.result(timeout=600)

        return {
            "status": "success",
            "video_location": target_video_uri,
            "result_location": result_storage_uri,
            "provider": "google"
        }


class AmazonVideoBackend(SegmenterBackend):
    """
    AWS 后端
    1. 如果视频在 GCS，搬运到 S3。
    2. 分析结果手动存入 S3 (因为 Rekognition 不支持直接存 S3)。
    """

    VALID_REGIONS = ['us-west-2', 'eu-central-1', 'ap-southeast-1']

    def __init__(self, region: str, storage_bucket: str):
        super().__init__(region, storage_bucket)
        if region not in self.VALID_REGIONS:
            raise ValueError(f"AWS Region error. Choose: {self.VALID_REGIONS}")

        self.rekognition = boto3.client('rekognition', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.gcs_client = storage.Client()  # 需要 GCS 客户端来下载

    def _transfer_gcs_to_s3(self, gcs_uri: str) -> str:
        """GCS -> S3 流式直传（不落盘），写入本项目的 S3 Bucket video_cache/"""
        print(f"[Transfer] GCS -> S3 流式直传: {gcs_uri} -> s3://{self.storage_bucket}/video_cache/...")

        filename = self._get_filename_from_uri(gcs_uri)
        parsed = urlparse(gcs_uri)
        gcs_bucket_name, gcs_blob_name = parsed.netloc, parsed.path.lstrip('/')
        bucket = self.gcs_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_blob_name)
        s3_key = f"video_cache/{filename}"
        with blob.open("rb") as stream:
            self.s3.upload_fileobj(stream, self.storage_bucket, s3_key)
        print(f"[Transfer] Done. New URI: s3://{self.storage_bucket}/{s3_key}")
        return s3_key  # 返回 Key 给 boto3 使用

    def process_segmentation(self, video_uri: str) -> Dict[str, Any]:
        # Step 1: 检查并搬运视频
        # AWS API 需要 Bucket 和 Name 分开，所以我们需要处理一下
        target_bucket = self.storage_bucket
        target_key = ""

        if video_uri.startswith("gs://"):
            target_key = self._transfer_gcs_to_s3(video_uri)
        elif video_uri.startswith("s3://"):
            # 如果已经在 S3，解析一下是否在当前 Bucket，如果不是可能也需要 copy，这里简化为直接使用
            parsed = urlparse(video_uri)
            target_bucket = parsed.netloc
            target_key = parsed.path.lstrip('/')
        else:
            raise ValueError("AWS backend requires s3:// or gs:// URI")

        print(f"[AWS {self.region}] Processing s3://{target_bucket}/{target_key}...")

        # Step 2: 启动任务
        response = self.rekognition.start_segment_detection(
            Video={'S3Object': {'Bucket': target_bucket, 'Name': target_key}},
            SegmentTypes=['SHOT']
        )
        job_id = response['JobId']

        # Step 3: 轮询等待结果 (AWS 特性)并手动存储
        print(f"[AWS] Job {job_id} started, waiting for completion...")
        result_json = self._poll_and_get_results(job_id)

        # Step 4: 将结果手动存入 S3 (实现“存储跟随计算”)
        result_filename = f"results/{os.path.basename(target_key)}.json"
        self.s3.put_object(
            Bucket=self.storage_bucket,
            Key=result_filename,
            Body=json.dumps(result_json)
        )

        result_uri = f"s3://{self.storage_bucket}/{result_filename}"
        print(f"[AWS] Results saved to {result_uri}")

        return {
            "status": "success",
            "video_location": f"s3://{target_bucket}/{target_key}",
            "result_location": result_uri,
            "provider": "amazon"
        }

    def _poll_and_get_results(self, job_id):
        """简单的轮询逻辑"""
        while True:
            response = self.rekognition.get_segment_detection(JobId=job_id)
            status = response['JobStatus']
            if status == 'SUCCEEDED':
                return response
            elif status == 'FAILED':
                raise RuntimeError(f"AWS Job failed: {response}")
            time.sleep(5)


class VideoSegment(Operation):
    PROVIDER_GOOGLE = 'google'
    PROVIDER_AMAZON = 'amazon'

    def __init__(self, provider: str, region: str, storage_bucket: str):
        """
        :param provider: google / amazon
        :param region: api region
        :param storage_bucket: 【重要】该 provider 下用于存放数据和结果的 Bucket 名称
        """
        self.provider = provider.lower()
        self.backend: Optional[SegmenterBackend] = None

        if self.provider == self.PROVIDER_GOOGLE:
            self.backend = GoogleVideoBackend(region, storage_bucket)
        elif self.provider == self.PROVIDER_AMAZON:
            self.backend = AmazonVideoBackend(region, storage_bucket)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def segment(self, video: str):
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        return self.backend.process_segmentation(video)


# --- 使用示例 ---

if __name__ == "__main__":
    # 场景 1: 在 Google 上跑，但是视频源在 AWS S3
    # 结果会自动存入 gs://my-gcp-data-bucket/results/...
    # gcp_task = VideoSegment(
    #     provider='google',
    #     region='asia-east1',
    #     storage_bucket='my-gcp-data-bucket' # 你的 GCP Bucket
    # )
    # gcp_task.segment("s3://external-client-data/video.mp4")

    # 场景 2: 在 AWS 上跑，但是视频源在 Google GCS
    # 结果会被拉取并存入 s3://my-aws-data-bucket/results/...
    # aws_task = VideoSegment(
    #     provider='amazon',
    #     region='us-west-2',
    #     storage_bucket='my-aws-data-bucket' # 你的 AWS Bucket
    # )
    # aws_task.segment("gs://external-partner-data/video.mp4")
    pass