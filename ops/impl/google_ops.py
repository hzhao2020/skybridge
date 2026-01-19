# ops/impl/google_ops.py
import os
import time
from typing import Dict, Any
from urllib.parse import urlparse
from ops.base import VideoSegmenter, VisualCaptioner, LLMQuery

try:
    from google.cloud import videointelligence
    from google.cloud import aiplatform
    try:
        from vertexai.generative_models import GenerativeModel, Part
    except ImportError:
        # 尝试旧版本的导入方式
        try:
            from vertexai.preview.generative_models import GenerativeModel, Part
        except ImportError:
            GenerativeModel = None
            Part = None
except ImportError:
    print("Warning: Google Cloud libraries not installed. Install with: pip install google-cloud-videointelligence google-cloud-aiplatform")
    GenerativeModel = None
    Part = None


class GoogleVideoSegmentImpl(VideoSegmenter):
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        # 初始化 Video Intelligence 客户端
        self._video_client = None
    
    @property
    def video_client(self):
        """懒加载 Google Video Intelligence 客户端"""
        if self._video_client is None:
            # 根据 region 设置 API endpoint
            region_endpoints = {
                'us-west1': 'us-west1-videointelligence.googleapis.com',
                'europe-west1': 'europe-west1-videointelligence.googleapis.com',
                'asia-east1': 'asia-east1-videointelligence.googleapis.com',
                'asia-southeast1': 'asia-southeast1-videointelligence.googleapis.com'
            }
            endpoint = region_endpoints.get(self.region)
            if endpoint:
                options = {"api_endpoint": endpoint}
                self._video_client = videointelligence.VideoIntelligenceServiceClient(client_options=options)
            else:
                self._video_client = videointelligence.VideoIntelligenceServiceClient()
        return self._video_client
    
    def _parse_uri(self, uri: str):
        """解析 URI，返回 bucket 和 blob/key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        return bucket, blob_path
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [Google Video Intelligence] Region: {self.region} ---")

        # 1. 确保数据在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")

        # 2. 解析 URI
        bucket, blob_path = self._parse_uri(target_uri)
        
        # 3. 准备结果存储路径
        filename = os.path.basename(blob_path)
        result_path = f"results/{filename}.json"
        result_uri = f"gs://{self.storage_bucket}/{result_path}"
        
        print(f"    Processing video: {target_uri}")
        print(f"    Results will be saved to: {result_uri}")

        # 4. 调用 Video Intelligence API
        features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
        operation = self.video_client.annotate_video(
            request={
                "features": features,
                "input_uri": target_uri,
                "output_uri": result_uri
            }
        )
        
        # 5. 等待结果（阻塞等待，实际生产环境可异步）
        print("    Waiting for video analysis to complete...")
        result = operation.result(timeout=600)
        
        # 6. 解析结果，提取 segments
        segments = []
        if result.annotation_results:
            for annotation_result in result.annotation_results:
                if annotation_result.shot_annotations:
                    for shot in annotation_result.shot_annotations:
                        segments.append({
                            "start": shot.start_time_offset.total_seconds() if shot.start_time_offset else 0.0,
                            "end": shot.end_time_offset.total_seconds() if shot.end_time_offset else 0.0
                        })
        
        print(f"    Found {len(segments)} segments")
        
        return {
            "provider": "google",
            "region": self.region,
            "segments": segments,
            "source_used": target_uri,
            "result_location": result_uri
        }


class GoogleVertexCaptionImpl(VisualCaptioner):
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        # 初始化 Vertex AI
        self._vertex_client = None
    
    def _init_vertex_ai(self):
        """初始化 Vertex AI 客户端"""
        if self._vertex_client is None:
            # 根据 region 设置 location
            region_locations = {
                'us-west1': 'us-west1',
                'europe-west1': 'europe-west1',
                'asia-east1': 'asia-east1',
                'asia-southeast1': 'asia-southeast1'
            }
            location = region_locations.get(self.region, 'us-central1')
            aiplatform.init(location=location)
            self._vertex_client = GenerativeModel(self.model_name)
        return self._vertex_client
    
    def _parse_uri(self, uri: str):
        """解析 URI，返回 bucket 和 blob/key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        return bucket, blob_path
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [Vertex AI Caption] Region: {self.region} | Model: {self.model_name} ---")

        # 1. 确保数据在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")

        # 2. 初始化 Vertex AI
        model = self._init_vertex_ai()
        
        # 3. 构建视频 Part
        if Part is None:
            raise ImportError("Vertex AI Part not available. Install with: pip install google-cloud-aiplatform")
        video_part = Part.from_uri(file_uri=target_uri, mime_type="video/mp4")
        
        # 4. 调用 Gemini API 生成视频描述
        print(f"    Generating caption with {self.model_name}...")
        response = model.generate_content(
            [video_part, "Describe this video in detail. Provide a comprehensive caption."],
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        caption = response.text if response.text else "Unable to generate caption."
        print(f"    Caption generated: {caption[:100]}...")
        
        return {
            "provider": "google_vertex",
            "model": self.model_name,
            "caption": caption,
            "source_used": target_uri
        }


class GoogleVertexLLMImpl(LLMQuery):
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._vertex_client = None
    
    def _init_vertex_ai(self):
        """初始化 Vertex AI 客户端"""
        if self._vertex_client is None:
            if GenerativeModel is None:
                raise ImportError("Vertex AI GenerativeModel not available. Install with: pip install google-cloud-aiplatform")
            # 根据 region 设置 location
            region_locations = {
                'us-west1': 'us-west1',
                'europe-west1': 'europe-west1',
                'asia-east1': 'asia-east1',
                'asia-southeast1': 'asia-southeast1'
            }
            location = region_locations.get(self.region, 'us-central1')
            aiplatform.init(location=location)
            self._vertex_client = GenerativeModel(self.model_name)
        return self._vertex_client
    
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # LLM 文本查询通常不需要搬运文件，直接调用
        print(f"--- [Vertex AI LLM] Region: {self.region} | Model: {self.model_name} ---")
        
        # 初始化 Vertex AI
        model = self._init_vertex_ai()
        
        # 调用 Gemini API
        print(f"    Sending prompt to {self.model_name}...")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        answer = response.text if response.text else "Unable to generate response."
        print(f"    Response received: {answer[:100]}...")
        
        return {
            "provider": "google_vertex",
            "model": self.model_name,
            "response": answer
        }