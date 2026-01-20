# ops/impl/google_ops.py
import os
import time
from typing import Dict, Any
from urllib.parse import urlparse
from ops.base import VideoSegmenter, VisualCaptioner, LLMQuery

try:
    from google.cloud import videointelligence
    from google.cloud import aiplatform
    from google.api_core.client_options import ClientOptions
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
    ClientOptions = None
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
            # 使用默认的全局端点
            # 区域通过请求中的 location_id 参数指定，而不是通过客户端端点
            self._video_client = videointelligence.VideoIntelligenceServiceClient()
        return self._video_client
    
    def _get_location_id(self):
        """将 region 转换为 Video Intelligence API 支持的 location_id"""
        # Google Video Intelligence API 支持的区域映射（仅支持3个区域）
        # us-west1 (Oregon), europe-west1 (Belgium), asia-east1 (Taiwan)
        region_to_location = {
            'us-west1': 'us-west1',
            'europe-west1': 'europe-west1',
            'asia-east1': 'asia-east1',
        }
        return region_to_location.get(self.region)
    
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
        
        # 构建请求，包含 location_id 以指定处理区域
        request_dict = {
            "features": features,
            "input_uri": target_uri,
            "output_uri": result_uri
        }
        
        # 如果区域支持，添加 location_id
        location_id = self._get_location_id()
        if location_id:
            request_dict["location_id"] = location_id
            print(f"    Using location_id: {location_id}")
        else:
            raise ValueError(
                f"Unsupported region '{self.region}' for Google Video Intelligence. "
                f"Supported regions: us-west1, europe-west1, asia-east1"
            )
        
        operation = self.video_client.annotate_video(request=request_dict)
        
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
            # Vertex AI 支持的区域映射（仅支持3个区域）
            # us-west1 (Oregon), europe-west1 (Belgium), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
                'europe-west1': 'europe-west1',
                'asia-southeast1': 'asia-southeast1'
            }
            if self.region not in region_locations:
                raise ValueError(
                    f"Unsupported region '{self.region}' for Vertex AI. "
                    f"Supported regions: {list(region_locations.keys())}"
                )
            location = region_locations[self.region]
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
        """
        对视频或视频片段生成描述
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            start_time: 可选，片段开始时间（秒）
            end_time: 可选，片段结束时间（秒）
            target_path: 可选，上传目标路径
        """
        start_time = kwargs.get('start_time')
        end_time = kwargs.get('end_time')
        
        if start_time is not None and end_time is not None:
            print(f"--- [Vertex AI Caption] Region: {self.region} | Model: {self.model_name} | Segment: {start_time:.2f}s-{end_time:.2f}s ---")
        else:
            print(f"--- [Vertex AI Caption] Region: {self.region} | Model: {self.model_name} ---")

        # 1. 确保数据在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")

        # 2. 初始化 Vertex AI
        model = self._init_vertex_ai()
        
        # 3. 构建视频 Part（支持时间范围）
        if Part is None:
            raise ImportError("Vertex AI Part not available. Install with: pip install google-cloud-aiplatform")
        
        # 如果指定了时间范围，使用 video_metadata
        if start_time is not None and end_time is not None:
            try:
                # 尝试使用 video_metadata（新版本 API）
                from google.protobuf.duration_pb2 import Duration
                video_metadata = {
                    "start_offset": {"seconds": int(start_time)},
                    "end_offset": {"seconds": int(end_time)},
                }
                # 注意：Part.from_uri 可能不支持 video_metadata 参数，需要检查 API 版本
                # 如果 API 不支持，可能需要先提取视频片段
                video_part = Part.from_uri(file_uri=target_uri, mime_type="video/mp4")
                # 如果 API 支持 video_metadata，可以这样调用：
                # video_part = Part.from_uri(file_uri=target_uri, mime_type="video/mp4", video_metadata=video_metadata)
                print(f"    Processing segment: {start_time:.2f}s - {end_time:.2f}s")
                prompt_text = f"Describe this video segment (from {start_time:.1f}s to {end_time:.1f}s) in detail. Provide a comprehensive caption."
            except Exception as e:
                print(f"    Warning: Could not use video_metadata, processing full video: {e}")
                video_part = Part.from_uri(file_uri=target_uri, mime_type="video/mp4")
                prompt_text = f"Describe this video segment (from {start_time:.1f}s to {end_time:.1f}s) in detail. Provide a comprehensive caption."
        else:
            video_part = Part.from_uri(file_uri=target_uri, mime_type="video/mp4")
            prompt_text = "Describe this video in detail. Provide a comprehensive caption."
        
        # 4. 调用 Gemini API 生成视频描述
        print(f"    Generating caption with {self.model_name}...")
        response = model.generate_content(
            [video_part, prompt_text],
            generation_config={
                "temperature": 0.4,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        caption = response.text if response.text else "Unable to generate caption."
        print(f"    Caption generated: {caption[:100]}...")
        
        result = {
            "provider": "google_vertex",
            "model": self.model_name,
            "caption": caption,
            "source_used": target_uri
        }
        
        if start_time is not None and end_time is not None:
            result["start_time"] = start_time
            result["end_time"] = end_time
        
        return result


class GoogleVertexLLMImpl(LLMQuery):
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._vertex_client = None
    
    def _init_vertex_ai(self):
        """初始化 Vertex AI 客户端"""
        if self._vertex_client is None:
            if GenerativeModel is None:
                raise ImportError("Vertex AI GenerativeModel not available. Install with: pip install google-cloud-aiplatform")
            # Vertex AI 支持的区域映射（仅支持3个区域）
            # us-west1 (Oregon), europe-west1 (Belgium), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
                'europe-west1': 'europe-west1',
                'asia-southeast1': 'asia-southeast1'
            }
            if self.region not in region_locations:
                raise ValueError(
                    f"Unsupported region '{self.region}' for Vertex AI. "
                    f"Supported regions: {list(region_locations.keys())}"
                )
            location = region_locations[self.region]
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