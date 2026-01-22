# ops/impl/google_ops.py
import os
import time
import subprocess
import tempfile
import json
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from ops.base import VideoSegmenter, VisualCaptioner, LLMQuery, VideoSplitter, VisualEncoder, TextEncoder, ObjectDetector

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
    try:
        from vertexai.vision_models import MultiModalEmbeddingModel, Video
    except ImportError:
        # 尝试旧版本的导入方式
        try:
            from vertexai.preview.vision_models import MultiModalEmbeddingModel, Video
        except ImportError:
            MultiModalEmbeddingModel = None
            Video = None
    try:
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
    except ImportError:
        # 尝试旧版本的导入方式
        try:
            from vertexai.preview.language_models import TextEmbeddingModel, TextEmbeddingInput
        except ImportError:
            TextEmbeddingModel = None
            TextEmbeddingInput = None
except ImportError:
    print("Warning: Google Cloud libraries not installed. Install with: pip install google-cloud-videointelligence google-cloud-aiplatform")
    ClientOptions = None
    GenerativeModel = None
    Part = None
    MultiModalEmbeddingModel = None
    Video = None
    TextEmbeddingModel = None
    TextEmbeddingInput = None


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
    
    def _get_gcs_signed_url(self, gcs_uri: str, expiration: int = 3600) -> str:
        """
        生成 GCS 的预签名 URL，用于 ffmpeg 直接访问
        
        Args:
            gcs_uri: GCS URI (gs://bucket/path)
            expiration: URL 过期时间（秒），默认1小时
            
        Returns:
            预签名 URL
        """
        from google.cloud import storage
        from datetime import datetime, timedelta
        
        bucket, blob_path = self._parse_uri(gcs_uri)
        storage_client = storage.Client()
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(blob_path)
        
        # 生成签名 URL
        url = blob.generate_signed_url(
            expiration=timedelta(seconds=expiration),
            method='GET'
        )
        return url
    
    def _extract_video_segment(self, video_uri: str, start_time: float, end_time: float, output_path: str) -> str:
        """
        使用 ffmpeg 提取视频片段（优化版：支持直接从云端读取）
        
        优化策略：
        1. 如果视频在 GCS 中，使用 signed URL + ffmpeg HTTP 输入，避免下载整个视频
        2. 如果视频在 S3 中，使用 presigned URL + ffmpeg HTTP 输入
        3. 如果是本地文件，直接使用
        
        注意：
        - ffmpeg 通过 HTTP 读取时，仍需要下载视频的元数据/索引（通常很小，<1MB）
        - 对于 MP4 等格式，如果 moov atom 在文件末尾，可能需要读取文件末尾的数据
        - 相比下载整个视频，这种方式大大减少了数据传输量
        - 如果需要完全云端处理（无本地ffmpeg），可以考虑使用 Cloud Run Jobs
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出文件路径
            
        Returns:
            提取的片段文件路径
        """
        # 计算持续时间
        duration = end_time - start_time
        
        # 确定输入源
        input_source = None
        temp_video_path = None
        
        if video_uri.startswith('gs://'):
            # GCS: 使用 signed URL，让 ffmpeg 直接从云端读取
            try:
                signed_url = self._get_gcs_signed_url(video_uri)
                input_source = signed_url
                print(f"    Using GCS signed URL for direct streaming (no download needed)")
            except Exception as e:
                print(f"    Warning: Failed to generate signed URL, falling back to download: {e}")
                # 降级：下载到本地
                bucket, blob_path = self._parse_uri(video_uri)
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_video_path = temp_video.name
                temp_video.close()
                
                from google.cloud import storage
                storage_client = storage.Client()
                bucket_obj = storage_client.bucket(bucket)
                blob = bucket_obj.blob(blob_path)
                blob.download_to_filename(temp_video_path)
                input_source = temp_video_path
                
        elif video_uri.startswith('s3://'):
            # S3: 使用 presigned URL
            try:
                import boto3
                parsed = urlparse(video_uri)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                s3_client = boto3.client('s3')
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': key},
                    ExpiresIn=3600
                )
                input_source = presigned_url
                print(f"    Using S3 presigned URL for direct streaming (no download needed)")
            except Exception as e:
                print(f"    Warning: Failed to generate presigned URL, falling back to download: {e}")
                # 降级：下载到本地
                temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_video_path = temp_video.name
                temp_video.close()
                
                s3_client = boto3.client('s3')
                s3_client.download_file(bucket, key, temp_video_path)
                input_source = temp_video_path
        else:
            # 本地路径
            if not os.path.exists(video_uri):
                raise FileNotFoundError(f"Video file not found: {video_uri}")
            input_source = video_uri
        
        # 使用 ffmpeg 提取片段
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # 构建 ffmpeg 命令
            # 注意：对于 HTTP 输入，ffmpeg 可能需要先下载部分数据来解析视频索引
            # 但相比下载整个视频，这仍然更高效
            cmd = [
                'ffmpeg',
                '-i', input_source,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # 使用 copy 模式以加快速度（如果可能）
                '-avoid_negative_ts', 'make_zero',
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            # 执行 ffmpeg 命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("ffmpeg extraction failed: output file is empty")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg extraction failed: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        finally:
            # 清理临时下载的文件（仅在降级情况下使用）
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
    
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

        # 2. 如果指定了时间范围，先提取视频片段
        segment_uri = target_uri
        temp_segment_path = None
        
        if start_time is not None and end_time is not None:
            print(f"    Extracting video segment: {start_time:.2f}s - {end_time:.2f}s")
            # 创建临时文件用于存储提取的片段
            temp_segment = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_segment_path = temp_segment.name
            temp_segment.close()
            
            try:
                # 提取片段（如果视频在云存储，需要先下载）
                self._extract_video_segment(video_uri, start_time, end_time, temp_segment_path)
                print(f"    Segment extracted to: {temp_segment_path}")
                
                # 将提取的片段上传到云存储
                segment_filename = f"segment_{int(start_time)}_{int(end_time)}_{os.path.basename(target_uri)}"
                if target_path:
                    segment_target_path = f"{target_path}/segments"
                else:
                    segment_target_path = "segments"
                
                segment_uri = self.transmitter.upload_local_to_cloud(
                    temp_segment_path,
                    'google',
                    self.storage_bucket,
                    segment_target_path
                )
                print(f"    Segment uploaded to: {segment_uri}")
            except Exception as e:
                print(f"    Warning: Failed to extract segment, processing full video: {e}")
                segment_uri = target_uri
        
        # 3. 初始化 Vertex AI
        model = self._init_vertex_ai()
        
        # 4. 构建视频 Part
        if Part is None:
            raise ImportError("Vertex AI Part not available. Install with: pip install google-cloud-aiplatform")
        
        # Vertex AI SDK 的 Part.from_uri 参数名在不同版本里不一致：
        # - 有的版本使用 from_uri(uri, mime_type=...)
        # - 有的版本接受关键字 uri=...
        # 这里用“位置参数 + mime_type”以获得更好的兼容性，避免 TypeError（如 file_uri 不被支持）。
        video_part = Part.from_uri(segment_uri, mime_type="video/mp4")
        
        if start_time is not None and end_time is not None:
            prompt_text = f"Describe this video segment in detail. Provide a comprehensive caption."
        else:
            prompt_text = "Describe this video in detail. Provide a comprehensive caption."
        
        # 5. 调用 Gemini API 生成视频描述
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
        
        # 6. 清理临时文件
        if temp_segment_path and os.path.exists(temp_segment_path):
            try:
                os.remove(temp_segment_path)
            except Exception:
                pass  # 忽略清理错误
        
        result = {
            "provider": "google_vertex",
            "model": self.model_name,
            "caption": caption,
            "source_used": segment_uri if (start_time is not None and end_time is not None) else target_uri
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


class GoogleCloudRunSplitImpl(VideoSplitter):
    """
    使用 Google Cloud Run（HTTP 服务）进行视频分割
    
    需要先部署一个 Cloud Run 服务，该服务接收视频 URI 和片段列表，
    使用 ffmpeg 在云端切割视频，并将结果上传到 GCS。
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, service_url: Optional[str] = None):
        super().__init__(provider, region, storage_bucket)
        # Cloud Run 的 URL 没有稳定的可推导规则（与项目/服务名/区域/是否自定义域名有关），
        # 因此这里不自动拼接默认 URL，而是由调用方显式传入。
        self.service_url = service_url
    
    def _parse_uri(self, uri: str):
        """解析 URI，返回 bucket 和 blob/key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        return bucket, blob_path
    
    def execute(self, video_uri: str, segments: List[Dict[str, float]], **kwargs) -> Dict[str, Any]:
        """
        执行视频分割
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            segments: 片段列表，每个片段包含 'start' 和 'end'（秒）
            **kwargs: 
                - target_path: 输出路径
                - output_format: 输出格式（默认 mp4）
                - service_url: 可选的 Cloud Run 服务 URL（覆盖默认）
        """
        print(f"--- [Google Cloud Run Video Split] Region: {self.region} ---")
        
        # 1. 确保视频在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
        print(f"    Video Ready: {target_uri}")
        
        # 2. 准备输出路径
        output_format = kwargs.get('output_format', 'mp4')
        if target_path:
            output_base_path = f"{target_path}/split_segments"
        else:
            output_base_path = "split_segments"
        
        # 3. 构建请求体
        request_body = {
            "video_uri": target_uri,
            "segments": segments,
            "output_bucket": self.storage_bucket,
            "output_path": output_base_path,
            "output_format": output_format
        }
        
        # 4. 调用 Cloud Run 服务
        service_url = kwargs.get('service_url', self.service_url)
        if not service_url:
            raise ValueError(
                "Missing Cloud Run service_url. "
                "请在构造 GoogleCloudRunSplitImpl(service_url=...) 或 execute(..., service_url=...) 时提供。"
            )
        print(f"    Calling Cloud Run: {service_url}")
        print(f"    Processing {len(segments)} segments...")
        
        try:
            response = requests.post(
                service_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=600  # 10分钟超时
            )
            response.raise_for_status()
            result = response.json()
            
            output_uris = result.get("output_uris", [])
            print(f"    Successfully split video into {len(output_uris)} segments")
            
            return {
                "provider": "google_cloud_run",
                "region": self.region,
                "input_video": target_uri,
                "segments": segments,
                "output_uris": output_uris,
                "output_count": len(output_uris)
            }
            
        except requests.exceptions.RequestException as e:
            # 尽量把 Cloud Run 返回的错误正文带出来，便于定位（常见：GCS 权限/ffmpeg 失败/输入 URI 不存在）
            resp = getattr(e, "response", None)
            if resp is not None:
                status = getattr(resp, "status_code", "unknown")
                try:
                    body_obj = resp.json()
                    body_text = json.dumps(body_obj, ensure_ascii=False)
                except Exception:
                    body_text = (getattr(resp, "text", "") or "").strip()
                if body_text:
                    body_text = body_text[:4000]
                    raise RuntimeError(f"Cloud Run call failed ({status}): {body_text}") from e
                raise RuntimeError(f"Cloud Run call failed ({status})") from e
            raise RuntimeError(f"Cloud Run call failed: {e}") from e
        except json.JSONDecodeError as e:
            # HTTP 成功但返回体不是 JSON
            raise RuntimeError(f"Invalid response from Cloud Run (not JSON): {e}") from e


class GoogleVertexEmbeddingImpl(VisualEncoder):
    """
    使用 Google Vertex AI 的 multimodalembedding@001 模型进行视觉编码
    """
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str = "multimodalembedding@001"):
        super().__init__(provider, region, storage_bucket, model_name)
        self._embedding_model = None
    
    def _init_embedding_model(self):
        """初始化 Vertex AI Embedding 模型"""
        if self._embedding_model is None:
            if MultiModalEmbeddingModel is None:
                raise ImportError("Vertex AI MultiModalEmbeddingModel not available. Install with: pip install google-cloud-aiplatform")
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
            self._embedding_model = MultiModalEmbeddingModel.from_pretrained(self.model_name)
        return self._embedding_model
    
    def _parse_uri(self, uri: str):
        """解析 URI，返回 bucket 和 blob/key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        return bucket, blob_path
    
    def _download_video_to_local(self, video_uri: str) -> str:
        """下载视频到本地临时文件"""
        if video_uri.startswith('gs://'):
            # GCS: 下载到本地
            bucket, blob_path = self._parse_uri(video_uri)
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            
            from google.cloud import storage
            storage_client = storage.Client()
            bucket_obj = storage_client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            blob.download_to_filename(temp_video_path)
            return temp_video_path
        elif video_uri.startswith('s3://'):
            # S3: 下载到本地
            import boto3
            parsed = urlparse(video_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket, key, temp_video_path)
            return temp_video_path
        else:
            # 本地路径
            if not os.path.exists(video_uri):
                raise FileNotFoundError(f"Video file not found: {video_uri}")
            return video_uri
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        对视频进行编码，生成向量
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            target_path: 可选，上传目标路径
            save_embedding: 可选，是否保存 embedding 到云存储（默认 False）
        """
        print(f"--- [Vertex AI Embedding] Region: {self.region} | Model: {self.model_name} ---")
        
        # 1. 确保数据在 Google Bucket（如果需要）
        target_path = kwargs.get('target_path')
        if target_path or not (video_uri.startswith('gs://') or os.path.exists(video_uri)):
            target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
            print(f"    Data Ready: {target_uri}")
        else:
            target_uri = video_uri
            print(f"    Using video: {target_uri}")
        
        # 2. 初始化 Embedding 模型
        model = self._init_embedding_model()
        
        # 3. 准备视频对象
        # Vertex AI 的 Video 类需要本地文件路径或 GCS URI
        temp_video_path = None
        try:
            if Video is None:
                raise ImportError("Vertex AI Video not available. Install with: pip install google-cloud-aiplatform")
            
            if target_uri.startswith('gs://'):
                # 直接使用 GCS URI
                video_obj = Video.load_from_uri(target_uri)
            else:
                # 如果是本地路径或 S3，需要先下载到本地
                if target_uri.startswith('s3://') or not os.path.exists(target_uri):
                    temp_video_path = self._download_video_to_local(target_uri)
                    video_obj = Video.load_from_file(temp_video_path)
                else:
                    video_obj = Video.load_from_file(target_uri)
            
            print(f"    Generating embedding for video...")
            
            # 4. 调用 API 生成 embedding
            embeddings = model.get_embeddings(video=video_obj)
            
            # 5. 提取 embedding 向量
            # multimodalembedding@001 返回 video_embeddings 是一个列表（每个帧一个 embedding）
            # 我们取第一个或平均所有帧的 embedding
            if embeddings.video_embeddings:
                if len(embeddings.video_embeddings) == 1:
                    embedding_vector = embeddings.video_embeddings[0]
                else:
                    # 如果有多个帧的 embedding，取平均（使用纯 Python 实现，避免 numpy 依赖）
                    num_frames = len(embeddings.video_embeddings)
                    num_dims = len(embeddings.video_embeddings[0])
                    embedding_vector = [
                        sum(frame_emb[i] for frame_emb in embeddings.video_embeddings) / num_frames
                        for i in range(num_dims)
                    ]
                embedding_dim = len(embedding_vector)
                print(f"    Embedding generated: dimension={embedding_dim}")
            else:
                raise RuntimeError("No video embeddings returned from model")
            
            # 6. 可选：保存 embedding 到云存储
            save_embedding = kwargs.get('save_embedding', False)
            embedding_uri = None
            if save_embedding:
                embedding_data = {
                    "embedding": embedding_vector,
                    "dimension": embedding_dim,
                    "source_video": target_uri,
                    "model": self.model_name,
                    "region": self.region
                }
                embedding_filename = f"{os.path.basename(target_uri)}.embedding.json"
                if target_path:
                    embedding_target_path = f"{target_path}/embeddings"
                else:
                    embedding_target_path = "embeddings"
                
                # 保存到临时文件然后上传
                temp_embedding_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(embedding_data, temp_embedding_file, indent=2)
                temp_embedding_file.close()
                
                try:
                    embedding_uri = self.transmitter.upload_local_to_cloud(
                        temp_embedding_file.name,
                        'google',
                        self.storage_bucket,
                        embedding_target_path
                    )
                    print(f"    Embedding saved to: {embedding_uri}")
                finally:
                    if os.path.exists(temp_embedding_file.name):
                        os.remove(temp_embedding_file.name)
            
            result = {
                "provider": "google_vertex",
                "model": self.model_name,
                "region": self.region,
                "embedding": embedding_vector,
                "embedding_dimension": embedding_dim,
                "source_video": target_uri
            }
            
            if embedding_uri:
                result["embedding_uri"] = embedding_uri
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e
        finally:
            # 清理临时文件
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass  # 忽略清理错误


class GoogleVertexTextEmbeddingImpl(TextEncoder):
    """
    使用 Google Vertex AI 的文本 embedding 模型进行文本编码
    
    支持的模型：
    - gemini-embedding-001: 3072 维向量
    - text-embedding-005: 更新的文本 embedding 模型
    """
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._embedding_model = None
    
    def _init_embedding_model(self):
        """初始化 Vertex AI Text Embedding 模型"""
        if self._embedding_model is None:
            if TextEmbeddingModel is None:
                raise ImportError("Vertex AI TextEmbeddingModel not available. Install with: pip install google-cloud-aiplatform")
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
            self._embedding_model = TextEmbeddingModel.from_pretrained(self.model_name)
        return self._embedding_model
    
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        对文本进行编码，生成向量
        
        Args:
            text: 要编码的文本
            save_embedding: 可选，是否保存 embedding 到云存储（默认 False）
            task_type: 可选，任务类型（如 "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY" 等）
            output_dimensionality: 可选，输出维度（仅 gemini-embedding-001 支持，默认 3072）
        """
        print(f"--- [Vertex AI Text Embedding] Region: {self.region} | Model: {self.model_name} ---")
        
        # 1. 初始化 Embedding 模型
        model = self._init_embedding_model()
        
        # 2. 准备参数
        task_type = kwargs.get('task_type', 'RETRIEVAL_DOCUMENT')
        output_dimensionality = kwargs.get('output_dimensionality')
        
        # 3. 构建输入
        if TextEmbeddingInput is None:
            raise ImportError("Vertex AI TextEmbeddingInput not available. Install with: pip install google-cloud-aiplatform")
        
        # 对于 gemini-embedding-001，需要使用 TextEmbeddingInput
        # 对于 text-embedding-005，可能直接使用文本
        try:
            if self.model_name == "gemini-embedding-001":
                text_input = TextEmbeddingInput(text, task_type)
                # 调用 API
                print(f"    Generating embedding for text (length: {len(text)} chars)...")
                if output_dimensionality:
                    embeddings = model.get_embeddings([text_input], output_dimensionality=output_dimensionality)
                else:
                    embeddings = model.get_embeddings([text_input])
            else:
                # text-embedding-005 或其他模型
                print(f"    Generating embedding for text (length: {len(text)} chars)...")
                embeddings = model.get_embeddings([text])
            
            # 4. 提取 embedding 向量
            if embeddings and len(embeddings) > 0:
                # 获取 embedding 值
                if hasattr(embeddings[0], 'values'):
                    embedding_vector = embeddings[0].values
                elif isinstance(embeddings[0], list):
                    embedding_vector = embeddings[0]
                else:
                    embedding_vector = list(embeddings[0])
                embedding_dim = len(embedding_vector)
                print(f"    Embedding generated: dimension={embedding_dim}")
            else:
                raise RuntimeError("No embedding returned from model")
            
            # 5. 可选：保存 embedding 到云存储
            save_embedding = kwargs.get('save_embedding', False)
            embedding_uri = None
            if save_embedding:
                embedding_data = {
                    "embedding": embedding_vector,
                    "dimension": embedding_dim,
                    "source_text": text[:100] + "..." if len(text) > 100 else text,  # 只保存前100个字符
                    "text_length": len(text),
                    "model": self.model_name,
                    "region": self.region,
                    "task_type": task_type
                }
                if output_dimensionality:
                    embedding_data["output_dimensionality"] = output_dimensionality
                
                # 保存到临时文件然后上传
                temp_embedding_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(embedding_data, temp_embedding_file, indent=2)
                temp_embedding_file.close()
                
                try:
                    embedding_uri = self.transmitter.upload_local_to_cloud(
                        temp_embedding_file.name,
                        'google',
                        self.storage_bucket,
                        "embeddings/text"
                    )
                    print(f"    Embedding saved to: {embedding_uri}")
                finally:
                    if os.path.exists(temp_embedding_file.name):
                        os.remove(temp_embedding_file.name)
            
            result = {
                "provider": "google_vertex",
                "model": self.model_name,
                "region": self.region,
                "embedding": embedding_vector,
                "embedding_dimension": embedding_dim,
                "text_length": len(text)
            }
            
            if embedding_uri:
                result["embedding_uri"] = embedding_uri
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embedding: {e}") from e


class GoogleVideoIntelligenceObjectDetectionImpl(ObjectDetector):
    """
    使用 Google Video Intelligence API 进行物体检测和跟踪
    """
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        self._video_client = None
    
    @property
    def video_client(self):
        """懒加载 Google Video Intelligence 客户端"""
        if self._video_client is None:
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
        """
        在视频中检测和跟踪物体
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            target_path: 可选，上传目标路径
            save_results: 可选，是否保存结果到云存储（默认 True）
        """
        print(f"--- [Google Video Intelligence Object Detection] Region: {self.region} ---")
        
        # 1. 确保数据在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")
        
        # 2. 解析 URI
        bucket, blob_path = self._parse_uri(target_uri)
        
        # 3. 准备结果存储路径
        filename = os.path.basename(blob_path)
        result_path = f"results/object_detection/{filename}.json"
        result_uri = f"gs://{self.storage_bucket}/{result_path}"
        
        print(f"    Processing video: {target_uri}")
        print(f"    Results will be saved to: {result_uri}")
        
        # 4. 调用 Video Intelligence API - Object Tracking
        features = [videointelligence.Feature.OBJECT_TRACKING]
        
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
        print("    Waiting for object detection to complete...")
        result = operation.result(timeout=600)
        
        # 6. 解析结果，提取检测到的物体
        detected_objects = []
        if result.annotation_results:
            for annotation_result in result.annotation_results:
                if annotation_result.object_annotations:
                    for obj_annotation in annotation_result.object_annotations:
                        # 提取物体信息
                        entity = obj_annotation.entity
                        object_info = {
                            "entity_id": entity.entity_id if entity.entity_id else None,
                            "description": entity.description if entity.description else None,
                            "language_code": entity.language_code if entity.language_code else None,
                            "confidence": obj_annotation.confidence if hasattr(obj_annotation, 'confidence') else None,
                            "segments": []
                        }
                        
                        # 提取时间片段和边界框
                        for segment in obj_annotation.segments:
                            segment_info = {
                                "start_time": segment.start_time_offset.total_seconds() if segment.start_time_offset else 0.0,
                                "end_time": segment.end_time_offset.total_seconds() if segment.end_time_offset else 0.0,
                            }
                            
                            # 提取边界框信息
                            if hasattr(segment, 'frames') and segment.frames:
                                frames = []
                                for frame in segment.frames:
                                    frame_info = {
                                        "time_offset": frame.time_offset.total_seconds() if frame.time_offset else 0.0,
                                    }
                                    if hasattr(frame, 'normalized_bounding_box') and frame.normalized_bounding_box:
                                        bbox = frame.normalized_bounding_box
                                        frame_info["bounding_box"] = {
                                            "left": bbox.left if hasattr(bbox, 'left') else None,
                                            "top": bbox.top if hasattr(bbox, 'top') else None,
                                            "right": bbox.right if hasattr(bbox, 'right') else None,
                                            "bottom": bbox.bottom if hasattr(bbox, 'bottom') else None,
                                        }
                                    frames.append(frame_info)
                                segment_info["frames"] = frames
                            
                            object_info["segments"].append(segment_info)
                        
                        detected_objects.append(object_info)
        
        print(f"    Found {len(detected_objects)} unique objects")
        
        # 7. 可选：保存结果到云存储
        save_results = kwargs.get('save_results', True)
        if not save_results:
            result_uri = None
        
        return {
            "provider": "google",
            "region": self.region,
            "detected_objects": detected_objects,
            "object_count": len(detected_objects),
            "source_used": target_uri,
            "result_location": result_uri if save_results else None
        }