# ops/impl/google_ops.py
import os
import re
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
        # Google Video Intelligence API 支持的区域映射（仅支持2个区域）
        # us-west1 (Oregon), asia-east1 (Taiwan)
        region_to_location = {
            'us-west1': 'us-west1',
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
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")

        # 2. 解析 URI
        bucket, blob_path = self._parse_uri(target_uri)
        
        # 3. 准备结果存储路径
        result_path = self._build_result_path(target_uri, "segment", "segment.json", target_path)
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
                f"Supported regions: us-west1, asia-east1"
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
            # Vertex AI 支持的区域映射（仅支持2个区域）
            # us-west1 (Oregon), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
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
        """
        import re

        start_time = kwargs.get('start_time')
        end_time = kwargs.get('end_time')
        target_path = kwargs.get('target_path')

        # 1. 准备基础数据
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
        segment_uri = target_uri
        temp_segment_path = None
        
        # 2. 如果指定了时间范围，提取视频片段
        if start_time is not None and end_time is not None:
            try:
                temp_segment = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                temp_segment_path = temp_segment.name
                temp_segment.close()
                
                self._extract_video_segment(video_uri, start_time, end_time, temp_segment_path)
                
                # 路径拼接逻辑：避免引入意外的双斜杠
                segment_filename = f"segment_{int(start_time)}_{int(end_time)}_{os.path.basename(target_uri)}"
                sub_dir = f"{target_path.strip('/')}/segments" if target_path else "segments"
                
                segment_uri = self.transmitter.upload_local_to_cloud(
                    temp_segment_path,
                    'google',
                    self.storage_bucket,
                    sub_dir
                )
            except Exception as e:
                print(f"Warning: Segment extraction failed, using full video. Error: {e}")
                segment_uri = target_uri

        # 3. 核心修正：URI 规范化逻辑
        # Vertex AI API 要求 GCS URI 格式正确，不接受路径中的双斜杠
        # 但GCS可能将双斜杠作为对象名的一部分存储
        # 策略：先检查原始URI是否存在，如果存在但包含双斜杠，需要特殊处理
        if segment_uri.startswith("gs://"):
            from google.cloud import storage
            try:
                # 先检查原始URI文件是否存在
                original_bucket, original_blob_path = self._parse_uri(segment_uri)
                storage_client = storage.Client()
                bucket = storage_client.bucket(original_bucket)
                original_blob = bucket.blob(original_blob_path)
                
                if original_blob.exists():
                    # 文件存在，检查路径中是否有双斜杠
                    if '//' in original_blob_path:
                        # 如果原始路径包含双斜杠，Vertex AI不接受，需要复制到规范化路径
                        # 规范化路径：将双斜杠替换为单斜杠
                        normalized_blob_path = re.sub(r'/+', '/', original_blob_path)
                        normalized_blob = bucket.blob(normalized_blob_path)
                        
                        if not normalized_blob.exists():
                            # 复制文件到规范化路径
                            print(f"Copying file from {original_blob_path} to {normalized_blob_path} for Vertex AI compatibility...")
                            bucket.copy_blob(original_blob, bucket, normalized_blob_path)
                            print(f"✓ File copied to normalized path")
                        
                        normalized_uri = f"gs://{original_bucket}/{normalized_blob_path}"
                    else:
                        # 路径已经规范化，直接使用
                        normalized_uri = segment_uri
                else:
                    # 文件不存在，尝试规范化路径
                    normalized_path = re.sub(r'/+', '/', original_blob_path.lstrip('/'))
                    normalized_uri = f"gs://{original_bucket}/{normalized_path}"
                    print(f"Warning: Original file not found, using normalized URI: {normalized_uri}")
            except Exception as e:
                # 如果检查失败，尝试规范化URI
                path_without_protocol = segment_uri[5:]
                while path_without_protocol.startswith('/'):
                    path_without_protocol = path_without_protocol[1:]
                normalized_path = re.sub(r'/+', '/', path_without_protocol)
                normalized_uri = f"gs://{normalized_path}"
                print(f"Warning: Could not verify file existence: {e}. Using normalized URI: {normalized_uri}")
        else:
            normalized_uri = segment_uri
        
        print(f"Using URI for Vertex AI: {normalized_uri}")
        
        # 5. 初始化模型与构建 Part
        model = self._init_vertex_ai()
        
        # 验证 URI 格式
        print(f"Normalized URI: {normalized_uri}")
        
        # 尝试最通用的位置参数调用，减少 400 错误的参数匹配风险
        try:
            # 优先使用 uri 关键字，这是目前 Vertex AI 比较标准的写法
            video_part = Part.from_uri(uri=normalized_uri, mime_type="video/mp4")
            print(f"✓ Created Part from URI successfully")
        except Exception as e:
            # 备选：直接传参
            try:
                video_part = Part.from_uri(normalized_uri, "video/mp4")
                print(f"✓ Created Part from URI (fallback method)")
            except Exception as e2:
                # 如果两种方式都失败，提供更详细的错误信息
                raise ValueError(
                    f"Failed to create Part from URI. "
                    f"URI: {normalized_uri}, "
                    f"Error 1: {str(e)}, "
                    f"Error 2: {str(e2)}"
                ) from e2

        prompt_text = "Describe this video in detail. Provide a comprehensive caption."
        
        # 打印完整的prompt
        print("\n" + "=" * 80)
        print("=== [Google Vertex AI Caption] Full Prompt ===")
        print("=" * 80)
        print(f"Video URI: {normalized_uri}")
        print(f"Prompt: {prompt_text}")
        print("=" * 80 + "\n")
        
        # 6. 调用 API 并增加安全拦截处理
        print(f"Generating caption for: {normalized_uri}...")
        try:
            # 尝试不同的调用方式，某些版本可能需要不同的参数格式
            # 强制设置temperature为0以确保确定性输出
            # 增加max_output_tokens到8192以确保完整的视频描述（1024可能不够）
            try:
                response = model.generate_content(
                    [video_part, prompt_text],
                    generation_config={
                        "temperature": 0,
                        "top_p": 0.95,
                        "max_output_tokens": 8192,  # 增加到8192以支持更长的视频描述
                    }
                )
            except Exception as e1:
                # 备选：尝试不使用generation_config（但会使用默认temperature，可能不是0）
                print(f"First attempt failed: {e1}, trying alternative format...")
                print("Warning: Using fallback method without explicit temperature=0")
                response = model.generate_content([video_part, prompt_text])
            
            # 检查响应状态
            finish_reason = None
            if response.candidates and len(response.candidates) > 0:
                finish_reason = response.candidates[0].finish_reason
                
                # finish_reason值：1=STOP(正常), 2=MAX_TOKENS(达到token限制), 3=SAFETY(安全拦截)
                if finish_reason == 3:  # SAFETY
                    caption = "Content blocked by safety filters."
                elif finish_reason == 2:  # MAX_TOKENS
                    caption = response.text if response.text else "No caption generated."
                    print(f"⚠ Warning: Caption may be truncated due to max_output_tokens limit (finish_reason=MAX_TOKENS)")
                else:
                    caption = response.text if response.text else "No caption generated."
            else:
                caption = response.text if response.text else "No caption generated."
            
            # 打印完整的响应和finish_reason信息
            print("\n" + "=" * 80)
            print("=== [Google Vertex AI Caption] Full Response ===")
            print("=" * 80)
            print(caption)
            if finish_reason is not None:
                finish_reason_names = {1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY"}
                print(f"\nFinish reason: {finish_reason_names.get(finish_reason, finish_reason)}")
            print("=" * 80 + "\n")
                
        except Exception as e:
            raise RuntimeError(f"Vertex AI API Error: {str(e)}\nVerified URI: {normalized_uri}") from e
        
        # 6. 清理
        if temp_segment_path and os.path.exists(temp_segment_path):
            os.remove(temp_segment_path)
        
        return {
            "provider": "google_vertex",
            "model": self.model_name,
            "caption": caption,
            "source_used": normalized_uri,
            "start_time": start_time,
            "end_time": end_time
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
            # Vertex AI 支持的区域映射（仅支持2个区域）
            # us-west1 (Oregon), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
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
        
        # 打印完整的prompt
        print("\n" + "=" * 80)
        print("=== [Google Vertex AI LLM] Full Prompt ===")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")
        
        # 初始化 Vertex AI
        model = self._init_vertex_ai()
        
        # 调用 Gemini API，强制设置temperature为0以确保确定性输出
        print(f"    Sending prompt to {self.model_name} (temperature=0)...")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": kwargs.get('max_output_tokens', 2048),
            }
        )
        
        answer = response.text if response.text else "Unable to generate response."
        
        # 打印完整的响应
        print("\n" + "=" * 80)
        print("=== [Google Vertex AI LLM] Full Response ===")
        print("=" * 80)
        print(answer)
        print("=" * 80 + "\n")
        
        return {
            "provider": "google_vertex",
            "model": self.model_name,
            "response": answer
        }


class GoogleCloudFunctionSplitImpl(VideoSplitter):
    """
    使用 Google Cloud Function Gen2（HTTP 触发器）进行视频分割。
    
    对应 deploy.sh 部署的 video-splitter Cloud Function（gcloud functions deploy --gen2）。
    接收视频 URI 和片段列表，使用 ffmpeg 在云端切割视频，并将结果上传到 GCS。
    service_url 可由 registry 根据项目与 region 自动获取（通过 gcloud 命令），
    或通过环境变量 GCP_VIDEOSPLIT_SERVICE_URLS 设置，或通过构造/execute 显式传入。
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, service_url: Optional[str] = None):
        super().__init__(provider, region, storage_bucket)
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
                - service_url: 可选的 Cloud Function 服务 URL（覆盖默认）
        """
        print(f"--- [Google Cloud Function Video Split] Region: {self.region} ---")
        
        # 1. 确保视频在 Google Bucket
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Video Ready: {target_uri}")
        
        # 2. 准备输出路径（规范化路径，避免双斜杠）
        output_format = kwargs.get('output_format', 'mp4')
        if target_path:
            # 移除末尾斜杠，避免拼接时产生双斜杠
            normalized_target_path = target_path.rstrip('/')
            output_base_path = f"{normalized_target_path}/split_segments"
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
        
        # 4. 调用 Cloud Function 服务（HTTP）
        service_url = kwargs.get('service_url', self.service_url)
        if not service_url:
            raise ValueError(
                "Missing Cloud Function service_url. "
                "请在构造 GoogleCloudFunctionSplitImpl(service_url=...) 或 execute(..., service_url=...) 时提供。"
            )
        print(f"    Calling Cloud Function: {service_url}")
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
                "provider": "google_cloud_function",
                "region": self.region,
                "input_video": target_uri,
                "segments": segments,
                "output_uris": output_uris,
                "output_count": len(output_uris)
            }
            
        except requests.exceptions.RequestException as e:
            # 尽量把 Cloud Function 返回的错误正文带出来，便于定位（常见：GCS 权限/ffmpeg 失败/输入 URI 不存在）
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
                    raise RuntimeError(f"Cloud Function call failed ({status}): {body_text}") from e
                raise RuntimeError(f"Cloud Function call failed ({status})") from e
            raise RuntimeError(f"Cloud Function call failed: {e}") from e
        except json.JSONDecodeError as e:
            # HTTP 成功但返回体不是 JSON
            raise RuntimeError(f"Invalid response from Cloud Function (not JSON): {e}") from e


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
            # Vertex AI 支持的区域映射（仅支持2个区域）
            # us-west1 (Oregon), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
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
        
        # 1. 确保数据在 Google Bucket 的正确bucket中
        # 无论视频是否已在云存储，都需要确保在正确的bucket中
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")
        
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
                # 使用新的路径格式：results/visual_embedding/[video_name]/embedding.json
                embedding_target_path = self._build_result_path(target_uri, "visual_embedding", "embedding.json", target_path)
                # 提取目录部分（去掉文件名）
                embedding_dir = os.path.dirname(embedding_target_path)
                
                # 保存到临时文件然后上传
                temp_embedding_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(embedding_data, temp_embedding_file, indent=2)
                temp_embedding_file.close()
                
                try:
                    embedding_uri = self.transmitter.upload_local_to_cloud(
                        temp_embedding_file.name,
                        'google',
                        self.storage_bucket,
                        embedding_dir
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
            # Vertex AI 支持的区域映射（仅支持2个区域）
            # us-west1 (Oregon), asia-southeast1 (Singapore)
            region_locations = {
                'us-west1': 'us-west1',
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
        # Google Video Intelligence API 支持的区域映射（仅支持2个区域）
        # us-west1 (Oregon), asia-east1 (Taiwan)
        region_to_location = {
            'us-west1': 'us-west1',
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
        target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")
        
        # 2. 解析 URI
        bucket, blob_path = self._parse_uri(target_uri)
        
        # 3. 准备结果存储路径
        result_path = self._build_result_path(target_uri, "object_detection", "result.json", target_path)
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
                f"Supported regions: us-west1, asia-east1"
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