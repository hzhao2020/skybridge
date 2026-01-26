# ops/impl/amazon_ops.py
import os
import time
import json
import base64
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from ops.base import VideoSegmenter, VisualCaptioner, LLMQuery, VideoSplitter, VisualEncoder, TextEncoder, ObjectDetector

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Warning: AWS boto3 not installed. Install with: pip install boto3")


class AmazonRekognitionSegmentImpl(VideoSegmenter):
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        self._rekognition_client = None
        self._s3_client = None
    
    @property
    def rekognition_client(self):
        """懒加载 AWS Rekognition 客户端"""
        if self._rekognition_client is None:
            self._rekognition_client = boto3.client('rekognition', region_name=self.region)
        return self._rekognition_client
    
    @property
    def s3_client(self):
        """懒加载 AWS S3 客户端"""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.region)
        return self._s3_client
    
    def _parse_uri(self, uri: str):
        """解析 S3 URI，返回 bucket 和 key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def _poll_segment_detection(self, job_id: str) -> Dict[str, Any]:
        """轮询 Rekognition 任务直到完成"""
        max_wait_time = 600  # 最大等待时间（秒）
        start_time = time.time()
        
        while True:
            response = self.rekognition_client.get_segment_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                return response
            elif status == 'FAILED':
                error_message = response.get('StatusMessage', 'Unknown error')
                raise RuntimeError(f"AWS Rekognition job failed: {error_message}")
            
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"Job {job_id} timed out after {max_wait_time} seconds")
            
            print(f"    Job {job_id} status: {status}, waiting...")
            time.sleep(5)
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [AWS Rekognition] Region: {self.region} ---")

        # 1. 确保数据在 AWS S3
        # 如果视频已经在云存储，直接使用；否则上传到临时路径（不放在results目录）
        if video_uri.startswith('s3://'):
            # 视频已在云存储，直接使用
            target_uri = video_uri
            print(f"    Data Ready (already in cloud): {target_uri}")
        else:
            # 本地视频，上传到临时路径（用于输入，不放在results目录）
            target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, None, target_region=self.region)
            print(f"    Data Ready (uploaded): {target_uri}")

        # 2. 解析 S3 URI
        bucket, key = self._parse_uri(target_uri)
        
        print(f"    Processing video: s3://{bucket}/{key}")

        # 3. 启动 Rekognition 视频分析任务
        response = self.rekognition_client.start_segment_detection(
            Video={'S3Object': {'Bucket': bucket, 'Name': key}},
            SegmentTypes=['SHOT']
        )
        job_id = response['JobId']
        print(f"    Job started: {job_id}")

        # 4. 轮询等待结果
        print("    Waiting for analysis to complete...")
        result = self._poll_segment_detection(job_id)
        
        # 5. 解析结果，提取 segments
        segments = []
        if 'Segments' in result:
            for segment in result['Segments']:
                start_time = segment.get('StartTimestampMillis', 0) / 1000.0
                end_time = segment.get('EndTimestampMillis', 0) / 1000.0
                segments.append({
                    "start": start_time,
                    "end": end_time
                })
        
        print(f"    Found {len(segments)} segments")
        
        # 6. 可选：保存结果到 S3（输出结果放在results目录）
        output_path = kwargs.get('target_path')  # target_path 是输出路径
        if kwargs.get('save_results', True):
            result_key = self._build_result_path(target_uri, "segment", "result.json", output_path)
            self.s3_client.put_object(
                Bucket=self.storage_bucket,
                Key=result_key,
                Body=json.dumps(result, default=str)
            )
            result_uri = f"s3://{self.storage_bucket}/{result_key}"
            print(f"    Results saved to: {result_uri}")
        else:
            result_uri = None
        
        return {
            "provider": "amazon",
            "region": self.region,
            "segments": segments,
            "source_used": target_uri,
            "result_location": result_uri
        }


class AmazonBedrockCaptionImpl(VisualCaptioner):
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._bedrock_client = None
        self._s3_client = None
    
    @property
    def bedrock_client(self):
        """懒加载 AWS Bedrock 客户端"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        return self._bedrock_client
    
    @property
    def s3_client(self):
        """懒加载 AWS S3 客户端"""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.region)
        return self._s3_client
    
    def _parse_uri(self, uri: str):
        """解析 S3 URI，返回 bucket 和 key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def _get_s3_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        """生成 S3 预签名 URL，用于 Bedrock 访问视频"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise RuntimeError(f"Failed to generate presigned URL: {e}")
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [AWS Bedrock Caption] Region: {self.region} | Model: {self.model_name} ---")

        # 1. 确保数据在 AWS S3
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")

        # 2. 解析 S3 URI
        bucket, key = self._parse_uri(target_uri)
        
        # 3. 生成预签名 URL（Bedrock 需要访问 S3 资源）
        presigned_url = self._get_s3_presigned_url(bucket, key)
        
        # 4. 构建 Bedrock 请求
        # 注意：Bedrock 的模型 ID 格式通常是 "anthropic.claude-3-5-sonnet-20241022-v2:0"
        # 对于 Nova 模型，使用标准模型 ID，但通过 Converse API 调用（支持 on-demand）
        model_id_map = {
            "nova-lite": "amazon.nova-lite-v1:0",
            "nova-pro": "amazon.nova-pro-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
        
        bedrock_model_id = model_id_map.get(self.model_name, self.model_name)
        is_nova_model = self.model_name in ["nova-lite", "nova-pro"]
        
        # 5. 对于 Nova 模型，使用 Converse API；对于其他模型，使用 InvokeModel API
        if is_nova_model:
            # 使用 Converse API（Nova 模型推荐使用此 API）
            # 注意：Converse API 支持视频输入，使用 S3 location
            print(f"    Generating caption with {bedrock_model_id} using Converse API...")
            try:
                # 使用 Converse API，直接传递 S3 URI
                response = self.bedrock_client.converse(
                    modelId=bedrock_model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "video": {
                                        "format": "mp4",
                                        "source": {
                                            "s3Location": {
                                                "uri": target_uri
                                            }
                                        }
                                    }
                                },
                                {
                                    "text": "Describe this video in detail. Provide a comprehensive caption."
                                }
                            ]
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": 1024
                    }
                )
                
                # Converse API 返回格式：response['output']['message']['content'][0]['text']
                caption = response.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', 'Unable to generate caption.')
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                print(f"    Warning: Converse API failed ({error_code}): {error_message}")
                # 如果 Converse API 失败，尝试使用 invoke_model 和 native message format
                try:
                    print(f"    Trying invoke_model with native message format...")
                    native_request = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "video": {
                                            "format": "mp4",
                                            "source": {
                                                "s3Location": {
                                                    "uri": target_uri
                                                }
                                            }
                                        }
                                    },
                                    {
                                        "text": "Describe this video in detail. Provide a comprehensive caption."
                                    }
                                ]
                            }
                        ],
                        "inferenceConfig": {
                            "maxTokens": 1024
                        }
                    }
                    response = self.bedrock_client.invoke_model(
                        modelId=bedrock_model_id,
                        body=json.dumps(native_request),
                        contentType="application/json",
                        accept="application/json"
                    )
                    response_body = json.loads(response['body'].read())
                    caption = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', 'Unable to generate caption.')
                except Exception as e2:
                    print(f"    Warning: invoke_model also failed. Error: {e2}")
                    caption = f"Video analysis requested for {target_uri} using {self.model_name}. Note: Full video captioning may require additional processing."
            except Exception as e:
                print(f"    Warning: Converse API failed. Error: {e}")
                caption = f"Video analysis requested for {target_uri} using {self.model_name}. Note: Full video captioning may require additional processing."
        else:
            # 对于 Claude 模型，使用 InvokeModel API（原有逻辑）
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": presigned_url
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this video in detail. Provide a comprehensive caption."
                            }
                        ]
                    }
                ]
            }
            
            print(f"    Generating caption with {bedrock_model_id}...")
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=bedrock_model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                caption = response_body.get('content', [{}])[0].get('text', 'Unable to generate caption.')
            except Exception as e:
                print(f"    Warning: Direct video analysis may not be supported. Error: {e}")
                # 降级处理：返回基本信息
                caption = f"Video analysis requested for {target_uri} using {self.model_name}. Note: Full video captioning may require additional processing."
        
        print(f"    Caption generated: {caption[:100]}...")
        
        return {
            "provider": "amazon_bedrock",
            "model": self.model_name,
            "caption": caption,
            "source_used": target_uri
        }


class AmazonBedrockLLMImpl(LLMQuery):
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._bedrock_client = None
    
    @property
    def bedrock_client(self):
        """懒加载 AWS Bedrock 客户端"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        return self._bedrock_client
    
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [AWS Bedrock LLM] Region: {self.region} | Model: {self.model_name} ---")
        
        # 映射模型名称到 Bedrock model ID
        model_id_map = {
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v2:0"
        }
        
        bedrock_model_id = model_id_map.get(self.model_name, self.model_name)
        
        # 构建请求体（Claude 3 格式）
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get('max_tokens', 2048),
            "temperature": kwargs.get('temperature', 0.7),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        print(f"    Sending prompt to {bedrock_model_id}...")
        try:
            response = self.bedrock_client.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            answer = response_body.get('content', [{}])[0].get('text', 'Unable to generate response.')
            
            print(f"    Response received: {answer[:100]}...")
            
            return {
                "provider": "amazon_bedrock",
                "model": self.model_name,
                "response": answer
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise RuntimeError(f"AWS Bedrock API error ({error_code}): {error_message}")
        except Exception as e:
            raise RuntimeError(f"Failed to invoke Bedrock model: {e}")


class AWSLambdaSplitImpl(VideoSplitter):
    """
    使用 AWS Lambda 进行视频分割
    
    需要先部署一个 Lambda 函数，该函数接收视频 URI 和片段列表，
    使用 ffmpeg 在云端切割视频，并将结果上传到 S3。
    
    默认函数名：video-splitter（与 deploy.sh 中部署的函数名一致）
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, function_name: Optional[str] = None):
        super().__init__(provider, region, storage_bucket)
        self._lambda_client = None
        
        # 如果提供了 function_name，直接使用；否则使用默认名称
        if function_name:
            self.function_name = function_name
        else:
            # 默认函数名称：video-splitter（与 deploy.sh 中部署的函数名一致）
            self.function_name = "video-splitter"
        
        print(f"    Lambda Function Name: {self.function_name} (Region: {self.region})")
    
    @property
    def lambda_client(self):
        """懒加载 AWS Lambda 客户端"""
        if self._lambda_client is None:
            import boto3
            self._lambda_client = boto3.client('lambda', region_name=self.region)
        return self._lambda_client
    
    def _parse_uri(self, uri: str):
        """解析 S3 URI，返回 bucket 和 key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def execute(self, video_uri: str, segments: List[Dict[str, float]], **kwargs) -> Dict[str, Any]:
        """
        执行视频分割
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            segments: 片段列表，每个片段包含 'start' 和 'end'（秒）
            **kwargs:
                - target_path: 输出路径（用于存放分割后的视频片段）
                - output_format: 输出格式（默认 mp4）
                - function_name: 可选的 Lambda 函数名称（覆盖默认）
        """
        print(f"--- [AWS Lambda Video Split] Region: {self.region} ---")
        
        # 1. 确保视频在 AWS S3
        # 如果视频已经在云存储，直接使用；否则上传到临时路径（不放在results目录）
        if video_uri.startswith('s3://'):
            # 视频已在云存储，直接使用
            target_uri = video_uri
            print(f"    Video Ready (already in cloud): {target_uri}")
        else:
            # 本地视频，上传到临时路径（用于输入，不放在results目录）
            target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, None, target_region=self.region)
            print(f"    Video Ready (uploaded): {target_uri}")
        
        # 2. 准备输出路径（用于存放分割后的视频片段）
        output_format = kwargs.get('output_format', 'mp4')
        output_path = kwargs.get('target_path')  # target_path 是输出路径
        if output_path:
            output_base_path = f"{output_path}/split_segments"
        else:
            output_base_path = "split_segments"
        
        # 3. 构建 Lambda 请求负载
        payload = {
            "video_uri": target_uri,
            "segments": segments,
            "output_bucket": self.storage_bucket,
            "output_path": output_base_path,
            "output_format": output_format
        }
        
        # 4. 调用 Lambda 函数
        function_name = kwargs.get('function_name', self.function_name)
        print(f"    Invoking Lambda function: {function_name}")
        print(f"    Processing {len(segments)} segments...")
        print(f"    Payload: video_uri={target_uri}, segments={len(segments)}, output_bucket={self.storage_bucket}")
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',  # 同步调用
                Payload=json.dumps(payload)
            )
            
            # 解析响应
            response_payload_str = response['Payload'].read()
            if isinstance(response_payload_str, bytes):
                response_payload_str = response_payload_str.decode('utf-8')
            
            response_payload = json.loads(response_payload_str)
            
            # 检查 Lambda 函数是否出错
            if response.get('FunctionError'):
                error_message = response_payload.get('errorMessage', 'Unknown error')
                error_type = response_payload.get('errorType', 'UnknownError')
                raise RuntimeError(f"Lambda function error ({error_type}): {error_message}")
            
            # 检查响应状态
            if response_payload.get('status') != 'success':
                error_msg = response_payload.get('error', 'Unknown error')
                raise RuntimeError(f"Lambda function returned non-success status: {error_msg}")
            
            # 获取输出 URI 列表
            output_uris = response_payload.get("output_uris", [])
            segment_count = response_payload.get("segment_count", len(output_uris))
            
            if not output_uris:
                print(f"    Warning: Lambda function returned no output URIs")
            else:
                print(f"    Successfully split video into {len(output_uris)} segments")
                if len(output_uris) <= 5:
                    for i, uri in enumerate(output_uris, 1):
                        print(f"      Segment {i}: {uri}")
                else:
                    for i, uri in enumerate(output_uris[:3], 1):
                        print(f"      Segment {i}: {uri}")
                    print(f"      ... (还有 {len(output_uris) - 3} 个片段)")
            
            return {
                "provider": "aws_lambda",
                "region": self.region,
                "input_video": target_uri,
                "segments": segments,
                "output_uris": output_uris,
                "output_count": len(output_uris),
                "segment_count": segment_count
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            raise RuntimeError(f"AWS Lambda API error ({error_code}): {error_message}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Lambda response: {e}")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            raise RuntimeError(f"Lambda invocation failed: {e}")


class AmazonBedrockEmbeddingImpl(VisualEncoder):
    """
    使用 AWS Bedrock 的 Titan Multimodal Embeddings G1 模型进行视觉编码
    
    注意：Titan Multimodal Embeddings G1 主要支持图像和文本。
    对于视频，我们将提取关键帧并使用图像 embedding。
    """
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str = "amazon.titan-embed-image-v1"):
        super().__init__(provider, region, storage_bucket, model_name)
        self._bedrock_client = None
        self._s3_client = None
    
    @property
    def bedrock_client(self):
        """懒加载 AWS Bedrock 客户端"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        return self._bedrock_client
    
    @property
    def s3_client(self):
        """懒加载 AWS S3 客户端"""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.region)
        return self._s3_client
    
    def _parse_uri(self, uri: str):
        """解析 S3 URI，返回 bucket 和 key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def _extract_video_frame(self, video_uri: str, frame_time: float = 0.0) -> str:
        """
        从视频中提取一帧作为图像
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            frame_time: 提取帧的时间点（秒），默认提取第一帧
            
        Returns:
            提取的帧图像文件路径
        """
        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image_path = temp_image.name
        temp_image.close()
        
        # 确定输入源
        input_source = None
        temp_video_path = None
        
        if video_uri.startswith('s3://'):
            # S3: 下载到本地
            bucket, key = self._parse_uri(video_uri)
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            
            self.s3_client.download_file(bucket, key, temp_video_path)
            input_source = temp_video_path
        elif video_uri.startswith('gs://'):
            # GCS: 下载到本地
            from google.cloud import storage
            bucket, blob_path = self._parse_uri(video_uri)
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            
            storage_client = storage.Client()
            bucket_obj = storage_client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            blob.download_to_filename(temp_video_path)
            input_source = temp_video_path
        else:
            # 本地路径
            if not os.path.exists(video_uri):
                raise FileNotFoundError(f"Video file not found: {video_uri}")
            input_source = video_uri
        
        # 使用 ffmpeg 提取帧
        try:
            cmd = [
                'ffmpeg',
                '-i', input_source,
                '-ss', str(frame_time),
                '-vframes', '1',
                '-q:v', '2',  # 高质量 JPEG
                '-y',
                temp_image_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if not os.path.exists(temp_image_path) or os.path.getsize(temp_image_path) == 0:
                raise RuntimeError("ffmpeg frame extraction failed: output file is empty")
            
            return temp_image_path
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg frame extraction failed: {error_msg}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        finally:
            # 清理临时视频文件
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图像文件转换为 base64 编码"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        对视频进行编码，生成向量
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            target_path: 可选，上传目标路径
            save_embedding: 可选，是否保存 embedding 到云存储（默认 False）
            frame_time: 可选，提取帧的时间点（秒），默认 0.0（第一帧）
            embedding_length: 可选，embedding 长度，可选 256, 384, 1024（默认 1024）
        """
        print(f"--- [AWS Bedrock Embedding] Region: {self.region} | Model: {self.model_name} ---")
        
        # 1. 确保数据在 AWS S3（如果需要）
        target_path = kwargs.get('target_path')
        if target_path or not (video_uri.startswith('s3://') or os.path.exists(video_uri)):
            target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path, target_region=self.region)
            print(f"    Data Ready: {target_uri}")
        else:
            target_uri = video_uri
            print(f"    Using video: {target_uri}")
        
        # 2. 从视频中提取关键帧
        frame_time = kwargs.get('frame_time', 0.0)
        print(f"    Extracting frame at {frame_time}s...")
        temp_image_path = None
        try:
            temp_image_path = self._extract_video_frame(target_uri, frame_time)
            
            # 3. 将图像转换为 base64
            image_base64 = self._image_to_base64(temp_image_path)
            
            # 4. 构建 Bedrock 请求
            embedding_length = kwargs.get('embedding_length', 1024)
            if embedding_length not in [256, 384, 1024]:
                raise ValueError(f"embedding_length must be one of [256, 384, 1024], got {embedding_length}")
            
            request_body = {
                "inputImage": image_base64,
                "embeddingConfig": {
                    "outputEmbeddingLength": embedding_length
                }
            }
            
            # 5. 调用 Bedrock API
            print(f"    Generating embedding with {self.model_name}...")
            response = self.bedrock_client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(request_body),
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            embedding_vector = response_body.get('embedding', [])
            input_text_token_count = response_body.get('inputTextTokenCount', 0)
            message = response_body.get('message', '')
            
            if not embedding_vector:
                raise RuntimeError(f"No embedding returned from model. Message: {message}")
            
            embedding_dim = len(embedding_vector)
            print(f"    Embedding generated: dimension={embedding_dim}")
            
            # 6. 可选：保存 embedding 到云存储
            save_embedding = kwargs.get('save_embedding', False)
            embedding_uri = None
            if save_embedding:
                embedding_data = {
                    "embedding": embedding_vector,
                    "dimension": embedding_dim,
                    "source_video": target_uri,
                    "frame_time": frame_time,
                    "model": self.model_name,
                    "region": self.region,
                    "embedding_length": embedding_length
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
                        'amazon',
                        self.storage_bucket,
                        embedding_dir
                    )
                    print(f"    Embedding saved to: {embedding_uri}")
                finally:
                    if os.path.exists(temp_embedding_file.name):
                        os.remove(temp_embedding_file.name)
            
            result = {
                "provider": "amazon_bedrock",
                "model": self.model_name,
                "region": self.region,
                "embedding": embedding_vector,
                "embedding_dimension": embedding_dim,
                "source_video": target_uri,
                "frame_time": frame_time,
                "embedding_length": embedding_length
            }
            
            if embedding_uri:
                result["embedding_uri"] = embedding_uri
            
            return result
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise RuntimeError(f"AWS Bedrock API error ({error_code}): {error_message}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e
        finally:
            # 清理临时文件
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception:
                    pass  # 忽略清理错误


class AmazonBedrockTextEmbeddingImpl(TextEncoder):
    """
    使用 AWS Bedrock 的 Titan Text Embeddings 模型进行文本编码
    
    支持的模型：
    - Titan Text Embeddings V2: amazon.titan-embed-text-v2:0
    - Titan Embeddings G1 - Text: amazon.titan-embed-text-v1
    """
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._bedrock_client = None
    
    @property
    def bedrock_client(self):
        """懒加载 AWS Bedrock 客户端"""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        return self._bedrock_client
    
    def execute(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        对文本进行编码，生成向量
        
        Args:
            text: 要编码的文本
            save_embedding: 可选，是否保存 embedding 到云存储（默认 False）
            dimensions: 可选，embedding 维度（仅 V2 支持：256, 512, 1024，默认 1024）
            normalize: 可选，是否归一化向量（仅 V2 支持，默认 False）
        """
        print(f"--- [AWS Bedrock Text Embedding] Region: {self.region} | Model: {self.model_name} ---")
        
        # 1. 构建请求体
        request_body = {
            "inputText": text
        }
        
        # 2. 对于 V2 模型，支持额外的配置
        if "v2" in self.model_name.lower():
            dimensions = kwargs.get('dimensions', 1024)
            if dimensions not in [256, 512, 1024]:
                raise ValueError(f"dimensions must be one of [256, 512, 1024], got {dimensions}")
            request_body["dimensions"] = dimensions
            
            normalize = kwargs.get('normalize', False)
            if normalize:
                request_body["normalize"] = True
        
        # 3. 调用 Bedrock API
        print(f"    Generating embedding for text (length: {len(text)} chars)...")
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(request_body),
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            embedding_vector = response_body.get('embedding', [])
            input_text_token_count = response_body.get('inputTextTokenCount', 0)
            message = response_body.get('message', '')
            
            if not embedding_vector:
                raise RuntimeError(f"No embedding returned from model. Message: {message}")
            
            embedding_dim = len(embedding_vector)
            print(f"    Embedding generated: dimension={embedding_dim}")
            
            # 4. 可选：保存 embedding 到云存储
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
                    "input_text_token_count": input_text_token_count
                }
                if "v2" in self.model_name.lower():
                    embedding_data["dimensions"] = dimensions
                    embedding_data["normalize"] = normalize
                
                # 保存到临时文件然后上传
                temp_embedding_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                json.dump(embedding_data, temp_embedding_file, indent=2)
                temp_embedding_file.close()
                
                try:
                    embedding_uri = self.transmitter.upload_local_to_cloud(
                        temp_embedding_file.name,
                        'amazon',
                        self.storage_bucket,
                        "embeddings/text"
                    )
                    print(f"    Embedding saved to: {embedding_uri}")
                finally:
                    if os.path.exists(temp_embedding_file.name):
                        os.remove(temp_embedding_file.name)
            
            result = {
                "provider": "amazon_bedrock",
                "model": self.model_name,
                "region": self.region,
                "embedding": embedding_vector,
                "embedding_dimension": embedding_dim,
                "text_length": len(text),
                "input_text_token_count": input_text_token_count
            }
            
            if embedding_uri:
                result["embedding_uri"] = embedding_uri
            
            return result
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise RuntimeError(f"AWS Bedrock API error ({error_code}): {error_message}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embedding: {e}") from e


class AmazonRekognitionObjectDetectionImpl(ObjectDetector):
    """
    使用 AWS Rekognition Video 进行物体检测
    """
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        self._rekognition_client = None
        self._s3_client = None
    
    @property
    def rekognition_client(self):
        """懒加载 AWS Rekognition 客户端"""
        if self._rekognition_client is None:
            self._rekognition_client = boto3.client('rekognition', region_name=self.region)
        return self._rekognition_client
    
    @property
    def s3_client(self):
        """懒加载 AWS S3 客户端"""
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', region_name=self.region)
        return self._s3_client
    
    def _parse_uri(self, uri: str):
        """解析 S3 URI，返回 bucket 和 key"""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def _poll_label_detection(self, job_id: str) -> Dict[str, Any]:
        """轮询 Rekognition 标签检测任务直到完成"""
        max_wait_time = 600  # 最大等待时间（秒）
        start_time = time.time()
        
        while True:
            response = self.rekognition_client.get_label_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status == 'SUCCEEDED':
                return response
            elif status == 'FAILED':
                error_message = response.get('StatusMessage', 'Unknown error')
                raise RuntimeError(f"AWS Rekognition job failed: {error_message}")
            
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"Job {job_id} timed out after {max_wait_time} seconds")
            
            print(f"    Job {job_id} status: {status}, waiting...")
            time.sleep(5)
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        在视频中检测物体
        
        Args:
            video_uri: 视频 URI（本地路径或云存储 URI）
            target_path: 可选，上传目标路径
            save_results: 可选，是否保存结果到云存储（默认 True）
            min_confidence: 可选，最小置信度阈值（默认 50）
        """
        print(f"--- [AWS Rekognition Object Detection] Region: {self.region} ---")
        
        # 1. 确保数据在 AWS S3
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")
        
        # 2. 解析 S3 URI
        bucket, key = self._parse_uri(target_uri)
        
        print(f"    Processing video: s3://{bucket}/{key}")
        
        # 3. 启动 Rekognition 标签检测任务
        min_confidence = kwargs.get('min_confidence', 50)
        response = self.rekognition_client.start_label_detection(
            Video={'S3Object': {'Bucket': bucket, 'Name': key}},
            MinConfidence=min_confidence,
            Features=['GENERAL_LABELS']  # 检测一般物体标签
        )
        job_id = response['JobId']
        print(f"    Job started: {job_id}")
        print(f"    Min confidence: {min_confidence}%")
        
        # 4. 轮询等待结果
        print("    Waiting for object detection to complete...")
        result = self._poll_label_detection(job_id)
        
        # 5. 解析结果，提取检测到的物体
        detected_objects = []
        if 'Labels' in result:
            # 按物体类型分组
            objects_by_name = {}
            for label in result['Labels']:
                label_name = label['Label']['Name']
                confidence = label['Label']['Confidence']
                instances = label.get('Instances', [])
                
                if label_name not in objects_by_name:
                    objects_by_name[label_name] = {
                        "name": label_name,
                        "confidence": confidence,
                        "instances": []
                    }
                
                # 提取每个实例的时间片段和边界框
                for instance in instances:
                    instance_info = {
                        "confidence": instance.get('Confidence', 0),
                        "bounding_box": instance.get('BoundingBox', {}),
                        "time_range": {}
                    }
                    
                    # 提取时间范围
                    if 'Timestamp' in instance:
                        timestamp_ms = instance['Timestamp']
                        instance_info["time_range"] = {
                            "timestamp_ms": timestamp_ms,
                            "timestamp_seconds": timestamp_ms / 1000.0
                        }
                    
                    objects_by_name[label_name]["instances"].append(instance_info)
            
            # 转换为列表格式
            for obj_name, obj_data in objects_by_name.items():
                detected_objects.append({
                    "name": obj_data["name"],
                    "overall_confidence": obj_data["confidence"],
                    "instance_count": len(obj_data["instances"]),
                    "instances": obj_data["instances"]
                })
        
        print(f"    Found {len(detected_objects)} unique object types")
        total_instances = sum(obj["instance_count"] for obj in detected_objects)
        print(f"    Total instances detected: {total_instances}")
        
        # 6. 可选：保存结果到 S3
        save_results = kwargs.get('save_results', True)
        result_uri = None
        if save_results:
            result_key = self._build_result_path(target_uri, "object_detection", "result.json", target_path)
            self.s3_client.put_object(
                Bucket=self.storage_bucket,
                Key=result_key,
                Body=json.dumps(result, default=str)
            )
            result_uri = f"s3://{self.storage_bucket}/{result_key}"
            print(f"    Results saved to: {result_uri}")
        
        return {
            "provider": "amazon",
            "region": self.region,
            "detected_objects": detected_objects,
            "object_type_count": len(detected_objects),
            "total_instance_count": total_instances,
            "source_used": target_uri,
            "result_location": result_uri
        }