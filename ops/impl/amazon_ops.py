# ops/impl/amazon_ops.py
import os
import time
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from ops.base import VideoSegmenter, VisualCaptioner, LLMQuery, VideoSplitter

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
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")

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
        
        # 6. 可选：保存结果到 S3
        if kwargs.get('save_results', True):
            result_key = f"results/{os.path.basename(key)}.json"
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
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path)
        print(f"    Data Ready: {target_uri}")

        # 2. 解析 S3 URI
        bucket, key = self._parse_uri(target_uri)
        
        # 3. 生成预签名 URL（Bedrock 需要访问 S3 资源）
        presigned_url = self._get_s3_presigned_url(bucket, key)
        
        # 4. 构建 Bedrock 请求
        # 注意：Bedrock 的模型 ID 格式通常是 "anthropic.claude-3-5-sonnet-20241022-v2:0"
        # 这里需要根据实际的 model_name 映射到正确的 model ID
        model_id_map = {
            "nova-lite": "amazon.nova-lite-v1:0",
            "nova-pro": "amazon.nova-pro-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
        
        bedrock_model_id = model_id_map.get(self.model_name, self.model_name)
        
        # 5. 构建请求体（使用 Claude 3 格式）
        # 注意：Bedrock 的视频分析可能需要使用特定的 API，这里使用文本生成 API 作为示例
        # 实际应用中，可能需要使用 Amazon Rekognition Video 或其他专门的视频分析服务
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
        
        # 注意：视频分析可能需要使用不同的 API，这里简化处理
        # 如果模型不支持视频，可能需要先提取关键帧
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
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, function_name: Optional[str] = None):
        super().__init__(provider, region, storage_bucket)
        self._lambda_client = None
        
        # 如果提供了 function_name，直接使用；否则根据 region 构建默认名称
        if function_name:
            self.function_name = function_name
        else:
            # 默认函数名称格式：video-split-{region}
            region_suffix = region.replace('-', '_')
            self.function_name = f"video-split-{region_suffix}"
        
        print(f"    Lambda Function Name: {self.function_name}")
    
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
                - target_path: 输出路径
                - output_format: 输出格式（默认 mp4）
                - function_name: 可选的 Lambda 函数名称（覆盖默认）
        """
        print(f"--- [AWS Lambda Video Split] Region: {self.region} ---")
        
        # 1. 确保视频在 AWS S3
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path)
        print(f"    Video Ready: {target_uri}")
        
        # 2. 准备输出路径
        output_format = kwargs.get('output_format', 'mp4')
        if target_path:
            output_base_path = f"{target_path}/split_segments"
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
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',  # 同步调用
                Payload=json.dumps(payload)
            )
            
            # 解析响应
            response_payload = json.loads(response['Payload'].read())
            
            if response.get('FunctionError'):
                error_message = response_payload.get('errorMessage', 'Unknown error')
                raise RuntimeError(f"Lambda function error: {error_message}")
            
            output_uris = response_payload.get("output_uris", [])
            print(f"    Successfully split video into {len(output_uris)} segments")
            
            return {
                "provider": "aws_lambda",
                "region": self.region,
                "input_video": target_uri,
                "segments": segments,
                "output_uris": output_uris,
                "output_count": len(output_uris)
            }
            
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            raise RuntimeError(f"Lambda invocation failed: {e}")