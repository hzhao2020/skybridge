# ops/impl/amazon_ops.py
import os
import time
import json
import base64
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import requests
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
        import time
        print(f"--- [AWS Rekognition] Region: {self.region} ---")

        # 1. 确保数据在 AWS S3 的正确bucket中（传输时间不包含在operation时间内）
        # 无论视频是否已在云存储，都需要确保在正确的bucket中
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Data Ready: {target_uri}")

        # 2. 传输完成，记录operation实际开始时间（排除传输时间）
        operation_start_time = time.time()

        # 3. 解析 S3 URI
        bucket, key = self._parse_uri(target_uri)
        
        print(f"    Processing video: s3://{bucket}/{key}")

        # 4. 启动 Rekognition 视频分析任务
        response = self.rekognition_client.start_segment_detection(
            Video={'S3Object': {'Bucket': bucket, 'Name': key}},
            SegmentTypes=['SHOT']
        )
        job_id = response['JobId']
        print(f"    Job started: {job_id}")

        # 5. 轮询等待结果
        print("    Waiting for analysis to complete...")
        result = self._poll_segment_detection(job_id)
        
        # 6. 解析结果，提取 segments
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
        
        # 7. 可选：保存结果到 S3（输出结果放在results目录）
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
        
        # 8. 记录operation实际开始时间（传输完成后），供workflow记录operation时间使用
        operation_end_time = time.time()
        # 从TimingRecorder获取当前operation信息并存储实际开始时间
        try:
            from utils.timing import TimingRecorder
            recorder = TimingRecorder()
            current_operation = recorder._current_operation
            if current_operation:
                # 将operation实际开始时间存储到TimingRecorder中
                # workflow可以在执行完成后获取并记录operation时间（排除传输时间）
                if not hasattr(recorder, '_operation_actual_start_times'):
                    recorder._operation_actual_start_times = {}
                recorder._operation_actual_start_times[current_operation] = operation_start_time
        except Exception:
            pass  # 如果记录失败，不影响主流程
        
        return {
            "provider": "amazon",
            "region": self.region,
            "segments": segments,
            "source_used": target_uri,
            "result_location": result_uri
        }


class AWSLambdaSplitImpl(VideoSplitter):
    """
    使用 AWS Lambda（通过 API Gateway URL）进行视频分割
    
    需要先部署一个 Lambda 函数并通过 API Gateway 暴露 HTTP 端点，该函数接收视频 URI 和片段列表，
    使用 ffmpeg 在云端切割视频，并将结果上传到 S3。
    
    优先使用 service_url（API Gateway URL）进行 HTTP 调用，如果没有提供则回退到 Lambda invoke。
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, service_url: Optional[str] = None, function_name: Optional[str] = None):
        super().__init__(provider, region, storage_bucket)
        self._lambda_client = None
        self.service_url = service_url
        
        # 如果提供了 function_name，直接使用；否则使用默认名称（用于回退到 Lambda invoke）
        if function_name:
            self.function_name = function_name
        else:
            # 默认函数名称：video-splitter（与 deploy.sh 中部署的函数名一致）
            self.function_name = "video-splitter"
        
        if self.service_url:
            print(f"    AWS API Gateway URL: {self.service_url} (Region: {self.region})")
        else:
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
                - service_url: 可选的 API Gateway URL（覆盖默认）
                - function_name: 可选的 Lambda 函数名称（仅在回退到 Lambda invoke 时使用）
        """
        print(f"--- [AWS Lambda Video Split] Region: {self.region} ---")
        
        # 1. 确保视频在 AWS S3 的正确bucket中
        # 无论视频是否已在云存储，都需要确保在正确的bucket中
        target_path = kwargs.get('target_path')
        target_uri = self.transmitter.smart_move(video_uri, 'amazon', self.storage_bucket, target_path, target_region=self.region)
        print(f"    Video Ready: {target_uri}")
        
        # 2. 准备输出路径（用于存放分割后的视频片段，规范化路径避免双斜杠）
        output_format = kwargs.get('output_format', 'mp4')
        output_path = kwargs.get('target_path')  # target_path 是输出路径
        if output_path:
            # 移除末尾斜杠，避免拼接时产生双斜杠
            normalized_output_path = output_path.rstrip('/')
            output_base_path = f"{normalized_output_path}/split_segments"
        else:
            output_base_path = "split_segments"
        
        # 3. 构建请求负载
        request_body = {
            "video_uri": target_uri,
            "segments": segments,
            "output_bucket": self.storage_bucket,
            "output_path": output_base_path,
            "output_format": output_format
        }
        
        # 4. 优先使用 API Gateway URL 调用，否则回退到 Lambda invoke
        service_url = kwargs.get('service_url', self.service_url)
        
        if service_url:
            # 使用 HTTP 调用 API Gateway
            print(f"    Calling API Gateway: {service_url}")
            print(f"    Processing {len(segments)} segments...")
            
            try:
                response = requests.post(
                    service_url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                    timeout=900  # 15分钟超时（与Google一致）
                )
                response.raise_for_status()
                result = response.json()
                
                # API Gateway 返回的格式可能是 {"statusCode": 200, "body": "..."} 或直接是结果
                if isinstance(result, dict) and "statusCode" in result:
                    # 解析 body 字段
                    body_str = result.get("body", "{}")
                    if isinstance(body_str, str):
                        result = json.loads(body_str)
                    else:
                        result = body_str
                
                output_uris = result.get("output_uris", [])
                print(f"    Successfully split video into {len(output_uris)} segments")
                
                return {
                    "provider": "aws_api_gateway",
                    "region": self.region,
                    "input_video": target_uri,
                    "segments": segments,
                    "output_uris": output_uris,
                    "output_count": len(output_uris)
                }
                
            except requests.exceptions.RequestException as e:
                # 尽量把 API Gateway 返回的错误正文带出来，便于定位
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
                        raise RuntimeError(f"API Gateway call failed ({status}): {body_text}") from e
                    raise RuntimeError(f"API Gateway call failed ({status})") from e
                raise RuntimeError(f"API Gateway call failed: {e}") from e
            except json.JSONDecodeError as e:
                # HTTP 成功但返回体不是 JSON
                raise RuntimeError(f"Invalid response from API Gateway (not JSON): {e}") from e
        else:
            # 回退到 Lambda invoke（保持向后兼容）
            function_name = kwargs.get('function_name', self.function_name)
            print(f"    Invoking Lambda function: {function_name}")
            print(f"    Processing {len(segments)} segments...")
            print(f"    Payload: video_uri={target_uri}, segments={len(segments)}, output_bucket={self.storage_bucket}")
            
            try:
                response = self.lambda_client.invoke(
                    FunctionName=function_name,
                    InvocationType='RequestResponse',  # 同步调用
                    Payload=json.dumps(request_body)
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