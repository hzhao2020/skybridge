"""
AWS Lambda Function for Video Splitting

部署说明：
1. 创建 Lambda 函数（Python 3.11）
2. 设置超时时间：10分钟（600秒）
3. 设置内存：2048 MB
4. 添加环境变量：OUTPUT_BUCKET=your-bucket-name
5. 添加 IAM 角色权限：S3 读写权限

Lambda Layer（如果需要）：
- 可以创建一个包含 ffmpeg 的 Lambda Layer
- 或者使用容器镜像部署（推荐）
"""

import os
import json
import subprocess
import tempfile
import boto3
from urllib.parse import urlparse


def lambda_handler(event, context):
    """
    Lambda 函数入口点
    
    事件格式：
    {
        "video_uri": "s3://bucket/path/to/video.mp4",
        "segments": [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 20.0}
        ],
        "output_bucket": "output-bucket",
        "output_path": "split_segments",
        "output_format": "mp4"
    }
    """
    try:
        # 解析事件
        video_uri = event.get("video_uri")
        segments = event.get("segments", [])
        output_bucket = event.get("output_bucket")
        output_path = event.get("output_path", "split_segments")
        output_format = event.get("output_format", "mp4")
        
        if not video_uri or not segments:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required fields: video_uri, segments"})
            }
        
        # 初始化 S3 客户端
        s3_client = boto3.client('s3')
        
        # 下载视频到临时文件
        parsed = urlparse(video_uri)
        bucket_name = parsed.netloc
        key = parsed.path.lstrip('/')
        
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        s3_client.download_file(bucket_name, key, temp_video_path)
        
        # 处理每个片段
        output_uris = []
        
        for idx, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            duration = end_time - start_time
            
            # 创建临时输出文件
            temp_output = tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False)
            temp_output_path = temp_output.name
            temp_output.close()
            
            try:
                # 使用 ffmpeg 提取片段
                cmd = [
                    'ffmpeg',
                    '-i', temp_video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    temp_output_path
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # 上传到 S3
                filename = os.path.basename(key)
                name_without_ext = os.path.splitext(filename)[0]
                output_key = f"{output_path}/{name_without_ext}_segment_{idx+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                
                s3_client.upload_file(temp_output_path, output_bucket, output_key)
                
                output_uri = f"s3://{output_bucket}/{output_key}"
                output_uris.append(output_uri)
                
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
        
        # 清理临时视频文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "output_uris": output_uris,
                "segment_count": len(output_uris)
            })
        }
        
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
