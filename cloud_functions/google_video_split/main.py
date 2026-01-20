"""
Google Cloud Function for Video Splitting

部署说明：
1. 安装依赖：pip install -r requirements.txt -t .
2. 部署函数：
   gcloud functions deploy video-split-us-west1 \
     --runtime python311 \
     --trigger-http \
     --allow-unauthenticated \
     --region us-west1 \
     --memory 2GB \
     --timeout 540s \
     --set-env-vars GCS_BUCKET=your-bucket-name

注意：需要确保 Cloud Function 有权限访问 GCS bucket
"""

import os
import json
import subprocess
import tempfile
from google.cloud import storage
from urllib.parse import urlparse


def video_split(request):
    """
    Cloud Function 入口点
    
    请求格式：
    {
        "video_uri": "gs://bucket/path/to/video.mp4",
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
        # 解析请求
        request_json = request.get_json(silent=True)
        if not request_json:
            return {"error": "Invalid JSON"}, 400
        
        video_uri = request_json.get("video_uri")
        segments = request_json.get("segments", [])
        output_bucket = request_json.get("output_bucket")
        output_path = request_json.get("output_path", "split_segments")
        output_format = request_json.get("output_format", "mp4")
        
        if not video_uri or not segments:
            return {"error": "Missing required fields: video_uri, segments"}, 400
        
        # 初始化 GCS 客户端
        storage_client = storage.Client()
        
        # 下载视频到临时文件
        parsed = urlparse(video_uri)
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip('/')
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        blob.download_to_filename(temp_video_path)
        
        # 处理每个片段
        output_uris = []
        output_bucket_obj = storage_client.bucket(output_bucket)
        
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
                
                # 上传到 GCS
                filename = os.path.basename(blob_path)
                name_without_ext = os.path.splitext(filename)[0]
                output_blob_name = f"{output_path}/{name_without_ext}_segment_{idx+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                
                output_blob = output_bucket_obj.blob(output_blob_name)
                output_blob.upload_from_filename(temp_output_path)
                
                output_uri = f"gs://{output_bucket}/{output_blob_name}"
                output_uris.append(output_uri)
                
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
        
        # 清理临时视频文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        return {
            "status": "success",
            "output_uris": output_uris,
            "segment_count": len(output_uris)
        }
        
    except Exception as e:
        return {"error": str(e)}, 500
