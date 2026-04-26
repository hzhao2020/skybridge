import json
import os
import subprocess
import tempfile
from urllib.parse import urlparse
import boto3

def _split_video_s3_to_s3(*, video_uri: str, segments: list, output_bucket: str, output_path: str, output_format: str):
    # (这部分逻辑保持不变，依然负责下载、FFmpeg分割和上传)
    if not str(video_uri).startswith("s3://"):
        raise ValueError("仅支持 s3:// 输入路径")
    
    s3_client = boto3.client("s3")
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    try:
        parsed = urlparse(video_uri)
        bucket_name = parsed.netloc
        key = parsed.path.lstrip("/")
        
        s3_client.download_file(bucket_name, key, temp_video_path)
        output_uris = []
        filename = os.path.basename(key)
        name_without_ext = os.path.splitext(filename)[0]

        for idx, segment in enumerate(segments):
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            duration = end_time - start_time
            if duration <= 0: continue

            temp_output = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
            temp_output_path = temp_output.name
            temp_output.close()

            try:
                cmd = ["ffmpeg", "-i", temp_video_path, "-ss", str(start_time), "-t", str(duration),
                       "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", temp_output_path]
                subprocess.run(cmd, check=True, capture_output=True)

                normalized_path = output_path.rstrip('/')
                output_key = f"{normalized_path}/{name_without_ext}_seg_{idx+1}.{output_format}"
                s3_client.upload_file(temp_output_path, output_bucket, output_key)
                output_uris.append(f"s3://{output_bucket}/{output_key}")
            finally:
                if os.path.exists(temp_output_path): os.remove(temp_output_path)

        return {"status": "success", "output_uris": output_uris}
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)

def lambda_handler(event, context):
    """
    针对 URL 调用 (API Gateway) 优化后的入口
    """
    try:
        # 1. 解析来自 URL 的 JSON 数据
        # API Gateway 传来的 body 可能是字符串，也可能是已解析的字典
        body = event.get("body", "{}")
        if isinstance(body, str):
            payload = json.loads(body)
        else:
            payload = body

        video_uri = payload.get("video_uri")
        segments = payload.get("segments") or []
        
        env_default_bucket = os.environ.get("DEFAULT_OUTPUT_BUCKET")
        output_bucket = payload.get("output_bucket") or env_default_bucket
        output_path = payload.get("output_path", "split_segments")
        output_format = payload.get("output_format", "mp4")

        if not video_uri or not segments or not output_bucket:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required fields"})
            }

        # 2. 执行分割业务
        result = _split_video_s3_to_s3(
            video_uri=video_uri,
            segments=segments,
            output_bucket=output_bucket,
            output_path=output_path,
            output_format=output_format,
        )

        # 3. 返回符合 API Gateway 规范的响应
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }