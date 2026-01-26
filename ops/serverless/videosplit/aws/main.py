import json
import os
import subprocess
import tempfile
from urllib.parse import urlparse

try:
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None


def _split_video_s3_to_s3(*, video_uri: str, segments: list, output_bucket: str, output_path: str, output_format: str):
    """
    Lambda 版本：输入/输出均为 S3（s3://）。

    成功返回业务结果 dict（包含 output_uris），以便调用侧 boto3.invoke 直接解析。
    """
    if not str(video_uri).startswith("s3://"):
        raise ValueError("Lambda (aws) 版本仅支持 s3:// 输入。")
    if boto3 is None:
        raise RuntimeError("缺少依赖：boto3（请在部署镜像中安装 requirements.txt）。")

    parsed = urlparse(video_uri)
    bucket_name = parsed.netloc
    key = parsed.path.lstrip("/")

    s3_client = boto3.client("s3")

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    try:
        s3_client.download_file(bucket_name, key, temp_video_path)

        output_uris = []
        filename = os.path.basename(key)
        name_without_ext = os.path.splitext(filename)[0]

        for idx, segment in enumerate(segments):
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            duration = end_time - start_time
            if duration <= 0:
                continue

            temp_output = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False)
            temp_output_path = temp_output.name
            temp_output.close()

            try:
                cmd = [
                    "ffmpeg",
                    "-i",
                    temp_video_path,
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-y",
                    temp_output_path,
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # 规范化路径，移除末尾斜杠避免双斜杠
                normalized_output_path = output_path.rstrip('/')
                output_key = (
                    f"{normalized_output_path}/{name_without_ext}_segment_{idx+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                )
                s3_client.upload_file(temp_output_path, output_bucket, output_key)
                output_uris.append(f"s3://{output_bucket}/{output_key}")
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        return {"status": "success", "output_uris": output_uris, "segment_count": len(output_uris)}
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def lambda_handler(event, context):
    """
    AWS Lambda 入口（供 boto3.invoke 同步调用）

    事件格式：
    {
        "video_uri": "s3://bucket/path/to/video.mp4",
        "segments": [{"start": 0.0, "end": 10.0}],
        "output_bucket": "output-bucket",
        "output_path": "split_segments",
        "output_format": "mp4"
    }
    """
    payload = event or {}
    video_uri = payload.get("video_uri")
    segments = payload.get("segments") or []
    output_bucket = payload.get("output_bucket")
    output_path = payload.get("output_path", "split_segments")
    output_format = payload.get("output_format", "mp4")

    if not video_uri or not segments or not output_bucket:
        raise ValueError("Missing required fields: video_uri, segments, output_bucket")

    try:
        return _split_video_s3_to_s3(
            video_uri=video_uri,
            segments=segments,
            output_bucket=output_bucket,
            output_path=output_path,
            output_format=output_format,
        )
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {detail[:2000]}")

