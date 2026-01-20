import os
import subprocess
import tempfile
from urllib.parse import urlparse

from flask import Flask, jsonify, request
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None


app = Flask(__name__)


def _split_video_gcs_to_gcs(*, video_uri: str, segments: list, output_bucket: str, output_path: str, output_format: str):
    """
    Cloud Run 版本：输入/输出均为 GCS（gs://）。

    请求体与输出结构保持与本地调用侧一致：
    - 入参：video_uri, segments, output_bucket, output_path, output_format
    - 出参：{"status": "success", "output_uris": [...], "segment_count": N}
    """
    if not str(video_uri).startswith("gs://"):
        raise ValueError("Cloud Run (google) 版本仅支持 gs:// 输入。")
    if storage is None:
        raise RuntimeError("缺少依赖：google-cloud-storage（请在部署镜像中安装 requirements.txt）。")

    parsed = urlparse(video_uri)
    bucket_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")

    storage_client = storage.Client()

    # 1) 下载原始视频到临时文件（简单/稳定；如要优化可改为 signed URL + ffmpeg 直读）
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(temp_video_path)

        # 2) 逐段切割并上传回 GCS
        output_uris = []
        output_bucket_obj = storage_client.bucket(output_bucket)

        filename = os.path.basename(blob_path)
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

                output_blob_name = (
                    f"{output_path}/{name_without_ext}_segment_{idx+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                )
                output_blob = output_bucket_obj.blob(output_blob_name)
                output_blob.upload_from_filename(temp_output_path)

                output_uris.append(f"gs://{output_bucket}/{output_blob_name}")
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        return {"status": "success", "output_uris": output_uris, "segment_count": len(output_uris)}
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.post("/")
@app.post("/video_split")
def video_split_http():
    """
    Cloud Run HTTP 入口：
    - POST / 或 POST /video_split
    """
    data = request.get_json(silent=True) or {}

    video_uri = data.get("video_uri")
    segments = data.get("segments") or []
    output_bucket = data.get("output_bucket")
    output_path = data.get("output_path", "split_segments")
    output_format = data.get("output_format", "mp4")

    if not video_uri or not segments or not output_bucket:
        return jsonify({"error": "Missing required fields: video_uri, segments, output_bucket"}), 400

    try:
        result = _split_video_gcs_to_gcs(
            video_uri=video_uri,
            segments=segments,
            output_bucket=output_bucket,
            output_path=output_path,
            output_format=output_format,
        )
        return jsonify(result), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ffmpeg failed", "detail": (e.stderr or b"").decode("utf-8", errors="ignore")}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 仅用于本地调试；Cloud Run 生产环境建议用 gunicorn
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

