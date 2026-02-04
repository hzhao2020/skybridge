import os
import subprocess
import tempfile
import json
from urllib.parse import urlparse

try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:  # pragma: no cover
    BlobServiceClient = None


def _get_azure_client(account_name: str):
    """从环境变量或配置获取Azure Blob Storage客户端"""
    if BlobServiceClient is None:
        raise RuntimeError("缺少依赖：azure-storage-blob（请在部署镜像中安装 requirements.txt）。")
    
    # 优先从环境变量获取连接字符串
    connection_string = os.getenv(f"AZURE_STORAGE_CONNECTION_STRING_{account_name.upper()}")
    
    if not connection_string:
        # 尝试从通用环境变量获取
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    
    if not connection_string:
        raise RuntimeError(
            f"缺少Azure存储连接字符串。请设置环境变量 AZURE_STORAGE_CONNECTION_STRING_{account_name.upper()} "
            f"或 AZURE_STORAGE_CONNECTION_STRING"
        )
    
    return BlobServiceClient.from_connection_string(connection_string)


def _get_account_name_from_container(container_name: str) -> str:
    """根据容器名推断账户名（eastasia -> videoea, westus2 -> videowu）"""
    # 从环境变量获取映射，或使用默认映射
    if container_name == "video-ea":
        return os.getenv("AZURE_STORAGE_ACCOUNT_EA", "videoea")
    elif container_name == "video-wu":
        return os.getenv("AZURE_STORAGE_ACCOUNT_WU", "videowu")
    else:
        # 默认尝试使用第一个环境变量
        return os.getenv("AZURE_STORAGE_ACCOUNT", "videoea")


def _split_video_blob_to_blob(*, video_uri: str, segments: list, output_bucket: str, output_path: str, output_format: str):
    """
    Azure Function 版本：输入/输出均为 Azure Blob Storage（azure:// 或 https://）。
    
    成功返回业务结果 dict（包含 output_uris），以便调用侧 HTTP 响应直接解析。
    """
    if not (str(video_uri).startswith("azure://") or str(video_uri).startswith("https://")):
        raise ValueError("Azure Function 版本仅支持 azure:// 或 https:// 输入。")
    if BlobServiceClient is None:
        raise RuntimeError("缺少依赖：azure-storage-blob（请在部署镜像中安装 requirements.txt）。")

    # 解析URI
    parsed = urlparse(video_uri)
    container_name = parsed.netloc
    blob_path = parsed.path.lstrip("/")
    
    # 获取Azure客户端
    account_name = _get_account_name_from_container(container_name)
    blob_service_client = _get_azure_client(account_name)
    
    # 下载视频到临时文件
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        blob_client.download_blob().readinto(open(temp_video_path, "wb"))

        output_uris = []
        filename = os.path.basename(blob_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 获取输出容器客户端
        output_account_name = _get_account_name_from_container(output_bucket)
        output_blob_service_client = _get_azure_client(output_account_name)
        output_container_client = output_blob_service_client.get_container_client(output_bucket)

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
                output_blob_name = (
                    f"{normalized_output_path}/{name_without_ext}_segment_{idx+1}_{int(start_time)}_{int(end_time)}.{output_format}"
                )
                
                # 上传到Azure Blob Storage
                output_blob_client = output_container_client.get_blob_client(output_blob_name)
                with open(temp_output_path, "rb") as data:
                    output_blob_client.upload_blob(data, overwrite=True)
                
                # 构建输出URI（使用https://格式）
                account_url = output_blob_service_client.account_name
                output_uris.append(f"https://{account_url}.blob.core.windows.net/{output_bucket}/{output_blob_name}")
            finally:
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        return {"status": "success", "output_uris": output_uris, "segment_count": len(output_uris)}
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def main(req):
    """
    Azure Function HTTP 入口函数（使用Azure Functions HTTP触发器）。
    
    支持：
    - POST /api/video_split 或 POST /
    - GET /api/healthz 或 GET /healthz (健康检查)
    
    请求体（JSON）：
    {
        "video_uri": "azure://container/path/to/video.mp4",
        "segments": [{"start": 0.0, "end": 10.0}],
        "output_bucket": "container",
        "output_path": "split_segments",
        "output_format": "mp4"
    }
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 处理健康检查
    if req.method == "GET":
        if "healthz" in req.url or req.route_params.get("path", "").endswith("/healthz"):
            return {"status": "ok"}, 200
    
    # 处理视频切割请求
    if req.method != "POST":
        return {"error": "Method not allowed"}, 405
    
    try:
        req_body = req.get_json()
    except Exception:
        req_body = {}
    
    video_uri = req_body.get("video_uri")
    segments = req_body.get("segments") or []
    output_bucket = req_body.get("output_bucket")
    output_path = req_body.get("output_path", "split_segments")
    output_format = req_body.get("output_format", "mp4")

    if not video_uri or not segments or not output_bucket:
        return {"error": "Missing required fields: video_uri, segments, output_bucket"}, 400

    try:
        result = _split_video_blob_to_blob(
            video_uri=video_uri,
            segments=segments,
            output_bucket=output_bucket,
            output_path=output_path,
            output_format=output_format,
        )
        return result, 200
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", errors="ignore")
        return {"error": "ffmpeg failed", "detail": detail[:2000]}, 500
    except Exception as e:
        return {"error": str(e)}, 500
