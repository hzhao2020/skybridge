import os
from azure.storage.blob import BlobServiceClient, ContentSettings

def upload_video_to_azure(connection_string, container_name, local_file_path):
    try:
        # 1. 初始化 BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # 2. 获取或创建容器
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            print(f"创建了新容器: {container_name}")

        # 3. 获取文件名称作为 Blob 名称
        blob_name = os.path.basename(local_file_path)
        blob_client = container_client.get_blob_client(blob_name)

        print(f"正在上传视频: {blob_name} ...")

        # 4. 设置内容类型（MIME type），方便浏览器或 AI 服务直接识别
        content_settings = ContentSettings(content_type='video/mp4')

        # 5. 上传文件
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)

        print(f"上传成功！Blob URL: {blob_client.url}")

    except Exception as ex:
        print(f"上传过程中出现错误: {ex}")

# --- 配置参数 ---
# 建议通过环境变量获取敏感信息，不要直接硬编码
MY_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=videoea;AccountKey=lzXwtXVrAvde7Kh+GVT8Eo9sq+sWseNI8LVwvOZhh9z9Zqor33p1VHtx4eRanDK4hTIR/tt/yB4F+AStWOm4mQ==;EndpointSuffix=core.windows.net"
MY_CONTAINER = "video-ea"
MY_LOCAL_VIDEO = "/home/heng/Documents/skybridge/datasets/EgoSchema/videos_sampled/0a8b2c9d-b54c-4811-acf3-5977895d2445.mp4"

upload_video_to_azure(MY_CONNECTION_STRING, MY_CONTAINER, MY_LOCAL_VIDEO)