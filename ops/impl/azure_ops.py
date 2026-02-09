import os
import json
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
from azure.identity import DefaultAzureCredential

from ops.base import VideoSegmenter, VideoSplitter, VisualCaptioner
from ops.utils import DataTransmission


class AzureVideoIndexerSegmentImpl(VideoSegmenter):
    """
    使用 Azure Video Indexer 进行视频分割（镜头检测）。
    
    设计假设：
    - 在 `config.py` 中配置：
        AZURE_CONFIG = {
            "subscription_id": "...",
            "resource_group": "...",
            "tenant_id": "..."  # 可选
        }
        REGION_CONFIGS = {
            "eastasia": {
                "vi_account_name": "...",
                "vi_account_id": "...",
                "location": "eastasia"
            },
            "westus2": {
                "vi_account_name": "...",
                "vi_account_id": "...",
                "location": "westus2"
            }
        }
    - Video Indexer 账户已在对应 region 创建（eastasia / westus2）。
    - 视频可以是：
        * 任何公网可访问的 HTTPS URL，或
        * 任意云（S3 / GCS / Azure），本类会自动搬运到对应 Azure Storage，
          再转换为 HTTPS Blob URL 交给 Video Indexer。
    - 使用 DefaultAzureCredential 进行身份验证（支持多种认证方式）。
    - 从 Video Indexer 的返回结果中提取 scenes 作为 segments。
    """

    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        # Azure Video Indexer 配置和认证逻辑
        self._credential: Optional[DefaultAzureCredential] = None
        self._account_id: Optional[str] = None
        self._account_name: Optional[str] = None
        self._location: Optional[str] = None
        self._vi_access_token: Optional[str] = None

    # ---------- 基础配置获取 ----------

    def _get_credential(self) -> DefaultAzureCredential:
        """获取 Azure 认证凭据。"""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    def _get_azure_config(self) -> Dict[str, str]:
        """从 config.AZURE_CONFIG 中读取 Azure 全局配置。"""
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法读取 Azure 配置。") from e

        azure_config = getattr(config, "AZURE_CONFIG", None)
        if not isinstance(azure_config, dict):
            raise RuntimeError(
                "请在 config.py 中配置 AZURE_CONFIG = { 'subscription_id': '...', 'resource_group': '...', ... }"
            )

        subscription_id = azure_config.get("subscription_id")
        resource_group = azure_config.get("resource_group")

        if not subscription_id or not resource_group:
            raise RuntimeError(
                "AZURE_CONFIG 中缺少必需的字段：subscription_id 或 resource_group"
            )

        return {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "tenant_id": azure_config.get("tenant_id"),  # 可选
        }

    def _get_region_config(self) -> Dict[str, str]:
        """从 config.REGION_CONFIGS 中读取当前 region 的配置。"""
        if self._account_id is not None and self._account_name is not None and self._location is not None:
            return {
                "account_id": self._account_id,
                "account_name": self._account_name,
                "location": self._location,
            }

        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法读取 Azure Video Indexer 配置。") from e

        region_configs = getattr(config, "REGION_CONFIGS", None)
        if not isinstance(region_configs, dict):
            raise RuntimeError(
                "请在 config.py 中配置 REGION_CONFIGS = { 'region': { 'vi_account_name': '...', 'vi_account_id': '...', 'location': '...' }, ... }"
            )

        region_config = region_configs.get(self.region)
        if not region_config:
            raise RuntimeError(
                f"未在 REGION_CONFIGS 中找到 region='{self.region}' 的配置，"
                f"请在 config.py 中添加。"
            )

        account_name = region_config.get("vi_account_name")
        account_id = region_config.get("vi_account_id")
        location = region_config.get("location")

        if not account_name or not account_id or not location:
            raise RuntimeError(
                f"REGION_CONFIGS['{self.region}'] 中缺少必需的字段：vi_account_name, vi_account_id 或 location"
            )

        self._account_name = account_name
        self._account_id = account_id
        self._location = location

        return {
            "account_id": account_id,
            "account_name": account_name,
            "location": location,
        }

    def _get_location(self) -> str:
        """获取 Video Indexer 的 location 字段。"""
        if self._location is None:
            self._get_region_config()
        return self._location

    def _get_account_id(self) -> str:
        """获取 Video Indexer 的 account_id。"""
        if self._account_id is None:
            self._get_region_config()
        return self._account_id

    def _get_account_name(self) -> str:
        """获取 Video Indexer 的 account_name。"""
        if self._account_name is None:
            self._get_region_config()
        return self._account_name

    def _get_access_token(self, force_refresh: bool = False) -> str:
        """
        使用 ARM API 获取 Video Indexer 的 Access Token。
        
        Args:
            force_refresh: 如果为 True，强制刷新 token（忽略缓存）。
                          默认每次调用都获取新 token，确保 token 始终有效。
        """
        if not force_refresh and self._vi_access_token is not None:
            return self._vi_access_token

        # 获取配置
        azure_config = self._get_azure_config()
        region_config = self._get_region_config()

        subscription_id = azure_config["subscription_id"]
        resource_group = azure_config["resource_group"]
        account_name = region_config["account_name"]

        # 1. 获取 ARM Token
        credential = self._get_credential()
        arm_token = credential.get_token("https://management.azure.com/.default").token

        # 2. 使用 ARM API 换取 VI Token
        url = (
            f"https://management.azure.com/subscriptions/{subscription_id}/"
            f"resourceGroups/{resource_group}/providers/Microsoft.VideoIndexer/"
            f"accounts/{account_name}/generateAccessToken?api-version=2024-01-01"
        )

        headers = {
            "Authorization": f"Bearer {arm_token}",
            "Content-Type": "application/json"
        }
        body = {
            "permissionType": "Contributor",
            "scope": "Account"
        }

        resp = requests.post(url, json=body, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"获取 Azure Video Indexer AccessToken 失败（subscription_id={subscription_id}, "
                f"resource_group={resource_group}, account_name={account_name}）。\n"
                f"HTTP {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()
        access_token = data.get("accessToken")
        if not access_token:
            raise RuntimeError(
                f"ARM API 返回结果中缺少 'accessToken' 字段：{data!r}"
            )

        self._vi_access_token = access_token
        return access_token

    # ---------- 视频 URL 处理 ----------

    @staticmethod
    def _is_https_url(uri: str) -> bool:
        return uri.startswith("https://") or uri.startswith("http://")

    def _azure_blob_https_from_uri(self, azure_uri: str) -> str:
        """
        将 azure://container/blob 或 HTTPS Blob URL 标准化为 HTTPS URL。
        如果配置了 SAS 令牌，会自动附加到 URL 中以便 Video Indexer 访问。
        """
        if self._is_https_url(azure_uri):
            # 已经是 HTTPS，检查是否已有 SAS 令牌
            if "?" in azure_uri and "sig=" in azure_uri:
                # 已有 SAS 令牌，直接返回
                return azure_uri
            
            # 如果是 Azure Blob Storage URL 但没有 SAS 令牌，尝试添加
            if ".blob.core.windows.net" in azure_uri:
                try:
                    import config
                    storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                    
                    # 从 URL 中提取账户名
                    parsed = urlparse(azure_uri)
                    account_name = parsed.netloc.split(".")[0]  # 例如 videoea.blob.core.windows.net -> videoea
                    
                    # 查找对应的 SAS 令牌
                    if account_name in storage_accounts:
                        sas_token = storage_accounts[account_name].get("sas_token")
                        if sas_token:
                            separator = "&" if "?" in azure_uri else "?"
                            return f"{azure_uri}{separator}{sas_token}"
                except Exception:
                    # 如果添加 SAS 令牌失败，继续使用原 URL
                    pass
            
            return azure_uri

        # 解析 azure://container/blob
        if not azure_uri.startswith("azure://"):
            raise ValueError(f"不支持的 Azure URI：{azure_uri}")

        parsed = urlparse(azure_uri)
        container = parsed.netloc
        blob_path = parsed.path.lstrip("/")

        # 从 config.AZURE_STORAGE_ACCOUNTS 中推断 account_name 和 SAS 令牌
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法解析 Azure Storage 账户。") from e

        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
        account_name = None
        sas_token = None
        
        for name, info in storage_accounts.items():
            if info.get("container") == container:
                account_name = name
                sas_token = info.get("sas_token")  # 获取 SAS 令牌
                break

        if not account_name:
            raise RuntimeError(
                f"无法根据容器名 '{container}' 在 AZURE_STORAGE_ACCOUNTS 中找到对应账户，"
                f"请检查 config.py 中的 AZURE_STORAGE_ACCOUNTS 配置。"
            )

        # 构建 HTTPS URL
        https_url = f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}"
        
        # 如果有 SAS 令牌，附加到 URL
        if sas_token:
            separator = "&" if "?" in https_url else "?"
            https_url = f"{https_url}{separator}{sas_token}"
        
        return https_url

    def _ensure_video_https_url(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """
        确保得到一个 Video Indexer 可访问的 HTTPS URL：
        - 如果 input 已是 HTTPS，检查是否需要添加 SAS 令牌（如果是 Azure Blob URL）；
        - 否则搬运到当前 region 对应的 Azure Blob 容器，并转成 HTTPS（自动添加 SAS 令牌）。
        """
        if self._is_https_url(video_uri):
            # 如果是 Azure Blob Storage URL，检查是否需要添加 SAS 令牌
            if ".blob.core.windows.net" in video_uri:
                # 检查是否已有 SAS 令牌
                if "?" not in video_uri or "sig=" not in video_uri:
                    # 没有 SAS 令牌，尝试添加
                    try:
                        import config
                        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                        
                        # 从 URL 中提取账户名和容器名
                        parsed = urlparse(video_uri)
                        account_name = parsed.netloc.split(".")[0]  # 例如 videoea.blob.core.windows.net -> videoea
                        
                        # 查找对应的 SAS 令牌
                        if account_name in storage_accounts:
                            sas_token = storage_accounts[account_name].get("sas_token")
                            if sas_token:
                                separator = "&" if "?" in video_uri else "?"
                                video_uri = f"{video_uri}{separator}{sas_token}"
                    except Exception:
                        # 如果添加 SAS 令牌失败，继续使用原 URL
                        pass
            return video_uri

        # 其它云（或本地） -> Azure Blob
        # 选择当前 region 对应的 Azure Storage 账户和容器
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法确定 Azure Storage 账户。") from e

        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})

        # 简单约定：eastasia -> videoea, westus2 -> videowu
        region_to_account = {
            "eastasia": "videoea",
            "westus2": "videowu",
        }
        account_name = region_to_account.get(self.region)
        if not account_name or account_name not in storage_accounts:
            raise RuntimeError(
                f"未在 AZURE_STORAGE_ACCOUNTS 中找到 region='{self.region}' 对应的账户，"
                f"请确保存在 key '{account_name}'。"
            )

        container = self.storage_bucket  # registry 中传入的容器名，例如 "video-ea" / "video-wu"

        # 使用 DataTransmission 智能搬运到 Azure
        target_uri = self.transmitter.smart_move(
            video_uri,
            target_provider="azure",
            target_bucket=container,
            target_path=target_path,
            azure_account_name=account_name,
        )

        return self._azure_blob_https_from_uri(target_uri)

    # ---------- Insights 缓存处理 ----------

    def _get_insights_cache_path(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """构建 insights 缓存的存储路径"""
        return self._build_result_path(video_uri, "azure_vi_insights", "insights.json", target_path)

    def _save_insights_to_cache(self, insights_json: Dict[str, Any], video_uri: str, target_path: Optional[str] = None) -> Optional[str]:
        """保存 insights JSON 到缓存"""
        # 由于两个类结构相同，可以直接复用相同的方法实现
        cache_path = self._get_insights_cache_path(video_uri, target_path)
        
        try:
            if video_uri.startswith("s3://"):
                import boto3
                parsed = urlparse(video_uri)
                bucket = parsed.netloc
                key = cache_path
                s3_client = boto3.client('s3')
                s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(insights_json, indent=2, ensure_ascii=False),
                    ContentType='application/json'
                )
                cache_uri = f"s3://{bucket}/{key}"
                print(f"    Insights cached to: {cache_uri}")
                return cache_uri
            elif video_uri.startswith("gs://"):
                from google.cloud import storage
                parsed = urlparse(video_uri)
                bucket = parsed.netloc
                blob_path = cache_path
                storage_client = storage.Client()
                bucket_obj = storage_client.bucket(bucket)
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_string(
                    json.dumps(insights_json, indent=2, ensure_ascii=False),
                    content_type='application/json'
                )
                cache_uri = f"gs://{bucket}/{blob_path}"
                print(f"    Insights cached to: {cache_uri}")
                return cache_uri
            elif video_uri.startswith("azure://"):
                parsed = urlparse(video_uri)
                container = parsed.netloc
                blob_path = cache_path
                try:
                    import config
                    storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                    account_name = None
                    for name, info in storage_accounts.items():
                        if info.get("container") == container:
                            account_name = name
                            break
                    if account_name:
                        azure_client = self.transmitter.get_azure_client(account_name)
                        blob_client = azure_client.get_blob_client(container=container, blob=blob_path)
                        blob_client.upload_blob(
                            json.dumps(insights_json, indent=2, ensure_ascii=False),
                            overwrite=True,
                            content_settings={"content_type": "application/json"}
                        )
                        cache_uri = f"azure://{container}/{blob_path}"
                        print(f"    Insights cached to: {cache_uri}")
                        return cache_uri
                except Exception as e:
                    print(f"    Warning: Failed to save insights to Azure cache: {e}")
            else:
                local_cache_path = cache_path
                os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
                with open(local_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(insights_json, f, indent=2, ensure_ascii=False)
                print(f"    Insights cached to: {local_cache_path}")
                return local_cache_path
        except Exception as e:
            print(f"    Warning: Failed to save insights cache: {e}")
        return None

    def _load_insights_from_cache(self, video_uri: str, target_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """从缓存加载 insights JSON"""
        cache_path = self._get_insights_cache_path(video_uri, target_path)
        
        try:
            if video_uri.startswith("s3://"):
                import boto3
                parsed = urlparse(video_uri)
                bucket = parsed.netloc
                key = cache_path
                s3_client = boto3.client('s3')
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    insights_json = json.loads(response['Body'].read().decode('utf-8'))
                    print(f"    Loaded insights from cache: s3://{bucket}/{key}")
                    return insights_json
                except s3_client.exceptions.NoSuchKey:
                    return None
            elif video_uri.startswith("gs://"):
                from google.cloud import storage
                parsed = urlparse(video_uri)
                bucket = parsed.netloc
                blob_path = cache_path
                storage_client = storage.Client()
                bucket_obj = storage_client.bucket(bucket)
                blob = bucket_obj.blob(blob_path)
                if blob.exists():
                    insights_json = json.loads(blob.download_as_text())
                    print(f"    Loaded insights from cache: gs://{bucket}/{blob_path}")
                    return insights_json
                return None
            elif video_uri.startswith("azure://"):
                parsed = urlparse(video_uri)
                container = parsed.netloc
                blob_path = cache_path
                try:
                    import config
                    storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                    account_name = None
                    for name, info in storage_accounts.items():
                        if info.get("container") == container:
                            account_name = name
                            break
                    if account_name:
                        azure_client = self.transmitter.get_azure_client(account_name)
                        blob_client = azure_client.get_blob_client(container=container, blob=blob_path)
                        if blob_client.exists():
                            insights_json = json.loads(blob_client.download_blob().readall().decode('utf-8'))
                            print(f"    Loaded insights from cache: azure://{container}/{blob_path}")
                            return insights_json
                except Exception:
                    pass
                return None
            else:
                local_cache_path = cache_path
                if os.path.exists(local_cache_path):
                    with open(local_cache_path, 'r', encoding='utf-8') as f:
                        insights_json = json.load(f)
                    print(f"    Loaded insights from cache: {local_cache_path}")
                    return insights_json
                return None
        except Exception as e:
            print(f"    Warning: Failed to load insights cache: {e}")
            return None

    def _process_video_with_indexer(self, video_https_url: str, language: str = "en-US", **kwargs) -> Dict[str, Any]:
        """统一的视频处理逻辑：上传到 Video Indexer 并轮询直到处理完成"""
        access_token = self._get_access_token()
        video_id = self._upload_to_video_indexer(
            video_https_url,
            access_token,
            language=language,
            name=kwargs.get("name"),
            description=kwargs.get("description"),
        )
        print(f"    Video accepted by Azure Video Indexer, video_id={video_id}")
        index_json = self._poll_video_index(video_id, access_token, language=language)
        return index_json

    # ---------- 与 Video Indexer 交互 ----------

    def _upload_to_video_indexer(self, video_https_url: str, access_token: str, **kwargs) -> str:
        """
        调用 Video Indexer 的 Upload by URL 接口，返回 video_id。
        """
        account_id = self._account_id or self._get_account_id()
        location = self._get_location()

        name = kwargs.get("name") or os.path.basename(urlparse(video_https_url).path) or "video"
        language = kwargs.get("language", "en-US")

        url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos"
        params = {
            "accessToken": access_token,
            "name": name,
            "description": kwargs.get("description", "Video segment detection by Azure Video Indexer"),
            "videoUrl": video_https_url,
            "language": language,
        }

        resp = requests.post(url, params=params, timeout=600)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"调用 Azure Video Indexer /Videos 接口失败。\n"
                f"HTTP {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()
        video_id = data.get("id")
        if not video_id:
            raise RuntimeError(f"Video Indexer /Videos 返回结果中缺少 'id' 字段：{data!r}")
        return video_id

    def _poll_video_index(self, video_id: str, access_token: str, language: str = "en-US", timeout_sec: int = 900) -> Dict[str, Any]:
        """
        轮询 Video Indexer，直到处理完成或超时，返回 index JSON。
        注意：每次轮询时都会刷新 token，确保长时间处理时 token 始终有效。
        """
        account_id = self._account_id or self._get_account_id()
        location = self._get_location()

        url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos/{video_id}/Index"

        start = time.time()
        while True:
            # 每次轮询时刷新 token，确保 token 始终有效
            current_token = self._get_access_token(force_refresh=True)
            params = {"accessToken": current_token, "language": language}

            resp = requests.get(url, params=params, timeout=60)
            try:
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(
                    f"轮询 Azure Video Indexer Index 接口失败。\n"
                    f"HTTP {resp.status_code}: {resp.text}"
                ) from e

            data = resp.json()
            state = data.get("state")
            
            # 显示处理进度
            if state == "Processing":
                progress = data.get("videos", [{}])[0].get("processingProgress", "0%")
                print(f"    [Azure Video Indexer] state={state}, progress={progress}", end="\r")
            else:
                print(f"    [Azure Video Indexer] state={state}")

            if state == "Processed":
                print("\n处理完成！")
                return data
            if state in ("Failed", "Error"):
                raise RuntimeError(f"Azure Video Indexer 处理失败：state={state}, details={data!r}")

            if time.time() - start > timeout_sec:
                raise TimeoutError(f"等待 Azure Video Indexer 处理超时（>{timeout_sec} 秒）。最后状态：{state}")

            time.sleep(10)

    @staticmethod
    def _extract_segments_from_index(index_json: Dict[str, Any]) -> List[Dict[str, float]]:
        """
        从 Video Indexer 的 index JSON 中提取 segments（scenes）。
        
        Args:
            index_json: Video Indexer 返回的完整 index JSON
        
        Returns:
            segments 列表，每个 segment 包含 "start" 和 "end" 字段（秒数）
        """
        segments = []
        
        videos = index_json.get("videos", [])
        if not videos:
            return segments
        
        video_data = videos[0]
        insights = video_data.get("insights", {})
        
        # 从 insights.scenes 中提取 segments
        scenes = insights.get("scenes", [])
        for scene in scenes:
            instances = scene.get("instances", [])
            for instance in instances:
                start_time = instance.get("start", "0:00:00")
                end_time = instance.get("end", "0:00:00")
                
                # 将时间字符串转换为秒数
                # 格式通常是 "HH:MM:SS" 或 "HH:MM:SS.mmm"
                def time_to_seconds(time_str: str) -> float:
                    """将时间字符串转换为秒数"""
                    if not time_str:
                        return 0.0
                    parts = time_str.split(":")
                    if len(parts) == 3:
                        hours = float(parts[0])
                        minutes = float(parts[1])
                        # 处理秒数部分（可能包含毫秒）
                        seconds_part = parts[2]
                        seconds = float(seconds_part)
                        return hours * 3600 + minutes * 60 + seconds
                    return 0.0
                
                start_seconds = time_to_seconds(start_time)
                end_seconds = time_to_seconds(end_time)
                
                if end_seconds > start_seconds:
                    segments.append({
                        "start": start_seconds,
                        "end": end_seconds
                    })
        
        return segments

    # ---------- 对外主入口 ----------

    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        使用 Azure Video Indexer 进行视频分割（镜头检测）。

        Args:
            video_uri: 视频 URI，可以是本地路径、S3/GCS/Azure URI 或 HTTPS URL。
            **kwargs:
                - target_path: 可选，将视频复制到 Azure 时使用的相对路径前缀。
                - language: 可选，视频语言（默认 'en-US'）。
                - use_cache: 可选，是否使用缓存（默认 True）。
        """
        print(f"--- [Azure Video Indexer Segment] Region: {self.region} ---")

        target_path = kwargs.get("target_path")
        language = kwargs.get("language", "en-US")
        use_cache = kwargs.get("use_cache", True)

        # 1. 检查缓存
        index_json = None
        video_https_url = video_uri
        if use_cache:
            index_json = self._load_insights_from_cache(video_uri, target_path=target_path)
            if index_json:
                print(f"    Using cached insights for segment extraction")
                # 从缓存的 index_json 中提取 video_https_url（如果存在）
                videos = index_json.get("videos", [])
                if videos:
                    video_https_url = videos[0].get("sourceUrl") or video_uri
        
        # 2. 如果没有缓存，则处理视频
        if index_json is None:
            # 判断 video_uri 类型，并进行 smart move 到对应的 bucket
            print(f"    Input video URI: {video_uri}")
            
            # 确定目标 Azure Storage 账户和容器
            try:
                import config
            except ImportError as e:
                raise RuntimeError("config.py 未找到，无法确定 Azure Storage 账户。") from e

            storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
            
            # 简单约定：eastasia -> videoea, westus2 -> videowu
            region_to_account = {
                "eastasia": "videoea",
                "westus2": "videowu",
            }
            account_name = region_to_account.get(self.region)
            if not account_name or account_name not in storage_accounts:
                raise RuntimeError(
                    f"未在 AZURE_STORAGE_ACCOUNTS 中找到 region='{self.region}' 对应的账户，"
                    f"请确保存在 key '{account_name}'。"
                )

            container = self.storage_bucket  # registry 中传入的容器名，例如 "video-ea" / "video-wu"

            # 进行 smart move：将视频移动到对应的 Azure bucket
            target_uri = self.transmitter.smart_move(
                video_uri,
                target_provider="azure",
                target_bucket=container,
                target_path=target_path,
                azure_account_name=account_name,
            )
            print(f"    Video moved to Azure bucket: {target_uri}")

            # 将移动后的 URI 转换为 HTTPS URL（供 Video Indexer 访问）
            video_https_url = self._azure_blob_https_from_uri(target_uri)
            print(f"    Video URL for Azure Video Indexer: {video_https_url}")

            # 处理视频（上传并轮询）
            index_json = self._process_video_with_indexer(
                video_https_url,
                language=language,
                name=kwargs.get("name"),
                description=kwargs.get("description"),
            )

            # 保存到缓存
            if use_cache:
                self._save_insights_to_cache(index_json, video_uri, target_path=target_path)

        # 3. 提取 segments
        segments = self._extract_segments_from_index(index_json)
        print(f"    Found {len(segments)} segments")

        # 从 index_json 中提取 video_id（如果存在）
        video_id = None
        videos = index_json.get("videos", [])
        if videos:
            video_id = videos[0].get("id")

        result: Dict[str, Any] = {
            "provider": "azure_video_indexer",
            "region": self.region,
            "segments": segments,
            "source_used": video_https_url,
            "video_id": video_id,
        }

        return result


class AzureContentUnderstandingCaptionImpl(VisualCaptioner):
    """
    使用 Azure AI Content Understanding (Foundry Tools) 进行视频字幕生成。
    
    设计假设：
    - 在 `config.py` 中配置：
        AZURE_CONTENT_UNDERSTANDING_CONFIG = {
            "eastasia": {
                "endpoint": "https://<resource-name>.cognitiveservices.azure.com",
                "resource_name": "<resource-name>",
            },
            "westus2": {
                "endpoint": "https://<resource-name>.cognitiveservices.azure.com",
                "resource_name": "<resource-name>",
            }
        }
    - 视频可以是：
        * 任何公网可访问的 HTTPS URL，或
        * 任意云（S3 / GCS / Azure），本类会自动搬运到对应 Azure Storage，
          再转换为 HTTPS Blob URL 交给 Content Understanding。
    - 使用 DefaultAzureCredential 进行身份验证（支持多种认证方式）。
    - 使用预构建的视频分析器 `prebuilt-videoAnalysis` 生成字幕。
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str):
        super().__init__(provider, region, storage_bucket)
        self._credential: Optional[DefaultAzureCredential] = None
        self._endpoint: Optional[str] = None
        self._resource_name: Optional[str] = None
        self._api_key: Optional[str] = None
    
    def _get_credential(self) -> DefaultAzureCredential:
        """获取 Azure 认证凭据。"""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential
    
    def _get_content_understanding_config(self) -> Dict[str, str]:
        """从 config.AZURE_CONTENT_UNDERSTANDING_CONFIG 中读取当前 region 的配置。"""
        if self._endpoint is not None and self._resource_name is not None:
            result = {
                "endpoint": self._endpoint,
                "resource_name": self._resource_name,
            }
            if self._api_key is not None:
                result["api_key"] = self._api_key
            return result
        
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法读取 Azure Content Understanding 配置。") from e
        
        cu_configs = getattr(config, "AZURE_CONTENT_UNDERSTANDING_CONFIG", None)
        if not isinstance(cu_configs, dict):
            raise RuntimeError(
                "请在 config.py 中配置 AZURE_CONTENT_UNDERSTANDING_CONFIG = { "
                "'region': { 'endpoint': '...', 'resource_name': '...', 'api_key': '...' }, ... }"
            )
        
        region_config = cu_configs.get(self.region)
        if not region_config:
            raise RuntimeError(
                f"未在 AZURE_CONTENT_UNDERSTANDING_CONFIG 中找到 region='{self.region}' 的配置，"
                f"请在 config.py 中添加。"
            )
        
        endpoint = region_config.get("endpoint")
        resource_name = region_config.get("resource_name")
        api_key = region_config.get("api_key")
        
        if not endpoint:
            raise RuntimeError(
                f"AZURE_CONTENT_UNDERSTANDING_CONFIG['{self.region}'] 中缺少必需的字段：endpoint"
            )
        
        # resource_name 可以从 endpoint 中提取，如果没有提供
        if not resource_name:
            # 从 endpoint 中提取 resource_name
            # 格式：https://<resource-name>.cognitiveservices.azure.com
            parsed = urlparse(endpoint)
            resource_name = parsed.netloc.split(".")[0]
        
        self._endpoint = endpoint.rstrip("/")  # 移除末尾的斜杠
        self._resource_name = resource_name
        self._api_key = api_key
        
        result = {
            "endpoint": self._endpoint,
            "resource_name": self._resource_name,
        }
        if api_key:
            result["api_key"] = api_key
        
        return result
    
    def _get_endpoint(self) -> str:
        """获取 Content Understanding 的 endpoint。"""
        if self._endpoint is None:
            self._get_content_understanding_config()
        return self._endpoint
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        获取认证头信息。
        优先使用 API Key，如果没有配置则使用 DefaultAzureCredential。
        """
        config_data = self._get_content_understanding_config()
        
        # 如果配置了 API Key，优先使用
        api_key = config_data.get("api_key")
        if api_key:
            return {
                "Ocp-Apim-Subscription-Key": api_key
            }
        
        # 否则使用 DefaultAzureCredential
        credential = self._get_credential()
        scope = "https://cognitiveservices.azure.com/.default"
        token = credential.get_token(scope).token
        return {
            "Authorization": f"Bearer {token}"
        }
    
    def _is_https_url(self, uri: str) -> bool:
        """检查 URI 是否为 HTTPS URL。"""
        return uri.startswith("https://") or uri.startswith("http://")
    
    def _azure_blob_https_from_uri(self, azure_uri: str) -> str:
        """
        将 azure://container/blob 或 HTTPS Blob URL 标准化为 HTTPS URL。
        如果配置了 SAS 令牌，会自动附加到 URL 中以便 Content Understanding 访问。
        """
        if self._is_https_url(azure_uri):
            # 已经是 HTTPS，检查是否已有 SAS 令牌
            if "?" in azure_uri and "sig=" in azure_uri:
                return azure_uri
            
            # 如果是 Azure Blob Storage URL 但没有 SAS 令牌，尝试添加
            if ".blob.core.windows.net" in azure_uri:
                try:
                    import config
                    storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                    
                    parsed = urlparse(azure_uri)
                    account_name = parsed.netloc.split(".")[0]
                    
                    if account_name in storage_accounts:
                        sas_token = storage_accounts[account_name].get("sas_token")
                        if sas_token:
                            separator = "&" if "?" in azure_uri else "?"
                            return f"{azure_uri}{separator}{sas_token}"
                except Exception:
                    pass
            
            return azure_uri
        
        # 解析 azure://container/blob
        if not azure_uri.startswith("azure://"):
            raise ValueError(f"不支持的 Azure URI：{azure_uri}")
        
        parsed = urlparse(azure_uri)
        container = parsed.netloc
        blob_path = parsed.path.lstrip("/")
        
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法解析 Azure Storage 账户。") from e
        
        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
        account_name = None
        sas_token = None
        
        for name, info in storage_accounts.items():
            if info.get("container") == container:
                account_name = name
                sas_token = info.get("sas_token")
                break
        
        if not account_name:
            raise RuntimeError(
                f"无法根据容器名 '{container}' 在 AZURE_STORAGE_ACCOUNTS 中找到对应账户，"
                f"请检查 config.py 中的 AZURE_STORAGE_ACCOUNTS 配置。"
            )
        
        https_url = f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}"
        
        if sas_token:
            separator = "&" if "?" in https_url else "?"
            https_url = f"{https_url}{separator}{sas_token}"
        
        return https_url
    
    def _ensure_video_https_url(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """
        确保得到一个 Content Understanding 可访问的 HTTPS URL：
        - 如果 input 已是 HTTPS，检查是否需要添加 SAS 令牌（如果是 Azure Blob URL）；
        - 否则搬运到当前 region 对应的 Azure Blob 容器，并转成 HTTPS（自动添加 SAS 令牌）。
        """
        if self._is_https_url(video_uri):
            if ".blob.core.windows.net" in video_uri:
                if "?" not in video_uri or "sig=" not in video_uri:
                    try:
                        import config
                        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
                        
                        parsed = urlparse(video_uri)
                        account_name = parsed.netloc.split(".")[0]
                        
                        if account_name in storage_accounts:
                            sas_token = storage_accounts[account_name].get("sas_token")
                            if sas_token:
                                separator = "&" if "?" in video_uri else "?"
                                video_uri = f"{video_uri}{separator}{sas_token}"
                    except Exception:
                        pass
            return video_uri
        
        # 其它云（或本地） -> Azure Blob
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法确定 Azure Storage 账户。") from e
        
        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
        
        region_to_account = {
            "eastasia": "videoea",
            "westus2": "videowu",
        }
        account_name = region_to_account.get(self.region)
        if not account_name or account_name not in storage_accounts:
            raise RuntimeError(
                f"未在 AZURE_STORAGE_ACCOUNTS 中找到 region='{self.region}' 对应的账户，"
                f"请确保存在 key '{account_name}'。"
            )
        
        container = self.storage_bucket
        
        target_uri = self.transmitter.smart_move(
            video_uri,
            target_provider="azure",
            target_bucket=container,
            target_path=target_path,
            azure_account_name=account_name,
        )
        
        return self._azure_blob_https_from_uri(target_uri)
    
    def _submit_analysis(self, video_url: str) -> str:
        """
        提交视频分析任务，返回 operation location URL。
        
        Args:
            video_url: 视频的 HTTPS URL
            
        Returns:
            operation_location: 用于查询分析结果的 URL
        """
        endpoint = self._get_endpoint()
        auth_headers = self._get_auth_headers()
        
        # API 端点
        # 注意：某些区域可能使用不同的 endpoint 格式
        # cognitiveservices.azure.com 或 services.ai.azure.com
        api_url = f"{endpoint}/contentunderstanding/analyzers/prebuilt-videoAnalysis:analyze"
        
        print(f"    API URL: {api_url}")
        
        # 根据区域选择 API 版本
        # 某些区域可能不支持 GA 版本，需要使用 preview 版本
        # 尝试先使用 GA 版本，如果失败再尝试 preview 版本
        # 注意：eastasia 区域可能不支持某些版本，需要尝试多个版本
        api_versions = [
            "2025-11-01",           # GA 版本
            "2025-05-01-preview",   # Preview 版本
            "2024-12-01-preview",   # 更早的 preview 版本
            "2024-11-01-preview",   # 更早的 preview 版本（如果存在）
            "2024-10-01-preview",   # 更早的 preview 版本（如果存在）
        ]
        
        # 从配置中获取首选 API 版本（如果配置了）
        config_data = self._get_content_understanding_config()
        preferred_api_version = config_data.get("api_version")
        if preferred_api_version:
            # 将首选版本移到最前面
            if preferred_api_version in api_versions:
                api_versions.remove(preferred_api_version)
            api_versions.insert(0, preferred_api_version)
        
        last_error = None
        for api_version in api_versions:
            params = {"api-version": api_version}
        
            # 请求体
            request_body = {
                "inputs": [
                    {
                        "url": video_url
                    }
                ]
            }
            
            headers = {
                **auth_headers,
                "Content-Type": "application/json"
            }
            
            print(f"    Submitting video analysis request to Azure Content Understanding...")
            print(f"    Video URL: {video_url}")
            print(f"    Trying API version: {api_version}")
            
            resp = requests.post(api_url, json=request_body, headers=headers, params=params, timeout=60)
            
            # 如果成功，返回结果
            if resp.status_code == 202:  # Accepted
                operation_location = resp.headers.get("Operation-Location")
                if not operation_location:
                    raise RuntimeError(
                        f"Azure Content Understanding 响应中缺少 'Operation-Location' 头：{resp.headers}"
                    )
                print(f"    ✓ Successfully submitted with API version: {api_version}")
                return operation_location
            
            # 如果是 404，检查错误信息
            if resp.status_code == 404:
                error_data = {}
                error_message = ""
                try:
                    if resp.content:
                        error_data = resp.json()
                        error_message = error_data.get("error", {}).get("message", "")
                except:
                    error_message = resp.text[:200] if resp.text else ""
                
                # 打印详细的错误信息
                print(f"    ✗ API version {api_version} failed:")
                print(f"      Status: {resp.status_code}")
                print(f"      Error: {error_message or resp.text[:200]}")
                
                # 如果错误信息提到 GA API 不支持或区域不支持，尝试下一个版本
                if ("GA API is not supported" in error_message or 
                    "not supported in this region" in error_message or
                    "not supported" in error_message.lower()):
                    print(f"    ⚠ API version {api_version} not supported, trying next version...")
                    last_error = resp
                    continue
                else:
                    # 其他类型的 404 错误，可能是 endpoint 或路径问题
                    print(f"    ⚠ 404 error with API version {api_version}, but error message doesn't indicate version issue")
                    print(f"      This might be an endpoint or path issue. Full response: {resp.text[:500]}")
                    last_error = resp
                    # 继续尝试下一个版本，但也可能是 endpoint 配置问题
                    continue
            
            # 其他错误，抛出异常
            try:
                resp.raise_for_status()
            except Exception as e:
                last_error = resp
                # 如果是最后一个版本，抛出异常
                if api_version == api_versions[-1]:
                    raise RuntimeError(
                        f"提交 Azure Content Understanding 分析请求失败。\n"
                        f"HTTP {resp.status_code}: {resp.text}\n"
                        f"已尝试所有 API 版本: {', '.join(api_versions)}"
                    ) from e
                # 否则继续尝试下一个版本
                continue
        
        # 如果所有版本都失败了
        if last_error:
            error_details = ""
            try:
                if last_error.content:
                    error_json = last_error.json()
                    error_details = f"\n错误详情: {json.dumps(error_json, indent=2, ensure_ascii=False)}"
            except:
                error_details = f"\n错误响应: {last_error.text[:500]}"
            
            raise RuntimeError(
                f"提交 Azure Content Understanding 分析请求失败。\n"
                f"HTTP {last_error.status_code}\n"
                f"Endpoint: {api_url}\n"
                f"Region: {self.region}\n"
                f"已尝试所有 API 版本: {', '.join(api_versions)}\n"
                f"{error_details}\n"
                f"\n可能的原因：\n"
                f"1. 该区域不支持 Azure Content Understanding API\n"
                f"2. Endpoint 配置不正确（当前: {endpoint}）\n"
                f"3. 资源未正确配置或未启用 Content Understanding 功能\n"
                f"\n建议：\n"
                f"- 检查 Azure Portal 中该资源是否支持 Content Understanding\n"
                f"- 尝试使用 westus2 区域（如果配置了）\n"
                f"- 确认 endpoint 格式是否正确（可能是 services.ai.azure.com 而不是 cognitiveservices.azure.com）"
            )
        else:
            raise RuntimeError("无法提交分析请求：未知错误")
    
    def _poll_analysis_result(self, operation_location: str, timeout_sec: int = 1800) -> Dict[str, Any]:
        """
        轮询分析结果，直到处理完成或超时。
        
        Args:
            operation_location: 操作位置 URL
            timeout_sec: 超时时间（秒），默认30分钟
            
        Returns:
            分析结果的 JSON 字典
        """
        auth_headers = self._get_auth_headers()
        
        headers = auth_headers.copy()
        
        start = time.time()
        while True:
            resp = requests.get(operation_location, headers=headers, timeout=60)
            try:
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(
                    f"轮询 Azure Content Understanding 结果失败。\n"
                    f"HTTP {resp.status_code}: {resp.text}"
                ) from e
            
            data = resp.json()
            status = data.get("status", "unknown")
            
            if status == "Processing":
                progress = data.get("progress", 0)
                print(f"    [Azure Content Understanding] status={status}, progress={progress}%", end="\r")
            else:
                print(f"    [Azure Content Understanding] status={status}")
            
            if status == "Succeeded":
                print("\n处理完成！")
                return data.get("result", {})
            
            if status in ("Failed", "Error"):
                error_info = data.get("error", {})
                error_message = error_info.get("message", "Unknown error")
                raise RuntimeError(
                    f"Azure Content Understanding 处理失败：status={status}, error={error_message}"
                )
            
            if time.time() - start > timeout_sec:
                raise TimeoutError(
                    f"等待 Azure Content Understanding 处理超时（>{timeout_sec} 秒）。最后状态：{status}"
                )
            
            time.sleep(10)
    
    def _extract_caption_from_result(self, result: Dict[str, Any]) -> str:
        """
        从分析结果中提取字幕文本。
        
        Azure Content Understanding 返回的结果包含：
        - transcript: WEBVTT 格式的字幕
        - description: 自然语言描述
        
        优先使用 description，如果没有则从 transcript 中提取文本。
        """
        # 尝试获取 description（自然语言描述）
        description = result.get("description")
        if description:
            return description
        
        # 如果没有 description，尝试从 transcript 中提取
        transcript = result.get("transcript")
        if transcript:
            # transcript 可能是 WEBVTT 格式的字符串
            # 简单提取：移除 WEBVTT 标记和时间戳，只保留文本
            if isinstance(transcript, str):
                lines = transcript.split("\n")
                text_lines = []
                for line in lines:
                    line = line.strip()
                    # 跳过 WEBVTT 头部、时间戳行、空行
                    if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
                        continue
                    text_lines.append(line)
                return " ".join(text_lines)
            return str(transcript)
        
        # 如果都没有，返回默认消息
        return "No caption generated from Azure Content Understanding."
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        使用 Azure AI Content Understanding 对视频生成字幕。
        
        Args:
            video_uri: 视频 URI，可以是本地路径、S3/GCS/Azure URI 或 HTTPS URL。
            **kwargs:
                - target_path: 可选，将视频复制到 Azure 时使用的相对路径前缀。
                - start_time: 可选，视频片段的开始时间（秒）。
                - end_time: 可选，视频片段的结束时间（秒）。
        
        Returns:
            包含字幕结果的字典
        """
        print(f"--- [Azure Content Understanding Caption] Region: {self.region} ---")
        
        target_path = kwargs.get("target_path")
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        
        # 1. 准备视频 URL
        print(f"    Input video URI: {video_uri}")
        
        # 如果指定了时间范围，需要先提取视频片段
        # 注意：Azure Content Understanding 可能不支持直接指定时间范围
        # 如果需要片段，可以先提取片段再上传
        video_https_url = self._ensure_video_https_url(video_uri, target_path=target_path)
        print(f"    Video URL for Azure Content Understanding: {video_https_url}")
        
        # 2. 提交分析任务
        operation_location = self._submit_analysis(video_https_url)
        
        # 3. 轮询结果
        result = self._poll_analysis_result(operation_location)
        
        # 4. 提取字幕
        caption = self._extract_caption_from_result(result)
        
        print("\n" + "=" * 80)
        print("=== [Azure Content Understanding Caption] Result ===")
        print("=" * 80)
        print(caption)
        print("=" * 80 + "\n")
        
        return {
            "provider": "azure_content_understanding",
            "region": self.region,
            "caption": caption,
            "source_used": video_https_url,
            "start_time": start_time,
            "end_time": end_time,
            "raw_result": result,  # 保留原始结果以便后续处理
        }
