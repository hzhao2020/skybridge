import os
import json
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
from azure.identity import DefaultAzureCredential

from ops.base import VisualCaptioner, VideoSegmenter
from ops.utils import DataTransmission


class AzureVideoIndexerCaptionImpl(VisualCaptioner):
    """
    使用 Azure Video Indexer 生成视频描述（caption）。

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
    """

    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: Optional[str] = "azure_video_indexer"):
        # provider="azure", region in {"eastasia", "westus2"}, storage_bucket 为 Azure 容器名
        super().__init__(provider, region, storage_bucket, model_name)
        # DataTransmission 默认已在 Operation 基类里初始化为 self.transmitter
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
            # 如果 URL 中已有查询参数，使用 & 连接，否则使用 ? 连接
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
                        from urllib.parse import urlparse
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
        """
        构建 insights 缓存的存储路径。
        
        Args:
            video_uri: 视频 URI
            target_path: 可选的原始目标路径
            
        Returns:
            insights JSON 的缓存路径（相对路径，格式：results/azure_vi_insights/{video_name}/insights.json）
        """
        return self._build_result_path(video_uri, "azure_vi_insights", "insights.json", target_path)

    def _save_insights_to_cache(self, insights_json: Dict[str, Any], video_uri: str, target_path: Optional[str] = None) -> Optional[str]:
        """
        保存 insights JSON 到缓存（本地或云存储）。
        
        Args:
            insights_json: Video Indexer 返回的完整 insights JSON
            video_uri: 视频 URI
            target_path: 可选的原始目标路径
            
        Returns:
            保存的缓存路径 URI（如果成功），否则返回 None
        """
        cache_path = self._get_insights_cache_path(video_uri, target_path)
        
        try:
            # 确定存储位置（本地或云存储）
            if video_uri.startswith("s3://"):
                # S3: 保存到 S3
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
                # GCS: 保存到 GCS
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
                # Azure: 保存到 Azure Blob Storage
                parsed = urlparse(video_uri)
                container = parsed.netloc
                blob_path = cache_path
                
                # 获取 Azure 客户端
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
                # 本地路径: 保存到本地文件系统
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
        """
        从缓存加载 insights JSON。
        
        Args:
            video_uri: 视频 URI
            target_path: 可选的原始目标路径
            
        Returns:
            insights JSON（如果存在），否则返回 None
        """
        cache_path = self._get_insights_cache_path(video_uri, target_path)
        
        try:
            # 确定存储位置（本地或云存储）
            if video_uri.startswith("s3://"):
                # S3: 从 S3 加载
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
                # GCS: 从 GCS 加载
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
                # Azure: 从 Azure Blob Storage 加载
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
                # 本地路径: 从本地文件系统加载
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
        """
        统一的视频处理逻辑：上传到 Video Indexer 并轮询直到处理完成。
        
        Args:
            video_https_url: Video Indexer 可访问的 HTTPS URL
            language: 视频语言（默认 'en-US'）
            **kwargs: 其他参数（name, description 等）
            
        Returns:
            Video Indexer 返回的完整 index JSON
        """
        # 获取访问 token
        access_token = self._get_access_token()

        # 上传/索引视频
        video_id = self._upload_to_video_indexer(
            video_https_url,
            access_token,
            language=language,
            name=kwargs.get("name"),
            description=kwargs.get("description"),
        )
        print(f"    Video accepted by Azure Video Indexer, video_id={video_id}")

        # 轮询直到处理完成
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
            "description": kwargs.get("description", "Video caption by Azure Video Indexer"),
            "videoUrl": video_https_url,
            "language": language,
            # 其它可选参数可以按需扩展：indexingPreset, streamingPreset, etc.
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
    def _extract_caption_from_index(index_json: Dict[str, Any]) -> str:
        """
        从 Video Indexer 的 index JSON 中提取字幕/描述。
        简化策略：
        - 优先使用 insights.transcript 文本；
        - 若不存在，则尝试 insights.keywords, insights.labels 等，做一个粗略摘要。
        """
        insights = index_json.get("videos", [{}])[0].get("insights", {})

        # 1) transcript
        transcript_entries = insights.get("transcript") or []
        if transcript_entries:
            fragments = [t.get("text", "") for t in transcript_entries if t.get("text")]
            caption = " ".join(fragments).strip()
            if caption:
                return caption

        # 2) keywords + labels 兜底
        keywords = [k.get("name", "") for k in insights.get("keywords") or [] if k.get("name")]
        labels = [l.get("name", "") for l in insights.get("labels") or [] if l.get("name")]

        parts = []
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords[:20]))
        if labels:
            parts.append("Labels: " + ", ".join(labels[:20]))

        return ". ".join(parts) if parts else "No caption or transcript found in Azure Video Indexer insights."

    # ---------- 对外主入口 ----------

    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        使用 Azure Video Indexer 生成视频 caption。

        Args:
            video_uri: 视频 URI，可以是本地路径、S3/GCS/Azure URI 或 HTTPS URL。
            **kwargs:
                - target_path: 可选，将视频复制到 Azure 时使用的相对路径前缀。
                - language: 可选，字幕语言（默认 'en-US'）。
                - return_raw_index: 可选，是否在结果中附带完整 index JSON。
                - use_cache: 可选，是否使用缓存（默认 True）。
        """
        print(f"--- [Azure Video Indexer Caption] Region: {self.region} ---")

        target_path = kwargs.get("target_path")
        language = kwargs.get("language", "en-US")
        return_raw = bool(kwargs.get("return_raw_index", False))
        use_cache = kwargs.get("use_cache", True)

        # 1. 检查缓存
        index_json = None
        video_https_url = video_uri
        if use_cache:
            index_json = self._load_insights_from_cache(video_uri, target_path=target_path)
            if index_json:
                print(f"    Using cached insights for caption extraction")
                # 从缓存的 index_json 中提取 video_https_url（如果存在）
                videos = index_json.get("videos", [])
                if videos:
                    video_https_url = videos[0].get("sourceUrl") or video_uri
        
        # 2. 如果没有缓存，则处理视频
        if index_json is None:
            # 确保有 HTTPS URL 供 Video Indexer 访问
            video_https_url = self._ensure_video_https_url(video_uri, target_path=target_path)
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

        # 3. 提取 caption
        caption = self._extract_caption_from_index(index_json)

        # 从 index_json 中提取 video_id（如果存在）
        video_id = None
        videos = index_json.get("videos", [])
        if videos:
            video_id = videos[0].get("id")

        result: Dict[str, Any] = {
            "provider": "azure_video_indexer",
            "region": self.region,
            "model": self.model_name,
            "caption": caption,
            "video_https_url": video_https_url,
            "video_id": video_id,
        }

        if return_raw:
            result["raw_index"] = index_json

        return result


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
        # 复用 AzureVideoIndexerCaptionImpl 的配置和认证逻辑
        self._credential: Optional[DefaultAzureCredential] = None
        self._account_id: Optional[str] = None
        self._account_name: Optional[str] = None
        self._location: Optional[str] = None
        self._vi_access_token: Optional[str] = None

    # ---------- 基础配置获取（复用 AzureVideoIndexerCaptionImpl 的逻辑）----------

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

    # ---------- 视频 URL 处理（复用 AzureVideoIndexerCaptionImpl 的逻辑）----------

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

    # ---------- Insights 缓存处理（复用 AzureVideoIndexerCaptionImpl 的逻辑）----------

    def _get_insights_cache_path(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """构建 insights 缓存的存储路径（复用基类方法）"""
        return self._build_result_path(video_uri, "azure_vi_insights", "insights.json", target_path)

    def _save_insights_to_cache(self, insights_json: Dict[str, Any], video_uri: str, target_path: Optional[str] = None) -> Optional[str]:
        """保存 insights JSON 到缓存（复用 AzureVideoIndexerCaptionImpl 的逻辑）"""
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
        """从缓存加载 insights JSON（复用 AzureVideoIndexerCaptionImpl 的逻辑）"""
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

