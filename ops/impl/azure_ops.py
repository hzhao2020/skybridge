import os
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import requests

from ops.base import VisualCaptioner
from ops.utils import DataTransmission


class AzureVideoIndexerCaptionImpl(VisualCaptioner):
    """
    使用 Azure Video Indexer 生成视频描述（caption）。

    设计假设：
    - 在 `config.py` 中配置（示例）：
        AZURE_VIDEO_INDEXER_KEYS = {
            "eastasia": "your-subscription-key-ea",
            "westus2": "your-subscription-key-wu",
        }
    - Video Indexer 账户已在对应 region 创建（eastasia / westus2）。
    - 视频可以是：
        * 任何公网可访问的 HTTPS URL，或
        * 任意云（S3 / GCS / Azure），本类会自动搬运到对应 Azure Storage，
          再转换为 HTTPS Blob URL 交给 Video Indexer。
    """

    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: Optional[str] = "azure_video_indexer"):
        # provider="azure", region in {"eastasia", "westus2"}, storage_bucket 为 Azure 容器名
        super().__init__(provider, region, storage_bucket, model_name)
        # DataTransmission 默认已在 Operation 基类里初始化为 self.transmitter
        self._subscription_key: Optional[str] = None
        self._account_id: Optional[str] = None

    # ---------- 基础配置获取 ----------

    def _get_subscription_key(self) -> str:
        """从 config.AZURE_VIDEO_INDEXER_KEYS 中读取当前 region 的 subscription key。"""
        if self._subscription_key is not None:
            return self._subscription_key

        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法读取 Azure Video Indexer 配置。") from e

        keys = getattr(config, "AZURE_VIDEO_INDEXER_KEYS", None)
        if not isinstance(keys, dict):
            raise RuntimeError(
                "请在 config.py 中配置 AZURE_VIDEO_INDEXER_KEYS = { 'location': { 'subscription_key': '...', 'account_id': '...' }, ... }"
            )

        region_config = keys.get(self.region)
        if not region_config:
            raise RuntimeError(
                f"未在 AZURE_VIDEO_INDEXER_KEYS 中找到 region='{self.region}' 的配置，"
                f"请在 config.py 中添加。"
            )

        # 支持新格式：字典包含 subscription_key 和 account_id
        if isinstance(region_config, dict):
            key = region_config.get("subscription_key")
            if not key:
                raise RuntimeError(
                    f"AZURE_VIDEO_INDEXER_KEYS['{self.region}'] 中缺少 'subscription_key' 字段"
                )
        # 兼容旧格式：直接是字符串
        elif isinstance(region_config, str):
            key = region_config
        else:
            raise RuntimeError(
                f"AZURE_VIDEO_INDEXER_KEYS['{self.region}'] 格式不正确，应为字典或字符串"
            )

        self._subscription_key = key
        return key

    def _get_location(self) -> str:
        """
        Video Indexer 的 location 字段。
        这里直接使用 region（eastasia / westus2），与用户在门户创建资源时的区域一致。
        """
        return self.region

    def _get_account_id(self) -> str:
        """
        从 config.AZURE_VIDEO_INDEXER_KEYS 中读取 account_id，如果配置中没有则通过 API 获取。
        """
        if self._account_id is not None:
            return self._account_id

        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法读取 Azure Video Indexer 配置。") from e

        keys = getattr(config, "AZURE_VIDEO_INDEXER_KEYS", None)
        if isinstance(keys, dict):
            region_config = keys.get(self.region)
            # 如果配置是字典格式且包含 account_id，直接使用
            if isinstance(region_config, dict) and "account_id" in region_config:
                account_id = region_config.get("account_id")
                if account_id:
                    self._account_id = account_id
                    return account_id

        # 如果配置中没有 account_id，则通过 API 获取
        subscription_key = self._get_subscription_key()
        location = self._get_location()

        url = f"https://api.videoindexer.ai/auth/{location}/Accounts"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"allowEdit": "true"}

        resp = requests.get(url, headers=headers, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"获取 Azure Video Indexer 账户列表失败（location={location}）。"
                f"HTTP {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()
        if not data or not isinstance(data, list):
            raise RuntimeError(f"Azure Video Indexer 未返回有效账户列表：{data!r}")

        account_id = data[0].get("id")
        if not account_id:
            raise RuntimeError(f"Azure Video Indexer 返回的账户信息中缺少 'id' 字段：{data[0]!r}")

        self._account_id = account_id
        return account_id

    def _get_access_token(self) -> str:
        """为指定 account 获取一次性访问 token。"""
        subscription_key = self._get_subscription_key()
        account_id = self._get_account_id()
        location = self._get_location()

        url = f"https://api.videoindexer.ai/auth/{location}/Accounts/{account_id}/AccessToken"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"allowEdit": "true"}

        resp = requests.get(url, headers=headers, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"获取 Azure Video Indexer AccessToken 失败（location={location}, account={account_id}）。"
                f"HTTP {resp.status_code}: {resp.text}"
            ) from e

        # token 可能是纯文本，也可能是 JSON 包装，做一下兼容处理
        try:
            data = resp.json()
            if isinstance(data, str):
                return data
            if isinstance(data, dict) and "accessToken" in data:
                return data["accessToken"]
        except ValueError:
            pass

        return resp.text.strip().strip('"')

    # ---------- 视频 URL 处理 ----------

    @staticmethod
    def _is_https_url(uri: str) -> bool:
        return uri.startswith("https://") or uri.startswith("http://")

    def _azure_blob_https_from_uri(self, azure_uri: str) -> str:
        """
        将 azure://container/blob 或 HTTPS Blob URL 标准化为 HTTPS URL。
        """
        if self._is_https_url(azure_uri):
            # 已经是 HTTPS，直接返回
            return azure_uri

        # 解析 azure://container/blob
        if not azure_uri.startswith("azure://"):
            raise ValueError(f"不支持的 Azure URI：{azure_uri}")

        parsed = urlparse(azure_uri)
        container = parsed.netloc
        blob_path = parsed.path.lstrip("/")

        # 从 config.AZURE_STORAGE_ACCOUNTS 中推断 account_name
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法解析 Azure Storage 账户。") from e

        storage_accounts = getattr(config, "AZURE_STORAGE_ACCOUNTS", {})
        account_name = None
        for name, info in storage_accounts.items():
            if info.get("container") == container:
                account_name = name
                break

        if not account_name:
            raise RuntimeError(
                f"无法根据容器名 '{container}' 在 AZURE_STORAGE_ACCOUNTS 中找到对应账户，"
                f"请检查 config.py 中的 AZURE_STORAGE_ACCOUNTS 配置。"
            )

        return f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}"

    def _ensure_video_https_url(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """
        确保得到一个 Video Indexer 可访问的 HTTPS URL：
        - 如果 input 已是 HTTPS，直接返回；
        - 否则搬运到当前 region 对应的 Azure Blob 容器，并转成 HTTPS。
        """
        if self._is_https_url(video_uri):
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
        """
        account_id = self._account_id or self._get_account_id()
        location = self._get_location()

        url = f"https://api.videoindexer.ai/{location}/Accounts/{account_id}/Videos/{video_id}/Index"
        params = {"accessToken": access_token, "language": language}

        start = time.time()
        while True:
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
            print(f"    [Azure Video Indexer] state={state}")

            if state == "Processed":
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
        """
        print(f"--- [Azure Video Indexer Caption] Region: {self.region} ---")

        target_path = kwargs.get("target_path")
        language = kwargs.get("language", "en-US")
        return_raw = bool(kwargs.get("return_raw_index", False))

        # 1. 确保有 HTTPS URL 供 Video Indexer 访问
        video_https_url = self._ensure_video_https_url(video_uri, target_path=target_path)
        print(f"    Video URL for Azure Video Indexer: {video_https_url}")

        # 2. 获取访问 token
        access_token = self._get_access_token()

        # 3. 上传/索引视频
        video_id = self._upload_to_video_indexer(
            video_https_url,
            access_token,
            language=language,
            name=kwargs.get("name"),
            description=kwargs.get("description"),
        )
        print(f"    Video accepted by Azure Video Indexer, video_id={video_id}")

        # 4. 轮询直到处理完成
        index_json = self._poll_video_index(video_id, access_token, language=language)

        # 5. 提取 caption
        caption = self._extract_caption_from_index(index_json)

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

