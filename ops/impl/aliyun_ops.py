# ops/impl/aliyun_ops.py
import os
import time
import json
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from ops.base import VisualCaptioner, LLMQuery

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: openai library not installed. Install with: pip install openai")


class AliyunQwenCaptionImpl(VisualCaptioner):
    """
    使用阿里云百炼平台 Qwen3-VL 模型进行视频字幕生成。
    
    设计假设：
    - 在 `config.py` 中配置：
        ALIYUN_AI_CONFIG = {
            "us-east-1": {
                "API key": "...",
            },
            "ap-southeast-1": {
                "API key": "...",
            }
        }
    - 视频可以是：
        * 任何公网可访问的 HTTPS URL，或
        * 任意云（S3 / GCS / OSS），本类会自动搬运到对应阿里云 OSS，
          再转换为 HTTPS URL 交给百炼平台。
    - 使用 OpenAI 兼容接口调用百炼平台 API
    - 使用 API Key 进行身份验证
    - 支持模型：qwen3-vl-flash
    - 根据区域自动选择对应的 base_url：
        * us-east-1 -> https://dashscope-us.aliyuncs.com/compatible-mode/v1
        * ap-southeast-1 -> https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    
    注意：
    - region 参数是 OSS 的实际区域（us-east-1 或 ap-southeast-1）
    - 配置文件中 ALIYUN_AI_CONFIG 的 key 也使用 OSS 区域名称
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._api_key = None
        self._openai_client = None
        # OSS区域 -> API base_url 映射
        self._base_url_mapping = {
            "us-east-1": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        }
    
    def _get_api_key(self) -> str:
        """获取阿里云 API Key"""
        if self._api_key is None:
            try:
                import config
                if not hasattr(config, 'ALIYUN_AI_CONFIG'):
                    raise RuntimeError("config.py 中未找到 ALIYUN_AI_CONFIG 配置")
                
                ai_config = config.ALIYUN_AI_CONFIG
                # 使用OSS区域名称（现在配置文件中已经使用OSS区域名称）
                if self.region not in ai_config:
                    raise RuntimeError(
                        f"未在 ALIYUN_AI_CONFIG 中找到 region='{self.region}' 的配置，"
                        f"支持的区域: {list(ai_config.keys())}"
                    )
                
                self._api_key = ai_config[self.region].get("API key") or ai_config[self.region].get("api_key")
                if not self._api_key:
                    raise RuntimeError(
                        f"ALIYUN_AI_CONFIG['{self.region}'] 中缺少必需的字段：API key"
                    )
            except ImportError as e:
                raise RuntimeError("config.py 未找到，无法读取阿里云 AI 配置。") from e
        
        return self._api_key
    
    def _get_openai_client(self):
        """初始化 OpenAI 兼容客户端"""
        if self._openai_client is None:
            if OpenAI is None:
                raise RuntimeError("openai library not installed. Install with: pip install openai")
            
            api_key = self._get_api_key()
            base_url = self._base_url_mapping.get(self.region)
            if not base_url:
                raise RuntimeError(
                    f"未找到区域 '{self.region}' 对应的 base_url，支持的区域: {list(self._base_url_mapping.keys())}"
                )
            
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
        return self._openai_client
    
    def _is_https_url(self, uri: str) -> bool:
        """检查是否为 HTTPS URL"""
        return uri.startswith('https://') or uri.startswith('http://')
    
    def _is_oss_uri(self, uri: str) -> bool:
        """检查是否为 OSS URI"""
        return uri.startswith('oss://')
    
    
    def _ensure_video_https_url(self, video_uri: str, target_path: Optional[str] = None) -> str:
        """
        确保视频 URL 是公网可访问的 HTTPS URL。
        
        - 如果 input 已是 HTTPS，直接返回；
        - 如果 input 是 OSS URI，直接生成预签名 URL（无需搬运）；
        - 否则搬运到当前 region 对应的阿里云 OSS，并生成预签名 URL。
        
        Args:
            video_uri: 视频 URI（本地路径、S3/GCS/OSS URI 或 HTTPS URL）
            target_path: 可选，将视频复制到 OSS 时使用的相对路径前缀
        
        Returns:
            HTTPS URL（预签名 URL 或公共 URL）
        """
        # 如果已经是 HTTPS URL，直接返回
        if self._is_https_url(video_uri):
            return video_uri
        
        # region 已经是 OSS 区域，直接使用
        oss_region = self.region
        
        try:
            import config
        except ImportError as e:
            raise RuntimeError("config.py 未找到，无法确定阿里云 OSS bucket。") from e
        
        storage_configs = getattr(config, "ALIYUN_STORAGE_CONFIG", {})
        if oss_region not in storage_configs:
            raise RuntimeError(
                f"未在 ALIYUN_STORAGE_CONFIG 中找到 region='{oss_region}' 对应的配置，"
                f"支持的区域: {list(storage_configs.keys())}"
            )
        
        storage_config = storage_configs[oss_region]
        bucket_name = storage_config["bucket"]
        
        # 如果已经是 OSS URI，直接生成预签名 URL（无需搬运）
        if self._is_oss_uri(video_uri):
            parsed = urlparse(video_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # 获取 OSS bucket 客户端并生成预签名 URL
            # 预签名 URL 有效期设置为 7 天（最大支持值）
            oss_bucket = self.transmitter.get_aliyun_bucket(oss_region)
            
            # 生成预签名 URL（用于 GET 请求，有效期 7 天 = 604800 秒）
            # oss2 的 sign_url 方法：sign_url(method, key, expires)
            try:
                presigned_url = oss_bucket.sign_url('GET', key, 604800)
                print(f"    Generated presigned URL (expires in 7 days)")
                return presigned_url
            except Exception as e:
                # 如果生成预签名 URL 失败，尝试使用公共 URL（fallback）
                print(f"    Warning: 生成预签名 URL 失败: {e}，尝试使用公共 URL")
                endpoint = storage_config["endpoint"]
                domain = endpoint.replace("https://", "").replace("http://", "")
                public_url = f"https://{bucket}.{domain}/{key}"
                return public_url
        
        # 其他云（或本地） -> 阿里云 OSS
        # 使用 DataTransmission 智能搬运到阿里云 OSS
        target_uri = self.transmitter.smart_move(
            source_uri=video_uri,
            target_provider="aliyun",
            target_bucket=bucket_name,
            target_path=target_path,
            target_region=oss_region
        )
        
        # 将 OSS URI 转换为预签名 HTTPS URL
        # oss://bucket/key -> 预签名 URL
        if target_uri.startswith("oss://"):
            parsed = urlparse(target_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # 获取 OSS bucket 客户端并生成预签名 URL
            # 预签名 URL 有效期设置为 7 天（最大支持值）
            oss_bucket = self.transmitter.get_aliyun_bucket(oss_region)
            
            # 生成预签名 URL（用于 GET 请求，有效期 7 天 = 604800 秒）
            # oss2 的 sign_url 方法：sign_url(method, key, expires)
            try:
                presigned_url = oss_bucket.sign_url('GET', key, 604800)
                print(f"    Generated presigned URL (expires in 7 days)")
                return presigned_url
            except Exception as e:
                # 如果生成预签名 URL 失败，尝试使用公共 URL（fallback）
                print(f"    Warning: 生成预签名 URL 失败: {e}，尝试使用公共 URL")
                endpoint = storage_config["endpoint"]
                domain = endpoint.replace("https://", "").replace("http://", "")
                public_url = f"https://{bucket}.{domain}/{key}"
                return public_url
        else:
            # 如果已经是 HTTPS URL，直接返回
            return target_uri
    
    def execute(self, video_uri: str, **kwargs) -> Dict[str, Any]:
        """
        使用阿里云百炼平台 Qwen3-VL 模型对视频生成字幕。
        
        Args:
            video_uri: 视频 URI，可以是本地路径、S3/GCS/OSS URI 或 HTTPS URL。
            **kwargs:
                - target_path: 可选，将视频复制到 OSS 时使用的相对路径前缀。
                - start_time: 可选，视频片段的开始时间（秒）。
                - end_time: 可选，视频片段的结束时间（秒）。
                - summarize: 可选，是否使用总结模式（默认False，生成字幕）
        
        Returns:
            包含字幕结果的字典
        """
        print(f"--- [Aliyun Qwen3-VL Caption] Region: {self.region}, Model: {self.model_name} ---")
        
        target_path = kwargs.get("target_path")
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        
        # 1. 准备视频 URL
        print(f"    Input video URI: {video_uri}")
        
        # 如果指定了时间范围，需要先提取视频片段
        # 注意：百炼平台可能不支持直接指定时间范围
        # 如果需要片段，可以先提取片段再上传
        video_https_url = self._ensure_video_https_url(video_uri, target_path=target_path)
        print(f"    Video URL for Aliyun Qwen3-VL: {video_https_url}")
        
        # 2. 初始化 OpenAI 客户端
        client = self._get_openai_client()
        
        # 3. 构建提示词（与 Google 保持一致）
        prompt = "Describe this video in detail. Provide a comprehensive caption."
        
        if start_time is not None or end_time is not None:
            time_info = []
            if start_time is not None:
                time_info.append(f"Start time: {start_time}s")
            if end_time is not None:
                time_info.append(f"End time: {end_time}s")
            prompt += f" Note: This is a video segment ({', '.join(time_info)})."
        
        # 4. 调用百炼平台 API（使用 OpenAI 兼容接口）
        base_url = self._base_url_mapping.get(self.region)
        print(f"    Calling Aliyun Qwen3-VL API (model: {self.model_name}, region: {self.region})...")
        print(f"    Base URL: {base_url}")
        print(f"    Prompt: {prompt}")
        
        # 构建消息内容（使用 video_url 格式，兼容 OpenAI API）
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_https_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # 重试配置
        max_retries = 3
        retry_delay = 2  # 初始延迟（秒）
        
        for attempt in range(max_retries):
            try:
                # 调用 OpenAI 兼容的 chat.completions API
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False  # 不使用流式输出，直接获取完整结果
                )
                
                # 5. 处理响应
                if response.choices and len(response.choices) > 0:
                    caption = response.choices[0].message.content
                    if not caption:
                        caption = "No caption generated from Aliyun Qwen3-VL."
                else:
                    caption = "No caption generated from Aliyun Qwen3-VL."
                
                print("\n" + "=" * 80)
                print("=== [Aliyun Qwen3-VL Caption] Result ===")
                print("=" * 80)
                print(caption)
                print("=" * 80 + "\n")
                
                return {
                    "provider": "aliyun_qwen",
                    "model": self.model_name,
                    "region": self.region,  # OSS区域
                    "caption": caption,
                    "source_used": video_https_url,
                    "start_time": start_time,
                    "end_time": end_time
                }
                
            except Exception as e:
                # 检查是否是连接错误（可重试的错误）
                is_retryable = False
                error_type = type(e).__name__
                
                # 可重试的错误类型
                retryable_errors = (
                    "APIConnectionError",
                    "ConnectionError",
                    "RemoteProtocolError",
                    "TimeoutError",
                    "ConnectionTimeout",
                    "ReadTimeout"
                )
                
                if any(err_type in error_type for err_type in retryable_errors):
                    is_retryable = True
                elif hasattr(e, 'response') and e.response is not None:
                    # HTTP 5xx 错误也可以重试
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and 500 <= status_code < 600:
                        is_retryable = True
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    print(f"    ⚠ 连接错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"    ⏳ {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 不可重试的错误或已达到最大重试次数
                    error_msg = f"Aliyun Qwen3-VL API Error: {str(e)}"
                    if attempt >= max_retries - 1:
                        error_msg += f" (已重试 {max_retries} 次)"
                    print(f"    ✗ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(error_msg) from e


class AliyunQwenLLMImpl(LLMQuery):
    """
    使用阿里云百炼平台 Qwen LLM 模型进行文本查询。
    
    设计假设：
    - 在 `config.py` 中配置：
        ALIYUN_AI_CONFIG = {
            "us-east-1": {
                "API key": "...",
            },
            "ap-southeast-1": {
                "API key": "...",
            }
        }
    - 使用 OpenAI 兼容接口调用百炼平台 API
    - 使用 API Key 进行身份验证
    - 支持模型：qwen-flash
    - 根据区域自动选择对应的 base_url：
        * us-east-1 -> https://dashscope-us.aliyuncs.com/compatible-mode/v1
        * ap-southeast-1 -> https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    
    注意：
    - region 参数是 OSS 的实际区域（us-east-1 或 ap-southeast-1）
    - 配置文件中 ALIYUN_AI_CONFIG 的 key 也使用 OSS 区域名称
    """
    
    def __init__(self, provider: str, region: str, storage_bucket: str, model_name: str):
        super().__init__(provider, region, storage_bucket, model_name)
        self._api_key = None
        self._openai_client = None
        # OSS区域 -> API base_url 映射
        self._base_url_mapping = {
            "us-east-1": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        }
    
    def _get_api_key(self) -> str:
        """获取阿里云 API Key"""
        if self._api_key is None:
            try:
                import config
                if not hasattr(config, 'ALIYUN_AI_CONFIG'):
                    raise RuntimeError("config.py 中未找到 ALIYUN_AI_CONFIG 配置")
                
                ai_config = config.ALIYUN_AI_CONFIG
                # 使用OSS区域名称（现在配置文件中已经使用OSS区域名称）
                if self.region not in ai_config:
                    raise RuntimeError(
                        f"未在 ALIYUN_AI_CONFIG 中找到 region='{self.region}' 的配置，"
                        f"支持的区域: {list(ai_config.keys())}"
                    )
                
                self._api_key = ai_config[self.region].get("API key") or ai_config[self.region].get("api_key")
                if not self._api_key:
                    raise RuntimeError(
                        f"ALIYUN_AI_CONFIG['{self.region}'] 中缺少必需的字段：API key"
                    )
            except ImportError as e:
                raise RuntimeError("config.py 未找到，无法读取阿里云 AI 配置。") from e
        
        return self._api_key
    
    def _get_openai_client(self):
        """初始化 OpenAI 兼容客户端"""
        if self._openai_client is None:
            if OpenAI is None:
                raise RuntimeError("openai library not installed. Install with: pip install openai")
            
            api_key = self._get_api_key()
            base_url = self._base_url_mapping.get(self.region)
            if not base_url:
                raise RuntimeError(
                    f"未找到区域 '{self.region}' 对应的 base_url，支持的区域: {list(self._base_url_mapping.keys())}"
                )
            
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
        return self._openai_client
    
    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        使用阿里云百炼平台 Qwen LLM 模型进行文本查询。
        
        Args:
            prompt: 查询提示词
            **kwargs:
                - max_tokens: 最大输出token数（默认2048）
                - temperature: 温度参数（默认0，确保确定性输出）
        
        Returns:
            包含查询结果的字典
        """
        print(f"--- [Aliyun Qwen LLM] Region: {self.region}, Model: {self.model_name} ---")
        
        # 打印完整的prompt
        print("\n" + "=" * 80)
        print("=== [Aliyun Qwen LLM] Full Prompt ===")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")
        
        # 初始化 OpenAI 客户端
        client = self._get_openai_client()
        
        # 重试配置
        max_retries = 3
        retry_delay = 2  # 初始延迟（秒）
        
        # 获取参数
        max_tokens = kwargs.get('max_tokens', 2048)
        temperature = kwargs.get('temperature', 0)  # 默认0，确保确定性输出
        
        base_url = self._base_url_mapping.get(self.region)
        print(f"    Calling Aliyun Qwen LLM API (model: {self.model_name}, region: {self.region})...")
        print(f"    Base URL: {base_url}")
        print(f"    Temperature: {temperature}, Max tokens: {max_tokens}")
        
        for attempt in range(max_retries):
            try:
                # 调用 OpenAI 兼容的 chat.completions API
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                # 处理响应
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content
                    if not answer:
                        answer = "No response generated from Aliyun Qwen LLM."
                else:
                    answer = "No response generated from Aliyun Qwen LLM."
                
                # 打印完整的响应
                print("\n" + "=" * 80)
                print("=== [Aliyun Qwen LLM] Full Response ===")
                print("=" * 80)
                print(answer)
                print("=" * 80 + "\n")
                
                return {
                    "provider": "aliyun_qwen",
                    "model": self.model_name,
                    "region": self.region,
                    "response": answer
                }
                
            except Exception as e:
                # 检查是否是连接错误（可重试的错误）
                is_retryable = False
                error_type = type(e).__name__
                
                # 可重试的错误类型
                retryable_errors = (
                    "APIConnectionError",
                    "ConnectionError",
                    "RemoteProtocolError",
                    "TimeoutError",
                    "ConnectionTimeout",
                    "ReadTimeout"
                )
                
                if any(err_type in error_type for err_type in retryable_errors):
                    is_retryable = True
                elif hasattr(e, 'response') and e.response is not None:
                    # HTTP 5xx 错误也可以重试
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and 500 <= status_code < 600:
                        is_retryable = True
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    print(f"    ⚠ 连接错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"    ⏳ {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 不可重试的错误或已达到最大重试次数
                    error_msg = f"Aliyun Qwen LLM API Error: {str(e)}"
                    if attempt >= max_retries - 1:
                        error_msg += f" (已重试 {max_retries} 次)"
                    print(f"    ✗ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(error_msg) from e
