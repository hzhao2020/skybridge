# ops/registry.py
import json
import os
import subprocess
from typing import Dict, Optional, List

from ops.base import Operation
from ops.impl.google_ops import (
    GoogleVideoSegmentImpl,
    GoogleVertexCaptionImpl,
    GoogleVertexLLMImpl,
    GoogleCloudFunctionSplitImpl,
    GoogleVertexEmbeddingImpl,
    GoogleVertexTextEmbeddingImpl,
    GoogleVideoIntelligenceObjectDetectionImpl,
)
from ops.impl.amazon_ops import (
    AmazonRekognitionSegmentImpl,
    AWSLambdaSplitImpl,
    AmazonRekognitionObjectDetectionImpl,
)
from ops.impl.openai_ops import OpenAILLMImpl
from ops.impl.azure_ops import AzureVideoIndexerCaptionImpl, AzureVideoIndexerSegmentImpl, AzureFunctionSplitImpl
# Storage 和 Transmission 操作直接使用 ops.utils 中的辅助类，不需要注册为 Operation

REGISTRY = {}

def register(pid: str, instance: Operation):
    REGISTRY[pid] = instance

def get_operation(pid: str) -> Operation:
    if pid not in REGISTRY:
        raise ValueError(f"Physical ID '{pid}' not found.")
    return REGISTRY[pid]

# --- 存储配置 (请在云端创建对应名称的 Bucket) ---
BUCKETS = {
    # Google (Region buckets)
    "gcp_us": "video_us",
    "gcp_tw": "video_tw",
    "gcp_sg": "video_sg",
    # AWS (Region buckets)
    # Note: boto3 S3 API 需要 Bucket Name，不是 ARN；下面已从 ARN 提取为 bucket 名称。
    "aws_us": "sky-video-us",
    "aws_sg": "sky-video-sg",
    # Azure (Region containers) - 与 config.AZURE_STORAGE_ACCOUNTS 中的 container 名保持一致
    # eastasia  -> 账户 videoea, 容器 "video-ea"
    # westus2   -> 账户 videowu, 容器 "video-wu"
    "azure_ea": "video-ea",
    "azure_wu": "video-wu",
}

# --- Serverless 服务 URL 配置 ---
# VideoSplit（Google Cloud Function Gen2）按区域部署的服务 URL
# - 与 deploy.sh 部署的 video-splitter Cloud Function 对应
# - 优先级：
#   1. 硬编码的 URL（见下面的 GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT）
#   2. 环境变量 GCP_VIDEOSPLIT_SERVICE_URLS（JSON 对象，格式：{"us-west1": "https://...", "asia-southeast1": "https://..."}）
#   3. 通过 gcloud functions describe 命令自动获取
#   4. 根据 GCP 项目号自动推导（旧格式，用于 Cloud Run）

# 硬编码的 Cloud Function URL（从 gcloud functions list 获取）
# 如果 URL 发生变化，请更新此处
GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT = {
    "us-west1": "https://video-splitter-nqis2t7p2a-uw.a.run.app",
    "asia-southeast1": "https://video-splitter-nqis2t7p2a-as.a.run.app",
}

def _get_gcp_project_number() -> Optional[str]:
    pn = os.getenv("GCP_PROJECT_NUMBER")
    if pn:
        return pn.strip()
    try:
        project_id = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if not project_id or project_id == "(unset)":
            return None
        out = subprocess.check_output(
            ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return out or None
    except Exception:
        return None


def _get_cloud_function_url(region: str, function_name: str = "video-splitter") -> Optional[str]:
    """通过 gcloud 命令获取 Cloud Function 的 URL"""
    try:
        url = subprocess.check_output(
            ["gcloud", "functions", "describe", function_name,
             "--gen2", "--region", region,
             "--format", "value(serviceConfig.uri)"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return url if url else None
    except Exception:
        return None


def _build_gcp_videosplit_urls() -> Dict[str, str]:
    # 1. 优先使用硬编码的 URL
    if GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT:
        return GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT.copy()
    
    # 2. 其次使用环境变量
    raw = os.getenv("GCP_VIDEOSPLIT_SERVICE_URLS")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return {k: str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    
    # 3. 尝试通过 gcloud 命令获取 Cloud Function URL
    urls = {}
    regions = ["us-west1", "asia-southeast1"]
    for region in regions:
        url = _get_cloud_function_url(region)
        if url:
            urls[region] = url
    
    # 如果成功获取了所有 URL，返回它们
    if len(urls) == len(regions):
        return urls
    
    # 4. 降级：根据项目号推导（旧格式，用于 Cloud Run）
    pn = _get_gcp_project_number()
    if not pn:
        return urls  # 返回已获取的部分 URL，或空字典
    base = f"https://video-splitter-{pn}.{{region}}.run.app/video_split"
    # 只填充未获取到的区域
    for region in regions:
        if region not in urls:
            urls[region] = base.format(region=region)
    return urls


GCP_VIDEOSPLIT_SERVICE_URLS = _build_gcp_videosplit_urls()

# Azure Function URLs（video-splitter）
# 硬编码的 Azure Function URL（从部署脚本获取）
AZURE_VIDEOSPLIT_FUNCTION_URLS_DEFAULT = {
    "eastasia": "https://video-splitter-eastasia.azurewebsites.net/api/video_split",
    "westus2": "https://video-splitter-westus2.azurewebsites.net/api/video_split",
}

def _build_azure_videosplit_urls() -> Dict[str, str]:
    """构建 Azure Function URLs"""
    # 1. 优先使用硬编码的 URL
    if AZURE_VIDEOSPLIT_FUNCTION_URLS_DEFAULT:
        return AZURE_VIDEOSPLIT_FUNCTION_URLS_DEFAULT.copy()
    
    # 2. 其次使用环境变量
    raw = os.getenv("AZURE_VIDEOSPLIT_FUNCTION_URLS")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return {k: str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    
    # 3. 尝试通过 Azure CLI 获取 Function App URL
    urls = {}
    regions = ["eastasia", "westus2"]
    for region in regions:
        try:
            function_name = f"video-splitter-{region}"
            url = subprocess.check_output(
                ["az", "functionapp", "show",
                 "--name", function_name,
                 "--resource-group", "vqa",
                 "--query", "defaultHostName",
                 "-o", "tsv"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
            if url:
                urls[region] = f"https://{url}/api/video_split"
        except Exception:
            pass
    
    # 如果成功获取了所有 URL，返回它们
    if len(urls) == len(regions):
        return urls
    
    # 降级：使用默认值
    return AZURE_VIDEOSPLIT_FUNCTION_URLS_DEFAULT.copy()


AZURE_VIDEOSPLIT_FUNCTION_URLS = _build_azure_videosplit_urls()

# =========================================================
# Catalog definitions: region + model selections
# =========================================================

# 基础区域配置，方便做笛卡尔积
# Vertex AI 支持的区域：us-west1, asia-southeast1
GCP_REGIONS = [
    {"region": "us-west1", "bucket_key": "gcp_us"},
    {"region": "asia-southeast1", "bucket_key": "gcp_sg"},
]

# Google Video Intelligence 支持的区域：us-west1, asia-east1
GCP_VIDEO_INTELLIGENCE_REGIONS = [
    {"region": "us-west1", "bucket_key": "gcp_us"},
    {"region": "asia-east1", "bucket_key": "gcp_tw"},
]

AWS_REGIONS = [
    {"region": "us-west-2", "bucket_key": "aws_us"},
    {"region": "ap-southeast-1", "bucket_key": "aws_sg"},
]

# 1) Video segmentation (shot detection)
# Google Video Intelligence: 仅支持 us-west1, asia-east1
# Azure Video Indexer: 支持 eastasia, westus2
VIDEO_SEGMENT_CATALOG = [
    # Google Video Intelligence (2个区域)
    {"pid": "seg_google_us", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "seg_google_tw", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "asia-east1", "bucket_key": "gcp_tw"},
    # Amazon Rekognition Video
    {"pid": "seg_aws_us", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us"},
    {"pid": "seg_aws_sg", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg"},
    # Azure Video Indexer (2个区域)
    {"pid": "seg_azure_vi_ea", "cls": AzureVideoIndexerSegmentImpl, "provider": "azure", "region": "eastasia", "bucket_key": "azure_ea"},
    {"pid": "seg_azure_vi_wu", "cls": AzureVideoIndexerSegmentImpl, "provider": "azure", "region": "westus2", "bucket_key": "azure_wu"},
]

# 1.5) Video splitting (physical cutting)
# Google Cloud Function (video-splitter): 支持 us-west1, asia-southeast1
# AWS Lambda: 支持多个区域
# Azure Function (video-splitter): 支持 eastasia, westus2
VIDEO_SPLIT_CATALOG = [
    # Google Cloud Function (service_url 由 GCP_VIDEOSPLIT_SERVICE_URLS 或 gcloud 项目号推导)
    {"pid": "split_google_us", "cls": GoogleCloudFunctionSplitImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "split_google_sg", "cls": GoogleCloudFunctionSplitImpl, "provider": "google", "region": "asia-southeast1", "bucket_key": "gcp_sg"},
    # AWS Lambda
    {"pid": "split_aws_us", "cls": AWSLambdaSplitImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us"},
    {"pid": "split_aws_sg", "cls": AWSLambdaSplitImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg"},
    # Azure Function (function_url 由 AZURE_VIDEOSPLIT_FUNCTION_URLS 或 Azure CLI 获取)
    {"pid": "split_azure_ea", "cls": AzureFunctionSplitImpl, "provider": "azure", "region": "eastasia", "bucket_key": "azure_ea"},
    {"pid": "split_azure_wu", "cls": AzureFunctionSplitImpl, "provider": "azure", "region": "westus2", "bucket_key": "azure_wu"},
]

# 1.6) Data Storage 和 Transmission
# 注意：Storage 和 Transmission 不需要注册为 Operation，直接使用 ops.utils 中的辅助类即可
# 
# Storage 使用示例：
#   from ops.utils import DataStorageHelper
#   storage = DataStorageHelper(aws_region="us-west-2")  # 或 gcp_project="your-project"
#   cloud_uri = storage.upload(local_path, provider, bucket, target_path)
#   storage.download(cloud_uri, local_path)
#   storage.delete(cloud_uri)
#   files = storage.list_files(cloud_uri, prefix)
#
# Transmission 使用示例：
#   from ops.utils import DataTransmission
#   transmission = DataTransmission()
#   target_uri = transmission.smart_move(source_uri, target_provider, target_bucket, target_path)
#
# 或者在其他 Operation 中直接使用：
#   self.transmitter.smart_move(...)  # Operation 基类已提供
#   self._storage_helper = DataStorageHelper(...)  # 按需创建

# 2) Visual captioning
VISUAL_CAPTION_CATALOG = []

# Google Vertex AI (Gemini 1.5) - 模型 x 区域 笛卡尔积
# Vertex AI 仅支持 us-west1, asia-southeast1
# 注意：flash_lite 和 flash 都使用相同的模型 gemini-2.5-flash
_gcp_cap_models = [
    ("gemini-2.5-flash", "flash_lite"),  # 更新为推荐的稳定版本
    ("gemini-2.5-flash", "flash"),  # 更新为推荐的稳定版本
]
for model, slug in _gcp_cap_models:
    for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的2个区域
        pid = f"cap_google_{slug}_{reg['region'].split('-')[0]}" if 'west1' in reg['region'] or 'east1' in reg['region'] else f"cap_google_{slug}_{reg['region'].split('-')[1]}"
        # 更直观的 pid：使用区域简称
        if reg["region"] == "us-west1":
            pid = f"cap_google_{slug}_us"
        elif reg["region"] == "asia-southeast1":
            pid = f"cap_google_{slug}_sg"
        VISUAL_CAPTION_CATALOG.append({
            "pid": pid,
            "cls": GoogleVertexCaptionImpl,
            "provider": "google",
            "region": reg["region"],
            "bucket_key": reg["bucket_key"],
            "model": model
        })

# Azure Video Indexer - 两个区域 eastasia / westus2
AZURE_CAPTION_REGIONS = [
    {"region": "eastasia", "bucket_key": "azure_ea", "pid_suffix": "ea"},
    {"region": "westus2", "bucket_key": "azure_wu", "pid_suffix": "wu"},
]

for reg in AZURE_CAPTION_REGIONS:
    pid = f"cap_azure_vi_{reg['pid_suffix']}"
    VISUAL_CAPTION_CATALOG.append({
        "pid": pid,
        "cls": AzureVideoIndexerCaptionImpl,
        "provider": "azure",
        "region": reg["region"],
        "bucket_key": reg["bucket_key"],
        "model": "azure_video_indexer",
    })


# 3) LLM querying
LLM_CATALOG = []

# Google Vertex AI (Gemini 1.5) - 模型 x 区域
# Vertex AI 仅支持 us-west1, asia-southeast1
_gcp_llm_models = {
    "gemini-2.5-flash": "flash",  # 更新为推荐的稳定版本
    "gemini-2.5-pro": "pro",  # 更新为推荐的稳定版本
}
for model, slug in _gcp_llm_models.items():
    for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的2个区域
        if reg["region"] == "us-west1":
            pid = f"llm_google_{slug}_us"
        elif reg["region"] == "asia-southeast1":
            pid = f"llm_google_{slug}_sg"
        else:
            pid = f"llm_google_{slug}_{reg['region'].replace('-', '_')}"
        LLM_CATALOG.append({
            "pid": pid,
            "cls": GoogleVertexLLMImpl,
            "provider": "google",
            "region": reg["region"],
            "bucket_key": reg["bucket_key"],
            "model": model
        })


# OpenAI (无区域概念)
LLM_CATALOG.extend([
    {"pid": "llm_openai_gpt4o_mini", "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o-mini"},
    {"pid": "llm_openai_gpt4o",      "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o"},
])

# 4) Visual encoding (embedding)
VISUAL_ENCODING_CATALOG = []

# Google Vertex AI (multimodalembedding@001) - 支持2个区域
# us-west1 (Oregon), asia-southeast1 (Singapore)
for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的2个区域
    if reg["region"] == "us-west1":
        pid = "embed_google_us"
    elif reg["region"] == "asia-southeast1":
        pid = "embed_google_sg"
    else:
        pid = f"embed_google_{reg['region'].replace('-', '_')}"
    VISUAL_ENCODING_CATALOG.append({
        "pid": pid,
        "cls": GoogleVertexEmbeddingImpl,
        "provider": "google",
        "region": reg["region"],
        "bucket_key": reg["bucket_key"],
        "model": "multimodalembedding@001"
    })


# 5) Text encoding (embedding)
TEXT_EMBEDDING_CATALOG = []

# Google Vertex AI - 2个模型 × 2个区域
# gemini-embedding-001: us-west1, asia-southeast1
# text-embedding-005: us-west1, asia-southeast1
_gcp_text_embedding_models = {
    "gemini-embedding-001": "gemini",
    "text-embedding-005": "text005",
}
for model, slug in _gcp_text_embedding_models.items():
    for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的2个区域
        if reg["region"] == "us-west1":
            pid = f"text_embed_google_{slug}_us"
        elif reg["region"] == "asia-southeast1":
            pid = f"text_embed_google_{slug}_sg"
        else:
            pid = f"text_embed_google_{slug}_{reg['region'].replace('-', '_')}"
        TEXT_EMBEDDING_CATALOG.append({
            "pid": pid,
            "cls": GoogleVertexTextEmbeddingImpl,
            "provider": "google",
            "region": reg["region"],
            "bucket_key": reg["bucket_key"],
            "model": model
        })


# 6) Object detection
OBJECT_DETECTION_CATALOG = []

# Google Video Intelligence - 2个区域
# us-west1 (Oregon), asia-east1 (Taiwan)
for reg in GCP_VIDEO_INTELLIGENCE_REGIONS:
    if reg["region"] == "us-west1":
        pid = "obj_detect_google_us"
    elif reg["region"] == "asia-east1":
        pid = "obj_detect_google_tw"
    else:
        pid = f"obj_detect_google_{reg['region'].replace('-', '_')}"
    OBJECT_DETECTION_CATALOG.append({
        "pid": pid,
        "cls": GoogleVideoIntelligenceObjectDetectionImpl,
        "provider": "google",
        "region": reg["region"],
        "bucket_key": reg["bucket_key"]
    })

# Amazon Rekognition Video - 2个区域
# us-west-2 (Oregon), ap-southeast-1 (Singapore)
for reg in AWS_REGIONS:
    if reg["region"] == "us-west-2":
        pid_suffix = "us"
    elif reg["region"] == "ap-southeast-1":
        pid_suffix = "sg"
    else:
        pid_suffix = reg["region"].replace("-", "_")
    pid = f"obj_detect_aws_{pid_suffix}"
    OBJECT_DETECTION_CATALOG.append({
        "pid": pid,
        "cls": AmazonRekognitionObjectDetectionImpl,
        "provider": "amazon",
        "region": reg["region"],
        "bucket_key": reg["bucket_key"]
    })

# =========================================================
# Register all combinations from catalogs
# =========================================================
for item in VIDEO_SEGMENT_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]]))

for item in VIDEO_SPLIT_CATALOG:
    # Google Cloud Function split: 为每个 region 注入 service_url（由 GCP_VIDEOSPLIT_SERVICE_URLS 或项目号推导）
    if item["provider"] == "google" and item["cls"] is GoogleCloudFunctionSplitImpl:
        service_url = GCP_VIDEOSPLIT_SERVICE_URLS.get(item["region"])
        register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], service_url=service_url))
    # Azure Function split: 为每个 region 注入 function_url（由 AZURE_VIDEOSPLIT_FUNCTION_URLS 或 Azure CLI 获取）
    elif item["provider"] == "azure" and item["cls"] is AzureFunctionSplitImpl:
        function_url = AZURE_VIDEOSPLIT_FUNCTION_URLS.get(item["region"])
        register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], function_url=function_url))
    else:
        register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]]))

# Storage 和 Transmission 操作不再注册，直接使用 ops.utils 中的辅助类
# 参见上面的注释说明

for item in VISUAL_CAPTION_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], item["model"]))

for item in LLM_CATALOG:
    bucket = BUCKETS[item["bucket_key"]] if item["bucket_key"] else None
    # OpenAI 的实现只需要 model_name，其余忽略 bucket/region
    if item["provider"] == "openai":
        register(item["pid"], item["cls"](item["model"]))
    else:
        register(item["pid"], item["cls"](item["provider"], item["region"], bucket, item["model"]))

for item in VISUAL_ENCODING_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], item["model"]))

for item in TEXT_EMBEDDING_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], item["model"]))

for item in OBJECT_DETECTION_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]]))


# =========================================================
# List all supported operations
# =========================================================

def list_supported_operations() -> str:
    """
    列出所有支持的 operation，按类别组织。
    只返回核心信息：PID、Provider、Region、Model。
    
    Returns:
        格式化的字符串，包含所有支持的 operation 信息
    """
    # 定义类别和对应的 catalog
    categories = {
        "1. Video Segmentation (视频分割 - 镜头检测)": VIDEO_SEGMENT_CATALOG,
        "2. Video Splitting (视频切割 - 物理切割)": VIDEO_SPLIT_CATALOG,
        "3. Visual Captioning (视觉字幕生成)": VISUAL_CAPTION_CATALOG,
        "4. LLM Querying (大语言模型查询)": LLM_CATALOG,
        "5. Visual Encoding (视觉嵌入)": VISUAL_ENCODING_CATALOG,
        "6. Text Embedding (文本嵌入)": TEXT_EMBEDDING_CATALOG,
        "7. Object Detection (目标检测)": OBJECT_DETECTION_CATALOG,
    }
    
    lines = []
    lines.append("=" * 80)
    lines.append("支持的 Operations")
    lines.append("=" * 80)
    lines.append("")
    
    total_count = 0
    for category, catalog in categories.items():
        lines.append(category)
        lines.append("-" * 80)
        
        if not catalog:
            lines.append("  (无)")
            lines.append("")
            continue
        
        # 按 provider 分组
        by_provider = {}
        for item in catalog:
            provider = item.get("provider", "unknown")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(item)
        
        for provider, items in sorted(by_provider.items()):
            lines.append(f"  Provider: {provider.upper()}")
            for item in items:
                pid = item.get("pid", "N/A")
                region = item.get("region", "N/A")
                model = item.get("model", "-")
                
                model_str = f" (Model: {model})" if model != "-" else ""
                lines.append(f"    - {pid:30s} | Region: {region:20s}{model_str}")
        
        lines.append("")
        total_count += len(catalog)
    
    lines.append("=" * 80)
    lines.append(f"总计: {total_count} 个 operations")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def get_operation_info(pid: str, include_class: bool = False) -> Optional[Dict]:
    """
    获取指定 operation 的详细信息。
    
    Args:
        pid: operation 的 physical ID
        include_class: 是否在返回结果中包含类对象（cls），默认 False（返回可序列化的字典）
    
    Returns:
        包含 operation 信息的字典，如果不存在则返回 None
        默认返回可序列化的字典（不包含 cls 对象），如果 include_class=True 则包含原始 cls 对象
    """
    # 搜索所有 catalog
    all_catalogs = [
        VIDEO_SEGMENT_CATALOG,
        VIDEO_SPLIT_CATALOG,
        VISUAL_CAPTION_CATALOG,
        LLM_CATALOG,
        VISUAL_ENCODING_CATALOG,
        TEXT_EMBEDDING_CATALOG,
        OBJECT_DETECTION_CATALOG,
    ]
    
    for catalog in all_catalogs:
        for item in catalog:
            if item.get("pid") == pid:
                if include_class:
                    return item.copy()
                else:
                    # 返回可序列化的版本
                    result = {
                        "pid": item.get("pid"),
                        "provider": item.get("provider"),
                        "region": item.get("region"),
                        "bucket_key": item.get("bucket_key"),
                    }
                    if "model" in item:
                        result["model"] = item["model"]
                    if "cls" in item:
                        result["class"] = item["cls"].__name__
                    return result
    
    return None
