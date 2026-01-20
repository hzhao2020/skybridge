# ops/registry.py
from ops.base import Operation
from ops.impl.google_ops import GoogleVideoSegmentImpl, GoogleVertexCaptionImpl, GoogleVertexLLMImpl
from ops.impl.amazon_ops import AmazonRekognitionSegmentImpl, AmazonBedrockCaptionImpl, AmazonBedrockLLMImpl
from ops.impl.openai_ops import OpenAILLMImpl

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
    "gcp_eu": "video_eu",
    "gcp_tw": "video_tw",
    "gcp_sg": "video_sg",
    # AWS (Region buckets)
    # Note: boto3 S3 API 需要 Bucket Name，不是 ARN；下面已从 ARN 提取为 bucket 名称。
    "aws_us": "sky-video-us",
    "aws_eu": "sky-video-eu",
    "aws_sg": "sky-video-sg"
}

# =========================================================
# Catalog definitions: region + model selections
# =========================================================

# 基础区域配置，方便做笛卡尔积
# Vertex AI 支持的区域：us-west1, europe-west1, asia-southeast1
GCP_REGIONS = [
    {"region": "us-west1", "bucket_key": "gcp_us"},
    {"region": "europe-west1", "bucket_key": "gcp_eu"},
    {"region": "asia-southeast1", "bucket_key": "gcp_sg"},
]

# Google Video Intelligence 支持的区域：us-west1, europe-west1, asia-east1
GCP_VIDEO_INTELLIGENCE_REGIONS = [
    {"region": "us-west1", "bucket_key": "gcp_us"},
    {"region": "europe-west1", "bucket_key": "gcp_eu"},
    {"region": "asia-east1", "bucket_key": "gcp_tw"},
]

AWS_REGIONS = [
    {"region": "us-west-2", "bucket_key": "aws_us"},
    {"region": "eu-central-1", "bucket_key": "aws_eu"},
    {"region": "ap-southeast-1", "bucket_key": "aws_sg"},
]

# 1) Video segmentation
# Google Video Intelligence: 仅支持 us-west1, europe-west1, asia-east1
VIDEO_SEGMENT_CATALOG = [
    # Google Video Intelligence (3个区域)
    {"pid": "vid_google_us", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "vid_google_eu", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "europe-west1", "bucket_key": "gcp_eu"},
    {"pid": "vid_google_tw", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "asia-east1", "bucket_key": "gcp_tw"},
    # Amazon Rekognition Video
    {"pid": "vid_aws_us", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us"},
    {"pid": "vid_aws_eu", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "eu-central-1", "bucket_key": "aws_eu"},
    {"pid": "vid_aws_sg", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg"},
]

# 2) Visual captioning
VISUAL_CAPTION_CATALOG = []

# Google Vertex AI (Gemini 2.5) - 模型 x 区域 笛卡尔积
# Vertex AI 仅支持 us-west1, europe-west1, asia-southeast1
_gcp_cap_models = {
    "gemini-2.5-flash-lite": "flash_lite",
    "gemini-2.5-flash": "flash",
}
for model, slug in _gcp_cap_models.items():
    for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的3个区域
        pid = f"cap_google_{slug}_{reg['region'].split('-')[0]}" if 'west1' in reg['region'] or 'east1' in reg['region'] else f"cap_google_{slug}_{reg['region'].split('-')[1]}"
        # 更直观的 pid：使用区域简称
        if reg["region"] == "us-west1":
            pid = f"cap_google_{slug}_us"
        elif reg["region"] == "europe-west1":
            pid = f"cap_google_{slug}_eu"
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

# Amazon Bedrock (Nova) - 模型 x 区域 笛卡尔积
_aws_cap_models = {
    "nova-lite": "nova_lite",
    "nova-pro": "nova_pro",
}
for model, slug in _aws_cap_models.items():
    for reg in AWS_REGIONS:
        if reg["region"] == "us-west-2":
            pid_suffix = "us"
        elif reg["region"] == "eu-central-1":
            pid_suffix = "eu"
        elif reg["region"] == "ap-southeast-1":
            pid_suffix = "sg"
        else:
            pid_suffix = reg["region"].replace("-", "_")
        pid = f"cap_aws_{slug}_{pid_suffix}"
        VISUAL_CAPTION_CATALOG.append({
            "pid": pid,
            "cls": AmazonBedrockCaptionImpl,
            "provider": "amazon",
            "region": reg["region"],
            "bucket_key": reg["bucket_key"],
            "model": model
        })

# 3) LLM querying
LLM_CATALOG = []

# Google Vertex AI (Gemini 2.5) - 模型 x 区域
# Vertex AI 仅支持 us-west1, europe-west1, asia-southeast1
_gcp_llm_models = {
    "gemini-2.5-flash": "flash",
    "gemini-2.5-pro": "pro",
}
for model, slug in _gcp_llm_models.items():
    for reg in GCP_REGIONS:  # GCP_REGIONS 已包含正确的3个区域
        if reg["region"] == "us-west1":
            pid = f"llm_google_{slug}_us"
        elif reg["region"] == "europe-west1":
            pid = f"llm_google_{slug}_eu"
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

# Amazon Bedrock (Claude) - 模型 x 区域
_aws_llm_models = {
    "claude-3-haiku": "haiku",
    "claude-3.5-sonnet": "sonnet",
}
for model, slug in _aws_llm_models.items():
    for reg in AWS_REGIONS:
        if reg["region"] == "us-west-2":
            pid_suffix = "us"
        elif reg["region"] == "eu-central-1":
            pid_suffix = "eu"
        elif reg["region"] == "ap-southeast-1":
            pid_suffix = "sg"
        else:
            pid_suffix = reg["region"].replace("-", "_")
        pid = f"llm_aws_{slug}_{pid_suffix}"
        LLM_CATALOG.append({
            "pid": pid,
            "cls": AmazonBedrockLLMImpl,
            "provider": "amazon",
            "region": reg["region"],
            "bucket_key": reg["bucket_key"],
            "model": model
        })

# OpenAI (无区域概念)
LLM_CATALOG.extend([
    {"pid": "llm_openai_gpt4o_mini", "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o-mini"},
    {"pid": "llm_openai_gpt4o",      "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o"},
])

# =========================================================
# Register all combinations from catalogs
# =========================================================
for item in VIDEO_SEGMENT_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]]))

for item in VISUAL_CAPTION_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], item["model"]))

for item in LLM_CATALOG:
    bucket = BUCKETS[item["bucket_key"]] if item["bucket_key"] else None
    # OpenAI 的实现只需要 model_name，其余忽略 bucket/region
    if item["provider"] == "openai":
        register(item["pid"], item["cls"](item["model"]))
    else:
        register(item["pid"], item["cls"](item["provider"], item["region"], bucket, item["model"]))