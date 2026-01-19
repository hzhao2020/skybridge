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

# 1) Video segmentation
VIDEO_SEGMENT_CATALOG = [
    # Google Video Intelligence
    {"pid": "vid_google_us", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "vid_google_eu", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "europe-west1", "bucket_key": "gcp_eu"},
    {"pid": "vid_google_tw", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "asia-east1", "bucket_key": "gcp_tw"},
    # Amazon Rekognition Video
    {"pid": "vid_aws_us", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us"},
    {"pid": "vid_aws_eu", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "eu-central-1", "bucket_key": "aws_eu"},
    {"pid": "vid_aws_sg", "cls": AmazonRekognitionSegmentImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg"},
]

# 2) Visual captioning
VISUAL_CAPTION_CATALOG = [
    # Google Vertex AI (Gemini 2.5)
    {"pid": "cap_google_flash_lite_us", "cls": GoogleVertexCaptionImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us", "model": "gemini-2.5-flash-lite"},
    {"pid": "cap_google_flash_eu",       "cls": GoogleVertexCaptionImpl, "provider": "google", "region": "europe-west1", "bucket_key": "gcp_eu", "model": "gemini-2.5-flash"},
    {"pid": "cap_google_flash_sg",       "cls": GoogleVertexCaptionImpl, "provider": "google", "region": "asia-southeast1", "bucket_key": "gcp_sg", "model": "gemini-2.5-flash"},
    # Amazon Bedrock (Nova)
    {"pid": "cap_aws_nova_lite_us", "cls": AmazonBedrockCaptionImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us", "model": "nova-lite"},
    {"pid": "cap_aws_nova_pro_eu", "cls": AmazonBedrockCaptionImpl, "provider": "amazon", "region": "eu-central-1", "bucket_key": "aws_eu", "model": "nova-pro"},
    {"pid": "cap_aws_nova_pro_sg", "cls": AmazonBedrockCaptionImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg", "model": "nova-pro"},
]

# 3) LLM querying
LLM_CATALOG = [
    # Google Vertex AI (Gemini 2.5)
    {"pid": "llm_google_flash_us", "cls": GoogleVertexLLMImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us", "model": "gemini-2.5-flash"},
    {"pid": "llm_google_pro_eu",   "cls": GoogleVertexLLMImpl, "provider": "google", "region": "europe-west1", "bucket_key": "gcp_eu", "model": "gemini-2.5-pro"},
    {"pid": "llm_google_pro_sg",   "cls": GoogleVertexLLMImpl, "provider": "google", "region": "asia-southeast1", "bucket_key": "gcp_sg", "model": "gemini-2.5-pro"},
    # Amazon Bedrock (Claude)
    {"pid": "llm_aws_haiku_us",   "cls": AmazonBedrockLLMImpl, "provider": "amazon", "region": "us-west-2", "bucket_key": "aws_us", "model": "claude-3-haiku"},
    {"pid": "llm_aws_sonnet_eu",  "cls": AmazonBedrockLLMImpl, "provider": "amazon", "region": "eu-central-1", "bucket_key": "aws_eu", "model": "claude-3.5-sonnet"},
    {"pid": "llm_aws_sonnet_sg",  "cls": AmazonBedrockLLMImpl, "provider": "amazon", "region": "ap-southeast-1", "bucket_key": "aws_sg", "model": "claude-3.5-sonnet"},
    # OpenAI
    {"pid": "llm_openai_gpt4o_mini", "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o-mini"},
    {"pid": "llm_openai_gpt4o",      "cls": OpenAILLMImpl, "provider": "openai", "region": "global", "bucket_key": None, "model": "gpt-4o"},
]

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