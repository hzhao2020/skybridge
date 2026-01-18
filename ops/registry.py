# ops/registry.py
from ops.base import Operation
from ops.impl.google_ops import GoogleVideoSegmentImpl, GoogleVertexCaptionImpl, GoogleVertexLLMImpl
from ops.impl.amazon_ops import AmazonRekognitionSegmentImpl, AmazonBedrockCaptionImpl, AmazonBedrockLLMImpl
from ops.impl.openai_ops import OpenAILLMImpl

# 注册表字典：Physical ID -> Operation Instance
REGISTRY: Dict[str, Operation] = {}

def register(physical_id: str, instance: Operation):
    REGISTRY[physical_id] = instance

# ==========================================
# 1. Video Segment (Google Video Intelligence & Amazon Rekognition)
# ==========================================
# Google
register("gcp_vid_us_west1", GoogleVideoSegmentImpl("google", "us-west1"))
register("gcp_vid_eu_west1", GoogleVideoSegmentImpl("google", "europe-west1"))
register("gcp_vid_asia_east1", GoogleVideoSegmentImpl("google", "asia-east1"))

# Amazon
register("aws_rek_us_west2", AmazonRekognitionSegmentImpl("amazon", "us-west-2"))
register("aws_rek_eu_central1", AmazonRekognitionSegmentImpl("amazon", "eu-central-1"))
register("aws_rek_ap_southeast1", AmazonRekognitionSegmentImpl("amazon", "ap-southeast-1"))


# ==========================================
# 2. Visual Caption (Google Vertex AI & Amazon Bedrock)
# ==========================================
# Google (Gemini 2.5 Flash-Lite / Flash)
register("gcp_cap_us_west1_flash_lite", GoogleVertexCaptionImpl("google", "us-west1", "gemini-2.5-flash-lite"))
register("gcp_cap_eu_west1_flash",      GoogleVertexCaptionImpl("google", "europe-west1", "gemini-2.5-flash"))
register("gcp_cap_sg_flash",            GoogleVertexCaptionImpl("google", "asia-southeast1", "gemini-2.5-flash")) # Assuming Flash in SG based on table

# Amazon (Nova Lite / Nova Pro)
register("aws_cap_us_west2_nova_lite",  AmazonBedrockCaptionImpl("amazon", "us-west-2", "nova-lite"))
register("aws_cap_eu_central1_nova_pro",AmazonBedrockCaptionImpl("amazon", "eu-central-1", "nova-pro"))
register("aws_cap_sg_nova_pro",         AmazonBedrockCaptionImpl("amazon", "ap-southeast-1", "nova-pro"))


# ==========================================
# 3. LLM Querying (Vertex, Bedrock, OpenAI)
# ==========================================
# Google (Gemini 2.5 Flash / Pro)
register("gcp_llm_us_west1_flash", GoogleVertexLLMImpl("google", "us-west1", "gemini-2.5-flash"))
register("gcp_llm_eu_west1_pro",   GoogleVertexLLMImpl("google", "europe-west1", "gemini-2.5-pro"))
register("gcp_llm_sg_pro",         GoogleVertexLLMImpl("google", "asia-southeast1", "gemini-2.5-pro"))

# Amazon (Claude 3 Haiku / Sonnet)
register("aws_llm_us_west2_haiku",    AmazonBedrockLLMImpl("amazon", "us-west-2", "claude-3-haiku"))
register("aws_llm_eu_central1_sonnet", AmazonBedrockLLMImpl("amazon", "eu-central-1", "claude-3.5-sonnet"))
register("aws_llm_sg_sonnet",          AmazonBedrockLLMImpl("amazon", "ap-southeast-1", "claude-3.5-sonnet"))

# OpenAI (Global)
register("openai_gpt4o_mini", OpenAILLMImpl("gpt-4o-mini"))
register("openai_gpt4o",      OpenAILLMImpl("gpt-4o"))


def get_operation(physical_id: str) -> Operation:
    if physical_id not in REGISTRY:
        raise ValueError(f"Physical ID {physical_id} not found in registry.")
    return REGISTRY[physical_id]