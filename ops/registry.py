from typing import Dict
from ops.base import Operation
from ops.impl.google_ops import GoogleVideoSegmentImpl, GoogleVertexCaptionImpl, GoogleVertexLLMImpl
from ops.impl.amazon_ops import AmazonRekognitionSegmentImpl, AmazonBedrockCaptionImpl, AmazonBedrockLLMImpl
from ops.impl.openai_ops import OpenAILLMImpl

REGISTRY: Dict[str, Operation] = {}


def register(physical_id: str, instance: Operation):
    REGISTRY[physical_id] = instance


# ==========================================
# Bucket Configuration (假设你已经在云端创建了这些 Bucket)
# ==========================================
BUCKETS = {
    # Google Buckets
    "gcp_us": "skybridge-gcp-us-west1",
    "gcp_eu": "skybridge-gcp-eu-west1",
    "gcp_asia": "skybridge-gcp-asia-east1",
    "gcp_sg": "skybridge-gcp-asia-se1",

    # AWS Buckets
    "aws_us": "skybridge-aws-us-west-2",
    "aws_eu": "skybridge-aws-eu-central-1",
    "aws_sg": "skybridge-aws-ap-se-1"
}

# ==========================================
# 1. Video Segment
# ==========================================
register("gcp_vid_us_west1", GoogleVideoSegmentImpl("google", "us-west1", BUCKETS["gcp_us"]))
register("gcp_vid_eu_west1", GoogleVideoSegmentImpl("google", "europe-west1", BUCKETS["gcp_eu"]))
register("gcp_vid_asia_east1", GoogleVideoSegmentImpl("google", "asia-east1", BUCKETS["gcp_asia"]))

register("aws_rek_us_west2", AmazonRekognitionSegmentImpl("amazon", "us-west-2", BUCKETS["aws_us"]))
register("aws_rek_eu_central1", AmazonRekognitionSegmentImpl("amazon", "eu-central-1", BUCKETS["aws_eu"]))
register("aws_rek_ap_southeast1", AmazonRekognitionSegmentImpl("amazon", "ap-southeast-1", BUCKETS["aws_sg"]))

# ==========================================
# 2. Visual Caption
# ==========================================
# Google
register("gcp_cap_us_west1_flash_lite",
         GoogleVertexCaptionImpl("google", "us-west1", BUCKETS["gcp_us"], "gemini-2.5-flash-lite"))
register("gcp_cap_eu_west1_flash",
         GoogleVertexCaptionImpl("google", "europe-west1", BUCKETS["gcp_eu"], "gemini-2.5-flash"))
register("gcp_cap_sg_flash",
         GoogleVertexCaptionImpl("google", "asia-southeast1", BUCKETS["gcp_sg"], "gemini-2.5-flash"))

# Amazon
register("aws_cap_us_west2_nova_lite", AmazonBedrockCaptionImpl("amazon", "us-west-2", BUCKETS["aws_us"], "nova-lite"))
register("aws_cap_eu_central1_nova_pro",
         AmazonBedrockCaptionImpl("amazon", "eu-central-1", BUCKETS["aws_eu"], "nova-pro"))
register("aws_cap_sg_nova_pro", AmazonBedrockCaptionImpl("amazon", "ap-southeast-1", BUCKETS["aws_sg"], "nova-pro"))

# ==========================================
# 3. LLM Query
# ==========================================
# Google
register("gcp_llm_us_west1_flash", GoogleVertexLLMImpl("google", "us-west1", BUCKETS["gcp_us"], "gemini-2.5-flash"))
register("gcp_llm_eu_west1_pro", GoogleVertexLLMImpl("google", "europe-west1", BUCKETS["gcp_eu"], "gemini-2.5-pro"))
register("gcp_llm_sg_pro", GoogleVertexLLMImpl("google", "asia-southeast1", BUCKETS["gcp_sg"], "gemini-2.5-pro"))

# Amazon
register("aws_llm_us_west2_haiku", AmazonBedrockLLMImpl("amazon", "us-west-2", BUCKETS["aws_us"], "claude-3-haiku"))
register("aws_llm_eu_central1_sonnet",
         AmazonBedrockLLMImpl("amazon", "eu-central-1", BUCKETS["aws_eu"], "claude-3.5-sonnet"))
register("aws_llm_sg_sonnet", AmazonBedrockLLMImpl("amazon", "ap-southeast-1", BUCKETS["aws_sg"], "claude-3.5-sonnet"))

# OpenAI (No bucket needed)
register("openai_gpt4o_mini", OpenAILLMImpl("gpt-4o-mini"))
register("openai_gpt4o", OpenAILLMImpl("gpt-4o"))


def get_operation(physical_id: str) -> Operation:
    if physical_id not in REGISTRY:
        raise ValueError(f"Physical ID {physical_id} not found.")
    return REGISTRY[physical_id]