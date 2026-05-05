"""
配置文件 - 存放访问URL和API密钥
此文件不应上传到GitHub，已添加到.gitignore
"""


# ========== OpenAI 配置 ==========
# OpenAI API密钥
OPENAI_API_KEY = "sk-YC4WBvVC69LM3A5YxwUhWYs6V081SDG8eMIdWqLPPBdsAGDq"

# OpenAI Base URL（可选，用于第三方代理或兼容平台）
# 例如: "https://api.openai-proxy.org/v1" （注意要带 /v1）
# 如果使用官方OpenAI API，可以设置为 None 或空字符串
OPENAI_BASE_URL = "https://api.openai-proxy.org/v1"  # 或 "https://api.openai-proxy.org/v1"

# ========== Google Cloud Platform (GCP) 配置 ==========
# GCP项目号（Project Number）
GCP_PROJECT_NUMBER = "project-ab73e1ce-e25c-48b5-a91"  # 例如: "123456789012"

VIDEO_SPLIT_URLS = {
    "aws": {
        "us-west-2": "https://69cwe8yx7f.execute-api.us-west-2.amazonaws.com/split",
        "ap-southeast-1": "https://tread0fep0.execute-api.ap-southeast-1.amazonaws.com/split",
    },
    "google": {
        "us-west1": "https://us-west1-project-ab73e1ce-e25c-48b5-a91.cloudfunctions.net/split_measurement",
        "asia-southeast1": "https://asia-southeast1-project-ab73e1ce-e25c-48b5-a91.cloudfunctions.net/videosplit",
    },
}

ALIYUN_CONFIG = {
    'AccessKeyID': "LTAI5tF8fYGgDgBokDD8qYxP", 
    'AccessKeySecret': "RgVDVXCYZs4tNmbf7EIA6Lc4HdPEDg",
}

ALIYUN_STORAGE_CONFIG = {
    "us-east-1": {
        "bucket": "vqa-store-us",
        "endpoint": "https://oss-us-east-1.aliyuncs.com",
        "region": "us-east-1",
    },
    "ap-southeast-1": {
        "bucket": "vqa-store-se",
        "endpoint": "https://oss-ap-southeast-1.aliyuncs.com",
        "region": "ap-southeast-1",
    }
}

ALIYUN_AI_CONFIG = {
    "ap-southeast-1": {
        "API key": "sk-ad3553499cbd4ee5bdad3f5b463a3caf",
    },
    "us-east-1": {
        "API key": "sk-8af7372f6efd49d0afe894f118cbaf0c",
    }
}
