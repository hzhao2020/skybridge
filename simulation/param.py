# 从 param.yaml 中读取配置参数
import yaml
from typing import Any, Sequence
from typing import Literal

Operation = Literal["segment", "split", "caption", "query"]

CLOUD_PROVIDERS = ("GCP", "AWS", "Aliyun")

CLOUD_REGIONS: dict[str, tuple[str, ...]] = {
    "GCP": ("us-east1", "us-west1", "europe-west1", "asia-east1"),
    "AWS": ("us-west-2", "us-east-2", "ap-southeast-1", "eu-central-1"),
    "Aliyun": ("cn-shanghai", "cn-beijing", "us-west-1", "ap-southeast-1"),
}

# 当前采用“云侧 provider+region 直接承载 LLM 节点”的方式，
# 不再使用独立的 LLM-only provider。
LLM_PROVIDER_TO_OPTIONS: dict[str, tuple[str, ...]] = {}

LLM_MODEL_KEYS = (
    "Gemini-1.5-Pro",
    "Gemini-1.5-Flash",
    "Claude-3.5-Sonnet",
    "Claude-3.5-Haiku",
    "Qwen-VL-Max",
    "Qwen-VL-Plus",
)


def get_simulation_config(path: str = "param.yaml") -> dict[str, Any]:
    """
    读取 param.yaml 配置文件，返回配置参数字典。
    默认从当前目录的 param.yaml 读取。
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_node_name():
    ops: Sequence[Operation] = ("segment", "split", "caption", "query")
    names: list[str] = []

    for operation in ops:
        for provider in CLOUD_PROVIDERS:
            for region in CLOUD_REGIONS[provider]:
                names.append(f"{provider}_{region}_{operation}")

        if operation in ("caption", "query"):
            for provider, options in LLM_PROVIDER_TO_OPTIONS.items():
                for opt in options:
                    names.append(f"{provider}_{opt}_{operation}")
    
    return names


def get_cloud_region_name():
    # 获取所有的(cloud provider, region) 组合
    names: list[str] = []
    for provider in CLOUD_PROVIDERS:
        for region in CLOUD_REGIONS[provider]:
            names.append(f"{provider}_{region}")
    return names



def get_llm_node_name():
    # 获取所有的 llm 节点名称（当前无 LLM-only provider）
    names: list[str] = []
    for operation in ("caption", "query"):
        for provider, options in LLM_PROVIDER_TO_OPTIONS.items():
            for opt in options:
                names.append(f"{provider}_{opt}_{operation}")
    return names





if __name__ == "__main__":
    print(get_cloud_region_name())