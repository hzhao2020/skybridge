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

# 每个 provider 的每个 region 支持两种 LLM。
PROVIDER_LLM_MODELS: dict[str, tuple[str, str]] = {
    "GCP": ("Gemini-1.5-Pro", "Gemini-1.5-Flash"),
    "AWS": ("Claude-3.5-Sonnet", "Claude-3.5-Haiku"),
    "Aliyun": ("Qwen-VL-Max", "Qwen-VL-Plus"),
}

LLM_MODEL_KEYS = tuple(
    k for models in PROVIDER_LLM_MODELS.values() for k in models
)


def build_llm_node_name(
    provider: str,
    region: str,
    model_key: str,
    operation: Literal["caption", "query"],
) -> str:
    return f"{provider}_{region}__{model_key}_{operation}"


def iter_cloud_llm_deployments() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for provider in CLOUD_PROVIDERS:
        for region in CLOUD_REGIONS[provider]:
            for model_key in PROVIDER_LLM_MODELS[provider]:
                out.append((provider, region, model_key))
    return out


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
        if operation in ("segment", "split"):
            for provider in CLOUD_PROVIDERS:
                for region in CLOUD_REGIONS[provider]:
                    names.append(f"{provider}_{region}_{operation}")
        else:
            for provider, region, model_key in iter_cloud_llm_deployments():
                names.append(build_llm_node_name(provider, region, model_key, operation))
    
    return names


def get_cloud_region_name():
    # 获取所有的(cloud provider, region) 组合
    names: list[str] = []
    for provider in CLOUD_PROVIDERS:
        for region in CLOUD_REGIONS[provider]:
            names.append(f"{provider}_{region}")
    return names



def get_llm_node_name():
    # 获取所有 llm 节点名称（云侧 2-model/region）。
    names: list[str] = []
    for operation in ("caption", "query"):
        for provider, region, model_key in iter_cloud_llm_deployments():
            names.append(build_llm_node_name(provider, region, model_key, operation))
    return names





if __name__ == "__main__":
    print(get_cloud_region_name())