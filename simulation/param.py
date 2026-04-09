# 从config.yaml 中读取配置参数
import yaml
from typing import Any, Sequence
from typing import Literal

Operation = Literal["segment", "split", "caption", "query"]

CLOUD_PROVIDERS = ("p1", "p2", "p3")

CLOUD_REGIONS: dict[str, tuple[str, ...]] = {
    "p1": ("r1", "r2", "r3", "r4"),
    "p2": ("r1", "r2", "r5", "r6"),
    "p3": ("r3", "r4", "r5", "r6"),
}

LLM_PROVIDER_TO_OPTIONS = {
    "p4": ("m1", "m2", "m3", "m4"),
    "p5": ("m3", "m4", "m5", "m6"),
}

LLM_MODEL_KEYS = ("m1", "m2", "m3", "m4", "m5", "m6", "p1", "p2", "p3")


def get_simulation_config(path: str = "config.yaml") -> dict[str, Any]:
    """
    读取 config.yaml 配置文件，返回配置参数字典。
    默认从当前目录的 config.yaml 读取。
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
    # 获取所有的llm节点的名称（仅 LLM-only provider: p4/p5，operation: caption/query）
    names: list[str] = []
    for operation in ("caption", "query"):
        for provider, options in LLM_PROVIDER_TO_OPTIONS.items():
            for opt in options:
                names.append(f"{provider}_{opt}_{operation}")
    return names





if __name__ == "__main__":
    print(get_cloud_region_name())