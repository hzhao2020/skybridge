"""仅含通信端点链路分类（供 ``measured_workflow_network`` 使用）。"""

from __future__ import annotations

import re
from enum import Enum


class NetworkLinkKind(str, Enum):
    INTRA_PROVIDER_CROSS_REGION = "intra_provider_cross_region"
    INTER_PROVIDER_SAME_REGION = "inter_provider_same_region"
    INTER_PROVIDER_CROSS_REGION = "inter_provider_cross_region"
    INTRA_PROVIDER_SAME_REGION = "intra_provider_same_region"


_AZ_SUFFIX = re.compile(r"^(.+-\d+)([a-z])$", re.IGNORECASE)


def normalize_region_strip_az(region: str) -> str:
    region = region.strip()
    m = _AZ_SUFFIX.match(region)
    if m:
        return m.group(1)
    return region


def parse_datacenter_node_name(name: str) -> tuple[str, str]:
    base = name.split("__", 1)[0].strip()
    if "_" not in base:
        raise ValueError(f"无法解析节点名 {name!r}，期望形如 Provider_region")
    provider, region = base.split("_", 1)
    if not provider or not region:
        raise ValueError(f"无法解析节点名 {name!r}：厂商或地域为空")
    return provider, region


def _region_token_for_inter_provider_compare(provider: str, region: str) -> str:
    r = normalize_region_strip_az(region)
    if provider.upper() == "GCP":
        m = re.match(r"^(.+?)(\d+)$", r)
        if m and "-" not in m.group(2) and not m.group(1).endswith("-"):
            r = f"{m.group(1)}-{m.group(2)}"
    return r.lower()


def classify_network_link(node_a: str, node_b: str) -> NetworkLinkKind:
    """根据两个端点名（``Provider_region``，无 local）划分链路类别。"""
    p1, r1 = parse_datacenter_node_name(node_a)
    p2, r2 = parse_datacenter_node_name(node_b)
    if p1.lower() == "local" or p2.lower() == "local":
        raise ValueError("local 节点请在外部单独处理")
    if p1 == p2:
        if normalize_region_strip_az(r1) == normalize_region_strip_az(r2):
            return NetworkLinkKind.INTRA_PROVIDER_SAME_REGION
        return NetworkLinkKind.INTRA_PROVIDER_CROSS_REGION
    t1 = _region_token_for_inter_provider_compare(p1, r1)
    t2 = _region_token_for_inter_provider_compare(p2, r2)
    if t1 == t2:
        return NetworkLinkKind.INTER_PROVIDER_SAME_REGION
    return NetworkLinkKind.INTER_PROVIDER_CROSS_REGION
