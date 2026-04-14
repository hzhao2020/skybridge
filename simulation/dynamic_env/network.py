"""
基于 InterdatacenterNetworkPerformanceDataset / PaperDataset 的 RTT 与带宽查询。

PaperDataset 目录下仅包含 **AWS / Azure 各自云内** 的测量结果，**不包含跨云厂商**链路。
跨厂商同地域 / 跨厂商跨地域 会给出明确分类，但无法从该数据集得到 RTT、带宽，将抛出
``PaperDatasetCrossProviderError``。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 类型与异常
# ---------------------------------------------------------------------------


class NetworkLinkKind(str, Enum):
    """链路在「厂商 / 地域」维度上的类别（与用户列出的三种场景对应）。"""

    INTRA_PROVIDER_CROSS_REGION = "intra_provider_cross_region"  # 同厂商跨区域
    INTER_PROVIDER_SAME_REGION = "inter_provider_same_region"  # 跨厂商同地域
    INTER_PROVIDER_CROSS_REGION = "inter_provider_cross_region"  # 跨厂商跨地域
    INTRA_PROVIDER_SAME_REGION = "intra_provider_same_region"  # 同厂商同地域（数据集中未必有条目）


class PaperDatasetError(LookupError):
    """PaperDataset 查询失败基类。"""


class PaperDatasetCrossProviderError(PaperDatasetError):
    """PaperDataset 不包含跨云厂商测量。"""


@dataclass(frozen=True)
class NetworkMetrics:
    """单条链路在数据集中聚合后的网络特性（论文数据为 synt.mean 等统计量）。"""

    link_kind: NetworkLinkKind
    rtt_ms: float
    bandwidth_mbps: float
    rtt_ms_std: float | None
    bandwidth_mbps_std: float | None
    sample_count: int
    protocol: str
    provider: str
    region_a: str
    region_b: str


# ---------------------------------------------------------------------------
# 路径与节点名解析
# ---------------------------------------------------------------------------

_DATASET_VENDORS = frozenset({"AWS", "Azure"})


def normalize_provider_for_dataset(provider: str) -> str:
    """与 PaperDataset 顶层目录一致：``AWS``、``Azure``。"""
    key = provider.strip().lower()
    if key == "aws":
        return "AWS"
    if key == "azure":
        return "Azure"
    return provider


def default_paper_dataset_root() -> Path:
    """默认 PaperDataset 根目录：``<SkyBridge>/InterdatacenterNetworkPerformanceDataset/PaperDataset``。"""
    env = os.environ.get("SKYBRIDGE_ICN_PAPER_DATASET")
    if env:
        return Path(env).expanduser().resolve()
    # network.py -> dynamic_env -> simulation -> src -> SkyBridge
    skybridge_root = Path(__file__).resolve().parents[3]
    return (skybridge_root / "InterdatacenterNetworkPerformanceDataset" / "PaperDataset").resolve()


def parse_datacenter_node_name(name: str) -> tuple[str, str]:
    """
    解析仿真中常见的节点名：``{Provider}_{region}`` 或 ``{Provider}_{region}__Model``。
    返回 ``(provider, region)``，provider 大小写保留原样首字母大写约定（如 AWS、GCP）。
    """
    base = name.split("__", 1)[0].strip()
    if "_" not in base:
        raise ValueError(f"无法解析节点名 {name!r}，期望形如 Provider_region")
    provider, region = base.split("_", 1)
    if not provider or not region:
        raise ValueError(f"无法解析节点名 {name!r}：厂商或地域为空")
    return provider, region


# ---------------------------------------------------------------------------
# 地域规范化
# ---------------------------------------------------------------------------

_AZ_SUFFIX = re.compile(r"^(.+-\d+)([a-z])$", re.IGNORECASE)


def normalize_region_strip_az(region: str) -> str:
    """
    去掉末尾可用区字母（如 ``ap-southeast-1a`` → ``ap-southeast-1``）。
    Azure 的 ``us-east-1`` 等保持不变。
    """
    region = region.strip()
    m = _AZ_SUFFIX.match(region)
    if m:
        return m.group(1)
    return region


def _region_token_for_inter_provider_compare(provider: str, region: str) -> str:
    """
    用于判断「跨厂商是否同地域」：在去掉 AZ 后，对 GCP 常见 ``us-east1`` 形式插入连字符，
    以便与 ``us-east-1`` 对齐。其它厂商保持小写规范化字符串。
    """
    r = normalize_region_strip_az(region)
    if provider.upper() == "GCP":
        # europe-west1 -> europe-west-1；us-east1 -> us-east-1
        m = re.match(r"^(.+?)(\d+)$", r)
        if m and "-" not in m.group(2) and not m.group(1).endswith("-"):
            r = f"{m.group(1)}-{m.group(2)}"
    return r.lower()


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------

_IndexKey = tuple[str, str, str, str]  # provider, reg_lo, reg_hi, proto


def _iter_result_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"PaperDataset 目录不存在: {root}")
    yield from root.rglob("*_results.json")


def _synt_mean(blob: dict[str, Any] | None, key: str) -> float | None:
    if not blob:
        return None
    s = blob.get("synt")
    if not isinstance(s, dict):
        return None
    m = s.get(key)
    if m is None or not isinstance(m, (int, float)):
        return None
    return float(m)


def _load_records_from_file(path: Path) -> list[tuple[str, str, str, str, float, float, float | None, float | None]]:
    """
    从单个 *_results.json 解析出若干条：
    (vendor, norm_snd, norm_rcv, proto, rtt_ms, tput_mbps, rtt_std, tput_std)
    """
    try:
        idx = path.parts.index("PaperDataset")
        vendor = path.parts[idx + 1]
    except (ValueError, IndexError):
        vendor = "Unknown"

    out: list[tuple[str, str, str, str, float, float, float | None, float | None]] = []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return out
    for rec in data:
        if not isinstance(rec, dict):
            continue
        camp = rec.get("camp")
        if not isinstance(camp, dict):
            continue
        rs = camp.get("region_snd")
        rr = camp.get("region_rcv")
        if not isinstance(rs, str) or not isinstance(rr, str):
            continue
        proto = str(rec.get("proto") or "").lower() or "unknown"
        results = rec.get("results")
        if not isinstance(results, dict):
            continue
        delay = results.get("delay")
        tput = results.get("tput")
        rtt = _synt_mean(delay if isinstance(delay, dict) else None, "mean")
        bw = _synt_mean(tput if isinstance(tput, dict) else None, "mean")
        if rtt is None or bw is None:
            continue
        rtt_std = None
        bw_std = None
        if isinstance(delay, dict) and isinstance(delay.get("synt"), dict):
            rs_ = delay["synt"].get("std")
            if isinstance(rs_, (int, float)):
                rtt_std = float(rs_)
        if isinstance(tput, dict) and isinstance(tput.get("synt"), dict):
            bs_ = tput["synt"].get("std")
            if isinstance(bs_, (int, float)):
                bw_std = float(bs_)
        ns = normalize_region_strip_az(rs)
        nr = normalize_region_strip_az(rr)
        lo, hi = sorted((ns, nr))
        out.append((vendor, lo, hi, proto, rtt, bw, rtt_std, bw_std))
    return out


_index_agg: dict[_IndexKey, list[tuple[float, float, float | None, float | None]]] | None = None
_index_root: Path | None = None


def clear_paper_dataset_index_cache() -> None:
    global _index_agg, _index_root
    _index_agg = None
    _index_root = None


def _ensure_index(root: Path) -> dict[_IndexKey, list[tuple[float, float, float | None, float | None]]]:
    global _index_agg, _index_root
    root = root.resolve()
    if _index_agg is not None and _index_root == root:
        return _index_agg
    agg: dict[_IndexKey, list[tuple[float, float, float | None, float | None]]] = {}
    for fp in _iter_result_files(root):
        for row in _load_records_from_file(fp):
            vendor, lo, hi, proto, rtt, bw, rs, bs = row
            key = (vendor, lo, hi, proto)
            agg.setdefault(key, []).append((rtt, bw, rs, bs))
    _index_agg = agg
    _index_root = root
    return agg


def _aggregate_samples(
    samples: list[tuple[float, float, float | None, float | None]],
) -> tuple[float, float, float | None, float | None, int]:
    n = len(samples)
    rtt_m = sum(s[0] for s in samples) / n
    bw_m = sum(s[1] for s in samples) / n
    rtt_stds = [s[2] for s in samples if s[2] is not None]
    bw_stds = [s[3] for s in samples if s[3] is not None]
    rtt_sd = sum(rtt_stds) / len(rtt_stds) if rtt_stds else None
    bw_sd = sum(bw_stds) / len(bw_stds) if bw_stds else None
    return rtt_m, bw_m, rtt_sd, bw_sd, n


def _lookup_intra_provider(
    index: dict[_IndexKey, list[tuple[float, float, float | None, float | None]]],
    provider: str,
    ra: str,
    rb: str,
    *,
    prefer_proto: str,
) -> tuple[float, float, float | None, float | None, int, str, NetworkLinkKind]:
    na = normalize_region_strip_az(ra)
    nb = normalize_region_strip_az(rb)
    lo, hi = sorted((na, nb))
    if lo == hi:
        kind = NetworkLinkKind.INTRA_PROVIDER_SAME_REGION
    else:
        kind = NetworkLinkKind.INTRA_PROVIDER_CROSS_REGION

    prefer_proto = prefer_proto.lower()
    for proto in (prefer_proto, "udp", "tcp"):
        key = (provider, lo, hi, proto)
        if key in index and index[key]:
            rtt_m, bw_m, rtt_sd, bw_sd, n = _aggregate_samples(index[key])
            return rtt_m, bw_m, rtt_sd, bw_sd, n, proto, kind

    raise PaperDatasetError(
        f"PaperDataset 中未找到 {provider} 地域对 ({na}, {nb}) 的测量"
        f"（已尝试协议 {prefer_proto} / udp / tcp）。"
    )


# ---------------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------------


def classify_network_link(node_a: str, node_b: str) -> NetworkLinkKind:
    """根据两个节点名划分链路类别（不读取数据集）。"""
    p1, r1 = parse_datacenter_node_name(node_a)
    p2, r2 = parse_datacenter_node_name(node_b)
    if p1.lower() == "local" or p2.lower() == "local":
        raise ValueError("local 节点不参与数据中心间 PaperDataset 查询")
    if p1 == p2:
        if normalize_region_strip_az(r1) == normalize_region_strip_az(r2):
            return NetworkLinkKind.INTRA_PROVIDER_SAME_REGION
        return NetworkLinkKind.INTRA_PROVIDER_CROSS_REGION
    t1 = _region_token_for_inter_provider_compare(p1, r1)
    t2 = _region_token_for_inter_provider_compare(p2, r2)
    if t1 == t2:
        return NetworkLinkKind.INTER_PROVIDER_SAME_REGION
    return NetworkLinkKind.INTER_PROVIDER_CROSS_REGION


def get_network_metrics(
    node_a: str,
    node_b: str,
    *,
    dataset_root: Path | str | None = None,
    prefer_proto: str = "udp",
) -> NetworkMetrics:
    """
    输入两个节点名（``Provider_region`` 或与 param.yaml 一致的带 ``__Model`` 后缀形式），
    返回 PaperDataset 中聚合得到的 RTT（ms）与带宽（Mbps）。

    - **同厂商跨区域**：可查表（AWS / Azure）。
    - **跨厂商**：PaperDataset 无跨云链路，抛出 ``PaperDatasetCrossProviderError``。
    - **同厂商同地域**：若数据集中无对应条目，抛出 ``PaperDatasetError``。

    ``prefer_proto``：优先使用的传输协议（``udp`` / ``tcp``），缺省时按 udp → tcp 回退。
    """
    root = Path(dataset_root).resolve() if dataset_root else default_paper_dataset_root()
    kind = classify_network_link(node_a, node_b)
    p1, r1 = parse_datacenter_node_name(node_a)
    p2, r2 = parse_datacenter_node_name(node_b)

    if kind in (
        NetworkLinkKind.INTER_PROVIDER_SAME_REGION,
        NetworkLinkKind.INTER_PROVIDER_CROSS_REGION,
    ):
        raise PaperDatasetCrossProviderError(
            "PaperDataset 仅包含单云厂商内部测量，不包含跨 AWS/Azure（或其它云）的链路。"
            f"当前链路类别为 {kind.value}（{node_a!r} ↔ {node_b!r}），无法提供 RTT 与带宽。"
        )

    prov = normalize_provider_for_dataset(p1)
    if prov not in _DATASET_VENDORS:
        raise PaperDatasetError(
            f"PaperDataset 仅索引 AWS 与 Azure，当前厂商为 {p1!r}。"
        )

    index = _ensure_index(root)
    rtt_m, bw_m, rtt_sd, bw_sd, n, proto, intra_kind = _lookup_intra_provider(
        index, prov, r1, r2, prefer_proto=prefer_proto
    )
    na = normalize_region_strip_az(r1)
    nb = normalize_region_strip_az(r2)
    return NetworkMetrics(
        link_kind=intra_kind,
        rtt_ms=rtt_m,
        bandwidth_mbps=bw_m,
        rtt_ms_std=rtt_sd,
        bandwidth_mbps_std=bw_sd,
        sample_count=n,
        protocol=proto,
        provider=prov,
        region_a=na,
        region_b=nb,
    )


def get_rtt_and_bandwidth(
    node_a: str,
    node_b: str,
    **kwargs: Any,
) -> tuple[float, float]:
    """
    便捷接口：仅返回 ``(rtt_ms, bandwidth_mbps)``，参数同 :func:`get_network_metrics`。
    """
    m = get_network_metrics(node_a, node_b, **kwargs)
    return m.rtt_ms, m.bandwidth_mbps
