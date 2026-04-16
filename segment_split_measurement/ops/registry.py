# 测量包精简版：仅注册 Google 镜头检测 + Cloud Function 物理切割
import json
import os
import subprocess
from typing import Dict, Optional, List

try:
    import config
except ImportError:
    config = None

from ops.base import Operation
from ops.impl.google_ops import GoogleVideoSegmentImpl, GoogleCloudFunctionSplitImpl

REGISTRY: Dict[str, Operation] = {}


def register(pid: str, instance: Operation) -> None:
    REGISTRY[pid] = instance


def get_operation(pid: str) -> Operation:
    if pid not in REGISTRY:
        raise ValueError(f"Physical ID '{pid}' not found.")
    return REGISTRY[pid]


BUCKETS = {
    "gcp_us": "video_us",
    "gcp_tw": "video_tw",
    "gcp_sg": "video_sg",
}

GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT = {
    "us-west1": "https://us-west1-project-ab73e1ce-e25c-48b5-a91.cloudfunctions.net/split_measurement",
    "asia-southeast1": "https://video-splitter-nqis2t7p2a-as.a.run.app",
}


def _get_gcp_project_number() -> Optional[str]:
    pn = os.getenv("GCP_PROJECT_NUMBER")
    if pn:
        return pn.strip()
    try:
        project_id = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if not project_id or project_id == "(unset)":
            return None
        out = subprocess.check_output(
            ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return out or None
    except Exception:
        return None


def _get_cloud_function_url(region: str, function_name: str = "video-splitter") -> Optional[str]:
    try:
        url = subprocess.check_output(
            [
                "gcloud",
                "functions",
                "describe",
                function_name,
                "--gen2",
                "--region",
                region,
                "--format",
                "value(serviceConfig.uri)",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        return url if url else None
    except Exception:
        return None


def _build_gcp_videosplit_urls() -> Dict[str, str]:
    if GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT:
        return GCP_VIDEOSPLIT_SERVICE_URLS_DEFAULT.copy()
    raw = os.getenv("GCP_VIDEOSPLIT_SERVICE_URLS")
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return {k: str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            pass
    urls = {}
    for region in ("us-west1", "asia-southeast1"):
        u = _get_cloud_function_url(region)
        if u:
            urls[region] = u
    if len(urls) == 2:
        return urls
    pn = _get_gcp_project_number()
    if not pn:
        return urls
    base = f"https://video-splitter-{pn}.{{region}}.run.app/video_split"
    for region in ("us-west1", "asia-southeast1"):
        if region not in urls:
            urls[region] = base.format(region=region)
    return urls


GCP_VIDEOSPLIT_SERVICE_URLS = _build_gcp_videosplit_urls()

VIDEO_SEGMENT_CATALOG = [
    {"pid": "seg_google_us", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "seg_google_tw", "cls": GoogleVideoSegmentImpl, "provider": "google", "region": "asia-east1", "bucket_key": "gcp_tw"},
]

VIDEO_SPLIT_CATALOG = [
    {"pid": "split_google_us", "cls": GoogleCloudFunctionSplitImpl, "provider": "google", "region": "us-west1", "bucket_key": "gcp_us"},
    {"pid": "split_google_sg", "cls": GoogleCloudFunctionSplitImpl, "provider": "google", "region": "asia-southeast1", "bucket_key": "gcp_sg"},
]

for item in VIDEO_SEGMENT_CATALOG:
    register(item["pid"], item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]]))

for item in VIDEO_SPLIT_CATALOG:
    service_url = None
    if config and hasattr(config, "VIDEO_SPLIT_URLS"):
        provider_urls = config.VIDEO_SPLIT_URLS.get(item["provider"], {})
        service_url = provider_urls.get(item["region"])
    if not service_url:
        service_url = GCP_VIDEOSPLIT_SERVICE_URLS.get(item["region"])
    register(
        item["pid"],
        item["cls"](item["provider"], item["region"], BUCKETS[item["bucket_key"]], service_url=service_url),
    )


def list_supported_operations() -> str:
    lines = ["segment/split only (measurement bundle):"]
    for item in VIDEO_SEGMENT_CATALOG + VIDEO_SPLIT_CATALOG:
        lines.append(f"  - {item['pid']}")
    return "\n".join(lines)


def get_operation_info(pid: str, include_class: bool = False) -> Optional[Dict]:
    for item in VIDEO_SEGMENT_CATALOG + VIDEO_SPLIT_CATALOG:
        if item.get("pid") == pid:
            if include_class:
                return item.copy()
            return {
                "pid": item.get("pid"),
                "provider": item.get("provider"),
                "region": item.get("region"),
                "bucket_key": item.get("bucket_key"),
                "class": item["cls"].__name__,
            }
    return None
