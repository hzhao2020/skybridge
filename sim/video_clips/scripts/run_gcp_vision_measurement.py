"""
对一批图像逐张调用 GCP Vision API 的单一功能（标签检测 或 文本检测/OCR），
记录每张图像该次调用的执行时长，写入 CSV。

设计：每次运行只跑一个 feature（--feature label / text），生成独立 CSV，
两个功能完全分开调用、互不影响（满足「分别单独调用」要求）。

Vision 单图标注为同步接口，故 execution_time_sec = API 调用墙钟时间
（无异步服务端时间戳）。

用法（在 video_clips/scripts 目录下）：
  conda activate sky
  cp config_example.py config.py     # 填写 GCP_CONFIG.gcs_bucket
  python run_gcp_vision_measurement.py --feature label --output results/gcp_label_detection_all.csv
  python run_gcp_vision_measurement.py --feature text  --output results/gcp_ocr_text_detection_all.csv
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
DEFAULT_VIDEO_DIR = (_ROOT.parent / "frames_sample").resolve()

CSV_FIELDS = [
    "run_id",
    "feature",
    "index",
    "image_name",
    "image_path",
    "gcs_uri",
    "image_size_bytes",
    "width",
    "height",
    "upload_sec",
    "start_utc",
    "end_utc",
    "execution_time_sec",
    "success",
    "detection_count",
    "response_size_bytes",
    "response_json_path",
    "error_message",
]


@dataclass(frozen=True)
class GcpConfig:
    project: str
    gcs_bucket: str
    upload_prefix: str


@dataclass
class ImageItem:
    image_path: str
    image_name: str
    gcs_uri: str
    gcs_key: str
    upload_sec: float
    size_bytes: int
    width: int
    height: int


def _load_gcp_config() -> GcpConfig:
    try:
        import config  # type: ignore
    except ImportError as e:
        raise RuntimeError("未找到 config.py。请复制 config_example.py 为 config.py 并填写 GCP_CONFIG。") from e
    raw = getattr(config, "GCP_CONFIG", None)
    if not raw:
        raise RuntimeError("请在 config.py 中配置 GCP_CONFIG。")
    project = str(raw.get("project", "") or "").strip()
    gcs_bucket = str(raw.get("gcs_bucket", "") or "").strip()
    upload_prefix = str(raw.get("vision_upload_prefix", "") or "").strip().strip("/") or "frames_sample"
    if not gcs_bucket:
        raise RuntimeError("GCP_CONFIG.gcs_bucket 不能为空。")
    return GcpConfig(project=project, gcs_bucket=gcs_bucket, upload_prefix=upload_prefix)


def _apply_proxy_from_config() -> None:
    proxy = None
    try:
        import config  # type: ignore

        proxy = getattr(config, "PROXY", None)
    except ImportError:
        pass
    if not proxy:
        return
    proxy = str(proxy).strip()
    for var in ("http_proxy", "https_proxy", "grpc_proxy", "ALL_PROXY"):
        if not os.environ.get(var):
            os.environ[var] = proxy


def _ffprobe_executable() -> str:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        ffprobe = Path(ffmpeg).with_name("ffprobe")
        if ffprobe.is_file():
            return str(ffprobe)
    except Exception:
        pass
    return ""


def probe_dimensions(path: str) -> Tuple[int, int]:
    ffprobe = _ffprobe_executable()
    if not ffprobe:
        return 0, 0
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0, 0
    try:
        s = json.loads(proc.stdout)["streams"][0]
        return int(s.get("width") or 0), int(s.get("height") or 0)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return 0, 0


def _client_options(gcp_cfg: GcpConfig) -> Any:
    from google.api_core.client_options import ClientOptions

    if gcp_cfg.project:
        return ClientOptions(quota_project_id=gcp_cfg.project)
    return None


def _create_storage_client(gcp_cfg: GcpConfig) -> Any:
    from google.cloud import storage

    opts = _client_options(gcp_cfg)
    if gcp_cfg.project:
        return storage.Client(project=gcp_cfg.project, client_options=opts)
    return storage.Client(client_options=opts)


def _create_vision_client(gcp_cfg: GcpConfig) -> Any:
    from google.cloud import vision

    return vision.ImageAnnotatorClient(client_options=_client_options(gcp_cfg))


def upload_image_to_gcs(storage_client: Any, gcp_cfg: GcpConfig, image_path: Path) -> Tuple[str, str, float, bool]:
    key = f"{gcp_cfg.upload_prefix}/{image_path.name}" if gcp_cfg.upload_prefix else image_path.name
    bucket = storage_client.bucket(gcp_cfg.gcs_bucket)
    blob = bucket.blob(key)
    gcs_uri = f"gs://{gcp_cfg.gcs_bucket}/{key}"
    local_size = image_path.stat().st_size
    if blob.exists():
        blob.reload()
        if blob.size is not None and int(blob.size) == int(local_size):
            return key, gcs_uri, 0.0, True
    t0 = time.perf_counter()
    blob.upload_from_filename(str(image_path))
    upload_sec = time.perf_counter() - t0
    return key, gcs_uri, upload_sec, False


def _natural_sort_key(path: Path) -> Tuple[str]:
    m = re.search(r"(\d+)", path.stem)
    return (f"{int(m.group(1)):08d}" if m else path.name.lower(),)


def _list_images(img_dir: Path, limit: Optional[int], glob_pattern: Optional[str]) -> List[Path]:
    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    paths = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if glob_pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.name, glob_pattern)]
    paths.sort(key=_natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"目录 {img_dir} 下未找到图像。")
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


def _response_to_json(resp: Any) -> str:
    from google.protobuf.json_format import MessageToJson

    pb = getattr(resp, "_pb", resp)
    return MessageToJson(pb, preserving_proto_field_name=True, ensure_ascii=False)


def annotate(vision_client: Any, feature: str, gcs_uri: str) -> Tuple[int, str, str]:
    """对单张图像调用指定 feature，返回 (detection_count, error_message, response_json)。"""
    from google.cloud import vision

    image = vision.Image(source=vision.ImageSource(image_uri=gcs_uri))
    if feature == "label":
        resp = vision_client.label_detection(image=image)
        err = resp.error.message or ""
        return len(resp.label_annotations), err, _response_to_json(resp)
    elif feature == "text":
        resp = vision_client.text_detection(image=image)
        err = resp.error.message or ""
        # text_annotations[0] 为整页文本，其余为词；这里记录条目数
        return len(resp.text_annotations), err, _response_to_json(resp)
    else:
        raise ValueError(f"未知 feature: {feature}")


def _preupload_all(storage_client: Any, gcp_cfg: GcpConfig, image_paths: List[Path], silent: bool) -> List[ImageItem]:
    items: List[ImageItem] = []
    if not silent:
        print(f"阶段 1/2：预上传 {len(image_paths)} 张图像到 GCS（不计入执行延迟）")
    for idx, p in enumerate(image_paths, start=1):
        gcs_key, gcs_uri, upload_sec, skipped = upload_image_to_gcs(storage_client, gcp_cfg, p)
        w, h = probe_dimensions(str(p))
        items.append(ImageItem(
            image_path=str(p.resolve()), image_name=p.name, gcs_uri=gcs_uri, gcs_key=gcs_key,
            upload_sec=upload_sec, size_bytes=p.stat().st_size, width=w, height=h,
        ))
        if not silent and (idx % 25 == 0 or idx == len(image_paths)):
            print(f"  uploaded {idx}/{len(image_paths)}", flush=True)
    return items


def _fmt(value: float) -> str:
    return f"{value:.6f}" if value == value else ""


def run_batch(*, image_dir: Path, feature: str, output_csv: Path, limit: Optional[int],
              glob_pattern: Optional[str], request_interval_sec: float, silent: bool = False) -> Path:
    gcp_cfg = _load_gcp_config()
    _apply_proxy_from_config()
    storage_client = _create_storage_client(gcp_cfg)
    vision_client = _create_vision_client(gcp_cfg)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    image_paths = _list_images(image_dir, limit, glob_pattern)
    items = _preupload_all(storage_client, gcp_cfg, image_paths, silent)

    if not silent:
        print(f"run_id={run_id}")
        print(f"feature={feature} images={len(items)}")
        print(f"api=Vision {'LABEL_DETECTION' if feature=='label' else 'TEXT_DETECTION'} (同步调用)")
        print(f"gcs_bucket={gcp_cfg.gcs_bucket}")
        print(f"csv={output_csv.resolve()}")
        print("timing: execution_time_sec = 单次 Vision API 调用墙钟时间")
        print(f"阶段 2/2：逐张调用 {feature} 并记录执行时长")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    response_dir = output_csv.with_suffix("").parent / f"{output_csv.with_suffix('').name}_json"
    response_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for idx, item in enumerate(items, start=1):
            if idx > 1 and request_interval_sec > 0:
                time.sleep(request_interval_sec)

            start = datetime.now(timezone.utc)
            row: Dict[str, Any] = {
                "run_id": run_id, "feature": feature, "index": idx,
                "image_name": item.image_name, "image_path": item.image_path, "gcs_uri": item.gcs_uri,
                "image_size_bytes": item.size_bytes, "width": item.width, "height": item.height,
                "upload_sec": _fmt(item.upload_sec), "start_utc": start.isoformat(),
                "end_utc": "", "execution_time_sec": "", "success": 0,
                "detection_count": "", "response_size_bytes": "", "response_json_path": "",
                "error_message": "",
            }
            t0 = time.perf_counter()
            try:
                count, err, response_json = annotate(vision_client, feature, item.gcs_uri)
                elapsed = time.perf_counter() - t0
                end = datetime.now(timezone.utc)
                response_bytes = response_json.encode("utf-8")
                response_json_path = response_dir / f"{idx:04d}_{item.image_name}.json"
                response_json_path.write_bytes(response_bytes)
                row["end_utc"] = end.isoformat()
                row["execution_time_sec"] = _fmt(elapsed)
                row["detection_count"] = count
                row["response_size_bytes"] = len(response_bytes)
                row["response_json_path"] = str(response_json_path.resolve())
                if err:
                    row["success"] = 0
                    row["error_message"] = err[:500]
                else:
                    row["success"] = 1
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                row["end_utc"] = datetime.now(timezone.utc).isoformat()
                row["execution_time_sec"] = _fmt(elapsed)
                message = getattr(exc, "message", None) or str(exc)
                row["error_message"] = (f"{type(exc).__name__}: {message}\n{traceback.format_exc()}")[:1000]

            writer.writerow(row)
            fp.flush()
            rows_written += 1
            if not silent and (idx % 20 == 0 or idx == len(items)):
                print(f"  {feature} [{idx}/{len(items)}] last exec={row['execution_time_sec']}s "
                      f"count={row['detection_count']} ok={row['success']}", flush=True)

    if not silent:
        print(f"\n完成：{feature} 共写入 {rows_written} 条，结果保存至 {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="GCP Vision 单功能批量测量（label / text 各自独立调用与 CSV）")
    parser.add_argument("--feature", required=True, choices=["label", "text"], help="label=标签检测；text=文本检测/OCR")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_VIDEO_DIR, help=f"图像目录（默认 {DEFAULT_VIDEO_DIR}）")
    parser.add_argument("--glob", dest="glob_pattern", default=None)
    parser.add_argument("--limit", type=int, default=0, help="最多处理张数（0=不限）")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument("--request-interval-sec", type=float, default=0.0, help="相邻调用间隔秒（默认 0）")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        name = "gcp_label_detection" if args.feature == "label" else "gcp_ocr_text_detection"
        args.output = _ROOT / "results" / f"{name}_{ts}.csv"

    try:
        run_batch(image_dir=args.image_dir.resolve(), feature=args.feature, output_csv=args.output.resolve(),
                  limit=limit, glob_pattern=args.glob_pattern, request_interval_sec=args.request_interval_sec)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
