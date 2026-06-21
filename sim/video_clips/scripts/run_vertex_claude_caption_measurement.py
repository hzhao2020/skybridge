"""
Call Anthropic Claude models on Vertex AI to caption sampled video frames.

Default input:
  - local frames: ../frames_sample
  - GCS provenance URI: gs://skyflow-ae/frames_sample/<image>

Usage from video_clips/scripts:
  conda activate sky
  python run_vertex_claude_caption_measurement.py --models haiku45 --limit 3
  python run_vertex_claude_caption_measurement.py --models haiku45,sonnet45,opus45

Authentication uses Google ADC:
  gcloud auth application-default login
or:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import fnmatch
import json
import mimetypes
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_IMAGE_DIR = (_ROOT.parent / "frames_sample").resolve()
DEFAULT_LOCATION = "us-east5"

MODEL_ALIASES: Dict[str, Tuple[str, str]] = {
    # Keep Haiku first, per measurement order requested.
    "haiku45": ("Claude Haiku 4.5", "claude-haiku-4-5@20251001"),
    "sonnet45": ("Claude Sonnet 4.5", "claude-sonnet-4-5@20250929"),
    "opus45": ("Claude Opus 4.5", "claude-opus-4-5@20251101"),
}

DEFAULT_PROMPT = (
    "Caption this video frame in one concise English sentence. "
    "Describe the visible scene, main subjects, action, and setting. "
    "Do not mention that this is an image or a frame."
)

CSV_FIELDS = [
    "run_id",
    "model_label",
    "model_id",
    "index",
    "image_name",
    "image_path",
    "gcs_uri",
    "image_size_bytes",
    "width",
    "height",
    "request_image_size_bytes",
    "request_width",
    "request_height",
    "start_utc",
    "end_utc",
    "execution_time_sec",
    "success",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "caption",
    "response_size_bytes",
    "response_json_path",
    "error_message",
]


@dataclass(frozen=True)
class GcpConfig:
    project: str
    location: str
    gcs_bucket: str
    gcs_prefix: str


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str


@dataclass(frozen=True)
class ImageItem:
    image_path: str
    image_name: str
    gcs_uri: str
    size_bytes: int
    width: int
    height: int
    request_size_bytes: int
    request_width: int
    request_height: int
    media_type: str
    data_b64: str


class VertexClaudeError(RuntimeError):
    pass


def _load_gcp_config(location_override: Optional[str], gcs_prefix_override: Optional[str]) -> GcpConfig:
    project = ""
    gcs_bucket = ""
    location = location_override or os.environ.get("VERTEX_LOCATION") or ""
    gcs_prefix = gcs_prefix_override or ""

    try:
        import config  # type: ignore

        raw = getattr(config, "GCP_CONFIG", None) or {}
        project = str(raw.get("project", "") or "").strip()
        gcs_bucket = str(raw.get("gcs_bucket", "") or "").strip()
        if not location:
            # Do not reuse GVI's location_id implicitly; partner models often
            # have a different region from Video Intelligence.
            location = str(raw.get("claude_location_id") or raw.get("vertex_location_id") or "").strip()
        if not gcs_prefix:
            gcs_prefix = str(
                raw.get("caption_gcs_prefix")
                or raw.get("vision_upload_prefix")
                or ""
            ).strip()
    except ImportError:
        pass

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", project).strip()
    if not project:
        raise RuntimeError(
            "未找到 GCP project。请在 config.py 的 GCP_CONFIG.project 中填写，"
            "或设置 GOOGLE_CLOUD_PROJECT。"
        )
    if not gcs_bucket:
        gcs_bucket = "skyflow-ae"
    if not location:
        location = DEFAULT_LOCATION
    # The sampled frames are already expected at gs://skyflow-ae/frames_sample.
    if not gcs_prefix:
        gcs_prefix = "frames_sample"
    return GcpConfig(
        project=project,
        location=location.strip(),
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix.strip().strip("/"),
    )


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
    for var in ("http_proxy", "https_proxy", "ALL_PROXY"):
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
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0, 0
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream.get("width") or 0), int(stream.get("height") or 0)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return 0, 0


def _natural_sort_key(path: Path) -> Tuple[str, str]:
    match = re.search(r"(\d+)", path.stem)
    number = f"{int(match.group(1)):08d}" if match else "99999999"
    return number, path.name.lower()


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


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed in {"image/jpeg", "image/png", "image/webp"}:
        return guessed
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix.lower() == ".png":
        return "image/png"
    if path.suffix.lower() == ".webp":
        return "image/webp"
    raise ValueError(f"不支持的图片类型: {path}")


def _prepare_request_image(path: Path, max_image_side: int, jpeg_quality: int) -> Tuple[bytes, str, int, int, int, int]:
    raw = path.read_bytes()
    width, height = probe_dimensions(str(path))
    if max_image_side <= 0:
        return raw, _media_type(path), len(raw), width, height, width, height
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return raw, _media_type(path), len(raw), width, height, width, height

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        orig_width, orig_height = img.size
        if width == 0 or height == 0:
            width, height = orig_width, orig_height
        if max(orig_width, orig_height) <= max_image_side:
            return raw, _media_type(path), len(raw), width, height, orig_width, orig_height

        img.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        if img.mode not in ("RGB", "L"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if "A" in img.getbands():
                background.paste(img, mask=img.getchannel("A"))
                img = background
            else:
                img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=max(1, min(jpeg_quality, 95)), optimize=True)
        request_bytes = out.getvalue()
        req_width, req_height = img.size
        return request_bytes, "image/jpeg", len(request_bytes), width, height, req_width, req_height


def _gcs_uri(gcp_cfg: GcpConfig, image_name: str) -> str:
    key = f"{gcp_cfg.gcs_prefix}/{image_name}" if gcp_cfg.gcs_prefix else image_name
    return f"gs://{gcp_cfg.gcs_bucket}/{key}"


def _load_image_items(
    image_paths: Sequence[Path],
    gcp_cfg: GcpConfig,
    max_image_side: int,
    jpeg_quality: int,
    silent: bool,
) -> List[ImageItem]:
    items: List[ImageItem] = []
    for idx, path in enumerate(image_paths, start=1):
        raw = path.read_bytes()
        request_bytes, media_type, request_size, width, height, req_width, req_height = _prepare_request_image(
            path, max_image_side=max_image_side, jpeg_quality=jpeg_quality
        )
        items.append(
            ImageItem(
                image_path=str(path.resolve()),
                image_name=path.name,
                gcs_uri=_gcs_uri(gcp_cfg, path.name),
                size_bytes=len(raw),
                width=width,
                height=height,
                request_size_bytes=request_size,
                request_width=req_width,
                request_height=req_height,
                media_type=media_type,
                data_b64=base64.b64encode(request_bytes).decode("ascii"),
            )
        )
        if not silent and (idx % 25 == 0 or idx == len(image_paths)):
            print(f"  prepared images {idx}/{len(image_paths)}", flush=True)
    return items


def _access_token(project: str) -> str:
    try:
        import google.auth
        from google.auth.transport.requests import Request as GoogleAuthRequest
    except ImportError as exc:
        raise RuntimeError(
            "缺少 google-auth。请运行: pip install google-auth requests"
        ) from exc

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    try:
        credentials, _ = google.auth.default(scopes=scopes, quota_project_id=project)
    except TypeError:
        credentials, _ = google.auth.default(scopes=scopes)
        if project and hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(project)
    credentials.refresh(GoogleAuthRequest())
    token = getattr(credentials, "token", None)
    if not token:
        raise RuntimeError("未能从 ADC 获取 access token。")
    return str(token)


def _extract_caption(resp_json: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for block in resp_json.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
    return "\n".join(part.strip() for part in chunks if part.strip()).strip()


def _usage_field(usage: Dict[str, Any], field: str) -> int:
    value = usage.get(field, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def call_vertex_claude(
    *,
    token: str,
    gcp_cfg: GcpConfig,
    model: ModelSpec,
    item: ImageItem,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_timeout_sec: float,
) -> Dict[str, Any]:
    import requests

    endpoint = (
        f"https://{gcp_cfg.location}-aiplatform.googleapis.com/v1/"
        f"projects/{gcp_cfg.project}/locations/{gcp_cfg.location}/"
        f"publishers/anthropic/models/{model.model_id}:rawPredict"
    )
    payload: Dict[str, Any] = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.media_type,
                            "data": item.data_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-goog-user-project": gcp_cfg.project,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=request_timeout_sec)
    if response.status_code >= 400:
        raise VertexClaudeError(
            f"HTTP {response.status_code} from Vertex AI for {model.model_id}: {response.text[:1200]}"
        )
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise VertexClaudeError(f"Vertex AI 返回了非 JSON 响应: {response.text[:1200]}") from exc


def _fmt(value: float) -> str:
    return f"{value:.6f}" if value == value else ""


def _parse_models(raw_models: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw in raw_models.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            label, model_id = [part.strip() for part in raw.split("=", 1)]
            specs.append(ModelSpec(label=label, model_id=model_id))
        elif raw in MODEL_ALIASES:
            label, model_id = MODEL_ALIASES[raw]
            specs.append(ModelSpec(label=label, model_id=model_id))
        else:
            known = ", ".join(MODEL_ALIASES)
            raise ValueError(f"未知模型别名 {raw!r}。可用别名: {known}；或使用 label=model_id。")
    if not specs:
        raise ValueError("至少指定一个模型。")
    return specs


def _load_completed(output_csv: Path) -> set[Tuple[str, str]]:
    completed: set[Tuple[str, str]] = set()
    if not output_csv.is_file():
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if str(row.get("success") or "") == "1":
                completed.add((str(row.get("model_id") or ""), str(row.get("image_name") or "")))
    return completed


def _write_jsonl(jsonl_path: Path, row: Dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def _writer(output_csv: Path, append: bool) -> Tuple[Any, csv.DictWriter]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and output_csv.exists() else "w"
    fp = output_csv.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
    if mode == "w":
        writer.writeheader()
        fp.flush()
    return fp, writer


def _summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_rows": 0,
        "success_rows": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "by_model": {},
    }
    for row in rows:
        model = str(row.get("model_label") or row.get("model_id") or "")
        by_model = summary["by_model"].setdefault(
            model, {"rows": 0, "success_rows": 0, "input_tokens": 0, "output_tokens": 0}
        )
        summary["total_rows"] += 1
        by_model["rows"] += 1
        success = str(row.get("success") or "") == "1"
        if success:
            summary["success_rows"] += 1
            by_model["success_rows"] += 1
        for field in ("input_tokens", "output_tokens"):
            try:
                value = int(row.get(field) or 0)
            except (TypeError, ValueError):
                value = 0
            summary[field] += value
            by_model[field] += value
    return summary


def run_batch(
    *,
    image_dir: Path,
    output_csv: Path,
    models: Sequence[ModelSpec],
    limit: Optional[int],
    glob_pattern: Optional[str],
    location_override: Optional[str],
    gcs_prefix_override: Optional[str],
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_interval_sec: float,
    request_timeout_sec: float,
    max_image_side: int,
    jpeg_quality: int,
    resume: bool,
    use_config_proxy: bool,
    silent: bool = False,
) -> Path:
    gcp_cfg = _load_gcp_config(location_override, gcs_prefix_override)
    if use_config_proxy:
        _apply_proxy_from_config()
    token = _access_token(gcp_cfg.project)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    image_paths = _list_images(image_dir, limit, glob_pattern)
    if not silent:
        print(f"阶段 1/2：准备 {len(image_paths)} 张图片的 base64 输入")
    items = _load_image_items(
        image_paths,
        gcp_cfg,
        max_image_side=max_image_side,
        jpeg_quality=jpeg_quality,
        silent=silent,
    )

    raw_dir = output_csv.with_suffix("").parent / f"{output_csv.with_suffix('').name}_json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_csv.with_suffix(".jsonl")
    completed = _load_completed(output_csv) if resume else set()
    append = resume and output_csv.exists()
    fp, writer = _writer(output_csv, append=append)
    rows_for_summary: List[Dict[str, Any]] = []
    rows_written = 0

    if not silent:
        print(f"run_id={run_id}")
        print(f"project={gcp_cfg.project} location={gcp_cfg.location}")
        print(f"gcs_prefix=gs://{gcp_cfg.gcs_bucket}/{gcp_cfg.gcs_prefix}")
        print("models=" + ", ".join(f"{m.label} ({m.model_id})" for m in models))
        print(f"csv={output_csv.resolve()}")
        if completed:
            print(f"resume: 已跳过成功记录 {len(completed)} 条")
        print("阶段 2/2：逐模型逐帧调用 Vertex AI Claude caption")

    try:
        for model in models:
            for idx, item in enumerate(items, start=1):
                if (model.model_id, item.image_name) in completed:
                    continue
                if rows_written > 0 and request_interval_sec > 0:
                    time.sleep(request_interval_sec)

                start = datetime.now(timezone.utc)
                row: Dict[str, Any] = {
                    "run_id": run_id,
                    "model_label": model.label,
                    "model_id": model.model_id,
                    "index": idx,
                    "image_name": item.image_name,
                    "image_path": item.image_path,
                    "gcs_uri": item.gcs_uri,
                    "image_size_bytes": item.size_bytes,
                    "width": item.width,
                    "height": item.height,
                    "request_image_size_bytes": item.request_size_bytes,
                    "request_width": item.request_width,
                    "request_height": item.request_height,
                    "start_utc": start.isoformat(),
                    "end_utc": "",
                    "execution_time_sec": "",
                    "success": 0,
                    "input_tokens": "",
                    "output_tokens": "",
                    "cache_creation_input_tokens": "",
                    "cache_read_input_tokens": "",
                    "caption": "",
                    "response_size_bytes": "",
                    "response_json_path": "",
                    "error_message": "",
                }
                t0 = time.perf_counter()
                try:
                    resp_json = call_vertex_claude(
                        token=token,
                        gcp_cfg=gcp_cfg,
                        model=model,
                        item=item,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        request_timeout_sec=request_timeout_sec,
                    )
                    elapsed = time.perf_counter() - t0
                    end = datetime.now(timezone.utc)
                    caption = _extract_caption(resp_json)
                    usage = resp_json.get("usage") or {}
                    response_bytes = json.dumps(resp_json, ensure_ascii=False, indent=2).encode("utf-8")
                    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model.model_id)
                    response_path = raw_dir / f"{safe_model}_{idx:04d}_{item.image_name}.json"
                    response_path.write_bytes(response_bytes)

                    row.update(
                        {
                            "end_utc": end.isoformat(),
                            "execution_time_sec": _fmt(elapsed),
                            "success": 1,
                            "input_tokens": _usage_field(usage, "input_tokens"),
                            "output_tokens": _usage_field(usage, "output_tokens"),
                            "cache_creation_input_tokens": _usage_field(usage, "cache_creation_input_tokens"),
                            "cache_read_input_tokens": _usage_field(usage, "cache_read_input_tokens"),
                            "caption": caption,
                            "response_size_bytes": len(response_bytes),
                            "response_json_path": str(response_path.resolve()),
                        }
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    row["end_utc"] = datetime.now(timezone.utc).isoformat()
                    row["execution_time_sec"] = _fmt(elapsed)
                    message = getattr(exc, "message", None) or str(exc)
                    row["error_message"] = (
                        f"{type(exc).__name__}: {message}\n{traceback.format_exc()}"
                    )[:2000]

                writer.writerow(row)
                fp.flush()
                _write_jsonl(jsonl_path, row)
                rows_for_summary.append(row)
                rows_written += 1
                if not silent and (rows_written % 10 == 0 or idx == len(items)):
                    print(
                        f"  {model.label} [{idx}/{len(items)}] ok={row['success']} "
                        f"in={row['input_tokens']} out={row['output_tokens']} "
                        f"exec={row['execution_time_sec']}s",
                        flush=True,
                    )
    finally:
        fp.close()

    summary = _summarize(rows_for_summary)
    summary.update(
        {
            "run_id": run_id,
            "output_csv": str(output_csv.resolve()),
            "output_jsonl": str(jsonl_path.resolve()),
            "raw_response_dir": str(raw_dir.resolve()),
        }
    )
    summary_path = output_csv.with_name(f"{output_csv.with_suffix('').name}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not silent:
        print(f"\n完成：写入 {rows_written} 条新记录，结果保存至 {output_csv}")
        print(f"JSONL={jsonl_path}")
        print(f"summary={summary_path}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI Claude frame caption measurement")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help=f"图像目录（默认 {DEFAULT_IMAGE_DIR}）")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="只处理匹配的文件名，如 'frame_00*.jpg'")
    parser.add_argument("--limit", type=int, default=0, help="最多处理张数（0=不限）")
    parser.add_argument(
        "--models",
        default="haiku45,sonnet45,opus45",
        help="逗号分隔模型。别名: haiku45,sonnet45,opus45；也可写 label=model_id",
    )
    parser.add_argument("--location", default=None, help=f"Vertex AI location（默认 config 或 {DEFAULT_LOCATION}）")
    parser.add_argument("--gcs-prefix", default=None, help="bucket 内 frame 前缀（默认 frames_sample）")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="caption prompt")
    parser.add_argument("--max-tokens", type=int, default=96, help="每张图输出 token 上限")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-interval-sec", type=float, default=0.0, help="相邻请求间隔秒")
    parser.add_argument("--request-timeout-sec", type=float, default=180.0, help="单请求 HTTP timeout 秒")
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="发送给 Claude 前的最长边像素上限；0 表示使用原图（默认 1024）",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="下采样 JPEG 质量（默认 85）")
    parser.add_argument("--resume", action="store_true", help="若输出 CSV 已存在，跳过其中 success=1 的记录")
    parser.add_argument("--no-config-proxy", action="store_true", help="忽略 config.py 里的 PROXY 设置")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    models = _parse_models(args.models)
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        args.output = _ROOT / "results" / f"vertex_claude_frame_captions_{ts}.csv"

    try:
        run_batch(
            image_dir=args.image_dir.resolve(),
            output_csv=args.output.resolve(),
            models=models,
            limit=limit,
            glob_pattern=args.glob_pattern,
            location_override=args.location,
            gcs_prefix_override=args.gcs_prefix,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            request_interval_sec=args.request_interval_sec,
            request_timeout_sec=args.request_timeout_sec,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            resume=args.resume,
            use_config_proxy=not args.no_config_proxy,
        )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
