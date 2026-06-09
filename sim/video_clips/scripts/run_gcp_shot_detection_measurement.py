"""
批量调用 GCP Video Intelligence 镜头检测（SHOT_CHANGE_DETECTION），
记录每个视频在不同时长下的执行延迟、上传时延与视频属性，写入 CSV。

API：google-cloud-videointelligence
  client.annotate_video(features=[SHOT_CHANGE_DETECTION], input_uri=gs://...) -> operation
  operation.result() 阻塞直到完成，返回 shot_annotations

测量语义（三类时延分列）：
  - upload_sec            ：本地视频上传至 GCS 的墙钟时间，单独记录，不计入执行耗时。
  - execution_latency_sec ：客户端观测的执行时延（提交完成 -> operation.result() 返回）。
  - server_execution_sec  ：服务端 update_time - start_time（operation.metadata），最纯净。
  - submit_sec            ：仅 annotate_video 提交调用的墙钟时间。

每条记录还包含：执行时间戳、视频时长、data size、帧率、码率、分辨率、镜头数。

流程：阶段 1 全部预上传到 GCS；阶段 2 依次提交镜头检测 + 等待，记录执行延迟。

用法（在 video_clips/scripts 目录下）：
  conda activate sky
  pip install -r requirements.txt
  gcloud auth application-default login        # 或设置 GOOGLE_APPLICATION_CREDENTIALS
  cp config_example.py config.py               # 填写 GCP_CONFIG.gcs_bucket
  python run_gcp_shot_detection_measurement.py --repeat 3 --request-interval-sec 60
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

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".flv", ".ts", ".webm", ".mpeg4"}
DEFAULT_LOCATION_ID = "us-west1"
SUPPORTED_LOCATIONS = {"us-west1", "asia-east1"}
DEFAULT_JOB_TIMEOUT_SEC = 1800

# 默认输入目录：上一级 video_clips（原始 mkv）。GVI 支持 mkv 及 ffmpeg 可解码的编码，
# 一般可直接用原片测量（data size/码率更真实）；若极少数情况 AV1 解码失败，
# 再用 --video-dir 指向 mp4_h264 等转码后的 H.264 mp4。
DEFAULT_VIDEO_DIR = _ROOT.parent.resolve()

CSV_FIELDS = [
    "run_id",
    "index",
    "repeat_index",
    "video_name",
    "video_path",
    "gcs_uri",
    "video_size_bytes",
    "video_duration_sec",
    "fps",
    "bitrate_bps",
    "width",
    "height",
    "upload_sec",
    "submit_utc",
    "done_utc",
    "submit_sec",
    "execution_latency_sec",
    "server_execution_sec",
    "server_start_utc",
    "server_update_utc",
    "shot_count",
    "success",
    "operation_name",
    "error_message",
]


@dataclass(frozen=True)
class GcpConfig:
    project: str
    gcs_bucket: str
    location_id: str
    upload_prefix: str


@dataclass
class VideoMeta:
    duration_sec: float
    size_bytes: int
    fps: float
    bitrate_bps: float
    width: int
    height: int


@dataclass
class VideoItem:
    video_path: str
    video_name: str
    gcs_uri: str
    gcs_key: str
    upload_sec: float
    meta: VideoMeta


def _load_gcp_config() -> GcpConfig:
    try:
        import config  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "未找到 config.py。请复制 config_example.py 为 config.py 并填写 GCP_CONFIG。"
        ) from e

    raw = getattr(config, "GCP_CONFIG", None)
    if not raw:
        raise RuntimeError("请在 config.py 中配置 GCP_CONFIG。")

    project = str(raw.get("project", "") or "").strip()
    gcs_bucket = str(raw.get("gcs_bucket", "") or "").strip()
    location_id = str(raw.get("location_id", DEFAULT_LOCATION_ID) or DEFAULT_LOCATION_ID).strip()
    upload_prefix = str(raw.get("upload_prefix", "gvi_shot_measure") or "").strip().strip("/")

    if not gcs_bucket:
        raise RuntimeError("GCP_CONFIG.gcs_bucket 不能为空（GVI 输入必须是 GCS bucket）。")
    if location_id not in SUPPORTED_LOCATIONS:
        raise RuntimeError(
            f"location_id={location_id} 不受支持，请使用 {sorted(SUPPORTED_LOCATIONS)} 之一。"
        )
    return GcpConfig(
        project=project,
        gcs_bucket=gcs_bucket,
        location_id=location_id,
        upload_prefix=upload_prefix,
    )


def _apply_proxy_from_config() -> None:
    """若 config.PROXY 配置了代理（如访问 Google 需 VPN），设进环境变量。

    同时设置 http_proxy/https_proxy/grpc_proxy，确保 REST 与 gRPC 都走代理。
    若环境中已存在代理变量，则不覆盖。
    """
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


def _load_job_timeout() -> float:
    timeout = float(DEFAULT_JOB_TIMEOUT_SEC)
    try:
        import config  # type: ignore

        timeout = float(getattr(config, "JOB_TIMEOUT_SEC", timeout))
    except (ImportError, TypeError, ValueError):
        pass
    return max(timeout, 1.0)


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
    raise RuntimeError("未找到 ffprobe，请安装 ffmpeg 或 imageio-ffmpeg。")


def _parse_fraction(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text or text == "0/0":
        return float("nan")
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else float("nan")
        except ValueError:
            return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def probe_video_meta(path: str) -> VideoMeta:
    ffprobe = _ffprobe_executable()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,r_frame_rate,avg_frame_rate,bit_rate,width,height",
        "-show_entries",
        "format=duration,bit_rate,size",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
    duration = float("nan")
    bitrate = float("nan")
    fps = float("nan")
    width = 0
    height = 0

    if proc.returncode == 0 and proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            data = {}
        fmt = data.get("format", {}) or {}
        try:
            duration = float(fmt.get("duration"))
        except (TypeError, ValueError):
            duration = float("nan")
        fmt_bitrate = fmt.get("bit_rate")
        try:
            size_bytes = int(fmt.get("size") or size_bytes)
        except (TypeError, ValueError):
            pass

        v_stream = None
        for s in data.get("streams", []) or []:
            if s.get("codec_type") == "video":
                v_stream = s
                break
        if v_stream is not None:
            width = int(v_stream.get("width") or 0)
            height = int(v_stream.get("height") or 0)
            fps = _parse_fraction(v_stream.get("r_frame_rate") or v_stream.get("avg_frame_rate"))
            stream_bitrate = v_stream.get("bit_rate")
            if stream_bitrate not in (None, "", "N/A"):
                try:
                    bitrate = float(stream_bitrate)
                except (TypeError, ValueError):
                    bitrate = float("nan")
        if bitrate != bitrate and fmt_bitrate not in (None, "", "N/A"):
            try:
                bitrate = float(fmt_bitrate)
            except (TypeError, ValueError):
                bitrate = float("nan")

    if bitrate != bitrate and duration == duration and duration > 0:
        bitrate = size_bytes * 8 / duration

    return VideoMeta(
        duration_sec=duration,
        size_bytes=size_bytes,
        fps=fps,
        bitrate_bps=bitrate,
        width=width,
        height=height,
    )


def _client_options(gcp_cfg: GcpConfig) -> Any:
    """ADC 用户凭证需要 quota project，否则 videointelligence 会返回 403。"""
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


def _create_video_client(gcp_cfg: GcpConfig) -> Any:
    from google.cloud import videointelligence

    return videointelligence.VideoIntelligenceServiceClient(
        client_options=_client_options(gcp_cfg)
    )


def upload_video_to_gcs(
    storage_client: Any,
    gcp_cfg: GcpConfig,
    run_id: str,
    video_path: Path,
) -> Tuple[str, str, float, bool]:
    """上传本地视频到 GCS，返回 (gcs_key, gcs_uri, upload_sec, skipped)。

    使用稳定路径（不含 run_id），若同名同大小对象已存在则跳过上传，
    使重复运行/中断重启时无需重传。
    """
    if gcp_cfg.upload_prefix:
        key = f"{gcp_cfg.upload_prefix}/{video_path.name}"
    else:
        key = video_path.name

    bucket = storage_client.bucket(gcp_cfg.gcs_bucket)
    blob = bucket.blob(key)
    gcs_uri = f"gs://{gcp_cfg.gcs_bucket}/{key}"
    local_size = video_path.stat().st_size

    blob.reload() if blob.exists() else None
    if blob.size is not None and int(blob.size) == int(local_size):
        return key, gcs_uri, 0.0, True

    t0 = time.perf_counter()
    blob.upload_from_filename(str(video_path))
    upload_sec = time.perf_counter() - t0
    return key, gcs_uri, upload_sec, False


def _natural_video_sort_key(path: Path) -> Tuple[int, int, str]:
    match = re.search(r"(\d+)min", path.stem, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def _list_videos(video_dir: Path, limit: Optional[int], glob_pattern: Optional[str]) -> List[Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(
            f"视频目录不存在: {video_dir}\n"
            "提示：默认目录是上一级 video_clips（原始 mkv）。GVI 支持 mkv 与 ffmpeg 可解码编码，"
            "通常可直接用原片；如需用转码后的 mp4，请用 --video-dir 指定。"
        )
    paths = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if glob_pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.name, glob_pattern)]
    paths.sort(key=_natural_video_sort_key)
    if not paths:
        hint = f"（glob={glob_pattern}）" if glob_pattern else ""
        raise FileNotFoundError(f"目录 {video_dir} 下未找到支持的视频{hint}。")
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


def detect_shots(
    video_client: Any,
    *,
    gcs_uri: str,
    location_id: str,
    job_timeout_sec: float,
) -> Tuple[int, float, float, Optional[datetime], Optional[datetime], str]:
    """提交镜头检测并等待结果。

    返回 (shot_count, submit_sec, execution_latency_sec,
          server_start, server_update, operation_name)。
    """
    from google.cloud import videointelligence

    features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
    request = {
        "features": features,
        "input_uri": gcs_uri,
        "location_id": location_id,
    }

    t_submit = time.perf_counter()
    operation = video_client.annotate_video(request=request)
    submit_sec = time.perf_counter() - t_submit

    operation_name = ""
    try:
        operation_name = operation.operation.name or ""
    except Exception:
        operation_name = ""

    t_wait = time.perf_counter()
    response = operation.result(timeout=job_timeout_sec)
    execution_latency_sec = time.perf_counter() - t_wait

    shot_count = 0
    if response.annotation_results:
        for ann in response.annotation_results:
            if ann.shot_annotations:
                shot_count += len(ann.shot_annotations)

    server_start: Optional[datetime] = None
    server_update: Optional[datetime] = None
    try:
        meta = operation.metadata
        if meta is not None and getattr(meta, "annotation_progress", None):
            prog = meta.annotation_progress[0]
            server_start = getattr(prog, "start_time", None)
            server_update = getattr(prog, "update_time", None)
    except Exception:
        pass

    return (
        shot_count,
        submit_sec,
        execution_latency_sec,
        server_start,
        server_update,
        operation_name,
    )


def _to_utc_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return str(dt)


def _server_execution_sec(start: Optional[datetime], update: Optional[datetime]) -> Optional[float]:
    if start is None or update is None:
        return None
    try:
        delta = (update - start).total_seconds()
        return delta if delta >= 0 else None
    except Exception:
        return None


def _preupload_all(
    storage_client: Any,
    gcp_cfg: GcpConfig,
    run_id: str,
    video_paths: List[Path],
    silent: bool,
) -> List[VideoItem]:
    items: List[VideoItem] = []
    if not silent:
        print(f"阶段 1/2：预上传 {len(video_paths)} 个视频到 GCS（不计入执行延迟）")

    for idx, video_path in enumerate(video_paths, start=1):
        gcs_key, gcs_uri, upload_sec, skipped = upload_video_to_gcs(
            storage_client, gcp_cfg, run_id, video_path
        )
        meta = probe_video_meta(str(video_path))
        items.append(
            VideoItem(
                video_path=str(video_path.resolve()),
                video_name=video_path.name,
                gcs_uri=gcs_uri,
                gcs_key=gcs_key,
                upload_sec=upload_sec,
                meta=meta,
            )
        )
        if not silent:
            dur = f"{meta.duration_sec:.1f}" if meta.duration_sec == meta.duration_sec else "nan"
            tag = "skip(已存在)" if skipped else f"upload_sec={upload_sec:.3f}"
            print(
                f"  upload [{idx}/{len(video_paths)}] {video_path.name} "
                f"dur={dur}s size={meta.size_bytes/1e6:.1f}MB {tag}",
                flush=True,
            )
    return items


def _default_output_path(run_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = _ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"gcp_shot_detection_latency_{ts}_{run_id[:8]}.csv"


def _fmt(value: float) -> str:
    return f"{value:.6f}" if value == value else ""


def run_batch(
    *,
    video_dir: Path,
    output_csv: Path,
    limit: Optional[int],
    glob_pattern: Optional[str],
    repeat: int,
    job_timeout_sec: float,
    request_interval_sec: float,
    append: bool = False,
    silent: bool = False,
) -> Path:
    gcp_cfg = _load_gcp_config()
    _apply_proxy_from_config()
    storage_client = _create_storage_client(gcp_cfg)
    video_client = _create_video_client(gcp_cfg)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    repeat = max(1, int(repeat))

    video_paths = _list_videos(video_dir, limit, glob_pattern)
    items = _preupload_all(storage_client, gcp_cfg, run_id, video_paths, silent)

    work_queue: List[Tuple[VideoItem, int]] = []
    for item in items:
        for rep in range(1, repeat + 1):
            work_queue.append((item, rep))

    if not silent:
        print(f"run_id={run_id}")
        print(f"unique_video={len(items)} repeat={repeat} total_runs={len(work_queue)}")
        print(f"api=GVI SHOT_CHANGE_DETECTION location_id={gcp_cfg.location_id}")
        print(f"gcs_bucket={gcp_cfg.gcs_bucket}")
        print(f"request_interval_sec={request_interval_sec} job_timeout_sec={job_timeout_sec}")
        print(f"csv={output_csv.resolve()}")
        print(
            "timing: execution_latency_sec=client(submit->result); "
            "server_execution_sec=server(update-start); upload_sec excluded"
        )
        print("阶段 2/2：annotate_video + 等待结果，记录执行延迟")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    write_header = True
    mode = "w"
    if append and output_csv.exists() and output_csv.stat().st_size > 0:
        mode = "a"
        write_header = False

    with open(output_csv, mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for idx, (item, repeat_index) in enumerate(work_queue, start=1):
            if idx > 1 and request_interval_sec > 0:
                if not silent:
                    print(f"  （间隔 {request_interval_sec:.0f}s 后提交下一次任务…）", flush=True)
                time.sleep(request_interval_sec)

            m = item.meta
            submit_utc = datetime.now(timezone.utc)
            row: Dict[str, Any] = {
                "run_id": run_id,
                "index": idx,
                "repeat_index": repeat_index,
                "video_name": item.video_name,
                "video_path": item.video_path,
                "gcs_uri": item.gcs_uri,
                "video_size_bytes": m.size_bytes,
                "video_duration_sec": _fmt(m.duration_sec),
                "fps": _fmt(m.fps),
                "bitrate_bps": _fmt(m.bitrate_bps),
                "width": m.width,
                "height": m.height,
                "upload_sec": _fmt(item.upload_sec),
                "submit_utc": submit_utc.isoformat(),
                "done_utc": "",
                "submit_sec": "",
                "execution_latency_sec": "",
                "server_execution_sec": "",
                "server_start_utc": "",
                "server_update_utc": "",
                "shot_count": "",
                "success": 0,
                "operation_name": "",
                "error_message": "",
            }

            try:
                (
                    shot_count,
                    submit_sec,
                    execution_latency_sec,
                    server_start,
                    server_update,
                    operation_name,
                ) = detect_shots(
                    video_client,
                    gcs_uri=item.gcs_uri,
                    location_id=gcp_cfg.location_id,
                    job_timeout_sec=job_timeout_sec,
                )
                done_utc = datetime.now(timezone.utc)
                server_exec = _server_execution_sec(server_start, server_update)

                row["done_utc"] = done_utc.isoformat()
                row["submit_sec"] = _fmt(submit_sec)
                row["execution_latency_sec"] = _fmt(execution_latency_sec)
                row["server_execution_sec"] = _fmt(server_exec) if server_exec is not None else ""
                row["server_start_utc"] = _to_utc_iso(server_start)
                row["server_update_utc"] = _to_utc_iso(server_update)
                row["shot_count"] = shot_count
                row["operation_name"] = operation_name
                row["success"] = 1
            except Exception as exc:
                done_utc = datetime.now(timezone.utc)
                message = getattr(exc, "message", None) or str(exc)
                err_detail = f"{type(exc).__name__}: {message}\n{traceback.format_exc()}"
                row["done_utc"] = done_utc.isoformat()
                row["success"] = 0
                row["error_message"] = err_detail[:2000] + ("…" if len(err_detail) > 2000 else "")

            writer.writerow(row)
            fp.flush()
            rows_written += 1

            if not silent:
                status = "OK" if row["success"] else "FAIL"
                print(
                    f"  shot [{idx}/{len(work_queue)}] {item.video_name} "
                    f"rep={repeat_index}/{repeat} {status} "
                    f"dur={row['video_duration_sec'] or '-'}s "
                    f"exec_client={row['execution_latency_sec'] or '-'}s "
                    f"exec_server={row['server_execution_sec'] or '-'}s "
                    f"shots={row['shot_count']}",
                    flush=True,
                )
                if not row["success"]:
                    print(f"    error: {row['error_message'][:200]}", flush=True)

    if not silent:
        print(f"\n完成：共写入 {rows_written} 条记录，结果已保存至 {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCP Video Intelligence 镜头检测批量测量：执行延迟 + 视频属性"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"本地视频目录（默认 {DEFAULT_VIDEO_DIR}）",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help='仅匹配指定文件名 glob，例如 "perfect_planet_*min*.mp4"',
    )
    parser.add_argument("--repeat", type=int, default=3, help="每个视频重复测量次数（默认 3）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理的视频数量（0 表示不限制）")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument(
        "--job-timeout-sec",
        type=float,
        default=None,
        help="单任务最长等待秒数（默认读取 config.JOB_TIMEOUT_SEC）",
    )
    parser.add_argument(
        "--request-interval-sec",
        type=float,
        default=60.0,
        help="相邻任务提交之间的间隔秒数（默认 60，即每次调用服务间隔 1 分钟）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加写入已存在的 CSV（配合固定 --output，用于分块运行汇总到同一文件）",
    )
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    job_timeout = args.job_timeout_sec if args.job_timeout_sec is not None else _load_job_timeout()
    run_id = uuid.uuid4().hex
    output_csv = args.output or _default_output_path(run_id)

    try:
        run_batch(
            video_dir=args.video_dir.resolve(),
            output_csv=output_csv.resolve(),
            limit=limit,
            glob_pattern=args.glob_pattern,
            repeat=args.repeat,
            job_timeout_sec=job_timeout,
            request_interval_sec=args.request_interval_sec,
            append=args.append,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
