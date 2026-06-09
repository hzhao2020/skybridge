"""
批量调用 GCP Speech-to-Text 语音转写（long_running_recognize），
记录每个音频在不同时长下的执行延迟、上传时延与音频属性，写入 CSV。

API：google-cloud-speech (v1)
  client.long_running_recognize(config=..., audio=RecognitionAudio(uri=gs://...)) -> operation
  operation.result() 阻塞直到完成，返回 results[].alternatives[].transcript

注意：>1 分钟音频必须用 long_running_recognize + GCS 输入；GCP 不支持 AAC，
故输入用 FLAC（16kHz 单声道，由 m4a 转码而来）。

测量语义（三类时延分列）：
  - upload_sec            ：本地音频上传至 GCS 的墙钟时间，单独记录，不计入执行耗时。
  - execution_latency_sec ：客户端观测的执行时延（提交完成 -> operation.result() 返回）。
  - server_execution_sec  ：服务端 last_update_time - start_time（operation.metadata），最纯净。
  - submit_sec            ：仅 long_running_recognize 提交调用的墙钟时间。

每条记录还包含：执行时间戳、音频时长、data size、采样率、声道数、码率、转写结果数/词数。

用法（在 video_clips/scripts 目录下）：
  conda activate sky
  pip install -r requirements.txt
  cp config_example.py config.py    # 填写 GCP_CONFIG.gcs_bucket
  python run_gcp_speech_measurement.py --repeat 3 --request-interval-sec 60
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

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a"}
DEFAULT_JOB_TIMEOUT_SEC = 1800
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SAMPLE_RATE = 16000
# 「标准识别模型」(Standard)：default 模型 + 非增强
DEFAULT_MODEL = "default"
DEFAULT_USE_ENHANCED = False

# 默认输入目录：上一级 video_clips/audio_flac（GCP 兼容的 FLAC）
DEFAULT_AUDIO_DIR = (_ROOT.parent / "audio_flac").resolve()

CSV_FIELDS = [
    "run_id",
    "index",
    "repeat_index",
    "audio_name",
    "audio_path",
    "gcs_uri",
    "audio_size_bytes",
    "audio_duration_sec",
    "sample_rate_hz",
    "channels",
    "bitrate_bps",
    "upload_sec",
    "submit_utc",
    "done_utc",
    "submit_sec",
    "execution_latency_sec",
    "server_execution_sec",
    "server_start_utc",
    "server_update_utc",
    "result_count",
    "word_count",
    "transcript_chars",
    "success",
    "operation_name",
    "error_message",
]


@dataclass(frozen=True)
class GcpConfig:
    project: str
    gcs_bucket: str
    upload_prefix: str


@dataclass
class AudioMeta:
    duration_sec: float
    size_bytes: int
    sample_rate_hz: int
    channels: int
    bitrate_bps: float


@dataclass
class AudioItem:
    audio_path: str
    audio_name: str
    gcs_uri: str
    gcs_key: str
    upload_sec: float
    meta: AudioMeta


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
    # 语音用独立子目录，避免与视频混放；可用 config.SPEECH_UPLOAD_PREFIX 覆盖
    upload_prefix = str(raw.get("speech_upload_prefix", "") or "").strip().strip("/")
    if not upload_prefix:
        upload_prefix = "gcp_speech_measure"

    if not gcs_bucket:
        raise RuntimeError("GCP_CONFIG.gcs_bucket 不能为空。")
    return GcpConfig(project=project, gcs_bucket=gcs_bucket, upload_prefix=upload_prefix)


def _load_str(name: str, default: str) -> str:
    try:
        import config  # type: ignore

        return str(getattr(config, name, default) or default)
    except ImportError:
        return default


def _load_int(name: str, default: int) -> int:
    try:
        import config  # type: ignore

        return int(getattr(config, name, default))
    except (ImportError, TypeError, ValueError):
        return default


def _load_bool(name: str, default: bool) -> bool:
    try:
        import config  # type: ignore

        return bool(getattr(config, name, default))
    except ImportError:
        return default


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


def probe_audio_meta(path: str) -> AudioMeta:
    ffprobe = _ffprobe_executable()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,bit_rate",
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
    sample_rate = 0
    channels = 0

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
        try:
            size_bytes = int(fmt.get("size") or size_bytes)
        except (TypeError, ValueError):
            pass
        fmt_bitrate = fmt.get("bit_rate")

        a_stream = None
        for s in data.get("streams", []) or []:
            if s.get("codec_type") == "audio":
                a_stream = s
                break
        if a_stream is not None:
            try:
                sample_rate = int(a_stream.get("sample_rate") or 0)
            except (TypeError, ValueError):
                sample_rate = 0
            try:
                channels = int(a_stream.get("channels") or 0)
            except (TypeError, ValueError):
                channels = 0
            sb = a_stream.get("bit_rate")
            if sb not in (None, "", "N/A"):
                try:
                    bitrate = float(sb)
                except (TypeError, ValueError):
                    bitrate = float("nan")
        if bitrate != bitrate and fmt_bitrate not in (None, "", "N/A"):
            try:
                bitrate = float(fmt_bitrate)
            except (TypeError, ValueError):
                bitrate = float("nan")

    if bitrate != bitrate and duration == duration and duration > 0:
        bitrate = size_bytes * 8 / duration

    return AudioMeta(
        duration_sec=duration,
        size_bytes=size_bytes,
        sample_rate_hz=sample_rate,
        channels=channels,
        bitrate_bps=bitrate,
    )


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


def _create_speech_client(gcp_cfg: GcpConfig) -> Any:
    from google.cloud import speech

    return speech.SpeechClient(client_options=_client_options(gcp_cfg))


def upload_audio_to_gcs(
    storage_client: Any,
    gcp_cfg: GcpConfig,
    audio_path: Path,
) -> Tuple[str, str, float, bool]:
    """上传本地音频到 GCS，稳定路径 + 已存在则跳过。返回 (key, uri, upload_sec, skipped)。"""
    key = f"{gcp_cfg.upload_prefix}/{audio_path.name}" if gcp_cfg.upload_prefix else audio_path.name
    bucket = storage_client.bucket(gcp_cfg.gcs_bucket)
    blob = bucket.blob(key)
    gcs_uri = f"gs://{gcp_cfg.gcs_bucket}/{key}"
    local_size = audio_path.stat().st_size

    if blob.exists():
        blob.reload()
        if blob.size is not None and int(blob.size) == int(local_size):
            return key, gcs_uri, 0.0, True

    t0 = time.perf_counter()
    blob.upload_from_filename(str(audio_path))
    upload_sec = time.perf_counter() - t0
    return key, gcs_uri, upload_sec, False


def _natural_sort_key(path: Path) -> Tuple[int, int, str]:
    match = re.search(r"(\d+)min", path.stem, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, 0, path.name.lower())


def _list_audio(audio_dir: Path, limit: Optional[int], glob_pattern: Optional[str]) -> List[Path]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"音频目录不存在: {audio_dir}\n"
            "提示：默认目录是 video_clips/audio_flac（FLAC）。"
        )
    paths = [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES]
    if glob_pattern:
        paths = [p for p in paths if fnmatch.fnmatch(p.name, glob_pattern)]
    paths.sort(key=_natural_sort_key)
    if not paths:
        hint = f"（glob={glob_pattern}）" if glob_pattern else ""
        raise FileNotFoundError(f"目录 {audio_dir} 下未找到支持的音频{hint}。")
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


def transcribe(
    speech_client: Any,
    *,
    gcs_uri: str,
    meta: AudioMeta,
    language: str,
    sample_rate: int,
    model: str,
    use_enhanced: bool,
    job_timeout_sec: float,
) -> Tuple[int, int, int, float, float, Optional[datetime], Optional[datetime], str]:
    """提交转写并等待结果。

    返回 (result_count, word_count, transcript_chars, submit_sec,
          execution_latency_sec, server_start, server_update, operation_name)。
    """
    from google.cloud import speech

    # model="default" + use_enhanced=False 即定价页所述「标准识别模型」(Standard)。
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=sample_rate or (meta.sample_rate_hz or DEFAULT_SAMPLE_RATE),
        language_code=language,
        audio_channel_count=meta.channels or 1,
        enable_automatic_punctuation=True,
        model=model,
        use_enhanced=use_enhanced,
    )
    audio = speech.RecognitionAudio(uri=gcs_uri)

    t_submit = time.perf_counter()
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    submit_sec = time.perf_counter() - t_submit

    operation_name = ""
    try:
        operation_name = operation.operation.name or ""
    except Exception:
        operation_name = ""

    t_wait = time.perf_counter()
    response = operation.result(timeout=job_timeout_sec)
    execution_latency_sec = time.perf_counter() - t_wait

    result_count = 0
    word_count = 0
    transcript_chars = 0
    for res in response.results:
        if not res.alternatives:
            continue
        result_count += 1
        text = res.alternatives[0].transcript or ""
        transcript_chars += len(text)
        word_count += len(text.split())

    server_start: Optional[datetime] = None
    server_update: Optional[datetime] = None
    try:
        meta_obj = operation.metadata
        if meta_obj is not None:
            server_start = getattr(meta_obj, "start_time", None)
            server_update = getattr(meta_obj, "last_update_time", None)
    except Exception:
        pass

    return (
        result_count,
        word_count,
        transcript_chars,
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
    audio_paths: List[Path],
    silent: bool,
) -> List[AudioItem]:
    items: List[AudioItem] = []
    if not silent:
        print(f"阶段 1/2：预上传 {len(audio_paths)} 个音频到 GCS（不计入执行延迟）")

    for idx, audio_path in enumerate(audio_paths, start=1):
        gcs_key, gcs_uri, upload_sec, skipped = upload_audio_to_gcs(
            storage_client, gcp_cfg, audio_path
        )
        meta = probe_audio_meta(str(audio_path))
        items.append(
            AudioItem(
                audio_path=str(audio_path.resolve()),
                audio_name=audio_path.name,
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
                f"  upload [{idx}/{len(audio_paths)}] {audio_path.name} "
                f"dur={dur}s size={meta.size_bytes/1e6:.1f}MB sr={meta.sample_rate_hz} "
                f"ch={meta.channels} {tag}",
                flush=True,
            )
    return items


def _default_output_path(run_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = _ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"gcp_speech_latency_{ts}_{run_id[:8]}.csv"


def _fmt(value: float) -> str:
    return f"{value:.6f}" if value == value else ""


def run_batch(
    *,
    audio_dir: Path,
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
    language = _load_str("SPEECH_LANGUAGE", DEFAULT_LANGUAGE)
    sample_rate = _load_int("SPEECH_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)
    model = _load_str("SPEECH_MODEL", DEFAULT_MODEL)
    use_enhanced = _load_bool("SPEECH_USE_ENHANCED", DEFAULT_USE_ENHANCED)

    storage_client = _create_storage_client(gcp_cfg)
    speech_client = _create_speech_client(gcp_cfg)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    repeat = max(1, int(repeat))

    audio_paths = _list_audio(audio_dir, limit, glob_pattern)
    items = _preupload_all(storage_client, gcp_cfg, audio_paths, silent)

    work_queue: List[Tuple[AudioItem, int]] = []
    for item in items:
        for rep in range(1, repeat + 1):
            work_queue.append((item, rep))

    if not silent:
        print(f"run_id={run_id}")
        print(f"unique_audio={len(items)} repeat={repeat} total_runs={len(work_queue)}")
        print(
            f"api=Speech long_running_recognize language={language} sample_rate={sample_rate} "
            f"model={model} use_enhanced={use_enhanced}（标准识别模型）"
        )
        print(f"gcs_bucket={gcp_cfg.gcs_bucket}")
        print(f"request_interval_sec={request_interval_sec} job_timeout_sec={job_timeout_sec}")
        print(f"csv={output_csv.resolve()}")
        print(
            "timing: execution_latency_sec=client(submit->result); "
            "server_execution_sec=server(last_update-start); upload_sec excluded"
        )
        print("阶段 2/2：long_running_recognize + 等待结果，记录执行延迟")

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
                "audio_name": item.audio_name,
                "audio_path": item.audio_path,
                "gcs_uri": item.gcs_uri,
                "audio_size_bytes": m.size_bytes,
                "audio_duration_sec": _fmt(m.duration_sec),
                "sample_rate_hz": m.sample_rate_hz,
                "channels": m.channels,
                "bitrate_bps": _fmt(m.bitrate_bps),
                "upload_sec": _fmt(item.upload_sec),
                "submit_utc": submit_utc.isoformat(),
                "done_utc": "",
                "submit_sec": "",
                "execution_latency_sec": "",
                "server_execution_sec": "",
                "server_start_utc": "",
                "server_update_utc": "",
                "result_count": "",
                "word_count": "",
                "transcript_chars": "",
                "success": 0,
                "operation_name": "",
                "error_message": "",
            }

            try:
                (
                    result_count,
                    word_count,
                    transcript_chars,
                    submit_sec,
                    execution_latency_sec,
                    server_start,
                    server_update,
                    operation_name,
                ) = transcribe(
                    speech_client,
                    gcs_uri=item.gcs_uri,
                    meta=m,
                    language=language,
                    sample_rate=sample_rate,
                    model=model,
                    use_enhanced=use_enhanced,
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
                row["result_count"] = result_count
                row["word_count"] = word_count
                row["transcript_chars"] = transcript_chars
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
                    f"  asr [{idx}/{len(work_queue)}] {item.audio_name} "
                    f"rep={repeat_index}/{repeat} {status} "
                    f"dur={row['audio_duration_sec'] or '-'}s "
                    f"exec_client={row['execution_latency_sec'] or '-'}s "
                    f"exec_server={row['server_execution_sec'] or '-'}s "
                    f"words={row['word_count']}",
                    flush=True,
                )
                if not row["success"]:
                    print(f"    error: {row['error_message'][:200]}", flush=True)

    if not silent:
        print(f"\n完成：共写入 {rows_written} 条记录，结果已保存至 {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCP Speech-to-Text 语音转写批量测量：执行延迟 + 音频属性"
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"本地音频目录（默认 {DEFAULT_AUDIO_DIR}）",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help='仅匹配指定文件名 glob，例如 "perfect_planet_*min.flac"',
    )
    parser.add_argument("--repeat", type=int, default=3, help="每个音频重复测量次数（默认 3）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理的音频数量（0 表示不限制）")
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
        help="相邻任务提交之间的间隔秒数（默认 60）",
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
            audio_dir=args.audio_dir.resolve(),
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
