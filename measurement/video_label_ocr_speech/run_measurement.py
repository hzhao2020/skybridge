"""
截断本地长视频 → 上传 GCS → 依次调用 Google Video Intelligence：
Label Detection、Text Detection(OCR)、Speech Transcription；
分项写入耗时（见 timings_sec 字段说明），结果写入本地 JSON；VI 结构化结果另写入 GCS output_uri。

计时语义（与 segment_split 中「客户端同步等待远端完成」一致）：
- vi_operation_wait_*：测量节点上从发起 annotate_video 到 operation.result() 返回的墙钟时间，
  即 Video Intelligence LRO 在客户端可观测的完成耗时（含排队与 Google 侧处理），不是 Google 控制台里的内部 CPU 分项。
- *_prep_sec：发起 VI 作业之前的准备（含 smart_move：对象就位 / 可能的跨桶搬运，以及请求组装）。

批量测量与 segment_split_measurement 一致：`python batch_sweep_measurements.py`（2→4→…→30 分钟，每种时长 10 次）。
默认不对 LRO 设客户端超时（无限等待）；单次调试可用 `--annotate-timeout-sec` 指定上限。

默认三项均使用美西 `video_us` 桶与 `us-west1`（可通过 *_pid / --bucket 覆盖）。

用法（建议在 video_label_ocr_speech 目录下）：
  cd video_label_ocr_speech
  python batch_sweep_measurements.py
  python run_measurement.py --minutes 10

请确保已配置 GCP 凭证；speech 默认语言 en-US，可用 --speech-language 指定（如 zh-CN）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import config  # noqa: F401

    if "GCP_PROJECT_NUMBER" not in os.environ and hasattr(config, "GCP_PROJECT_NUMBER") and config.GCP_PROJECT_NUMBER:
        os.environ["GCP_PROJECT_NUMBER"] = str(config.GCP_PROJECT_NUMBER)
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ops.registry import (  # noqa: E402
    DEFAULT_LABEL_PID,
    DEFAULT_OCR_PID,
    DEFAULT_SPEECH_LANGUAGE_CODE,
    DEFAULT_SPEECH_PID,
    get_operation,
)

_MEASURE_TIMING_KEYS = frozenset({"timing_prep_sec", "timing_vi_operation_wait_sec"})


def _strip_measure_timing_fields(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in _MEASURE_TIMING_KEYS}


def _speech_language_default() -> str:
    """优先 config.DEFAULT_SPEECH_LANGUAGE_CODE，否则 ops.registry.DEFAULT_SPEECH_LANGUAGE_CODE。"""
    try:
        import config as cfg

        v = getattr(cfg, "DEFAULT_SPEECH_LANGUAGE_CODE", None)
        if v:
            return str(v).strip()
    except ImportError:
        pass
    return DEFAULT_SPEECH_LANGUAGE_CODE


def _cfg_measurement_pid(attr: str, fallback: str) -> str:
    """从 config.py 读取 MEASUREMENT_*_PID（若存在）。"""
    try:
        import config as cfg

        v = getattr(cfg, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    except ImportError:
        pass
    return fallback


def _ffmpeg_executable() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("未找到 ffmpeg：请安装系统 ffmpeg 或 pip install imageio-ffmpeg。")


def _run_ffmpeg_truncate(
    src: str,
    dst: str,
    duration_sec: float,
    reencode: bool,
) -> None:
    ffmpeg = _ffmpeg_executable()
    if duration_sec <= 0:
        raise ValueError("截取时长必须为正数（秒）。")
    cmd: List[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-t", str(duration_sec)]
    if reencode:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.extend(["-c", "copy"])
    cmd.append(dst)
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run(
    *,
    video_path: str,
    duration_sec: float,
    label_pid: str,
    ocr_pid: str,
    speech_pid: str,
    upload_prefix: str,
    bucket: Optional[str],
    output_json: Optional[str],
    reencode: bool,
    speech_language_code: str,
    annotate_timeout_sec: Optional[int] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    label_op = get_operation(label_pid)
    ocr_op = get_operation(ocr_pid)
    speech_op = get_operation(speech_pid)

    buckets = {label_op.storage_bucket, ocr_op.storage_bucket, speech_op.storage_bucket}
    buckets.discard(None)
    buckets.discard("")
    if len(buckets) > 1:
        raise ValueError(
            f"三项操作的 storage_bucket 不一致：{buckets}。"
            "请使用同一 region 对应的 pid（例如均为 *_google_us）。"
        )

    resolved_bucket = bucket or label_op.storage_bucket
    if not resolved_bucket:
        raise ValueError("无法解析 GCS bucket：请设置 --bucket 或使用带 storage_bucket 的 pid。")

    if bucket is not None and bucket != label_op.storage_bucket:
        raise ValueError(
            f"--bucket ({bucket}) 与 {label_pid} 的 storage_bucket ({label_op.storage_bucket}) 不一致。"
            "请去掉 --bucket 或更换 *_pid。"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    prefix = upload_prefix.strip("/")
    target_path = f"{prefix}/{run_id}" if prefix else run_id

    suffix = ".mp4"
    base = os.path.splitext(os.path.basename(video_path))[0]
    truncated_name = f"{base}_trunc_{duration_sec:g}s{suffix}"
    truncated_path = os.path.join(tempfile.gettempdir(), f"sb_vi_measure_{run_id}_{truncated_name}")

    t0 = time.perf_counter()
    _run_ffmpeg_truncate(video_path, truncated_path, duration_sec, reencode=reencode)
    t_trunc = time.perf_counter() - t0

    from ops.utils import DataTransmission  # noqa: E402

    transmitter = DataTransmission()
    t1 = time.perf_counter()
    gcs_uri = transmitter.upload_local_to_cloud(
        truncated_path, "google", resolved_bucket, target_path=target_path
    )
    t_upload = time.perf_counter() - t1

    speech_kw: Dict[str, Any] = {
        "target_path": target_path,
        "speech_language_code": speech_language_code,
        "annotate_timeout_sec": annotate_timeout_sec,
    }
    common_kw: Dict[str, Any] = {
        "target_path": target_path,
        "annotate_timeout_sec": annotate_timeout_sec,
    }

    try:
        label_result = label_op.execute(gcs_uri, **common_kw)
        ocr_result = ocr_op.execute(gcs_uri, **common_kw)
        speech_result = speech_op.execute(gcs_uri, **speech_kw)
    finally:
        try:
            os.remove(truncated_path)
        except OSError:
            pass

    record: Dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "video_source": os.path.abspath(video_path),
            "duration_requested_sec": duration_sec,
            "label_pid": label_pid,
            "ocr_pid": ocr_pid,
            "speech_pid": speech_pid,
            "speech_language_code": speech_language_code,
            "bucket": resolved_bucket,
            "upload_prefix": target_path,
            "gcs_uri": gcs_uri,
            "reencode": reencode,
            "annotate_timeout_sec": annotate_timeout_sec,
            "timing_semantics": {
                "vi_operation_wait": (
                    "测量 VM 上 annotate_video 调用结束至 operation.result() 返回的墙钟时间；"
                    "为 VI LRO 在客户端可观测的完成耗时（含排队与远端处理），非 Google 另行公布的内部 CPU 秒。"
                ),
                "prep": "发起 VI 前准备：含 smart_move（对象就位/跨桶搬运）与请求组装。",
            },
            "utc_iso": datetime.now(timezone.utc).isoformat(),
        },
        "timings_sec": {
            "truncate": round(t_trunc, 6),
            "upload": round(t_upload, 6),
            "label_prep_sec": round(float(label_result["timing_prep_sec"]), 6),
            "label_vi_operation_wait_sec": round(float(label_result["timing_vi_operation_wait_sec"]), 6),
            "ocr_prep_sec": round(float(ocr_result["timing_prep_sec"]), 6),
            "ocr_vi_operation_wait_sec": round(float(ocr_result["timing_vi_operation_wait_sec"]), 6),
            "speech_prep_sec": round(float(speech_result["timing_prep_sec"]), 6),
            "speech_vi_operation_wait_sec": round(float(speech_result["timing_vi_operation_wait_sec"]), 6),
        },
        "label_detection": _strip_measure_timing_fields({k: v for k, v in label_result.items() if k != "provider"}),
        "ocr": _strip_measure_timing_fields({k: v for k, v in ocr_result.items() if k != "provider"}),
        "speech_transcription": _strip_measure_timing_fields({k: v for k, v in speech_result.items() if k != "provider"}),
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        if not silent:
            print(f"已保存测量结果: {output_json}")

    return record


def main() -> None:
    default_video = os.path.join(_ROOT, "video", "merged.mp4")
    p = argparse.ArgumentParser(description="上传 GCS 并测量 VI Label / OCR / Speech 耗时")
    p.add_argument(
        "--video",
        default=default_video,
        help="本地长视频路径（默认本目录下 video/merged.mp4）",
    )
    p.add_argument(
        "--minutes",
        type=float,
        required=True,
        help="从原片开头截取的时长（分钟）",
    )
    p.add_argument(
        "--label-pid",
        default=_cfg_measurement_pid("MEASUREMENT_LABEL_PID", DEFAULT_LABEL_PID),
        help=f"registry Label Detection pid（默认 {DEFAULT_LABEL_PID}；可在 config 设 MEASUREMENT_LABEL_PID）",
    )
    p.add_argument(
        "--ocr-pid",
        default=_cfg_measurement_pid("MEASUREMENT_OCR_PID", DEFAULT_OCR_PID),
        help=f"registry OCR(Text Detection) pid（默认 {DEFAULT_OCR_PID}；可在 config 设 MEASUREMENT_OCR_PID）",
    )
    p.add_argument(
        "--speech-pid",
        default=_cfg_measurement_pid("MEASUREMENT_SPEECH_PID", DEFAULT_SPEECH_PID),
        help=f"registry Speech Transcription pid（默认 {DEFAULT_SPEECH_PID}；可在 config 设 MEASUREMENT_SPEECH_PID）",
    )
    p.add_argument(
        "--upload-prefix",
        default="video_label_ocr_speech",
        help="上传到 bucket 内的一级目录前缀（每次运行会再建子目录）",
    )
    p.add_argument("--bucket", default=None, help="覆盖 GCS bucket；默认与 pid 一致")
    p.add_argument(
        "--output",
        default=None,
        help="测量结果 JSON 路径；默认本目录下 results/measurement/<run_id>.json",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="截断时使用 libx264/aac 重编码；默认 -c copy",
    )
    p.add_argument(
        "--speech-language",
        default=_speech_language_default(),
        help=f"Speech Transcription BCP-47 语言码（默认 {DEFAULT_SPEECH_LANGUAGE_CODE}；可在 config.py 设 DEFAULT_SPEECH_LANGUAGE_CODE）",
    )
    p.add_argument(
        "--annotate-timeout-sec",
        type=int,
        default=None,
        metavar="SEC",
        help="每项 LRO（operation.result）客户端最长等待秒数；省略则不限制（批量脚本默认）",
    )

    args = p.parse_args()

    out = args.output
    if not out:
        rid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out = os.path.join(_ROOT, "results", "measurement", f"measure_{rid}.json")

    record = run(
        video_path=args.video,
        duration_sec=float(args.minutes) * 60.0,
        label_pid=args.label_pid,
        ocr_pid=args.ocr_pid,
        speech_pid=args.speech_pid,
        upload_prefix=args.upload_prefix,
        bucket=args.bucket,
        output_json=out,
        reencode=args.reencode,
        speech_language_code=args.speech_language,
        annotate_timeout_sec=args.annotate_timeout_sec,
    )
    print(json.dumps(record["timings_sec"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
