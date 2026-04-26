"""
截断本地长视频 → 上传 GCS → Google Video Intelligence（镜头检测）→ Cloud Function 物理切割，
并记录各阶段耗时；结果写入 JSON。

默认 segment / split 使用 Google美西：`seg_google_us`、`split_google_us`（可通过参数覆盖）。

用法（建议在 segment_split_measurement 目录下）：
  cd segment_split_measurement
  python run_measurement.py --minutes 10
  # 即从原片开头截取 10 分钟；默认读取本目录下 video/merged.mp4；也可用 --video 指定路径。

请确保已配置 GCP 凭证，且 --segment-pid / --split-pid 对应同一 GCS bucket（例如均为 *_us），
否则 split 前可能触发跨桶传输，耗时统计会混入搬运时间。

split 的「节点执行时间」在此脚本中取客户端观测的 HTTP 往返时间（同步等待 Cloud Function 返回），
与 Cloud Function 内实际 CPU 时间一致量级；若需服务端精确耗时需在 Function 内自行上报。
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

# 本目录含 ops/ 子包与 config.py，单独上传本文件夹即可运行
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 加载本目录 config.py（VIDEO_SPLIT_URLS、代理等）
try:
    import config  # noqa: F401

    if "GCP_PROJECT_NUMBER" not in os.environ and hasattr(config, "GCP_PROJECT_NUMBER") and config.GCP_PROJECT_NUMBER:
        os.environ["GCP_PROJECT_NUMBER"] = str(config.GCP_PROJECT_NUMBER)
    if "GCP_VIDEOSPLIT_SERVICE_URLS" not in os.environ and hasattr(config, "GCP_VIDEOSPLIT_SERVICE_URLS") and config.GCP_VIDEOSPLIT_SERVICE_URLS:
        os.environ["GCP_VIDEOSPLIT_SERVICE_URLS"] = json.dumps(config.GCP_VIDEOSPLIT_SERVICE_URLS)
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ops.registry import get_operation  # noqa: E402
from ops.utils import DataTransmission  # noqa: E402

# 默认与 registry 中 Google us-west1 的 segment / split 成对，避免跨桶搬运
DEFAULT_SEGMENT_PID = "seg_google_us"
DEFAULT_SPLIT_PID = "split_google_us"


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


def _prepare_split_segments(
    segments: List[Dict[str, Any]],
    duration_requested: float,
    max_segments: Optional[int],
) -> List[Dict[str, float]]:
    valid: List[Dict[str, float]] = []
    for s in segments:
        try:
            st = float(s.get("start", 0))
            en = float(s.get("end", 0))
        except (TypeError, ValueError):
            continue
        if en > st:
            valid.append({"start": st, "end": en})
    valid.sort(key=lambda x: x["start"])
    if not valid:
        valid = [{"start": 0.0, "end": float(duration_requested)}]
    if max_segments is not None and max_segments > 0 and len(valid) > max_segments:
        valid = valid[:max_segments]
    return valid


def run(
    *,
    video_path: str,
    duration_sec: float,
    segment_pid: str = DEFAULT_SEGMENT_PID,
    split_pid: str = DEFAULT_SPLIT_PID,
    upload_prefix: str,
    bucket: Optional[str],
    output_json: Optional[str],
    reencode: bool,
    max_split_segments: Optional[int],
    silent: bool = False,
) -> Dict[str, Any]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    segmenter = get_operation(segment_pid)
    splitter = get_operation(split_pid)

    resolved_bucket = bucket or segmenter.storage_bucket
    if not resolved_bucket:
        raise ValueError("无法解析 GCS bucket：请设置 --bucket 或使用带 storage_bucket 的 segment_pid。")
    if bucket is not None and bucket != segmenter.storage_bucket:
        raise ValueError(
            f"--bucket ({bucket}) 与 {segment_pid} 的 storage_bucket ({segmenter.storage_bucket}) 不一致。"
            "上传与 segment.execute 的目标桶必须相同，否则会触发跨桶复制；请去掉 --bucket 或更换 segment_pid。"
        )
    if segmenter.storage_bucket != splitter.storage_bucket and bucket is None:
        print(
            f"警告：{segment_pid} 与 {split_pid} 的默认 bucket 不同 "
            f"({segmenter.storage_bucket} vs {splitter.storage_bucket})，"
            "split 时可能对视频做跨桶搬运。建议成对使用例如 seg_google_us + split_google_us。",
            file=sys.stderr,
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    prefix = upload_prefix.strip("/")
    target_path = f"{prefix}/{run_id}" if prefix else run_id

    suffix = ".mp4"
    base = os.path.splitext(os.path.basename(video_path))[0]
    truncated_name = f"{base}_trunc_{duration_sec:g}s{suffix}"
    truncated_path = os.path.join(tempfile.gettempdir(), f"sb_measure_{run_id}_{truncated_name}")

    t0 = time.perf_counter()
    _run_ffmpeg_truncate(video_path, truncated_path, duration_sec, reencode=reencode)
    t_trunc = time.perf_counter() - t0

    transmitter = DataTransmission()
    t1 = time.perf_counter()
    gcs_uri = transmitter.upload_local_to_cloud(
        truncated_path, "google", resolved_bucket, target_path=target_path
    )
    t_upload = time.perf_counter() - t1

    try:
        t2 = time.perf_counter()
        seg_result = segmenter.execute(gcs_uri, target_path=target_path)
        t_segment = time.perf_counter() - t2

        raw_segments = list(seg_result.get("segments") or [])
        split_segments = _prepare_split_segments(raw_segments, duration_sec, max_split_segments)

        t3 = time.perf_counter()
        split_result = splitter.execute(
            gcs_uri,
            split_segments,
            target_path=target_path,
        )
        t_split = time.perf_counter() - t3
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
            "segment_pid": segment_pid,
            "split_pid": split_pid,
            "bucket": resolved_bucket,
            "upload_prefix": target_path,
            "gcs_uri": gcs_uri,
            "reencode": reencode,
            "max_split_segments": max_split_segments,
            "utc_iso": datetime.now(timezone.utc).isoformat(),
        },
        "timings_sec": {
            "truncate": round(t_trunc, 6),
            "upload": round(t_upload, 6),
            "segment_execute": round(t_segment, 6),
            "split_execute_http_observed": round(t_split, 6),
        },
        "segment": {
            "shot_count": len(raw_segments),
            "result_location": seg_result.get("result_location"),
            "source_used": seg_result.get("source_used"),
        },
        "split": {
            "segments_used_count": len(split_segments),
            "output_count": split_result.get("output_count"),
            "output_uris": split_result.get("output_uris"),
        },
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
    p = argparse.ArgumentParser(description="截断视频、上传 GCS、测量 segment 与 split 耗时")
    p.add_argument(
        "--video",
        default=default_video,
        help=f"本地长视频路径（默认本目录下 video/merged.mp4）",
    )
    p.add_argument(
        "--minutes",
        type=float,
        required=True,
        help="从原片开头截取的时长（分钟），例如 2 表示 0–120s",
    )
    p.add_argument(
        "--segment-pid",
        default=DEFAULT_SEGMENT_PID,
        help=f"registry 中的 segment 操作 pid（默认 {DEFAULT_SEGMENT_PID}）",
    )
    p.add_argument(
        "--split-pid",
        default=DEFAULT_SPLIT_PID,
        help=f"registry 中的 split 操作 pid（默认 {DEFAULT_SPLIT_PID}）",
    )
    p.add_argument(
        "--upload-prefix",
        default="segment_split_measurement",
        help="上传到 bucket 内的一级目录前缀（每次运行会再建子目录）",
    )
    p.add_argument("--bucket", default=None, help="覆盖 GCS bucket；默认与 segment 操作一致")
    p.add_argument(
        "--output",
        default=None,
        help="测量结果 JSON 路径；默认本目录下 results/measurement/<run_id>.json",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="截断时使用 libx264/aac 重编码（较慢但更稳）；默认 -c copy",
    )
    p.add_argument(
        "--max-split-segments",
        type=int,
        default=80,
        help="参与切割的最大镜头数（按时间排序后截取前 N 个）；0 表示不限制",
    )

    args = p.parse_args()
    max_seg = args.max_split_segments if args.max_split_segments > 0 else None

    out = args.output
    if not out:
        rid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out = os.path.join(_ROOT, "results", "measurement", f"measure_{rid}.json")

    record = run(
        video_path=args.video,
        duration_sec=float(args.minutes) * 60.0,
        segment_pid=args.segment_pid,
        split_pid=args.split_pid,
        upload_prefix=args.upload_prefix,
        bucket=args.bucket,
        output_json=out,
        reencode=args.reencode,
        max_split_segments=max_seg,
    )
    print(json.dumps(record["timings_sec"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
