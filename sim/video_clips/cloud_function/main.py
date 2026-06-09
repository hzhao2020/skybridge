"""
Cloud Function / Cloud Run 服务：split & sample。

接收 video（GCS）与 segments（时间区间），对整段视频按指定 fps 一次性采样帧
（仅解码+重采样，不保存输出帧），再按采样时间戳把帧归属到各个 segment。
返回各阶段执行时长。

请求 JSON：
{
  "video_uri": "gs://bucket/path/video.mkv",
  "segments": [{"start": 0.0, "end": 4.5}, ...],
  "sample_fps": 0.2 | 0.5 | 1.0
}

返回 JSON：
{
  "elapsed_total_sec": ...,      # 端到端（函数内总耗时）
  "split_sample_sec": ...,       # ffmpeg 一次性采样耗时
  "segments": N, "sample_fps": f, "total_frames": K,
  "frames_per_segment": [...]
}

内存友好：ffmpeg 通过 v4 签名 URL 以 HTTP range 流式读取视频（不把整段 4K 视频落到
/tmp）；采样结果写入 null sink，不落盘。长任务期间以 chunked 心跳保活，避免代理 idle 断连。
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import functions_framework
import google.auth
from google.auth.transport import requests as auth_requests
from google.cloud import storage
from flask import Response

_storage_client = storage.Client()
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
_HEARTBEAT_SEC = 15


def _parse_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// uri, got {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"invalid gs uri: {uri}")
    return bucket, key


def _signed_url(bucket: str, key: str) -> str:
    blob = _storage_client.bucket(bucket).blob(key)
    credentials, _ = google.auth.default()
    auth_request = auth_requests.Request()
    credentials.refresh(auth_request)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=2),
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )


def _run_ffmpeg_sample(src_url: str, duration: float, fps: float) -> None:
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", src_url,
        "-t", f"{duration:.3f}",
        "-an",
        "-vf", f"fps={fps}",
        "-f", "null",
        "-",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _valid_segments(segments: List[Dict[str, float]]) -> List[Tuple[float, float, int]]:
    out: List[Tuple[float, float, int]] = []
    for idx, seg in enumerate(segments):
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if e > s:
            out.append((s, e, idx))
    return out


def _sample_timestamps(duration: float, fps: float) -> List[float]:
    if duration <= 0 or fps <= 0:
        return []
    count = max(0, math.ceil(duration * fps - 1e-9))
    return [i / fps for i in range(count)]


def _frames_per_segment(
    segment_count: int,
    valid_segments: List[Tuple[float, float, int]],
    fps: float,
) -> List[int]:
    if not valid_segments:
        return []
    sorted_segments = sorted(valid_segments, key=lambda item: item[0])
    duration = max(e for _, e, _ in sorted_segments)
    counts = [0 for _ in range(segment_count)]
    seg_idx = 0
    for ts in _sample_timestamps(duration, fps):
        while seg_idx < len(sorted_segments) and ts >= sorted_segments[seg_idx][1]:
            seg_idx += 1
        if seg_idx >= len(sorted_segments):
            break
        start, end, original_idx = sorted_segments[seg_idx]
        if start <= ts < end:
            counts[original_idx] += 1
    return counts


def _process(
    video_uri: str,
    segments: List[Dict[str, float]],
    sample_fps: float,
) -> Tuple[Dict[str, Any], int]:
    bkt, key = _parse_gs(video_uri)
    src_url = _signed_url(bkt, key)

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    valid_segments = _valid_segments(segments)
    if not valid_segments:
        raise ValueError("no positive-duration segments")

    duration = max(e for _, e, _ in valid_segments)
    t0 = time.perf_counter()
    _run_ffmpeg_sample(src_url, duration, sample_fps)
    split_sample_sec = time.perf_counter() - t0

    frames_per_segment = _frames_per_segment(len(segments), valid_segments, sample_fps)
    total_frames = sum(frames_per_segment)

    body = {
        "split_sample_sec": round(split_sample_sec, 6),
        "segments": len(segments),
        "valid_segments": len(valid_segments),
        "sample_fps": sample_fps,
        "total_frames": total_frames,
        "frames_per_segment": frames_per_segment,
        "method": "single_pass_fps",
    }
    return body, 200


def _json_response(payload: dict, status: int) -> Tuple[str, int, dict]:
    return (json.dumps(payload), status, {"Content-Type": "application/json"})


def _stream_response(body: dict, status: int) -> Response:
    data = json.dumps(body).encode("utf-8")

    def generate():
        yield data

    return Response(generate(), status=status, mimetype="application/json")


@functions_framework.http
def split_and_sample(request):
    t_total0 = time.perf_counter()
    data: Dict[str, Any] = request.get_json(silent=True) or {}

    try:
        video_uri = data["video_uri"]
        segments: List[Dict[str, float]] = data.get("segments", [])
        sample_fps = float(data.get("sample_fps", 1.0))
    except (KeyError, TypeError, ValueError) as e:
        return _json_response({"error": f"bad request: {e}"}, 400)

    if not segments:
        return _json_response({"error": "no segments"}, 400)

    holder: Dict[str, Any] = {}
    error: Dict[str, Any] = {}
    done = threading.Event()

    def worker() -> None:
        try:
            body, status = _process(video_uri, segments, sample_fps)
            body["elapsed_total_sec"] = round(time.perf_counter() - t_total0, 6)
            holder["body"] = body
            holder["status"] = status
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", "ignore")[:1000]
            error["body"] = {"error": "ffmpeg failed", "detail": stderr}
            error["status"] = 500
        except Exception as e:  # noqa: BLE001
            error["body"] = {"error": f"{type(e).__name__}: {e}"}
            error["status"] = 500
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while not done.wait(timeout=_HEARTBEAT_SEC):
            yield b" \n"
        if error:
            payload, status = error["body"], error["status"]
        else:
            payload, status = holder["body"], holder["status"]
        yield json.dumps(payload).encode("utf-8")

    return Response(generate(), status=200, mimetype="application/json")
