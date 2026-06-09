"""
调用已部署的 split & sample 云函数，测量「split + sample」运行时长。

矩阵：5 个时长(2/4/6/8/10min) × 3 个采样率(0.2/0.5/1.0 fps) × 3 次 = 45 次。
segment 数量取自 shot detection 测量的 shot_count（gcp_shot_detection_all.csv 中位数），
将视频均分为对应段数，再对每个 segment 按 sample_fps 采样 frames（null sink，不落盘）。
记录：
  - 服务端 split_sample_sec（ffmpeg 切分+采样）
  - 服务端 elapsed_total_sec（函数内端到端）
  - 客户端 round-trip（含网络）

用法：
  python run_split_sample_measurement.py --repeat 3
  python run_split_sample_measurement.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
import urllib.request
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DURATIONS_MIN = [2, 4, 6, 8, 10]
SAMPLE_FPS = [0.2, 0.5, 1.0]
SHOT_CSV = _ROOT / "results" / "gcp_shot_detection_all.csv"
VIDEO_URI_TMPL = "gs://{bucket}/gvi_shot_measure/perfect_planet_{m}min.mkv"

CSV_FIELDS = [
    "run_id", "index", "duration_min", "sample_fps", "repeat_index",
    "video_uri", "segments", "client_roundtrip_sec",
    "server_elapsed_total_sec", "server_split_sample_sec", "server_upload_sec",
    "total_frames", "http_status", "success",
    "submit_utc", "done_utc", "error_message",
]


def _load_url(arg_url: Optional[str]) -> str:
    if arg_url:
        return arg_url
    try:
        import config  # type: ignore

        url = getattr(config, "SPLIT_SAMPLE_URL", "") or ""
    except ImportError:
        url = ""
    if not url:
        raise RuntimeError("缺少函数 URL：请用 --url 传入，或在 config.py 设置 SPLIT_SAMPLE_URL。")
    return url.rstrip("/")


def _load_bucket() -> str:
    try:
        import config  # type: ignore

        return str(config.GCP_CONFIG.get("gcs_bucket", "") or "")
    except (ImportError, AttributeError):
        return ""


def _load_shot_counts() -> Dict[int, int]:
    """从 shot detection CSV 读取每个时长的 shot_count（取中位数）。"""
    counts: Dict[int, List[int]] = defaultdict(list)
    if SHOT_CSV.exists():
        with SHOT_CSV.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("success") != "1":
                    continue
                try:
                    m = round(float(r["video_duration_sec"]) / 60)
                    counts[m].append(int(r["shot_count"]))
                except (KeyError, ValueError):
                    continue
    fallback = {2: 24, 4: 46, 6: 82, 8: 97, 10: 110}
    out: Dict[int, int] = {}
    for m in DURATIONS_MIN:
        out[m] = int(st.median(counts[m])) if counts.get(m) else fallback[m]
    return out


def _make_equal_segments(duration_min: int, n: int) -> List[Dict[str, float]]:
    """将视频 [0, duration] 均分为 n 段（n = shot_count）。"""
    total = duration_min * 60.0
    n = max(1, n)
    step = total / n
    segs: List[Dict[str, float]] = []
    for i in range(n):
        s = i * step
        e = min((i + 1) * step, total)
        segs.append({"start": round(s, 3), "end": round(e, 3)})
    return segs


def _load_all_segments(shot_counts: Dict[int, int]) -> Dict[int, List[Dict[str, float]]]:
    return {m: _make_equal_segments(m, shot_counts[m]) for m in DURATIONS_MIN}


def _iter_jobs(
    repeat: int,
    bucket: str,
    all_segments: Dict[int, List[Dict[str, float]]],
) -> Iterator[Tuple[int, int, float, int, List, str]]:
    idx = 0
    for m in DURATIONS_MIN:
        segments = all_segments[m]
        video_uri = VIDEO_URI_TMPL.format(bucket=bucket, m=m)
        for fps in SAMPLE_FPS:
            for rep in range(1, repeat + 1):
                idx += 1
                yield idx, m, fps, rep, segments, video_uri


def _last_done_index(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("success") != "1":
                continue
            try:
                last = max(last, int(r["index"]))
            except (KeyError, ValueError):
                continue
    return last


def _post(url: str, payload: dict, timeout: float) -> Tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.getcode()
        body = resp.read().decode("utf-8").strip()
    start = body.rfind("{")
    if start >= 0:
        body = body[start:]
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, {"raw": body[:500]}


def main() -> None:
    p = argparse.ArgumentParser(description="split & sample 云函数运行时长测量（5×3×3=45）")
    p.add_argument("--url", default=None, help="函数 URL（默认读 config.SPLIT_SAMPLE_URL）")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--output", type=Path, default=_ROOT / "results" / "gcp_split_sample_all.csv")
    p.add_argument("--request-interval-sec", type=float, default=5.0)
    p.add_argument("--timeout-sec", type=float, default=3600.0)
    p.add_argument("--start-index", type=int, default=1, help="从第 N 条开始（含）")
    p.add_argument("--end-index", type=int, default=None, help="跑到第 N 条结束（含）")
    p.add_argument("--index-shard-count", type=int, default=1, help="按 index 取模分片的分片总数")
    p.add_argument("--index-shard", type=int, default=0, help="当前分片编号：运行 index %% shard_count == shard 的条目")
    p.add_argument("--resume", action="store_true", help="从 output CSV 最后成功条目的下一条续跑并追加")
    args = p.parse_args()

    url = _load_url(args.url)
    bucket = _load_bucket()
    if not bucket:
        print("错误: 无法从 config.GCP_CONFIG 读取 gcs_bucket", file=sys.stderr)
        sys.exit(1)
    if args.index_shard_count < 1:
        print("错误: --index-shard-count 必须 >= 1", file=sys.stderr)
        sys.exit(1)
    if not (0 <= args.index_shard < args.index_shard_count):
        print("错误: --index-shard 必须满足 0 <= shard < shard_count", file=sys.stderr)
        sys.exit(1)

    shot_counts = _load_shot_counts()
    all_segments = _load_all_segments(shot_counts)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    start_index = args.start_index
    append_mode = False
    if args.resume:
        start_index = _last_done_index(out) + 1
        append_mode = start_index > 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    try:
        import config  # type: ignore

        proxy = getattr(config, "PROXY", None)
        if proxy:
            for v in ("http_proxy", "https_proxy", "ALL_PROXY"):
                os.environ.setdefault(v, str(proxy))
    except ImportError:
        pass

    total = len(DURATIONS_MIN) * len(SAMPLE_FPS) * args.repeat
    print(f"url={url}")
    print(f"run_id={run_id} bucket={bucket}")
    print(f"shot_counts(segments)={shot_counts}  # 来自 {SHOT_CSV.name}")
    print(f"矩阵: {len(DURATIONS_MIN)}时长 × {len(SAMPLE_FPS)}采样率 × {args.repeat}次 = {total} 次")
    print(f"csv={out.resolve()}")
    if append_mode:
        print(f"续跑: 从 index={start_index} 追加写入\n")
    else:
        print()

    file_mode = "a" if append_mode else "w"
    with out.open(file_mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        if not append_mode:
            writer.writeheader()

        for idx, m, fps, rep, segments, video_uri in _iter_jobs(args.repeat, bucket, all_segments):
            if idx < start_index:
                continue
            if args.end_index is not None and idx > args.end_index:
                continue
            if idx % args.index_shard_count != args.index_shard:
                continue
            if idx > start_index and args.request_interval_sec > 0:
                time.sleep(args.request_interval_sec)

            payload = {
                "video_uri": video_uri,
                "segments": segments,
                "sample_fps": fps,
            }
            row = {k: "" for k in CSV_FIELDS}
            row.update({
                "run_id": run_id, "index": idx, "duration_min": m,
                "sample_fps": fps, "repeat_index": rep, "video_uri": video_uri,
                "segments": len(segments), "success": 0,
                "submit_utc": datetime.now(timezone.utc).isoformat(),
            })
            t0 = time.perf_counter()
            try:
                status, body = _post(url, payload, args.timeout_sec)
                rt = time.perf_counter() - t0
                row["client_roundtrip_sec"] = f"{rt:.6f}"
                row["http_status"] = status
                row["done_utc"] = datetime.now(timezone.utc).isoformat()
                if status == 200 and "split_sample_sec" in body:
                    row["server_elapsed_total_sec"] = body.get("elapsed_total_sec", "")
                    row["server_split_sample_sec"] = body.get("split_sample_sec", "")
                    row["server_upload_sec"] = 0
                    row["total_frames"] = body.get("total_frames", "")
                    row["success"] = 1
                else:
                    row["error_message"] = json.dumps(body, ensure_ascii=False)[:500]
            except Exception as e:  # noqa: BLE001
                row["client_roundtrip_sec"] = f"{time.perf_counter()-t0:.6f}"
                row["done_utc"] = datetime.now(timezone.utc).isoformat()
                row["error_message"] = f"{type(e).__name__}: {e}"[:500]

            writer.writerow(row)
            fp.flush()
            print(f"  [{idx}/{total}] {m}min fps={fps} rep={rep} segs={len(segments)} "
                  f"ok={row['success']} split_sample={row['server_split_sample_sec'] or '-'}s "
                  f"frames={row['total_frames'] or '-'}",
                  flush=True)
            if not row["success"] and row["error_message"]:
                print(f"      err: {row['error_message'][:160]}", flush=True)

    print(f"\n完成：结果写入 {out}")


if __name__ == "__main__":
    main()
