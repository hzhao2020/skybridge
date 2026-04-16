"""
按 2→4→6→…→30 分钟批量执行 segment/split 测量，写入 CSV 与每次运行的 JSON。

视频默认本目录 video/merged.mp4；每种时长默认重复 10 次。改时长列表或重复次数请编辑本文件顶部常量。

用法：
  cd segment_split_measurement
  python batch_sweep_measurements.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run_measurement import DEFAULT_SEGMENT_PID, DEFAULT_SPLIT_PID, run  # noqa: E402

VIDEO_PATH = os.path.join(_ROOT, "video", "merged.mp4")
DURATION_MINUTES_SWEEP = list(range(2, 31, 2))  # 2, 4, …, 30
TRIALS_PER_DURATION = 10


def _row_from_record(
    batch_id: str,
    trial_index: int,
    duration_min: float,
    detail_json_path: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    meta = record.get("meta") or {}
    timings = record.get("timings_sec") or {}
    seg = record.get("segment") or {}
    spl = record.get("split") or {}
    return {
        "batch_id": batch_id,
        "duration_minutes": duration_min,
        "duration_sec": meta.get("duration_requested_sec", ""),
        "trial_index": trial_index,
        "run_id": meta.get("run_id", ""),
        "utc_iso": meta.get("utc_iso", ""),
        "success": True,
        "error_message": "",
        "node_truncate_sec": timings.get("truncate", ""),
        "node_upload_sec": timings.get("upload", ""),
        "node_segment_execute_sec": timings.get("segment_execute", ""),
        "node_split_execute_http_observed_sec": timings.get("split_execute_http_observed", ""),
        "segment_shot_count": seg.get("shot_count", ""),
        "split_segments_used_count": spl.get("segments_used_count", ""),
        "split_output_count": spl.get("output_count", ""),
        "segment_pid": meta.get("segment_pid", ""),
        "split_pid": meta.get("split_pid", ""),
        "bucket": meta.get("bucket", ""),
        "gcs_uri": meta.get("gcs_uri", ""),
        "segment_result_location": seg.get("result_location", ""),
        "detail_json_path": detail_json_path,
    }


def _error_row(
    batch_id: str,
    trial_index: int,
    duration_min: float,
    duration_sec: float,
    err: str,
) -> Dict[str, Any]:
    return {
        "batch_id": batch_id,
        "duration_minutes": duration_min,
        "duration_sec": duration_sec,
        "trial_index": trial_index,
        "run_id": "",
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "error_message": err,
        "node_truncate_sec": "",
        "node_upload_sec": "",
        "node_segment_execute_sec": "",
        "node_split_execute_http_observed_sec": "",
        "segment_shot_count": "",
        "split_segments_used_count": "",
        "split_output_count": "",
        "segment_pid": "",
        "split_pid": "",
        "bucket": "",
        "gcs_uri": "",
        "segment_result_location": "",
        "detail_json_path": "",
    }


CSV_FIELDS = [
    "batch_id",
    "duration_minutes",
    "duration_sec",
    "trial_index",
    "run_id",
    "utc_iso",
    "success",
    "error_message",
    "node_truncate_sec",
    "node_upload_sec",
    "node_segment_execute_sec",
    "node_split_execute_http_observed_sec",
    "segment_shot_count",
    "split_segments_used_count",
    "split_output_count",
    "segment_pid",
    "split_pid",
    "bucket",
    "gcs_uri",
    "segment_result_location",
    "detail_json_path",
]


def main() -> None:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "Z"
    video = os.path.abspath(VIDEO_PATH)
    if not os.path.isfile(video):
        print(f"找不到视频文件: {video}", file=sys.stderr)
        sys.exit(1)

    upload_prefix = f"segment_split_sweep/{batch_id}"
    csv_path = os.path.join(_ROOT, "results", f"sweep_{batch_id}.csv")
    json_dir = os.path.join(_ROOT, "results", f"sweep_{batch_id}_json")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    durations_sec: List[float] = [float(m * 60) for m in DURATION_MINUTES_SWEEP]

    print(f"batch_id={batch_id}")
    print(f"video={video}")
    print(f"minutes={DURATION_MINUTES_SWEEP} trials_each={TRIALS_PER_DURATION}")
    print(f"csv={csv_path}")
    print(f"json_dir={json_dir}")

    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for d_sec in durations_sec:
            d_min = d_sec / 60.0
            for trial in range(1, TRIALS_PER_DURATION + 1):
                prefix = f"{upload_prefix}/d{int(d_sec)}s_t{trial}"
                detail_path = os.path.join(json_dir, f"pending_{batch_id}_d{int(d_sec)}_t{trial}.json")
                try:
                    record = run(
                        video_path=video,
                        duration_sec=d_sec,
                        segment_pid=DEFAULT_SEGMENT_PID,
                        split_pid=DEFAULT_SPLIT_PID,
                        upload_prefix=prefix,
                        bucket=None,
                        output_json=detail_path,
                        reencode=False,
                        max_split_segments=80,
                        silent=True,
                    )
                    rid = (record.get("meta") or {}).get("run_id", "")
                    final_json = os.path.join(json_dir, f"{rid}.json")
                    if os.path.abspath(detail_path) != os.path.abspath(final_json):
                        if os.path.isfile(detail_path):
                            os.replace(detail_path, final_json)
                    elif not os.path.isfile(final_json):
                        with open(final_json, "w", encoding="utf-8") as jf:
                            json.dump(record, jf, ensure_ascii=False, indent=2)
                    row = _row_from_record(
                        batch_id,
                        trial,
                        d_min,
                        os.path.abspath(final_json),
                        record,
                    )
                except Exception as e:
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    if os.path.isfile(detail_path):
                        try:
                            os.remove(detail_path)
                        except OSError:
                            pass
                    fail_log = os.path.join(json_dir, f"error_{batch_id}_d{int(d_sec)}_t{trial}.txt")
                    with open(fail_log, "w", encoding="utf-8") as ef:
                        ef.write(err)
                    row = _error_row(batch_id, trial, d_min, d_sec, err[:2000] + ("…" if len(err) > 2000 else ""))
                    row["detail_json_path"] = os.path.abspath(fail_log)
                writer.writerow(row)
                cf.flush()
                print(
                    f"  d={int(d_sec)}s trial={trial}/{TRIALS_PER_DURATION} "
                    f"ok={row['success']} segment={row.get('node_segment_execute_sec', '')}"
                )

    print(f"完成。CSV: {csv_path}")


if __name__ == "__main__":
    main()
