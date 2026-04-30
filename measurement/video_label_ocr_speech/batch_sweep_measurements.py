"""
按 2→4→6→…→30 分钟批量执行 Label / OCR / Speech 测量，写入 CSV 与每次运行的 JSON。

视频默认本目录 video/merged.mp4；每种时长默认重复 10 次。改时长列表或重复次数请编辑本文件顶部常量。

用法：
  cd video_label_ocr_speech
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

from ops.registry import DEFAULT_LABEL_PID, DEFAULT_OCR_PID, DEFAULT_SPEECH_PID  # noqa: E402
from run_measurement import _cfg_measurement_pid, _speech_language_default, run  # noqa: E402

VIDEO_PATH = os.path.join(_ROOT, "video", "merged.mp4")
DURATION_MINUTES_SWEEP = list(range(2, 31, 2))  # 2, 4, …, 30（与 segment_split_measurement 一致）
TRIALS_PER_DURATION = 10
SPEECH_LANGUAGE_CODE = _speech_language_default()


def _row_from_record(
    batch_id: str,
    trial_index: int,
    duration_min: float,
    detail_json_path: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    meta = record.get("meta") or {}
    timings = record.get("timings_sec") or {}
    lab = record.get("label_detection") or {}
    ocr = record.get("ocr") or {}
    sp = record.get("speech_transcription") or {}
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
        "node_label_prep_sec": timings.get("label_prep_sec", ""),
        "node_label_vi_operation_wait_sec": timings.get("label_vi_operation_wait_sec", ""),
        "node_ocr_prep_sec": timings.get("ocr_prep_sec", ""),
        "node_ocr_vi_operation_wait_sec": timings.get("ocr_vi_operation_wait_sec", ""),
        "node_speech_prep_sec": timings.get("speech_prep_sec", ""),
        "node_speech_vi_operation_wait_sec": timings.get("speech_vi_operation_wait_sec", ""),
        "label_segment_label_count": lab.get("segment_label_count", ""),
        "label_shot_label_count": lab.get("shot_label_count", ""),
        "ocr_text_annotation_count": ocr.get("text_annotation_count", ""),
        "speech_transcription_count": sp.get("speech_transcription_count", ""),
        "label_pid": meta.get("label_pid", ""),
        "ocr_pid": meta.get("ocr_pid", ""),
        "speech_pid": meta.get("speech_pid", ""),
        "speech_language_code": meta.get("speech_language_code", ""),
        "bucket": meta.get("bucket", ""),
        "gcs_uri": meta.get("gcs_uri", ""),
        "label_result_location": lab.get("result_location", ""),
        "ocr_result_location": ocr.get("result_location", ""),
        "speech_result_location": sp.get("result_location", ""),
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
        "node_label_prep_sec": "",
        "node_label_vi_operation_wait_sec": "",
        "node_ocr_prep_sec": "",
        "node_ocr_vi_operation_wait_sec": "",
        "node_speech_prep_sec": "",
        "node_speech_vi_operation_wait_sec": "",
        "label_segment_label_count": "",
        "label_shot_label_count": "",
        "ocr_text_annotation_count": "",
        "speech_transcription_count": "",
        "label_pid": "",
        "ocr_pid": "",
        "speech_pid": "",
        "speech_language_code": "",
        "bucket": "",
        "gcs_uri": "",
        "label_result_location": "",
        "ocr_result_location": "",
        "speech_result_location": "",
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
    "node_label_prep_sec",
    "node_label_vi_operation_wait_sec",
    "node_ocr_prep_sec",
    "node_ocr_vi_operation_wait_sec",
    "node_speech_prep_sec",
    "node_speech_vi_operation_wait_sec",
    "label_segment_label_count",
    "label_shot_label_count",
    "ocr_text_annotation_count",
    "speech_transcription_count",
    "label_pid",
    "ocr_pid",
    "speech_pid",
    "speech_language_code",
    "bucket",
    "gcs_uri",
    "label_result_location",
    "ocr_result_location",
    "speech_result_location",
    "detail_json_path",
]


def main() -> None:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "Z"
    video = os.path.abspath(VIDEO_PATH)
    if not os.path.isfile(video):
        print(f"找不到视频文件: {video}", file=sys.stderr)
        sys.exit(1)

    upload_prefix = f"video_label_ocr_speech_sweep/{batch_id}"
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
                        label_pid=_cfg_measurement_pid("MEASUREMENT_LABEL_PID", DEFAULT_LABEL_PID),
                        ocr_pid=_cfg_measurement_pid("MEASUREMENT_OCR_PID", DEFAULT_OCR_PID),
                        speech_pid=_cfg_measurement_pid("MEASUREMENT_SPEECH_PID", DEFAULT_SPEECH_PID),
                        upload_prefix=prefix,
                        bucket=None,
                        output_json=detail_path,
                        reencode=False,
                        speech_language_code=SPEECH_LANGUAGE_CODE,
                        annotate_timeout_sec=None,
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
                    f"ok={row['success']} label_vi_wait={row.get('node_label_vi_operation_wait_sec', '')}"
                )

    print(f"完成。CSV: {csv_path}")


if __name__ == "__main__":
    main()
