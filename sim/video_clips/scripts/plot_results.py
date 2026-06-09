"""读取两个测量 CSV，绘制「执行延迟 vs 视频/音频时长」对比图。"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
RESULTS = _ROOT / "results"
SHOT_CSV = RESULTS / "gcp_shot_detection_all.csv"
SPEECH_CSV = RESULTS / "gcp_speech_all.csv"


def load(csv_path: Path, dur_col: str):
    by_dur = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("success") != "1":
                continue
            dur_min = round(float(r[dur_col]) / 60)
            by_dur[dur_min].append(
                (float(r["server_execution_sec"]), float(r["execution_latency_sec"]))
            )
    durs = sorted(by_dur)
    server_mean = [st.mean(v[0] for v in by_dur[d]) for d in durs]
    client_mean = [st.mean(v[1] for v in by_dur[d]) for d in durs]
    server_all = {d: [v[0] for v in by_dur[d]] for d in durs}
    return durs, server_mean, client_mean, server_all


def main() -> None:
    s_dur, s_srv, s_cli, s_all = load(SHOT_CSV, "video_duration_sec")
    a_dur, a_srv, a_cli, a_all = load(SPEECH_CSV, "audio_duration_sec")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：两服务对比（服务端执行时间）
    ax = axes[0]
    ax.plot(s_dur, s_srv, "o-", color="#d62728", label="Shot Detection (server)", linewidth=2)
    ax.plot(a_dur, a_srv, "s-", color="#1f77b4", label="Speech-to-Text (server)", linewidth=2)
    for d in s_dur:
        for y in s_all[d]:
            ax.plot(d, y, ".", color="#d62728", alpha=0.35)
    for d in a_dur:
        for y in a_all[d]:
            ax.plot(d, y, ".", color="#1f77b4", alpha=0.35)
    ax.set_xlabel("Video/Audio duration (min)")
    ax.set_ylabel("Server-side execution time (s)")
    ax.set_title("GCP execution latency vs duration\n(Shot Detection vs Speech-to-Text)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(s_dur)

    # 右：客户端 vs 服务端（各自）
    ax = axes[1]
    ax.plot(s_dur, s_srv, "o-", color="#d62728", label="Shot: server", linewidth=2)
    ax.plot(s_dur, s_cli, "o--", color="#ff9896", label="Shot: client", linewidth=1.5)
    ax.plot(a_dur, a_srv, "s-", color="#1f77b4", label="Speech: server", linewidth=2)
    ax.plot(a_dur, a_cli, "s--", color="#aec7e8", label="Speech: client", linewidth=1.5)
    ax.set_xlabel("Video/Audio duration (min)")
    ax.set_ylabel("Execution time (s)")
    ax.set_title("Client vs Server execution time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(s_dur)

    fig.tight_layout()
    out = RESULTS / "execution_latency_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
