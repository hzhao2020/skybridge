#!/usr/bin/env bash
# 在 Debian/Ubuntu VM（us-west1）上初始化 Python 依赖与本目录运行环境。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv ffmpeg
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "完成。运行前执行: source ${ROOT}/.venv/bin/activate"
echo "将 video/merged.mp4 放好（或 bash scripts/create_placeholder_merged_mp4.sh）后:"
echo "  python batch_sweep_measurements.py"
echo "（与 segment_split_measurement 一致：2–30 分钟步进 2，每种 10 次；LRO 默认不限时）"
