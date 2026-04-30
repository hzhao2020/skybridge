#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/video/merged.mp4"
mkdir -p "${ROOT}/video"

FFMPEG="$(command -v ffmpeg 2>/dev/null || true)"
if [[ -z "${FFMPEG}" ]]; then
  FFMPEG="$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
fi
if [[ -z "${FFMPEG}" ]]; then
  echo "未找到 ffmpeg：请 apt install ffmpeg，或 pip install imageio-ffmpeg 后重试。" >&2
  exit 1
fi

exec "${FFMPEG}" -y \
  -f lavfi -i "testsrc=duration=20:size=640x480:rate=25" \
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=20" \
  -pix_fmt yuv420p -c:v libx264 -preset ultrafast -c:a aac -shortest \
  "${OUT}"
