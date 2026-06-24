#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-skyflow-prototype}"
REGIONS="${REGIONS:-cn-shanghai cn-beijing us-east-1 ap-southeast-1}"
OBJECT="${OBJECT:-serverless/split_sample_fc.zip}"
BUILD_DIR="${BUILD_DIR:-build/aliyun_split_sample}"
PACKAGE_DIR="${BUILD_DIR}/package"
ZIP_PATH="${BUILD_DIR}/split_sample_fc.zip"
FFMPEG_ARCHIVE="${BUILD_DIR}/ffmpeg-release-amd64-static.tar.xz"
FFMPEG_URL="${FFMPEG_URL:-https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz}"

if ! command -v aliyun >/dev/null 2>&1; then
  echo "Aliyun CLI is required. Install it with: brew install aliyun-cli" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip is required to build the FC package." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

if [[ -n "${LINUX_FFMPEG:-}" ]]; then
  mkdir -p "${BUILD_DIR}/ffmpeg-static"
  cp "${LINUX_FFMPEG}" "${BUILD_DIR}/ffmpeg-static/ffmpeg"
elif ! find "${BUILD_DIR}" -path '*ffmpeg*amd64-static/ffmpeg' -type f | grep -q .; then
  echo "Downloading Linux static ffmpeg..."
  curl -L "${FFMPEG_URL}" -o "${FFMPEG_ARCHIVE}"
  tar -xf "${FFMPEG_ARCHIVE}" -C "${BUILD_DIR}"
fi

FFMPEG_BIN="$(find "${BUILD_DIR}" -path '*ffmpeg*amd64-static/ffmpeg' -type f | head -n 1)"
if [[ -z "${FFMPEG_BIN}" && -f "${BUILD_DIR}/ffmpeg-static/ffmpeg" ]]; then
  FFMPEG_BIN="${BUILD_DIR}/ffmpeg-static/ffmpeg"
fi
if [[ -z "${FFMPEG_BIN}" ]]; then
  echo "Could not find Linux ffmpeg. Set LINUX_FFMPEG=/path/to/linux/ffmpeg and retry." >&2
  exit 1
fi

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/bin"
cp cloud/split_sample/app.py "${PACKAGE_DIR}/app.py"
cp "${FFMPEG_BIN}" "${PACKAGE_DIR}/bin/ffmpeg"
chmod +x "${PACKAGE_DIR}/bin/ffmpeg"

cat > "${PACKAGE_DIR}/bootstrap" <<'EOF'
#!/bin/sh
set -e
export PATH="$(pwd)/bin:/code/bin:/mnt/auto/bin:$PATH"
export PORT="${FC_SERVER_PORT:-${PORT:-9000}}"
export SAMPLES_PER_SHOT="${SAMPLES_PER_SHOT:-3}"
cd /code 2>/dev/null || cd "$(dirname "$0")"
exec python3 app.py
EOF
chmod +x "${PACKAGE_DIR}/bootstrap"

(cd "${PACKAGE_DIR}" && zip -qr "../$(basename "${ZIP_PATH}")" .)

for region in ${REGIONS}; do
  bucket="${PREFIX}-aliyun-${region}"
  fn="${PREFIX}-split-sample-${region}"
  echo "== ${region} =="
  aliyun oss cp "${ZIP_PATH}" "oss://${bucket}/${OBJECT}" --region "${region}" --force

  if aliyun fc get-function --region "${region}" --function-name "${fn}" >/dev/null 2>&1; then
    echo "Updating function ${fn}"
    aliyun fc update-function \
      --region "${region}" \
      --function-name "${fn}" \
      --runtime custom.debian10 \
      --handler bootstrap \
      --code ossBucketName="${bucket}" ossObjectName="${OBJECT}" \
      --cpu 2 \
      --memory-size 2048 \
      --timeout 900 \
      --disk-size 512 \
      --instance-concurrency 1 \
      --internet-access true \
      --environment-variables SAMPLES_PER_SHOT=3 PORT=9000 \
      --custom-runtime-config '{"command":["./bootstrap"],"port":9000}' >/dev/null
  else
    echo "Creating function ${fn}"
    aliyun fc create-function \
      --region "${region}" \
      --function-name "${fn}" \
      --runtime custom.debian10 \
      --handler bootstrap \
      --code ossBucketName="${bucket}" ossObjectName="${OBJECT}" \
      --cpu 2 \
      --memory-size 2048 \
      --timeout 900 \
      --disk-size 512 \
      --instance-concurrency 1 \
      --internet-access true \
      --environment-variables SAMPLES_PER_SHOT=3 PORT=9000 \
      --custom-runtime-config '{"command":["./bootstrap"],"port":9000}' >/dev/null
  fi

  trigger_json="$(mktemp)"
  if aliyun fc get-trigger --region "${region}" --function-name "${fn}" --trigger-name http >"${trigger_json}" 2>/dev/null; then
    :
  else
    aliyun fc create-trigger \
      --region "${region}" \
      --function-name "${fn}" \
      --trigger-name http \
      --trigger-type http \
      --trigger-config '{"authType":"anonymous","methods":["GET","POST"],"disableURLInternet":false}' >"${trigger_json}"
  fi
  python - <<'PY' "${region}" "${trigger_json}"
import json
import sys

region = sys.argv[1]
with open(sys.argv[2], encoding="utf-8") as f:
    data = json.load(f)
print(f"{region}: {data['httpTrigger']['urlInternet']}")
PY
  rm -f "${trigger_json}"
done
