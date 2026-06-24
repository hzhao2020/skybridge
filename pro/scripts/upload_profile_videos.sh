#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-experiments/activitynet_splits/upload_manifests/profile_videos_upload_manifest_seed20260622.json}"
PROVIDERS="${PROVIDERS:-gcp aws aliyun}"
PREFIX="${PREFIX:-profile/videos}"
GSUTIL_PARALLEL_PROCESS_COUNT="${GSUTIL_PARALLEL_PROCESS_COUNT:-1}"
GSUTIL_PARALLEL_THREAD_COUNT="${GSUTIL_PARALLEL_THREAD_COUNT:-4}"
AWS_UPLOAD_RETRIES="${AWS_UPLOAD_RETRIES:-6}"
AWS_CLI_CONNECT_TIMEOUT="${AWS_CLI_CONNECT_TIMEOUT:-60}"
AWS_CLI_READ_TIMEOUT="${AWS_CLI_READ_TIMEOUT:-300}"
ALIYUN_CONNECT_TIMEOUT="${ALIYUN_CONNECT_TIMEOUT:-30}"
ALIYUN_READ_TIMEOUT="${ALIYUN_READ_TIMEOUT:-300}"
ALIYUN_RETRY_TIMES="${ALIYUN_RETRY_TIMES:-20}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

paths_file="$(mktemp)"
python - <<'PY' "${MANIFEST}" "${paths_file}"
import json
import sys

manifest_path, paths_path = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)
with open(paths_path, "w", encoding="utf-8") as f:
    for item in manifest["items"]:
        f.write(item["path"] + "\n")
PY

upload_gcp() {
  command -v gsutil >/dev/null 2>&1 || { echo "gsutil not found" >&2; return 1; }
  local regions=(${GCP_REGIONS:-us-east1 us-west1 europe-west1 asia-east1})
  for region in "${regions[@]}"; do
    local bucket="skyflow-prototype-gcp-${region}"
    echo "== GCP ${region}: gs://${bucket}/${PREFIX}/ =="
    python - <<'PY' "${MANIFEST}" "${bucket}" "${PREFIX}" "${GSUTIL_PARALLEL_PROCESS_COUNT}" "${GSUTIL_PARALLEL_THREAD_COUNT}"
import json
import subprocess
import sys

manifest_path, bucket, prefix, process_count, thread_count = sys.argv[1:6]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)
total = len(manifest["items"])
base = [
    "gsutil",
    "-o",
    f"GSUtil:parallel_process_count={process_count}",
    "-o",
    f"GSUtil:parallel_thread_count={thread_count}",
]
for index, item in enumerate(manifest["items"], start=1):
    dest = f"gs://{bucket}/{prefix}/{item['filename']}"
    stat = subprocess.run(base + ["-q", "stat", dest])
    if stat.returncode == 0:
        print(f"[{index}/{total}] skip {dest}", flush=True)
        continue
    print(f"[{index}/{total}] upload {item['filename']} -> {dest}", flush=True)
    subprocess.run(base + ["cp", "-n", item["path"], dest], check=True)
PY
    gsutil ls "gs://${bucket}/${PREFIX}/*.mp4" | wc -l
  done
}

upload_aws() {
  command -v aws >/dev/null 2>&1 || { echo "aws not found" >&2; return 1; }
  local regions=(${AWS_REGIONS:-us-west-2 us-east-2 ap-southeast-1 eu-central-1})
  for region in "${regions[@]}"; do
    local bucket="skyflow-prototype-aws-${region}"
    echo "== AWS ${region}: s3://${bucket}/${PREFIX}/ =="
    AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-10}" AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}" \
    python - <<'PY' "${MANIFEST}" "${region}" "${bucket}" "${PREFIX}" "${AWS_UPLOAD_RETRIES}" "${AWS_CLI_CONNECT_TIMEOUT}" "${AWS_CLI_READ_TIMEOUT}"
import json
import os
import subprocess
import sys
import time

manifest_path, region, bucket, prefix, retries, connect_timeout, read_timeout = sys.argv[1:8]
retries = int(retries)
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)
total = len(manifest["items"])
for index, item in enumerate(manifest["items"], start=1):
    key = f"{prefix}/{item['filename']}"
    head = subprocess.run(
        ["aws", "s3api", "head-object", "--region", region, "--bucket", bucket, "--key", key],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if head.returncode == 0:
        print(f"[{index}/{total}] skip s3://{bucket}/{key}", flush=True)
        continue
    print(f"[{index}/{total}] upload {item['filename']} -> s3://{bucket}/{key}", flush=True)
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                item["path"],
                f"s3://{bucket}/{key}",
                "--region",
                region,
                "--cli-connect-timeout",
                connect_timeout,
                "--cli-read-timeout",
                read_timeout,
            ],
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            break
        head = subprocess.run(
            ["aws", "s3api", "head-object", "--region", region, "--bucket", bucket, "--key", key],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if head.returncode == 0:
            print(f"[{index}/{total}] object exists after failed upload, continue", flush=True)
            break
        if attempt == retries:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        sleep_s = min(120, 5 * attempt)
        print(f"[{index}/{total}] retry {attempt}/{retries} after {sleep_s}s", flush=True)
        time.sleep(sleep_s)
PY
    aws s3 ls "s3://${bucket}/${PREFIX}/" --region "${region}" | grep -c '.mp4$' || true
  done
}

upload_aliyun() {
  command -v aliyun >/dev/null 2>&1 || { echo "aliyun not found" >&2; return 1; }
  local regions=(${ALIYUN_REGIONS:-cn-shanghai cn-beijing us-east-1 ap-southeast-1})
  for region in "${regions[@]}"; do
    local bucket="skyflow-prototype-aliyun-${region}"
    echo "== Aliyun ${region}: oss://${bucket}/${PREFIX}/ =="
    python - <<'PY' "${MANIFEST}" "${region}" "${bucket}" "${PREFIX}" "${ALIYUN_CONNECT_TIMEOUT}" "${ALIYUN_READ_TIMEOUT}" "${ALIYUN_RETRY_TIMES}"
import json
import subprocess
import sys

manifest_path, region, bucket, prefix, connect_timeout, read_timeout, retry_times = sys.argv[1:8]
endpoints = {
    "cn-shanghai": "oss-cn-shanghai.aliyuncs.com",
    "cn-beijing": "oss-cn-beijing.aliyuncs.com",
    "us-east-1": "oss-us-east-1.aliyuncs.com",
    "ap-southeast-1": "oss-ap-southeast-1.aliyuncs.com",
}
endpoint = endpoints[region]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)
total = len(manifest["items"])
for index, item in enumerate(manifest["items"], start=1):
    key = f"{prefix}/{item['filename']}"
    dest = f"oss://{bucket}/{key}"
    exists = subprocess.run(
        [
            "aliyun", "oss", "stat", dest,
            "--region", region,
            "--endpoint", endpoint,
            "--connect-timeout", connect_timeout,
            "--read-timeout", read_timeout,
            "--retry-times", retry_times,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode == 0:
        print(f"[{index}/{total}] skip {dest}", flush=True)
        continue
    print(f"[{index}/{total}] upload {item['filename']} -> {dest}", flush=True)
    subprocess.run(
        [
            "aliyun", "oss", "cp", item["path"], dest,
            "--region", region,
            "--endpoint", endpoint,
            "--connect-timeout", connect_timeout,
            "--read-timeout", read_timeout,
            "--retry-times", retry_times,
            "--force",
        ],
        check=True,
    )
PY
    aliyun oss ls "oss://${bucket}/${PREFIX}/" \
      --region "${region}" \
      --endpoint "oss-${region}.aliyuncs.com" \
      --connect-timeout "${ALIYUN_CONNECT_TIMEOUT}" \
      --read-timeout "${ALIYUN_READ_TIMEOUT}" \
      --retry-times "${ALIYUN_RETRY_TIMES}" | grep -c '.mp4' || true
  done
}

for provider in ${PROVIDERS}; do
  case "${provider}" in
    gcp) upload_gcp ;;
    aws) upload_aws ;;
    aliyun) upload_aliyun ;;
    *) echo "Unknown provider: ${provider}" >&2; exit 1 ;;
  esac
done

rm -f "${paths_file}"
