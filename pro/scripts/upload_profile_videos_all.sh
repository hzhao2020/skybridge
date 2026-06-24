#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${MANIFEST:-experiments/activitynet_splits/upload_manifests/profile_videos_upload_manifest_seed20260622.json}"
PREFIX="${PREFIX:-profile/videos}"

GCP_SEED_REGION="${GCP_SEED_REGION:-asia-east1}"
AWS_SEED_REGION="${AWS_SEED_REGION:-ap-southeast-1}"
ALIYUN_SEED_REGION="${ALIYUN_SEED_REGION:-cn-shanghai}"

GCP_REGIONS="${GCP_REGIONS:-us-east1 us-west1 europe-west1 asia-east1}"
AWS_REGIONS="${AWS_REGIONS:-us-west-2 us-east-2 ap-southeast-1 eu-central-1}"
ALIYUN_REGIONS="${ALIYUN_REGIONS:-cn-shanghai cn-beijing us-east-1 ap-southeast-1}"
ALIYUN_CONNECT_TIMEOUT="${ALIYUN_CONNECT_TIMEOUT:-30}"
ALIYUN_READ_TIMEOUT="${ALIYUN_READ_TIMEOUT:-300}"
ALIYUN_RETRY_TIMES="${ALIYUN_RETRY_TIMES:-20}"

expected_count="$(
  python - <<'PY' "${MANIFEST}"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)
print(len(manifest["items"]))
PY
)"

echo "Profile videos: ${expected_count}"
echo "Object prefix: ${PREFIX}"
echo

echo "== Step 1/3: upload local videos to seed buckets =="
PROVIDERS=gcp GCP_REGIONS="${GCP_SEED_REGION}" MANIFEST="${MANIFEST}" PREFIX="${PREFIX}" \
  "${SCRIPT_DIR}/upload_profile_videos.sh"
PROVIDERS=aws AWS_REGIONS="${AWS_SEED_REGION}" MANIFEST="${MANIFEST}" PREFIX="${PREFIX}" \
  "${SCRIPT_DIR}/upload_profile_videos.sh"
PROVIDERS=aliyun ALIYUN_REGIONS="${ALIYUN_SEED_REGION}" MANIFEST="${MANIFEST}" PREFIX="${PREFIX}" \
  ALIYUN_CONNECT_TIMEOUT="${ALIYUN_CONNECT_TIMEOUT}" \
  ALIYUN_READ_TIMEOUT="${ALIYUN_READ_TIMEOUT}" \
  ALIYUN_RETRY_TIMES="${ALIYUN_RETRY_TIMES}" \
  "${SCRIPT_DIR}/upload_profile_videos.sh"

count_gcp() {
  local region="$1"
  local bucket="skyflow-prototype-gcp-${region}"
  gsutil ls "gs://${bucket}/${PREFIX}/*.mp4" 2>/dev/null | wc -l | tr -d ' '
}

count_aws() {
  local region="$1"
  local bucket="skyflow-prototype-aws-${region}"
  aws s3 ls "s3://${bucket}/${PREFIX}/" --region "${region}" 2>/dev/null | grep -c '.mp4$' || true
}

count_aliyun() {
  local region="$1"
  local bucket="skyflow-prototype-aliyun-${region}"
  aliyun oss ls "oss://${bucket}/${PREFIX}/" \
    --region "${region}" \
    --endpoint "oss-${region}.aliyuncs.com" \
    --connect-timeout "${ALIYUN_CONNECT_TIMEOUT}" \
    --read-timeout "${ALIYUN_READ_TIMEOUT}" \
    --retry-times "${ALIYUN_RETRY_TIMES}" 2>/dev/null | grep -c '.mp4' || true
}

check_count() {
  local label="$1"
  local count="$2"
  echo "${label}: ${count}/${expected_count}"
  if [[ "${count}" != "${expected_count}" ]]; then
    echo "Count check failed for ${label}" >&2
    return 1
  fi
}

echo
echo "== Seed bucket checks =="
check_count "GCP ${GCP_SEED_REGION}" "$(count_gcp "${GCP_SEED_REGION}")"
check_count "AWS ${AWS_SEED_REGION}" "$(count_aws "${AWS_SEED_REGION}")"
check_count "Aliyun ${ALIYUN_SEED_REGION}" "$(count_aliyun "${ALIYUN_SEED_REGION}")"

echo
echo "== Step 2/3: replicate seed buckets to other regions =="
PROVIDERS="gcp aws aliyun" \
  GCP_SEED_REGION="${GCP_SEED_REGION}" \
  AWS_SEED_REGION="${AWS_SEED_REGION}" \
  ALIYUN_SEED_REGION="${ALIYUN_SEED_REGION}" \
  GCP_REGIONS="${GCP_REGIONS}" \
  AWS_REGIONS="${AWS_REGIONS}" \
  ALIYUN_REGIONS="${ALIYUN_REGIONS}" \
  ALIYUN_CONNECT_TIMEOUT="${ALIYUN_CONNECT_TIMEOUT}" \
  ALIYUN_READ_TIMEOUT="${ALIYUN_READ_TIMEOUT}" \
  ALIYUN_RETRY_TIMES="${ALIYUN_RETRY_TIMES}" \
  PREFIX="${PREFIX}" \
  "${SCRIPT_DIR}/replicate_profile_videos.sh"

echo
echo "== Step 3/3: final bucket checks =="
for region in ${GCP_REGIONS}; do
  check_count "GCP ${region}" "$(count_gcp "${region}")"
done
for region in ${AWS_REGIONS}; do
  check_count "AWS ${region}" "$(count_aws "${region}")"
done
for region in ${ALIYUN_REGIONS}; do
  check_count "Aliyun ${region}" "$(count_aliyun "${region}")"
done

echo
echo "Done. All configured buckets contain ${expected_count} profile videos."
