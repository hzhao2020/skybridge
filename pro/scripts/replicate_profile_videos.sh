#!/usr/bin/env bash
set -euo pipefail

PREFIX="${PREFIX:-profile/videos}"
PROVIDERS="${PROVIDERS:-gcp aws aliyun}"

GCP_SEED_REGION="${GCP_SEED_REGION:-asia-east1}"
AWS_SEED_REGION="${AWS_SEED_REGION:-ap-southeast-1}"
ALIYUN_SEED_REGION="${ALIYUN_SEED_REGION:-cn-shanghai}"

GCP_REGIONS="${GCP_REGIONS:-us-east1 us-west1 europe-west1 asia-east1}"
AWS_REGIONS="${AWS_REGIONS:-us-west-2 us-east-2 ap-southeast-1 eu-central-1}"
ALIYUN_REGIONS="${ALIYUN_REGIONS:-cn-shanghai cn-beijing us-east-1 ap-southeast-1}"
ALIYUN_CONNECT_TIMEOUT="${ALIYUN_CONNECT_TIMEOUT:-30}"
ALIYUN_READ_TIMEOUT="${ALIYUN_READ_TIMEOUT:-300}"
ALIYUN_RETRY_TIMES="${ALIYUN_RETRY_TIMES:-20}"

replicate_gcp() {
  command -v gsutil >/dev/null 2>&1 || { echo "gsutil not found" >&2; return 1; }
  local source_bucket="skyflow-prototype-gcp-${GCP_SEED_REGION}"
  for region in ${GCP_REGIONS}; do
    if [[ "${region}" == "${GCP_SEED_REGION}" ]]; then
      continue
    fi
    local target_bucket="skyflow-prototype-gcp-${region}"
    echo "== GCP ${GCP_SEED_REGION} -> ${region} =="
    gsutil -m rsync -r "gs://${source_bucket}/${PREFIX}" "gs://${target_bucket}/${PREFIX}"
    gsutil ls "gs://${target_bucket}/${PREFIX}/*.mp4" | wc -l
  done
}

replicate_aws() {
  command -v aws >/dev/null 2>&1 || { echo "aws not found" >&2; return 1; }
  local source_bucket="skyflow-prototype-aws-${AWS_SEED_REGION}"
  for region in ${AWS_REGIONS}; do
    if [[ "${region}" == "${AWS_SEED_REGION}" ]]; then
      continue
    fi
    local target_bucket="skyflow-prototype-aws-${region}"
    echo "== AWS ${AWS_SEED_REGION} -> ${region} =="
    aws s3 sync \
      "s3://${source_bucket}/${PREFIX}/" \
      "s3://${target_bucket}/${PREFIX}/" \
      --source-region "${AWS_SEED_REGION}" \
      --region "${region}"
    aws s3 ls "s3://${target_bucket}/${PREFIX}/" --region "${region}" | grep -c '.mp4$' || true
  done
}

replicate_aliyun() {
  command -v aliyun >/dev/null 2>&1 || { echo "aliyun not found" >&2; return 1; }
  local source_bucket="skyflow-prototype-aliyun-${ALIYUN_SEED_REGION}"
  for region in ${ALIYUN_REGIONS}; do
    if [[ "${region}" == "${ALIYUN_SEED_REGION}" ]]; then
      continue
    fi
    local target_bucket="skyflow-prototype-aliyun-${region}"
    echo "== Aliyun ${ALIYUN_SEED_REGION} -> ${region} =="
    aliyun oss cp \
      "oss://${source_bucket}/${PREFIX}/" \
      "oss://${target_bucket}/${PREFIX}/" \
      --recursive \
      --region "${region}" \
      --endpoint "oss-${region}.aliyuncs.com" \
      --connect-timeout "${ALIYUN_CONNECT_TIMEOUT}" \
      --read-timeout "${ALIYUN_READ_TIMEOUT}" \
      --retry-times "${ALIYUN_RETRY_TIMES}" \
      --force
    aliyun oss ls "oss://${target_bucket}/${PREFIX}/" \
      --region "${region}" \
      --endpoint "oss-${region}.aliyuncs.com" \
      --connect-timeout "${ALIYUN_CONNECT_TIMEOUT}" \
      --read-timeout "${ALIYUN_READ_TIMEOUT}" \
      --retry-times "${ALIYUN_RETRY_TIMES}" | grep -c '.mp4' || true
  done
}

for provider in ${PROVIDERS}; do
  case "${provider}" in
    gcp) replicate_gcp ;;
    aws) replicate_aws ;;
    aliyun) replicate_aliyun ;;
    *) echo "Unknown provider: ${provider}" >&2; exit 1 ;;
  esac
done
