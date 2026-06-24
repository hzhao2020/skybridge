#!/usr/bin/env bash
set -euo pipefail

REGIONS="${REGIONS:-us-west-2 us-east-2 ap-southeast-1 eu-central-1}"
FUNCTION_PREFIX="${FUNCTION_PREFIX:-skyflow-prototype-split-sample}"
ROLE_NAME="${ROLE_NAME:-skyflow-prototype-lambda-role}"
ZIP_PATH="${ZIP_PATH:-build/aws_split_sample/split_sample_lambda.zip}"

if [[ ! -f "${ZIP_PATH}" ]]; then
  cat >&2 <<EOF
Missing ${ZIP_PATH}.

Build it first with:
  mkdir -p build/aws_split_sample/package/bin
  cp cloud/split_sample/app.py cloud/split_sample/lambda_function.py build/aws_split_sample/package/
  cd build/aws_split_sample
  curl -L -o ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
  tar -xf ffmpeg-release-amd64-static.tar.xz
  cp ffmpeg-*-amd64-static/ffmpeg package/bin/ffmpeg
  chmod +x package/bin/ffmpeg
  cd package && zip -qr ../split_sample_lambda.zip .
EOF
  exit 1
fi

TRUST_POLICY="$(mktemp)"
cat > "${TRUST_POLICY}" <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
JSON

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  ROLE_ARN="$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)"
else
  ROLE_ARN="$(aws iam create-role --role-name "${ROLE_NAME}" --assume-role-policy-document "file://${TRUST_POLICY}" --query Role.Arn --output text)"
  aws iam attach-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  sleep 15
fi

for region in ${REGIONS}; do
  fn="${FUNCTION_PREFIX}-${region}"
  echo "Deploying ${fn} to ${region}"
  if aws lambda get-function --region "${region}" --function-name "${fn}" >/dev/null 2>&1; then
    aws lambda update-function-code \
      --region "${region}" \
      --function-name "${fn}" \
      --zip-file "fileb://${ZIP_PATH}" >/dev/null
    aws lambda wait function-updated --region "${region}" --function-name "${fn}"
    aws lambda update-function-configuration \
      --region "${region}" \
      --function-name "${fn}" \
      --runtime python3.11 \
      --handler lambda_function.handler \
      --memory-size 2048 \
      --timeout 900 \
      --architectures x86_64 \
      --environment "Variables={SAMPLES_PER_SHOT=3,PATH=/var/task/bin:/var/lang/bin:/usr/local/bin:/usr/bin/:/bin:/opt/bin}" >/dev/null
    aws lambda wait function-updated --region "${region}" --function-name "${fn}"
  else
    aws lambda create-function \
      --region "${region}" \
      --function-name "${fn}" \
      --runtime python3.11 \
      --handler lambda_function.handler \
      --zip-file "fileb://${ZIP_PATH}" \
      --role "${ROLE_ARN}" \
      --memory-size 2048 \
      --timeout 900 \
      --architectures x86_64 \
      --environment "Variables={SAMPLES_PER_SHOT=3,PATH=/var/task/bin:/var/lang/bin:/usr/local/bin:/usr/bin/:/bin:/opt/bin}" >/dev/null
    aws lambda wait function-active --region "${region}" --function-name "${fn}"
  fi

  if aws lambda get-function-url-config --region "${region}" --function-name "${fn}" >/dev/null 2>&1; then
    url="$(aws lambda get-function-url-config --region "${region}" --function-name "${fn}" --query FunctionUrl --output text)"
  else
    url="$(aws lambda create-function-url-config \
      --region "${region}" \
      --function-name "${fn}" \
      --auth-type NONE \
      --cors 'AllowOrigins=["*"],AllowMethods=["GET","POST"],AllowHeaders=["content-type"]' \
      --query FunctionUrl \
      --output text)"
  fi
  aws lambda add-permission \
    --region "${region}" \
    --function-name "${fn}" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE >/dev/null 2>&1 || true
  aws lambda add-permission \
    --region "${region}" \
    --function-name "${fn}" \
    --statement-id FunctionURLAllowPublicInvoke \
    --action lambda:InvokeFunction \
    --principal "*" \
    --invoked-via-function-url >/dev/null 2>&1 || true
  echo "${region} ${url}"
done
