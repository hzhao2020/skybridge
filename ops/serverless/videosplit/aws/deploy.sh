#!/bin/bash

# === 脚本配置 ===
APP_NAME="video-splitter"
ROLE_NAME="lambda-video-splitter-role"
REGIONS=("us-west-2" "ap-southeast-1")

# 获取当前账户 ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "=== AWS 部署工具 (Root 模式) ==="
echo "当前账户: $AWS_ACCOUNT_ID"

# ---------------------------------------------------------
# 第一步：检查 IAM 角色
# ---------------------------------------------------------
echo "Checking IAM Role..."
aws iam get-role --role-name $ROLE_NAME > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "创建 IAM 角色..."
    # (省略创建代码，假设你角色已经存在)
else
    echo "IAM 角色已存在，跳过创建。"
fi
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"

# ---------------------------------------------------------
# 第二步：删除已有的 Lambda 函数
# ---------------------------------------------------------
echo "=== 删除已有的 Lambda 函数 ==="
for REGION in "${REGIONS[@]}"
do
  echo "检查区域 $REGION 的函数..."
  if aws lambda get-function --function-name $APP_NAME --region $REGION > /dev/null 2>&1; then
    echo "删除函数: $APP_NAME (区域: $REGION)"
    aws lambda delete-function --function-name $APP_NAME --region $REGION || true
  else
    echo "函数 $APP_NAME 在区域 $REGION 不存在，跳过删除"
  fi
done

# ---------------------------------------------------------
# 第三步：本地构建 Docker 镜像 (关键修改)
# ---------------------------------------------------------
echo "=== 本地构建 Docker 镜像 ==="

# 核心修改：添加 --network=host 以解决 yum 连接超时问题
docker build --network=host --platform linux/amd64 -t local-video-splitter .

if [ $? -ne 0 ]; then
    echo "❌ Docker 构建失败！请检查 Dockerfile 是否已更新，以及网络连接。"
    exit 1
fi

# ---------------------------------------------------------
# 第四步：循环部署到各个 Region
# ---------------------------------------------------------
for REGION in "${REGIONS[@]}"
do
  echo "========================================================"
  echo "🚀 正在部署到区域: $REGION"
  
  # 1. 登录 ECR
  aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

  # 2. 确保 ECR 仓库存在
  aws ecr describe-repositories --repository-names $APP_NAME --region $REGION > /dev/null 2>&1
  if [ $? -ne 0 ]; then
      aws ecr create-repository --repository-name $APP_NAME --region $REGION > /dev/null
  fi

  # 3. 推送镜像
  ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}:latest"
  docker tag local-video-splitter:latest $ECR_URI
  docker push $ECR_URI

  # 4. 更新/创建 Lambda
  aws lambda get-function --function-name $APP_NAME --region $REGION > /dev/null 2>&1
  if [ $? -eq 0 ]; then
      echo "更新现有函数代码..."
      aws lambda update-function-code --function-name $APP_NAME --image-uri $ECR_URI --region $REGION > /dev/null
  else
      echo "创建新函数..."
      aws lambda create-function \
          --function-name $APP_NAME \
          --package-type Image \
          --code ImageUri=$ECR_URI \
          --role $ROLE_ARN \
          --memory-size 2048 \
          --timeout 300 \
          --architectures x86_64 \
          --region $REGION > /dev/null
  fi
  echo "区域 $REGION 部署完成！"
done

echo "🎉 所有 AWS 节点部署完毕！"