#!/bin/bash

# 遇到任何错误立即停止脚本
set -e

echo "🚀 开始部署 AWS Lambda 函数..."

# 检查 serverless framework 是否安装
if ! command -v sls &> /dev/null; then
    echo "❌ 错误: serverless framework 未安装"
    echo "请运行: npm install -g serverless"
    exit 1
fi

# 部署到各个区域
REGIONS=("ap-southeast-1" "us-west-2")

for REGION in "${REGIONS[@]}"
do
    echo "========================================================"
    echo "🚀 正在部署到区域: $REGION"
    sls deploy --region "$REGION"
    echo "✅ 区域 $REGION 部署完成！"
done

echo "🎉 所有区域部署完成！"