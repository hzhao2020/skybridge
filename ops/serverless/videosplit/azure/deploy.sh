#!/bin/bash

# 遇到任何错误立即停止脚本
set -e

# 切换到脚本所在目录（确保 Dockerfile 在正确的位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "工作目录: $(pwd)"

# === 脚本配置 ===
FUNCTION_NAME="video-splitter"
RESOURCE_GROUP="vqa"
REGIONS=("eastasia" "westus2")
ACR_NAME="vqaregistryvqa"  # Azure Container Registry 名称，请根据实际情况修改
SUBSCRIPTION_ID="5f52a986-5b81-46cb-83f2-20c17e8d58d9"

# 如果设置了 SKIP_BUILD=true，将跳过构建步骤
SKIP_BUILD_ENV="${SKIP_BUILD:-false}"

# 存储账户配置（与 config.py 中的配置对应）
declare -A STORAGE_ACCOUNTS
STORAGE_ACCOUNTS["eastasia"]="videoea"
STORAGE_ACCOUNTS["westus2"]="videowu"

declare -A STORAGE_CONTAINERS
STORAGE_CONTAINERS["eastasia"]="video-ea"
STORAGE_CONTAINERS["westus2"]="video-wu"

echo "=== Azure Functions 部署工具 ==="
echo "订阅 ID: $SUBSCRIPTION_ID"
echo "资源组: $RESOURCE_GROUP"

# 配置Docker使用阿里云镜像加速器
echo "=== 配置Docker镜像加速器 ==="
if sudo -n true 2>/dev/null; then
    if [ -f /etc/docker/daemon.json ]; then
        echo "检测到现有Docker配置，备份为 /etc/docker/daemon.json.bak"
        sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak 2>/dev/null || true
    fi
    
    # 创建或更新Docker daemon配置使用阿里云镜像加速器
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<'DOCKER_CONFIG'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com",
    "https://registry.docker-cn.com"
  ]
}
DOCKER_CONFIG
    
    # 重启Docker服务以应用配置
    echo "重启Docker服务以应用镜像加速器配置..."
    sudo systemctl daemon-reload 2>/dev/null || true
    sudo systemctl restart docker 2>/dev/null || true
    
    # 等待Docker服务启动
    sleep 3
    echo "✅ Docker镜像加速器配置完成"
else
    echo "⚠️  无法自动配置Docker镜像加速器（需要sudo权限）"
    echo "   请手动执行以下命令配置镜像加速器："
    echo "   sudo mkdir -p /etc/docker"
    echo "   sudo tee /etc/docker/daemon.json <<'EOF'"
    echo "   {"
    echo "     \"registry-mirrors\": ["
    echo "       \"https://docker.mirrors.ustc.edu.cn\","
    echo "       \"https://hub-mirror.c.163.com\""
    echo "     ]"
    echo "   }"
    echo "   EOF"
    echo "   sudo systemctl daemon-reload && sudo systemctl restart docker"
    echo ""
    echo "   或者继续使用当前配置（可能较慢）..."
fi

# 设置默认订阅
az account set --subscription $SUBSCRIPTION_ID

# ---------------------------------------------------------
# 第一步：删除已有的 Function Apps
# ---------------------------------------------------------
echo "=== 删除已有的 Function Apps ==="
for REGION in "${REGIONS[@]}"
do
  FUNCTION_APP_NAME="${FUNCTION_NAME}-${REGION}"
  echo "检查 Function App: $FUNCTION_APP_NAME"
  FUNCTION_APP_EXISTS=$(az functionapp list --resource-group $RESOURCE_GROUP --query "[?name=='$FUNCTION_APP_NAME'].name" -o tsv)
  if [ -n "$FUNCTION_APP_EXISTS" ]; then
    echo "删除 Function App: $FUNCTION_APP_NAME"
    echo "删除 Function App: $FUNCTION_APP_NAME"
    az functionapp delete --name $FUNCTION_APP_NAME --resource-group $RESOURCE_GROUP || true
  else
    echo "Function App $FUNCTION_APP_NAME 不存在，跳过删除"
  fi
done

# ---------------------------------------------------------
# 第二步：确保 Azure Container Registry 存在并登录
# ---------------------------------------------------------
echo "=== 检查 Azure Container Registry ==="
ACR_EXISTS=$(az acr list --resource-group $RESOURCE_GROUP --query "[?name=='$ACR_NAME'].name" -o tsv)
if [ -z "$ACR_EXISTS" ]; then
    echo "创建 Azure Container Registry: $ACR_NAME"
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true
else
    echo "Azure Container Registry 已存在: $ACR_NAME"
fi

# 登录到 ACR
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer -o tsv)
echo "登录到 ACR: $ACR_LOGIN_SERVER"
# 检查Docker权限，如果没有权限则使用token方式
if docker ps > /dev/null 2>&1; then
    az acr login --name $ACR_NAME
else
    echo "⚠️  Docker权限不足，使用token方式登录ACR"
    ACR_TOKEN=$(az acr login --name $ACR_NAME --expose-token --query accessToken -o tsv)
    echo "$ACR_TOKEN" | docker login $ACR_LOGIN_SERVER -u 00000000-0000-0000-0000-000000000000 --password-stdin || {
        echo "❌ ACR登录失败，请检查Docker权限或手动执行: sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    }
fi

# ---------------------------------------------------------
# 第三步：本地构建 Docker 镜像
# ---------------------------------------------------------
echo "=== 本地构建 Docker 镜像 ==="

# 检查镜像是否已存在
IMAGE_EXISTS=$(docker images -q local-video-splitter:latest 2>/dev/null)
SKIP_BUILD=false

if [ -n "$IMAGE_EXISTS" ]; then
    echo "检测到已存在的镜像: local-video-splitter:latest"
    # 检查 Dockerfile 和相关文件是否有更新
    DOCKERFILE_TIME=$(stat -c %Y Dockerfile 2>/dev/null || echo 0)
    IMAGE_TIME=$(docker inspect -f '{{ .Created }}' local-video-splitter:latest 2>/dev/null | xargs -I {} date -d {} +%s 2>/dev/null || echo 0)
    
    if [ "$DOCKERFILE_TIME" -le "$IMAGE_TIME" ] 2>/dev/null; then
        echo "Dockerfile 和相关文件未更新，使用现有镜像"
        SKIP_BUILD=true
    else
        echo "检测到文件更新，需要重新构建镜像"
    fi
fi

# 如果设置了环境变量 SKIP_BUILD=true，强制跳过构建
if [ "$SKIP_BUILD_ENV" = "true" ]; then
    echo "⚠️  环境变量 SKIP_BUILD=true，跳过构建步骤"
    SKIP_BUILD=true
fi

if [ "$SKIP_BUILD" = false ]; then
    # 尝试使用代理加速（如果可用）
    PROXY_URL="${https_proxy:-${HTTPS_PROXY:-http://127.0.0.1:7897}}"
    BUILD_ARGS=""
    if curl -s --connect-timeout 2 --proxy "$PROXY_URL" https://www.google.com > /dev/null 2>&1; then
        echo "检测到代理可用，使用代理构建: $PROXY_URL"
        BUILD_ARGS="--build-arg HTTP_PROXY=$PROXY_URL --build-arg HTTPS_PROXY=$PROXY_URL --build-arg http_proxy=$PROXY_URL --build-arg https_proxy=$PROXY_URL"
    else
        echo "未检测到可用代理，使用直连构建"
    fi

    echo "开始构建 Docker 镜像（这可能需要几分钟）..."
    echo "构建命令: docker build --platform linux/amd64 $BUILD_ARGS -t local-video-splitter:latest ."
    
    # 设置超时（30分钟）
    if timeout 1800 docker build --platform linux/amd64 $BUILD_ARGS -t local-video-splitter:latest .; then
        echo "✅ Docker 镜像构建成功！"
    else
        BUILD_EXIT_CODE=$?
        if [ $BUILD_EXIT_CODE -eq 124 ]; then
            echo "⚠️  Docker 构建超时（超过30分钟）！"
        else
            echo "⚠️  Docker 构建失败！"
        fi
        
        # 检查是否有现有镜像可以使用
        if [ -n "$IMAGE_EXISTS" ]; then
            echo "检测到现有镜像，将使用现有镜像继续部署..."
            echo "提示：如果需要使用最新构建的镜像，请稍后手动运行构建命令："
            echo "  docker build --platform linux/amd64 -t local-video-splitter:latest ."
        else
            echo "❌ 没有可用的镜像，无法继续部署！"
            echo "提示：如果网络较慢，可以尝试："
            echo "  1. 配置 Docker 镜像加速器（脚本已尝试自动配置）"
            echo "  2. 检查代理设置是否正确"
            echo "  3. 手动运行: docker build --platform linux/amd64 -t local-video-splitter:latest ."
            exit 1
        fi
    fi
else
    echo "✅ 跳过构建，使用现有镜像"
fi

# ---------------------------------------------------------
# 第四步：循环部署到各个 Region
# ---------------------------------------------------------
for REGION in "${REGIONS[@]}"
do
  echo "========================================================"
  echo "🚀 正在部署到区域: $REGION"
  
  STORAGE_ACCOUNT=${STORAGE_ACCOUNTS[$REGION]}
  STORAGE_CONTAINER=${STORAGE_CONTAINERS[$REGION]}
  
  # 构建函数应用名称（Azure Functions 名称必须全局唯一）
  FUNCTION_APP_NAME="${FUNCTION_NAME}-${REGION}"
  
  # 1. 标记并推送镜像到 ACR
  IMAGE_TAG="${ACR_LOGIN_SERVER}/${FUNCTION_NAME}:${REGION}-latest"
  echo "标记镜像: $IMAGE_TAG"
  docker tag local-video-splitter:latest $IMAGE_TAG
  
  echo "推送镜像到 ACR..."
  docker push $IMAGE_TAG
  
  # 2. 检查存储账户是否存在
  STORAGE_ACCOUNT_EXISTS=$(az storage account list --resource-group $RESOURCE_GROUP --query "[?name=='$STORAGE_ACCOUNT'].name" -o tsv)
  if [ -z "$STORAGE_ACCOUNT_EXISTS" ]; then
      echo "⚠️  警告: 存储账户 $STORAGE_ACCOUNT 不存在，请先创建。"
  fi
  
  # 3. 获取存储账户连接字符串（用于设置环境变量）
  STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
      --name $STORAGE_ACCOUNT \
      --resource-group $RESOURCE_GROUP \
      --query connectionString -o tsv 2>/dev/null || echo "")
  
  # 4. 检查 Function App 是否存在
  FUNCTION_APP_EXISTS=$(az functionapp list --resource-group $RESOURCE_GROUP --query "[?name=='$FUNCTION_APP_NAME'].name" -o tsv)
  
  if [ -z "$FUNCTION_APP_EXISTS" ]; then
      echo "创建新的 Function App: $FUNCTION_APP_NAME (使用 Consumption Plan - 按调用计费)"
      
      # 创建 Function App (使用 Consumption Plan)
      # Consumption Plan 不需要单独的 App Service Plan，使用 --consumption-plan-location 即可
      az functionapp create \
          --resource-group $RESOURCE_GROUP \
          --name $FUNCTION_APP_NAME \
          --storage-account $STORAGE_ACCOUNT \
          --consumption-plan-location $REGION \
          --runtime python \
          --runtime-version 3.11 \
          --functions-version 4 \
          --os-type Linux \
          --deployment-container-image-name $IMAGE_TAG
      
      # 配置 Function App 设置
      echo "配置 Function App 设置..."
      az functionapp config appsettings set \
          --resource-group $RESOURCE_GROUP \
          --name $FUNCTION_APP_NAME \
          --settings \
              "AZURE_STORAGE_CONNECTION_STRING_${STORAGE_ACCOUNT^^}=$STORAGE_CONNECTION_STRING" \
              "AZURE_STORAGE_ACCOUNT_EA=videoea" \
              "AZURE_STORAGE_ACCOUNT_WU=videowu" \
              "FUNCTIONS_WORKER_RUNTIME=python" \
              "FUNCTIONS_EXTENSION_VERSION=~4" \
              "WEBSITES_ENABLE_APP_SERVICE_STORAGE=false" \
              "DOCKER_REGISTRY_SERVER_URL=https://${ACR_LOGIN_SERVER}" \
              "DOCKER_REGISTRY_SERVER_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)" \
              "DOCKER_REGISTRY_SERVER_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)"
      
      # 配置容器镜像（对于 Consumption Plan，使用应用设置）
      echo "配置容器镜像设置..."
      az functionapp config appsettings set \
          --resource-group $RESOURCE_GROUP \
          --name $FUNCTION_APP_NAME \
          --settings \
              "DOCKER_CUSTOM_IMAGE_NAME=$IMAGE_TAG" \
              "DOCKER_REGISTRY_SERVER_URL=https://${ACR_LOGIN_SERVER}" \
              "DOCKER_REGISTRY_SERVER_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)" \
              "DOCKER_REGISTRY_SERVER_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)" \
          > /dev/null 2>&1 || echo "⚠️  容器配置警告（Consumption Plan 可能需要重启后生效）"
      
  else
      echo "更新现有 Function App: $FUNCTION_APP_NAME"
      
      # 更新容器镜像（对于 Consumption Plan，使用应用设置）
      echo "更新容器镜像设置..."
      az functionapp config appsettings set \
          --resource-group $RESOURCE_GROUP \
          --name $FUNCTION_APP_NAME \
          --settings \
              "DOCKER_CUSTOM_IMAGE_NAME=$IMAGE_TAG" \
              "DOCKER_REGISTRY_SERVER_URL=https://${ACR_LOGIN_SERVER}" \
              "DOCKER_REGISTRY_SERVER_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)" \
              "DOCKER_REGISTRY_SERVER_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)" \
          > /dev/null 2>&1 || echo "⚠️  容器配置警告（Consumption Plan 可能需要重启后生效）"
      
      # 重启 Function App 以应用新镜像
      echo "重启 Function App..."
      az functionapp restart \
          --resource-group $RESOURCE_GROUP \
          --name $FUNCTION_APP_NAME
  fi
  
  # 5. 获取 Function App URL
  FUNCTION_URL=$(az functionapp show --resource-group $RESOURCE_GROUP --name $FUNCTION_APP_NAME --query defaultHostName -o tsv)
  echo "✅ 区域 $REGION 部署完成！"
  echo "   Function URL: https://${FUNCTION_URL}/api/video_split"
done

echo "🎉 所有 Azure Functions 部署完毕！"
