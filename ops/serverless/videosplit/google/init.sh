# 1. 设置环境变量
export PROJECT_ID=$(gcloud config get-value project)
# 你的目标 Region
REGIONS=("us-west1" "asia-southeast1")
# 统一的仓库名称
REPO_NAME="experiment-repo"

echo "当前项目 ID: $PROJECT_ID"

# 2. 配置 Docker 认证 (一次性配置所有区域的域名)
echo "正在配置 Docker 认证..."
gcloud auth configure-docker \
    us-west1-docker.pkg.dev,asia-southeast1-docker.pkg.dev,asia-east1-docker.pkg.dev --quiet

# 3. 循环创建 Artifact Registry 仓库
# 注意：如果仓库已存在，它会报错但不会中断，忽略即可
for REGION in "${REGIONS[@]}"
do
    echo "正在检查/创建仓库 in $REGION ..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Experiment Repo for $REGION" || echo "仓库可能已存在，跳过创建步骤。"
done
