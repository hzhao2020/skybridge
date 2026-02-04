#!/bin/bash
# Docker镜像加速器配置脚本

echo "=== 配置Docker镜像加速器 ==="

# 检查是否有sudo权限
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  需要sudo权限来配置Docker镜像加速器"
    echo "请手动执行以下命令："
    echo ""
    echo "sudo mkdir -p /etc/docker"
    echo "sudo tee /etc/docker/daemon.json <<EOF"
    echo "{"
    echo "  \"registry-mirrors\": ["
    echo "    \"https://docker.mirrors.ustc.edu.cn\","
    echo "    \"https://hub-mirror.c.163.com\","
    echo "    \"https://mirror.baidubce.com\""
    echo "  ]"
    echo "}"
    echo "EOF"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl restart docker"
    exit 1
fi

# 创建或更新daemon.json
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
EOF

echo "✅ Docker镜像加速器配置已更新"
echo "正在重启Docker服务..."

sudo systemctl daemon-reload
sudo systemctl restart docker

echo "✅ Docker服务已重启"
echo ""
echo "验证配置："
docker info | grep -A 10 "Registry Mirrors"
