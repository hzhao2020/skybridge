# AWS Lambda Video Split Function

## 部署说明

### 方法1：使用 ZIP 文件部署

1. 安装依赖：
```bash
pip install boto3 -t .
```

2. 创建部署包：
```bash
zip -r lambda_function.zip lambda_function.py boto3* botocore*
```

3. 创建 Lambda 函数：
```bash
aws lambda create-function \
  --function-name video-split-us-west-2 \
  --runtime python3.11 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda_function.zip \
  --timeout 600 \
  --memory-size 2048
```

### 方法2：使用容器镜像部署（推荐，支持 ffmpeg）

1. 创建 Dockerfile：
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# 安装 ffmpeg
RUN yum install -y ffmpeg

# 复制函数代码
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# 设置处理程序
CMD [ "lambda_function.lambda_handler" ]
```

2. 构建和推送镜像：
```bash
docker build -t video-split-lambda .
aws ecr create-repository --repository-name video-split-lambda
docker tag video-split-lambda:latest YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/video-split-lambda:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/video-split-lambda:latest
```

3. 创建 Lambda 函数：
```bash
aws lambda create-function \
  --function-name video-split-us-west-2 \
  --package-type Image \
  --code ImageUri=YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/video-split-lambda:latest \
  --timeout 600 \
  --memory-size 2048
```

## IAM 权限

Lambda 执行角色需要以下权限：
- `s3:GetObject` - 读取输入视频
- `s3:PutObject` - 写入输出片段
- `s3:ListBucket` - 列出 bucket 内容（可选）

## 环境变量

- `OUTPUT_BUCKET`: 默认输出 bucket（可选）
