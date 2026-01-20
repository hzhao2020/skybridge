# Serverless（按应用组织）

你计划创建多个 serverless 应用，因此 `ops/serverless/` 作为“应用集合目录”，每个应用在其子目录中自包含代码与部署文件（Dockerfile/requirements/README）。

## 当前已内置：VideoSplit

- **Google Cloud Run**：`ops/serverless/videosplit/google/`
- **AWS Lambda**：`ops/serverless/videosplit/aws/`

每个目录下都有自己的部署说明与依赖文件，请分别查看对应 README。

