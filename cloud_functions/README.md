# Cloud Functions 部署说明

## 两种部署方式

### 方式1：每个函数单独文件夹（推荐）

**结构：**
```
cloud_functions/
  ├── google_video_split/
  │   ├── main.py
  │   ├── requirements.txt
  │   └── README.md
  └── other_function/
      ├── main.py
      └── requirements.txt
```

**优点：**
- ✅ 每个函数可以有独立的依赖
- ✅ 结构清晰，便于管理
- ✅ 部署时指定不同的 source 目录

**部署命令：**
```bash
cd cloud_functions/google_video_split
gcloud functions deploy video-split-us-west1 \
  --gen2 \
  --runtime python311 \
  --region us-west1 \
  --source . \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated
```

### 方式2：统一函数文件（可选）

**结构：**
```
cloud_functions/
  ├── unified_functions.py  # 所有函数在一个文件
  └── requirements.txt      # 统一的依赖
```

**优点：**
- ✅ 代码集中，减少重复
- ✅ 便于共享公共代码

**缺点：**
- ❌ 所有函数必须使用相同的依赖
- ❌ 如果函数需要不同的依赖，不适用

**部署命令：**
```bash
cd cloud_functions
gcloud functions deploy video-split-us-west1 \
  --gen2 \
  --runtime python311 \
  --region us-west1 \
  --source . \
  --entry-point video_split \
  --trigger-http \
  --allow-unauthenticated
```

## 推荐方案

**建议使用方式1（单独文件夹）**，因为：
1. 视频分割函数可能需要 ffmpeg，其他函数可能不需要
2. 不同函数可能有不同的依赖版本要求
3. 更符合微服务的最佳实践

## 注意事项

1. **ffmpeg 支持**：
   - Cloud Functions Gen2 需要使用容器镜像才能包含 ffmpeg
   - 或者使用 Cloud Run Jobs（更适合长时间运行的任务）

2. **部署位置**：
   - 可以在项目根目录下创建 `cloud_functions/` 文件夹
   - 也可以放在任何位置，部署时指定 `--source` 参数

3. **入口点**：
   - 函数文件中的函数名就是入口点名称
   - 部署时通过 `--entry-point` 指定
