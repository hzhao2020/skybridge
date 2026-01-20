# Storage 和 Transmission 分离设计总结

## 问题分析

### 当前设计的问题

`DataTransmission` 类混合了两种职责：
1. **存储操作**：`upload_local_to_cloud` - 上传文件到云存储
2. **传输操作**：`transfer_s3_to_gcs`, `transfer_gcs_to_s3` - 跨云数据传输

### 职责划分

| 操作类型 | 职责 | 示例 |
|---------|------|------|
| **Storage** | 文件的存储和检索 | 上传、下载、删除、列出文件 |
| **Transmission** | 跨云/跨区域数据传输 | S3→GCS, GCS→S3, 跨区域复制 |

## 推荐方案：折中方案

### 架构设计

```
core/
  ├── storage.py          # DataStorageHelper (辅助类)
  └── transmission.py     # DataTransmissionHelper (辅助类，重命名现有类)

ops/
  ├── base.py
  └── impl/
      ├── storage_ops.py      # Storage Operation (新增)
      └── transmission_ops.py # Transmission Operation (新增)
```

### 设计原则

1. **辅助类**：供其他 Operation 内部使用（保持向后兼容）
2. **Operation**：作为独立的操作使用（新增功能）

## 优点

### ✅ 职责清晰
- Storage 专注于存储操作
- Transmission 专注于传输操作

### ✅ 可以作为独立 Operation 使用
```python
# 存储操作
storage_op = get_operation("storage_google_us")
result = storage_op.execute(
    operation="upload",
    local_path="/path/to/file.mp4",
    target_path="videos/"
)

# 传输操作
transmission_op = get_operation("transmission_s3_to_gcs")
result = transmission_op.execute(
    source_uri="s3://bucket/file.mp4",
    target_bucket="gcs-bucket"
)
```

### ✅ 保持向后兼容
- 现有的 Operation 可以继续使用 `self.transmitter`
- 不需要修改现有代码

### ✅ 灵活组合
- 可以独立使用
- 也可以在其他 Operation 内部使用辅助类

## 缺点

### ❌ 代码复杂度增加
- 需要维护两套接口（辅助类 + Operation）
- 代码量增加

### ❌ 可能过度设计
- 如果存储和传输总是一起使用，分离可能不必要

## 使用场景

### 场景1：其他 Operation 内部使用（当前方式）

```python
class VideoSegmenter(Operation):
    def execute(self, video_uri):
        # 使用辅助类（保持现有方式）
        target_uri = self.transmitter.smart_move(...)
```

### 场景2：作为独立的 Operation 使用（新方式）

```python
# 存储操作
storage_op = get_operation("storage_google_us")
result = storage_op.execute(operation="upload", ...)

# 传输操作
transmission_op = get_operation("transmission_s3_to_gcs")
result = transmission_op.execute(source_uri="s3://...", ...)
```

## 实施建议

### 阶段1：创建新结构（不破坏现有代码）
1. ✅ 创建 `core/storage.py`（辅助类）
2. ✅ 创建 `ops/impl/storage_ops.py`（Operation）
3. ✅ 创建 `ops/impl/transmission_ops.py`（Operation）
4. ✅ 保留现有的 `core/transmission.py`（重命名为 Helper）

### 阶段2：注册新的 Operation
在 `ops/registry.py` 中注册：
```python
STORAGE_CATALOG = [
    {"pid": "storage_google_us", "cls": GoogleStorageImpl, ...},
    {"pid": "storage_aws_us", "cls": AmazonStorageImpl, ...},
]

TRANSMISSION_CATALOG = [
    {"pid": "transmission_s3_to_gcs", "cls": S3ToGCSImpl, ...},
    {"pid": "transmission_gcs_to_s3", "cls": GCSToS3Impl, ...},
]
```

### 阶段3：逐步迁移（可选）
- 新代码使用新的 Operation
- 现有代码保持不变（使用辅助类）

## 结论

**推荐采用折中方案**：
- ✅ 保持向后兼容
- ✅ 提供新的独立 Operation 功能
- ✅ 职责清晰
- ✅ 灵活使用

**不建议完全分离**（如果完全移除辅助类）：
- ❌ 会破坏现有代码
- ❌ 迁移成本高
