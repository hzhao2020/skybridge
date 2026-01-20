# Storage 和 Transmission 分离设计

## 当前设计分析

### 当前问题

`DataTransmission` 类混合了两种职责：
1. **存储操作**：上传本地文件到云存储
2. **传输操作**：跨云数据传输（S3 ↔ GCS）

### 职责划分

| 操作类型 | 职责 | 示例 |
|---------|------|------|
| **Storage** | 文件的存储和检索 | 上传、下载、删除、列出文件 |
| **Transmission** | 跨云/跨区域数据传输 | S3→GCS, GCS→S3, 跨区域复制 |

## 推荐方案：分离为两个 Operation

### 方案1：完全分离（推荐）

创建两个独立的 Operation 类：

```python
class DataStorage(Operation):
    """数据存储操作"""
    def execute(self, operation: str, **kwargs):
        # operation: 'upload', 'download', 'delete', 'list'
        pass

class DataTransmission(Operation):
    """数据传输操作"""
    def execute(self, source_uri: str, target_uri: str, **kwargs):
        # 跨云/跨区域传输
        pass
```

**优点：**
- ✅ 职责清晰
- ✅ 可以作为独立的 operation 使用
- ✅ 便于独立测试和扩展

**缺点：**
- ❌ 如果经常一起使用，需要组合调用

### 方案2：保留辅助类 + 新增 Operation（折中方案）

保留 `DataTransmission` 作为辅助类（用于其他 Operation 内部使用），同时创建独立的 Operation：

```python
# 辅助类（内部使用）
class DataTransmissionHelper:
    """数据传输辅助类，供其他 Operation 内部使用"""
    def smart_move(...): ...
    def upload_local_to_cloud(...): ...

# Operation（对外使用）
class DataStorage(Operation):
    """数据存储 Operation"""
    def execute(...): ...

class DataTransmission(Operation):
    """数据传输 Operation"""
    def execute(...): ...
```

**优点：**
- ✅ 保持向后兼容
- ✅ 既支持独立使用，也支持内部辅助

## 实现建议

### 推荐：方案2（折中方案）

原因：
1. **向后兼容**：现有的 Operation 可以继续使用 `self.transmitter`
2. **灵活性**：既支持独立使用，也支持内部辅助
3. **渐进式迁移**：可以逐步迁移现有代码

### 实现结构

```
core/
  ├── storage.py          # DataStorageHelper (辅助类)
  └── transmission.py     # DataTransmissionHelper (辅助类)

ops/
  ├── base.py
  └── impl/
      ├── storage_ops.py      # DataStorage Operation
      └── transmission_ops.py # DataTransmission Operation
```

## 使用场景对比

### 场景1：其他 Operation 内部使用（当前方式）

```python
class VideoSegmenter(Operation):
    def execute(self, video_uri):
        # 使用辅助类
        target_uri = self.transmitter.smart_move(...)
```

### 场景2：作为独立的 Operation 使用（新方式）

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
    target_bucket="gcs-bucket",
    target_path="videos/"
)
```

## 迁移策略

1. **阶段1**：创建新的 Operation 类，保留现有辅助类
2. **阶段2**：新代码使用新的 Operation
3. **阶段3**：逐步迁移现有代码（可选）
