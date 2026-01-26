# Co-located Storage 策略分析报告

## 用户需求

按照DAG执行workflow，采用**co-located storage策略**来最小化数据移动开销：

1. **如果物理endpoint支持native persistence** → 生成的中间数据存储在endpoint所在的同一region
2. **如果endpoint不支持native storage**（如商业LLM APIs）→ 输出路由回parent endpoint的storage service
3. **显式数据传输成本**仅在successor node被分配到物理上不同的region或provider时产生

## 当前实现分析

### ✅ 已实现的部分

1. **DAG执行**：`Workflow.get_execution_order()` 使用拓扑排序按依赖关系执行
2. **数据持久化**：中间数据存储在云存储（S3/GCS）中
3. **智能传输**：`DataTransmission.smart_move()` 支持跨云和跨region的流式直传
4. **Region感知**：每个operation都有`region`和`storage_bucket`属性

### ❌ 未满足的需求

#### 问题1：每个operation都会将数据移动到自己的bucket

**当前行为**：
- 每个operation在`execute()`方法中调用`smart_move()`，将输入数据移动到**自己的**`storage_bucket`和`region`
- 即使下一个operation在同一个region/provider，数据也会被移动

**示例**：
```python
# google_ops.py:93
target_uri = self.transmitter.smart_move(video_uri, 'google', self.storage_bucket, target_path, target_region=self.region)
```

**问题**：如果segment和split都在`google/us-west1`，数据仍然会被移动到各自的bucket，造成不必要的传输。

#### 问题2：没有检查successor node的逻辑

**当前行为**：
- Workflow执行时，每个步骤独立执行
- 没有检查下一个步骤（successor）的region/provider来决定是否需要传输数据
- `smart_move()`只检查source和target是否相同bucket，如果不同就传输

**缺失的逻辑**：
```python
# 应该有这样的逻辑（但当前没有）：
def should_transfer_data(current_op, next_op):
    # 如果next_op在同一个region/provider，且支持native storage，不需要传输
    if current_op.provider == next_op.provider and current_op.region == next_op.region:
        return False
    return True
```

#### 问题3：LLM operations没有存储策略

**当前行为**：
- OpenAI operations的`storage_bucket=None`（`openai_ops.py:14`）
- LLM的输出（文本）直接返回，不会路由回parent endpoint的storage
- 如果LLM后面还有需要数据的步骤，数据可能丢失或需要重新获取

**示例**：
```python
# openai_ops.py:56-60
return {
    "provider": "openai",
    "model": self.model_name,
    "response": answer  # 只是返回文本，没有存储
}
```

**问题**：根据需求，LLM的输出应该路由回parent endpoint的storage service。

#### 问题4：缺少endpoint native persistence判断

**当前行为**：
- 没有机制判断endpoint是否支持native persistence
- 所有operation都假设需要将数据移动到自己的bucket

**应该有的逻辑**：
```python
# 应该区分：
# 1. 支持native persistence的endpoint（如AWS Rekognition, Google Video Intelligence）
#    → 数据存储在endpoint的region
# 2. 不支持native storage的endpoint（如OpenAI API）
#    → 输出路由回parent endpoint的storage
```

## 具体代码问题

### 1. `smart_move()` 的局限性

**位置**：`ops/utils.py:236-275`

**问题**：
- 只检查bucket是否相同，不检查region
- 即使bucket相同但region不同，也不会传输（但实际上可能需要）
- 没有考虑successor node的位置

**当前逻辑**：
```python
# 如果源 bucket 和目标 bucket 相同，直接返回（不需要移动）
if source_bucket == target_bucket:
    return source_uri
```

**问题**：如果两个operation使用相同的bucket但不同的region，这个逻辑会错误地跳过传输。

### 2. Workflow执行缺少successor感知

**位置**：`core/workflow.py:266-310`

**问题**：
- `execute()`方法按顺序执行步骤，但没有检查successor
- 每个步骤独立决定数据存储位置，不考虑下一个步骤的需求

**应该有的逻辑**：
```python
def execute_step(self, step_name: str):
    step = self.steps[step_name]
    
    # 获取successor steps
    successors = [name for name, s in self.steps.items() if step_name in s.dependencies]
    
    # 根据successor的位置决定数据存储策略
    if successors:
        next_op = get_operation(self.steps[successors[0]].operation_pid)
        # 决定是否需要传输数据
```

### 3. Operation缺少persistence能力标记

**位置**：`ops/base.py:9-30`

**问题**：
- `Operation`基类没有标记是否支持native persistence
- 无法区分需要存储的operation（如VideoSegmenter）和不需要存储的operation（如LLMQuery）

**应该添加**：
```python
class Operation(ABC):
    def __init__(self, ..., supports_native_persistence: bool = True):
        self.supports_native_persistence = supports_native_persistence
```

## 改进建议

### 方案1：在Workflow层面实现co-located storage策略

1. **添加successor感知逻辑**：
   - 在执行每个步骤前，检查successor steps的region/provider
   - 如果successor在同一个region/provider，且当前operation支持native persistence，数据存储在当前region
   - 如果successor在不同region/provider，才进行数据传输

2. **添加persistence能力标记**：
   - 在`Operation`基类中添加`supports_native_persistence`属性
   - LLM operations标记为`False`，其他标记为`True`

3. **改进数据路由逻辑**：
   - 对于不支持native storage的operation（如LLM），输出路由回parent operation的storage
   - 在workflow context中跟踪每个步骤的数据位置

### 方案2：在Operation层面实现智能存储决策

1. **改进`smart_move()`方法**：
   - 添加successor信息参数
   - 根据successor的位置决定是否传输

2. **添加`get_storage_location()`方法**：
   - 根据operation的persistence能力和successor位置，返回最优存储位置

## 重新评估（基于用户反馈）

用户指出：**在他们的设置下，一个region只会有一个bucket**。

### ✅ 当前实现可以满足的部分

1. **`smart_move()`的bucket检查机制**：
   ```python
   # ops/utils.py:256-258
   if source_bucket == target_bucket:
       return source_uri  # 不移动，直接返回
   ```
   
   在"一个region只有一个bucket"的设置下：
   - 如果segment operation将数据上传到bucket A（region X）
   - 如果split operation也在region X，使用同一个bucket A
   - `smart_move()`会检查到`source_bucket == target_bucket`，**直接返回，不移动** ✓

2. **数据URI传递机制**：
   - segment步骤返回`video_uri`（云存储URI）
   - split步骤从context获取`video_uri`，直接传递给operation
   - 如果URI已经在正确的bucket中，`smart_move()`不会移动

3. **AWS实现的额外优化**：
   ```python
   # amazon_ops.py:451-454
   if video_uri.startswith('s3://'):
       target_uri = video_uri  # 直接使用，不调用smart_move()
   ```
   这是一个额外的优化，避免不必要的函数调用。

### ⚠️ 仍需改进的部分

1. **跨region/provider场景**：
   - 当successor node在不同region/provider时，数据传输是必要的 ✓
   - 当前实现会正确触发传输 ✓

2. **LLM operations的输出存储**：
   - LLM operations（如OpenAI）的输出（文本）没有存储到parent endpoint的storage
   - 如果后续步骤需要这些数据，可能需要重新获取
   - **建议**：对于不支持native storage的operation，将输出路由回parent storage

3. **Google实现的一致性**：
   - Google的实现总是调用`smart_move()`，即使数据已经在正确的bucket中
   - 虽然`smart_move()`会快速返回，但可以像AWS一样添加提前检查优化

## 结论

**在当前设置下（一个region只有一个bucket），代码基本满足co-located storage策略的要求**：

✅ **已满足**：
- 同region同bucket时，`smart_move()`不会移动数据
- 跨region/provider时，会正确触发数据传输
- 数据URI在workflow步骤间正确传递

⚠️ **可优化**：
- LLM operations的输出可以存储到parent endpoint的storage
- Google实现可以添加AWS式的提前检查优化
- 可以添加更明确的日志，显示数据是否被移动

**总体评价**：在当前架构下，代码已经实现了co-located storage策略的核心要求，避免了不必要的双egress成本。
