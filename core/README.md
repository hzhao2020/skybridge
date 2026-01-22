# Workflow 抽象框架

## 概述

这个框架提供了一种抽象的方法来定义和实例化logical video workflow。通过这个框架，你可以：

1. **定义workflow结构**：描述workflow中的步骤和步骤之间的依赖关系
2. **配置operation**：为每个步骤指定使用的具体operation（通过pid）
3. **执行workflow**：自动处理步骤之间的数据传递和执行顺序

## 文件结构

- `workflow.py`: 核心抽象类（Workflow, WorkflowStep）
- `video_qa_workflow.py`: VideoQAWorkflow的具体实现（对应main.py中的run_demo()）
- `WORKFLOW_USAGE.md`: 详细的使用文档和示例
- `__init__.py`: 模块导出

## 快速开始

### 基本使用

```python
from core.video_qa_workflow import VideoQAWorkflow

# 创建workflow
workflow = VideoQAWorkflow()

# 配置workflow（指定每个步骤使用的operation）
config = {
    "segment": {"operation_pid": "seg_aws_sg", "enabled": True},
    "split": {"operation_pid": "split_aws_sg", "enabled": True},
    "caption": {"operation_pid": "cap_aws_nova_lite_sg", "enabled": True},
    "llm_query": {"operation_pid": "llm_openai_gpt4o_mini", "enabled": True},
}
workflow.configure(config)

# 准备输入数据
input_data = {
    "video_path": "path/to/video.mp4",
    "question": "视频中发生了什么？",
    "options": ["选项A", "选项B", "选项C"],
    "answer": "选项A",
    "answer_idx": 0,
}

# 执行workflow
result = workflow.execute(input_data)
```

### 运行不同配置的实验

```python
# 实验1：使用Google服务
config_google = {
    "segment": {"operation_pid": "seg_google_us", "enabled": True},
    "split": {"operation_pid": "split_google_us", "enabled": True},
    "caption": {"operation_pid": "cap_google_flash_us", "enabled": True},
    "llm_query": {"operation_pid": "llm_google_pro_us", "enabled": True},
}

# 实验2：使用AWS服务
config_aws = {
    "segment": {"operation_pid": "seg_aws_eu", "enabled": True},
    "split": {"operation_pid": "split_aws_eu", "enabled": True},
    "caption": {"operation_pid": "cap_aws_nova_pro_eu", "enabled": True},
    "llm_query": {"operation_pid": "llm_aws_sonnet_eu", "enabled": True},
}

# 使用相同的workflow，不同的配置
workflow = VideoQAWorkflow()
for config_name, config in [("Google", config_google), ("AWS", config_aws)]:
    workflow.configure(config)
    result = workflow.execute(input_data)
    print(f"{config_name} 结果: {result['context'].get('pred_letter')}")
```

## 与原始run_demo()的对比

### 原始方式（main.py中的run_demo()）
- 硬编码的步骤执行逻辑
- 直接在函数中指定operation pid
- 难以复用和扩展
- 难以进行批量实验

### 新方式（使用Workflow框架）
- 抽象的workflow定义
- 通过配置指定operation
- 易于复用和扩展
- 方便进行批量实验
- 自动处理依赖关系
- 统一的错误处理和结果管理

## VideoQAWorkflow 步骤

1. **segment**: 视频分割（检测场景切换）
2. **split**: 视频物理切割
3. **caption**: 视频描述生成
4. **concentrate_captions**: 合并所有描述
5. **llm_query**: LLM查询
6. **parse_result**: 解析结果

## 更多信息

- 详细使用文档：`WORKFLOW_USAGE.md`
- 示例代码：`../examples/workflow_example.py`
- 原始实现：`../main.py` 中的 `run_demo()`
