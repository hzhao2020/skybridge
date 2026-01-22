# Workflow 框架使用指南

## 概述

Workflow框架提供了一种抽象的方法来定义和实例化logical video workflow。通过这个框架，你可以：

1. **定义workflow结构**：描述workflow中的步骤和步骤之间的依赖关系
2. **配置operation**：为每个步骤指定使用的具体operation（通过pid）
3. **执行workflow**：自动处理步骤之间的数据传递和执行顺序

## 核心概念

### Workflow
- 定义逻辑步骤和步骤之间的数据流
- 管理执行上下文（context）
- 处理步骤的配置和验证

### WorkflowStep
- 定义单个步骤的输入、输出和执行逻辑
- 指定依赖关系
- 通过`operation_pid`指定使用的具体operation

### 配置（Config）
- 为每个步骤指定`operation_pid`
- 控制步骤的启用/禁用
- 传递步骤特定的额外参数

## 使用示例

### 1. 使用预定义的VideoQAWorkflow

```python
from core.video_qa_workflow import VideoQAWorkflow

# 创建workflow实例
workflow = VideoQAWorkflow()

# 配置workflow（指定每个步骤使用的operation）
config = {
    "segment": {
        "operation_pid": "seg_aws_sg",  # 视频分割
        "enabled": True,
    },
    "split": {
        "operation_pid": "split_aws_sg",  # 视频切割
        "enabled": True,
    },
    "caption": {
        "operation_pid": "cap_aws_nova_lite_sg",  # 视频描述
        "enabled": True,
    },
    "llm_query": {
        "operation_pid": "llm_openai_gpt4o_mini",  # LLM查询
        "enabled": True,
    },
}
workflow.configure(config)

# 准备输入数据
input_data = {
    "video_path": "path/to/video.mp4",
    "question": "视频中发生了什么？",
    "options": ["选项A", "选项B", "选项C"],
    "answer": "选项A",
    "answer_idx": 0,
    "upload_target_path": "videos/egoschema/",
    "max_segments": 12,
}

# 执行workflow
result = workflow.execute(input_data)

# 获取结果
print(f"预测结果: {result['context']['pred_letter']}")
print(f"是否正确: {result['context']['correct']}")
```

### 2. 创建自定义Workflow

```python
from core.workflow import Workflow, WorkflowStep
from ops.registry import get_operation

class MyCustomWorkflow(Workflow):
    def __init__(self):
        super().__init__(
            name="MyCustomWorkflow",
            description="我的自定义workflow"
        )
        self._setup_steps()
    
    def _setup_steps(self):
        # 添加步骤1
        self.add_step(
            name="step1",
            description="第一步",
            dependencies=[],  # 无依赖
            execute_func=self._execute_step1,
            input_keys=["input_data"],
            output_keys=["intermediate_result"],
        )
        
        # 添加步骤2（依赖步骤1）
        self.add_step(
            name="step2",
            description="第二步",
            dependencies=["step1"],  # 依赖步骤1
            execute_func=self._execute_step2,
            input_keys=["intermediate_result"],
            output_keys=["final_result"],
        )
    
    def get_default_config(self):
        return {
            "step1": {
                "operation_pid": "some_operation_pid",
                "enabled": True,
            },
            "step2": {
                "operation_pid": "another_operation_pid",
                "enabled": True,
            },
        }
    
    def _execute_step1(self, workflow, step):
        """执行步骤1"""
        input_data = workflow.context.get("input_data")
        
        # 获取operation
        op = get_operation(step.operation_pid)
        
        # 执行操作
        result = op.execute(input_data)
        
        return {"intermediate_result": result}
    
    def _execute_step2(self, workflow, step):
        """执行步骤2"""
        intermediate = workflow.context.get("intermediate_result")
        
        # 获取operation
        op = get_operation(step.operation_pid)
        
        # 执行操作
        result = op.execute(intermediate)
        
        return {"final_result": result}
```

### 3. 动态配置不同的Operation组合

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

# 实验3：混合使用
config_mixed = {
    "segment": {"operation_pid": "seg_google_us", "enabled": True},
    "split": {"operation_pid": "split_aws_sg", "enabled": True},
    "caption": {"operation_pid": "cap_aws_nova_lite_sg", "enabled": True},
    "llm_query": {"operation_pid": "llm_openai_gpt4o", "enabled": True},
}

# 使用相同的workflow，不同的配置
workflow = VideoQAWorkflow()

for config_name, config in [("Google", config_google), ("AWS", config_aws), ("Mixed", config_mixed)]:
    print(f"\n运行实验: {config_name}")
    workflow.configure(config)
    result = workflow.execute(input_data)
    print(f"结果: {result['context'].get('pred_letter')}")
```

## VideoQAWorkflow 步骤说明

VideoQAWorkflow包含以下步骤：

1. **segment** (视频分割)
   - 检测视频中的场景切换
   - 输出：segments（时间段列表）
   - 依赖：无

2. **split** (视频切割)
   - 根据segments将视频物理切割成多个文件
   - 输出：segment_video_uris（片段视频URI列表）
   - 依赖：segment

3. **caption** (视频描述)
   - 生成视频的描述文本
   - 输出：captions（描述列表）
   - 依赖：无

4. **concentrate_captions** (合并描述)
   - 将所有captions合并成一个字符串
   - 输出：concentrated_captions
   - 依赖：caption

5. **llm_query** (LLM查询)
   - 将问题、选项和描述组合成prompt，提交给LLM
   - 输出：llm_response, prompt
   - 依赖：concentrate_captions

6. **parse_result** (解析结果)
   - 解析LLM输出，提取预测选项
   - 输出：pred_letter, pred_idx, correct
   - 依赖：llm_query

## 输入数据格式

VideoQAWorkflow的输入数据应包含以下字段：

```python
{
    "video_path": str,              # 必需：视频文件路径
    "question": str,                # 必需：问题文本
    "options": List[str],          # 必需：选项列表
    "answer": Optional[str],       # 可选：正确答案
    "answer_idx": Optional[int],   # 可选：正确答案索引
    "qid": Optional[str],          # 可选：问题ID
    "upload_target_path": Optional[str],  # 可选：上传路径，默认"videos/egoschema/"
    "max_segments": Optional[int],  # 可选：最大片段数，默认12
    "google_videosplit_service_url": Optional[str],  # 可选：Google Cloud Function VideoSplit 服务 URL
    "aws_videosplit_function_name": Optional[str],   # 可选：AWS Lambda函数名
}
```

## 输出结果格式

Workflow执行后返回的结果包含：

```python
{
    "workflow_name": str,           # Workflow名称
    "context": {                    # 执行上下文（包含所有步骤的输出）
        "segments": List[Dict],
        "segment_video_uris": List[str],
        "captions": List[Dict],
        "concentrated_captions": str,
        "llm_response": str,
        "prompt": str,
        "pred_letter": str,
        "pred_idx": int,
        "correct": bool,
        # ... 以及所有输入数据
    },
    "step_results": {               # 每个步骤的执行结果
        "step_name": {
            "status": str,          # "completed", "skipped", "failed"
            "result": Dict,
            "error": Optional[str],
        },
        ...
    }
}
```

## 查看可用Operation

```python
from ops.registry import REGISTRY

# 列出所有可用的operation pid
for pid in sorted(REGISTRY.keys()):
    print(pid)
```

## 注意事项

1. **依赖关系**：确保步骤的依赖关系正确设置，workflow会自动按拓扑顺序执行
2. **配置验证**：执行前会自动验证配置，确保所有operation_pid都存在
3. **错误处理**：如果某个步骤失败，workflow会停止执行并抛出异常
4. **上下文管理**：步骤之间的数据通过context传递，确保input_keys和output_keys正确设置
