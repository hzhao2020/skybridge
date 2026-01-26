# core/workflow.py
"""
Workflow抽象框架：用于定义和实例化logical video workflow

核心概念：
- Workflow: 定义逻辑步骤和步骤之间的数据流
- WorkflowStep: 定义单个步骤的输入、输出和执行逻辑
- WorkflowConfig: 配置每个步骤使用的具体operation（通过pid）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import os


class StepStatus(Enum):
    """步骤执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """
    工作流步骤定义
    
    Attributes:
        name: 步骤名称（唯一标识）
        description: 步骤描述
        operation_pid: 使用的operation的pid（可选，可以在运行时通过config指定）
        enabled: 是否启用此步骤（默认True）
        dependencies: 依赖的其他步骤名称列表
        execute_func: 执行函数，接收context和kwargs，返回结果
        input_keys: 从context中读取的输入键列表
        output_keys: 写入context的输出键列表
    """
    name: str
    description: str = ""
    operation_pid: Optional[str] = None
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    execute_func: Optional[Callable] = None
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class Workflow(ABC):
    """
    抽象Workflow基类
    
    子类需要：
    1. 在__init__中定义所有步骤（通过add_step）
    2. 实现get_default_config()返回默认配置
    3. 可选：实现validate_config()进行配置验证
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.context: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        
    def add_step(
        self,
        name: str,
        description: str = "",
        operation_pid: Optional[str] = None,
        enabled: bool = True,
        dependencies: Optional[List[str]] = None,
        execute_func: Optional[Callable] = None,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
    ) -> WorkflowStep:
        """
        添加一个步骤到workflow
        
        Args:
            name: 步骤名称
            description: 步骤描述
            operation_pid: 默认的operation pid（可被config覆盖）
            enabled: 是否默认启用
            dependencies: 依赖的步骤名称列表
            execute_func: 执行函数
            input_keys: 输入键列表
            output_keys: 输出键列表
            
        Returns:
            创建的WorkflowStep对象
        """
        step = WorkflowStep(
            name=name,
            description=description,
            operation_pid=operation_pid,
            enabled=enabled,
            dependencies=dependencies or [],
            execute_func=execute_func,
            input_keys=input_keys or [],
            output_keys=output_keys or [],
        )
        self.steps[name] = step
        return step
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        返回默认配置
        
        子类应该重写此方法，返回包含所有步骤的operation_pid的字典
        格式：{"step_name": {"operation_pid": "xxx", "enabled": True, ...}}
        """
        return {}
    
    def configure(self, config: Dict[str, Any]):
        """
        配置workflow（设置每个步骤的operation_pid等）
        
        Args:
            config: 配置字典，格式：
                {
                    "step_name": {
                        "operation_pid": "xxx",
                        "enabled": True,
                        "extra_params": {...}  # 步骤特定的额外参数
                    },
                    ...
                }
        """
        self.config = config
        
        # 应用配置到步骤
        for step_name, step_config in config.items():
            if step_name in self.steps:
                step = self.steps[step_name]
                if "operation_pid" in step_config:
                    step.operation_pid = step_config["operation_pid"]
                if "enabled" in step_config:
                    step.enabled = step_config["enabled"]
                # 存储额外参数到context
                if "extra_params" in step_config:
                    self.context[f"{step_name}_params"] = step_config["extra_params"]
    
    def validate_config(self) -> bool:
        """
        验证配置是否有效
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        from ops.registry import REGISTRY
        
        for step_name, step in self.steps.items():
            if not step.enabled:
                continue
                
            # 如果步骤有 execute_func 但没有 operation_pid，允许跳过 operation_pid 验证
            # （这种情况适用于纯数据处理步骤，不依赖外部 operation）
            if not step.operation_pid:
                if step.execute_func:
                    # 有 execute_func 但没有 operation_pid，允许（可能是纯数据处理步骤）
                    continue
                else:
                    # 既没有 operation_pid 也没有 execute_func，报错
                    raise ValueError(
                        f"步骤 '{step_name}' 未设置 operation_pid，且没有定义 execute_func"
                    )
            
            # 检查operation是否存在
            if step.operation_pid not in REGISTRY:
                raise ValueError(
                    f"步骤 '{step_name}' 的 operation_pid '{step.operation_pid}' 不存在。"
                    f"可用pid列表：{sorted(REGISTRY.keys())}"
                )
            
            # 检查依赖是否满足
            for dep in step.dependencies:
                if dep not in self.steps:
                    raise ValueError(f"步骤 '{step_name}' 依赖的步骤 '{dep}' 不存在")
        
        return True
    
    def get_execution_order(self) -> List[str]:
        """
        根据依赖关系计算执行顺序（拓扑排序）
        
        Returns:
            步骤名称的有序列表
        """
        # 简单的拓扑排序
        in_degree = {name: len(step.dependencies) for name, step in self.steps.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # 更新依赖此步骤的其他步骤的入度
            for name, step in self.steps.items():
                if current in step.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        # 检查是否有循环依赖
        if len(order) != len(self.steps):
            raise ValueError("Workflow中存在循环依赖")
        
        return order
    
    def execute_step(self, step_name: str) -> Dict[str, Any]:
        """
        执行单个步骤
        
        Args:
            step_name: 步骤名称
            
        Returns:
            步骤执行结果
        """
        if step_name not in self.steps:
            raise ValueError(f"步骤 '{step_name}' 不存在")
        
        step = self.steps[step_name]
        
        # 检查是否启用
        if not step.enabled:
            step.status = StepStatus.SKIPPED
            return {"status": "skipped", "message": "步骤已禁用"}
        
        # 检查依赖是否完成
        for dep_name in step.dependencies:
            dep_step = self.steps[dep_name]
            if dep_step.status != StepStatus.COMPLETED and dep_step.status != StepStatus.SKIPPED:
                raise ValueError(f"步骤 '{step_name}' 的依赖 '{dep_name}' 尚未完成")
        
        # 执行步骤
        step.status = StepStatus.RUNNING
        try:
            if step.execute_func:
                result = step.execute_func(self, step)
            else:
                result = {"status": "no_execute_func", "message": "步骤未定义执行函数"}
            
            step.status = StepStatus.COMPLETED
            step.result = result
            
            # 将结果写入context
            if step.output_keys and isinstance(result, dict):
                for key in step.output_keys:
                    if key in result:
                        self.context[key] = result[key]
            
            return result
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = e
            raise
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        执行整个workflow
        
        Args:
            input_data: 初始输入数据（会写入context）
            **kwargs: 额外的执行参数
            
        Returns:
            最终结果字典
        """
        # 初始化context
        self.context = input_data.copy()
        self.context.update(kwargs)
        
        # 验证配置
        self.validate_config()
        
        # 获取执行顺序
        execution_order = self.get_execution_order()
        
        # 按顺序执行步骤
        for step_name in execution_order:
            step = self.steps[step_name]
            if step.enabled:
                print(f"\n{'='*60}")
                print(f"执行步骤: {step_name} - {step.description}")
                print(f"{'='*60}")
                self.execute_step(step_name)
            else:
                step.status = StepStatus.SKIPPED
                print(f"\n{'='*60}")
                print(f"跳过步骤: {step_name} - {step.description}")
                print(f"{'='*60}")
        
        # 返回最终结果
        return {
            "workflow_name": self.name,
            "context": self.context,
            "step_results": {name: {
                "status": step.status.value,
                "result": step.result,
                "error": str(step.error) if step.error else None
            } for name, step in self.steps.items()}
        }
    
    def get_step_info(self) -> Dict[str, Any]:
        """
        获取workflow的步骤信息（用于调试和展示）
        
        Returns:
            包含所有步骤信息的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "steps": {
                name: {
                    "description": step.description,
                    "operation_pid": step.operation_pid,
                    "enabled": step.enabled,
                    "dependencies": step.dependencies,
                    "input_keys": step.input_keys,
                    "output_keys": step.output_keys,
                }
                for name, step in self.steps.items()
            }
        }
