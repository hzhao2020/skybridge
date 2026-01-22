#!/usr/bin/env python3
"""
Workflow框架使用示例

这个脚本展示了如何使用抽象的Workflow框架来执行视频问答任务。
你可以通过修改配置来使用不同的provider/region/model组合。
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.video_qa_workflow import VideoQAWorkflow
from ops.registry import REGISTRY

# 尝试导入dataset（可选）
try:
    from utils.dataset import build_dataset
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False
    print("注意: 无法导入dataset模块，某些示例将跳过实际执行")


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 1. 创建workflow
    workflow = VideoQAWorkflow()
    
    # 2. 配置workflow（使用默认配置）
    default_config = workflow.get_default_config()
    workflow.configure(default_config)
    
    if not HAS_DATASET:
        print("(跳过实际执行，因为无法导入dataset模块)")
        return None
    
    # 3. 准备输入数据
    sample = build_dataset("EgoSchema", "train")[0]
    input_data = {
        "video_path": sample.get("video_path"),
        "question": sample.get("question", ""),
        "options": sample.get("options", []),
        "answer": sample.get("answer"),
        "answer_idx": sample.get("answer_idx"),
        "qid": sample.get("qid"),
        "upload_target_path": "videos/egoschema/",
        "max_segments": 12,
    }
    
    # 4. 执行workflow
    result = workflow.execute(input_data)
    
    # 5. 查看结果
    print(f"\n预测结果: {result['context'].get('pred_letter')}")
    print(f"是否正确: {result['context'].get('correct')}")
    
    return result


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "="*60)
    print("示例2: 自定义配置（使用Google服务）")
    print("="*60)
    
    workflow = VideoQAWorkflow()
    
    # 自定义配置：使用Google服务
    config = {
        "segment": {
            "operation_pid": "seg_google_us",
            "enabled": True,
        },
        "split": {
            "operation_pid": "split_google_us",
            "enabled": True,
        },
        "caption": {
            "operation_pid": "cap_google_flash_us",
            "enabled": True,
        },
        "llm_query": {
            "operation_pid": "llm_google_pro_us",
            "enabled": True,
        },
    }
    workflow.configure(config)
    
    # 显示配置信息
    print("\nWorkflow配置:")
    info = workflow.get_step_info()
    for step_name, step_info in info["steps"].items():
        if step_info["enabled"]:
            print(f"  {step_name}: {step_info['operation_pid']}")
    
    # 注意：这里不实际执行，因为需要配置相应的云服务凭证
    print("\n(注意：实际执行需要配置相应的云服务凭证)")


def example_disable_steps():
    """禁用某些步骤的示例"""
    print("\n" + "="*60)
    print("示例3: 禁用某些步骤（仅运行LLM）")
    print("="*60)
    
    workflow = VideoQAWorkflow()
    
    # 禁用视频相关步骤，仅运行LLM
    config = {
        "segment": {
            "operation_pid": "seg_aws_sg",
            "enabled": False,  # 禁用
        },
        "split": {
            "operation_pid": "split_aws_sg",
            "enabled": False,  # 禁用
        },
        "caption": {
            "operation_pid": "cap_aws_nova_lite_sg",
            "enabled": False,  # 禁用
        },
        "llm_query": {
            "operation_pid": "llm_openai_gpt4o_mini",
            "enabled": True,
        },
    }
    workflow.configure(config)
    
    print("\nWorkflow配置:")
    info = workflow.get_step_info()
    for step_name, step_info in info["steps"].items():
        status = "启用" if step_info["enabled"] else "禁用"
        print(f"  {step_name}: {status}")


def example_list_available_operations():
    """列出所有可用的operation"""
    print("\n" + "="*60)
    print("示例4: 列出所有可用的Operation")
    print("="*60)
    
    print("\n可用的operation pid:")
    for pid in sorted(REGISTRY.keys()):
        op = REGISTRY[pid]
        print(f"  {pid:30s} -> provider={op.provider:8s} region={op.region:20s} model={op.model_name or 'N/A'}")


def example_multiple_experiments():
    """运行多个实验（不同配置）的示例"""
    print("\n" + "="*60)
    print("示例5: 运行多个实验（不同配置）")
    print("="*60)
    
    # 定义多个实验配置
    experiments = {
        "实验1 - AWS": {
            "segment": {"operation_pid": "seg_aws_sg", "enabled": True},
            "split": {"operation_pid": "split_aws_sg", "enabled": True},
            "caption": {"operation_pid": "cap_aws_nova_lite_sg", "enabled": True},
            "llm_query": {"operation_pid": "llm_aws_haiku_sg", "enabled": True},
        },
        "实验2 - Google": {
            "segment": {"operation_pid": "seg_google_us", "enabled": True},
            "split": {"operation_pid": "split_google_us", "enabled": True},
            "caption": {"operation_pid": "cap_google_flash_us", "enabled": True},
            "llm_query": {"operation_pid": "llm_google_flash_us", "enabled": True},
        },
        "实验3 - 混合": {
            "segment": {"operation_pid": "seg_google_us", "enabled": True},
            "split": {"operation_pid": "split_aws_sg", "enabled": True},
            "caption": {"operation_pid": "cap_aws_nova_lite_sg", "enabled": True},
            "llm_query": {"operation_pid": "llm_openai_gpt4o_mini", "enabled": True},
        },
    }
    
    if not HAS_DATASET:
        print("(跳过实际执行，因为无法导入dataset模块)")
        return {}
    
    # 准备输入数据
    sample = build_dataset("EgoSchema", "train")[0]
    input_data = {
        "video_path": sample.get("video_path"),
        "question": sample.get("question", ""),
        "options": sample.get("options", []),
        "answer": sample.get("answer"),
        "answer_idx": sample.get("answer_idx"),
        "qid": sample.get("qid"),
        "upload_target_path": "videos/egoschema/",
        "max_segments": 12,
    }
    
    # 运行每个实验
    results = {}
    for exp_name, config in experiments.items():
        print(f"\n运行: {exp_name}")
        workflow = VideoQAWorkflow()
        workflow.configure(config)
        
        # 注意：这里不实际执行，因为需要配置相应的云服务凭证
        # result = workflow.execute(input_data)
        # results[exp_name] = result
        
        print(f"  配置已设置（实际执行需要云服务凭证）")
    
    print("\n(注意：实际执行需要配置相应的云服务凭证)")
    return results


if __name__ == "__main__":
    print("Workflow框架使用示例")
    print("="*60)
    
    # 列出可用的operation
    example_list_available_operations()
    
    # 基本使用示例
    # example_basic_usage()  # 取消注释以实际运行
    
    # 自定义配置示例
    example_custom_config()
    
    # 禁用步骤示例
    example_disable_steps()
    
    # 多个实验示例
    example_multiple_experiments()
    
    print("\n" + "="*60)
    print("更多信息请参考: core/WORKFLOW_USAGE.md")
    print("="*60)
