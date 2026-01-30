import os
import re
import sys

from utils.dataset import (
    build_dataset,
    EgoSchemaDataset,
    NExTQADataset,
    ActivityNetQADataset,
)
from ops.registry import get_operation, REGISTRY
from core.workflows.lvqa import LVQA

# 加载配置文件（如果存在）
try:
    import config
    # 配置代理
    if "https_proxy" not in os.environ and hasattr(config, 'HTTPS_PROXY'):
        os.environ["https_proxy"] = config.HTTPS_PROXY
    # 配置OpenAI API Key
    if "OPENAI_API_KEY" not in os.environ and hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    # 配置OpenAI Base URL
    if "OPENAI_BASE_URL" not in os.environ and hasattr(config, 'OPENAI_BASE_URL') and config.OPENAI_BASE_URL:
        os.environ["OPENAI_BASE_URL"] = config.OPENAI_BASE_URL
    # 配置GCP Project Number
    if "GCP_PROJECT_NUMBER" not in os.environ and hasattr(config, 'GCP_PROJECT_NUMBER') and config.GCP_PROJECT_NUMBER:
        os.environ["GCP_PROJECT_NUMBER"] = config.GCP_PROJECT_NUMBER
    # 配置GCP VideoSplit Service URLs
    if "GCP_VIDEOSPLIT_SERVICE_URLS" not in os.environ and hasattr(config, 'GCP_VIDEOSPLIT_SERVICE_URLS') and config.GCP_VIDEOSPLIT_SERVICE_URLS:
        import json
        os.environ["GCP_VIDEOSPLIT_SERVICE_URLS"] = json.dumps(config.GCP_VIDEOSPLIT_SERVICE_URLS)
except ImportError:
    # 如果config.py不存在，使用默认值
    if "https_proxy" not in os.environ:
        os.environ["https_proxy"] = "http://127.0.0.1:7897"


# 尽量在 Windows 控制台下正确输出中文
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def list_available_ops():
    """打印当前可用的 (provider, region, model) 组合及其 pid。"""
    print("\n=== 可用的操作 pid 列表 ===")
    for pid in sorted(REGISTRY.keys()):
        print(f"- {pid}")
    print("========================\n")


def run_workflow_demo():
    """
    使用抽象的Workflow框架运行demo
    
    这个函数展示了如何使用 LVQA 来执行 workflow。
    你可以通过配置来指定每个步骤使用的operation（通过pid）。
    """
    # ========== 1) 创建workflow实例 ==========
    workflow = LVQA()
    
    # ========== 2) 配置workflow（指定每个步骤使用的operation） ==========
    config = {
        "segment": {
            # "operation_pid": "seg_aws_us",  # seg_google_us, seg_aws_us 等
            "operation_pid": "seg_google_us",  # seg_google_us, seg_aws_us 等
            "enabled": True,
        },
        "split": {
            "operation_pid": "split_google_sg",  # split_google_us, split_aws_us 等
            "enabled": True,
        },
        "caption": {
            "operation_pid": "cap_google_flash_sg",  # cap_google_flash_us 等
            "enabled": True,
        },
        "llm_query": {
            "operation_pid": "llm_google_flash_sg",  # llm_google_pro_us  等
            "enabled": True,
        },
    }
    workflow.configure(config)
    
    # 打印workflow信息
    print("\n" + "="*60)
    print("Workflow 配置信息")
    print("="*60)
    info = workflow.get_step_info()
    for step_name, step_info in info["steps"].items():
        if step_info["enabled"]:
            print(f"{step_name}: {step_info['operation_pid']} - {step_info['description']}")
        else:
            print(f"{step_name}: (disabled)")
    print("="*60 + "\n")
    
    # ========== 3) 准备输入数据 ==========
    # 从数据集获取样本
    sample = build_dataset("EgoSchema", "train")[0]
    qid = sample.get("qid") or "egoschema_train_0"
    question = sample.get("question") or ""
    options = sample.get("options") or []
    answer = sample.get("answer")
    answer_idx = sample.get("answer_idx")
    video_path = sample.get("video_path")
    
    print("=== EgoSchema 样本 ===")
    print(f"qid: {qid}")
    print(f"video_path: {video_path}")
    print(f"question: {question}")
    if options:
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            print(f"  {letter}. {opt}")
    if answer is not None or answer_idx is not None:
        print(f"gt(answer): {answer} | gt(answer_idx): {answer_idx}")
    print()
    
    # 准备workflow输入
    input_data = {
        "video_path": video_path,
        "question": question,
        "options": options,
        "answer": answer,
        "answer_idx": answer_idx,
        "qid": qid,
        "upload_target_path": "videos/egoschema/",
        "max_segments": 12,
        # 可选：Google Cloud Function VideoSplit 服务 URL
        # "google_videosplit_service_url": os.getenv("GCP_VIDEOSPLIT_SERVICE_URL"),
        # 可选：AWS Lambda function name
        # "aws_videosplit_function_name": os.getenv("AWS_VIDEOSPLIT_FUNCTION_NAME"),
    }
    
    # ========== 4) 执行workflow ==========
    result = workflow.execute(input_data)
    
    # ========== 5) 保存时间记录 ==========
    try:
        from utils.timing import TimingRecorder
        import os
        from datetime import datetime
        
        recorder = TimingRecorder()
        timing = recorder.get_timing()
        
        if timing:
            # 构建时间记录文件路径
            qid = input_data.get("qid", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timing_dir = "timing_logs"
            os.makedirs(timing_dir, exist_ok=True)
            timing_filepath = os.path.join(timing_dir, f"timing_{qid}_{timestamp}.json")
            
            # 保存时间记录
            recorder.save_to_file(timing_filepath)
    except Exception as e:
        print(f"警告：保存时间记录失败: {e}")
    
    # ========== 6) 处理结果 ==========
    print("\n" + "="*60)
    print("Workflow 执行完成！")
    print("="*60 + "\n")
    
    # 返回结果（与run_demo()格式兼容）
    return {
        "qid": result["context"].get("qid"),
        "question": result["context"].get("question"),
        "segments": result["context"].get("segments"),
        "segment_video_uris": result["context"].get("segment_video_uris"),
        "captions": result["context"].get("captions"),
        "concentrated_captions": result["context"].get("concentrated_captions"),
        "llm_response": result["context"].get("llm_response"),
        "pred_letter": result["context"].get("pred_letter"),
        "pred_idx": result["context"].get("pred_idx"),
        "answer": result["context"].get("answer"),
        "answer_idx": result["context"].get("answer_idx"),
        "correct": result["context"].get("correct"),
    }


if __name__ == "__main__":
    # 打印可用的 (provider, region, model) 组合
    list_available_ops()

    result = run_workflow_demo()

    # 有条理地打印 result 的结果
    if result:
        print("\n===== Workflow 执行结果 =====")
        for key in [
            "qid", "question", "segments", "segment_video_uris",
            "captions", "concentrated_captions", "llm_response",
            "pred_letter", "pred_idx", "answer", "answer_idx", "correct"
        ]:
            val = result.get(key)
            print(f"{key}: {val}")
        print("============================\n")
    else:
        print("无 result 返回或为空。")