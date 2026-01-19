from pprint import pprint

from utils.dataset import (
    build_dataset,
    EgoSchemaDataset,
    NExTQADataset,
    ActivityNetQADataset,
)
from ops.registry import get_operation, REGISTRY


def list_available_ops():
    """打印当前可用的 (provider, region, model) 组合及其 pid。"""
    print("\n=== 可用的操作 pid 列表 ===")
    for pid in sorted(REGISTRY.keys()):
        print(f"- {pid}")
    print("========================\n")


def run_demo():
    """
    演示如何选择 provider, region, model 来运行 workflow。
    
    pid 命名规则：
        - 视频分割: vid_{provider}_{region简称}
            例: vid_google_us, vid_aws_eu
        
        - 视觉描述: cap_{provider}_{model简称}_{region简称}
            例: cap_google_flash_us, cap_aws_nova_lite_sg
        
        - LLM查询: llm_{provider}_{model简称}_{region简称}
            例: llm_google_pro_us, llm_aws_sonnet_eu, llm_openai_gpt4o
    
    可用的 provider/region/model 组合:
        Google:  us (us-west1), eu (europe-west1), sg (asia-southeast1)
        Amazon:  us (us-west-2), eu (eu-central-1), sg (ap-southeast-1)
        OpenAI:  global (无区域)
    """
    
    # ========== 1. 选择你的配置 ==========
    # 修改以下变量来选择不同的 provider, region, model
    
    # 视频分割 (Video Segmentation)
    # 可选: vid_google_us, vid_google_eu, vid_google_tw, vid_aws_us, vid_aws_eu, vid_aws_sg
    segment_pid = "vid_google_us"
    
    # 视觉描述 (Visual Captioning)  
    # Google: cap_google_flash_lite_us, cap_google_flash_us, cap_google_flash_lite_eu, ...
    # Amazon: cap_aws_nova_lite_us, cap_aws_nova_pro_us, cap_aws_nova_lite_eu, ...
    caption_pid = "cap_google_flash_us"
    
    # LLM 查询
    # Google: llm_google_flash_us, llm_google_pro_us, llm_google_flash_eu, ...
    # Amazon: llm_aws_haiku_us, llm_aws_sonnet_us, llm_aws_haiku_eu, ...
    # OpenAI: llm_openai_gpt4o_mini, llm_openai_gpt4o
    llm_pid = "llm_openai_gpt4o_mini"
    
    # ========== 2. 获取操作实例 ==========
    segment_op = get_operation(segment_pid)
    caption_op = get_operation(caption_pid)
    llm_op = get_operation(llm_pid)
    
    print(f"=== Workflow 配置 ===")
    print(f"视频分割: {segment_pid} -> provider={segment_op.provider}, region={segment_op.region}")
    print(f"视觉描述: {caption_pid} -> provider={caption_op.provider}, region={caption_op.region}, model={caption_op.model_name}")
    print(f"LLM查询:  {llm_pid} -> provider={llm_op.provider}, model={llm_op.model_name}")
    print()
    
    # ========== 3. 运行 Workflow ==========
    # 示例视频路径 (本地或云端 URI)
    video_path = "datasets/EgoSchema/videos_sampled/sample.mp4"  # 修改为你的视频路径
    
    # 步骤 1: 视频分割
    # segment_result = segment_op.execute(video_path)
    # print("分割结果:", segment_result)
    
    # 步骤 2: 视觉描述
    # caption_result = caption_op.execute(video_path)
    # print("描述结果:", caption_result)
    
    # 步骤 3: LLM 查询
    prompt = "请用中文总结这段视频的主要内容。"
    # llm_result = llm_op.execute(prompt)
    # print("LLM 回答:", llm_result)
    
    print("--- Demo 配置完成，取消上面的注释即可运行实际调用 ---")


if __name__ == "__main__":
    # 示例：加载 EgoSchema 训练集的第一条样本
    egoschema_dataset_train = build_dataset("EgoSchema", "train")
    pprint(egoschema_dataset_train[0])

    # 打印可用的 (provider, region, model) 组合
    list_available_ops()

    # 运行一个示例调用（按需修改 pid 和输入）
    run_demo()