import os
import re
import sys
from pprint import pprint

from utils.dataset import (
    build_dataset,
    EgoSchemaDataset,
    NExTQADataset,
    ActivityNetQADataset,
)
from ops.registry import get_operation, REGISTRY

os.environ["https_proxy"] = "http://127.0.0.1:7897"
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


def run_demo():
    """
    一个“可直接运行”的 EgoSchema demo：拿 train[0] 的一个样本跑一次 QA workflow。

    你需要在哪里选择 provider / region / model？
    - 就在下面的 `segment_pid` / `caption_pid` / `llm_pid` 这三个变量里选择。
    - 也可以先运行 `list_available_ops()`，从输出的 pid 列表里挑你想要的组合。
    
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
    
    # ========== 1) 选择你的 provider / region / model（改这里即可） ==========
    # 默认仅跑 LLM（最容易跑通）；如果你已配置好 GCP/AWS 凭证，可打开视频相关步骤。
    
    # 视频分割 (Video Segmentation)
    # 可选: vid_google_us, vid_google_eu, vid_google_tw, vid_aws_us, vid_aws_eu, vid_aws_sg
    segment_pid = "vid_google_tw"
    
    # 视觉描述 (Visual Captioning)  
    # Google: cap_google_flash_lite_us, cap_google_flash_us, cap_google_flash_lite_eu, ...
    # Amazon: cap_aws_nova_lite_us, cap_aws_nova_pro_us, cap_aws_nova_lite_eu, ...
    caption_pid = "cap_google_flash_sg"
    
    # LLM 查询
    # Google: llm_google_flash_us, llm_google_pro_us, llm_google_flash_eu, ...
    # Amazon: llm_aws_haiku_us, llm_aws_sonnet_us, llm_aws_haiku_eu, ...
    # OpenAI: llm_openai_gpt4o_mini, llm_openai_gpt4o
    llm_pid = "llm_openai_gpt4o_mini"

    # 是否执行视频相关云服务（需要你本机已配置好对应云的凭证）
    enable_video_segment = True
    enable_video_caption = True

    # ========== 2) 取 EgoSchema 第 1 条样本（train[0]） ==========
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

    # 上传路径说明：我们会把视频上传到 bucket 下的这个目录（避免散落在根目录）
    # 例如 GCP: gs://video_us/inputs/egoschema/<qid>/<filename>
    # 例如 AWS: s3://sky-video-us/inputs/egoschema/<qid>/<filename>
    upload_target_path = f"videos/egoschema/"

    # ========== 3) 获取操作实例（按 pid 选择 provider/region/model） ==========
    segment_op = get_operation(segment_pid) if enable_video_segment else None
    caption_op = get_operation(caption_pid) if enable_video_caption else None
    llm_op = get_operation(llm_pid)
    
    print("=== Workflow 配置 ===")
    if segment_op is not None:
        print(f"视频分割: {segment_pid} -> provider={segment_op.provider}, region={segment_op.region}")
    else:
        print("视频分割: (disabled)")

    if caption_op is not None:
        print(f"视觉描述: {caption_pid} -> provider={caption_op.provider}, region={caption_op.region}, model={caption_op.model_name}")
    else:
        print("视觉描述: (disabled)")

    print(f"LLM查询:  {llm_pid} -> provider={llm_op.provider}, model={llm_op.model_name}")
    print(f"视频上传目录: {upload_target_path}")
    print()
    
    # ========== 4) 执行（可选）视频分割 / 视觉描述 ==========
    segments = None
    caption = None
    segment_captions = []  # 存储每个片段的 caption

    if (enable_video_segment or enable_video_caption) and (not video_path or not os.path.exists(video_path)):
        raise FileNotFoundError(
            f"找不到视频文件：{video_path}\n"
            f"请确认 datasets/EgoSchema/videos_sampled 下存在对应视频，或先关闭 enable_video_segment/enable_video_caption"
        )

    # 先执行视频分割
    if enable_video_segment and segment_op is not None:
        seg_res = segment_op.execute(video_path, target_path=upload_target_path)
        segments = seg_res.get("segments")
        print(f"\n=== 视频分割完成，共 {len(segments) if segments else 0} 个片段 ===\n")

    # 如果启用了 caption 且有 segments，对每个片段进行 caption
    if enable_video_caption and caption_op is not None:
        if segments and len(segments) > 0:
            # 对每个 segment 进行 caption
            print(f"=== 开始对 {len(segments)} 个片段进行描述 ===\n")
            for idx, seg in enumerate(segments):
                start_time = seg.get('start', 0)
                end_time = seg.get('end', 0)
                print(f"  [{idx+1}/{len(segments)}] 处理片段: {start_time:.2f}s - {end_time:.2f}s")
                
                try:
                    cap_res = caption_op.execute(
                        video_path, 
                        target_path=upload_target_path,
                        start_time=start_time,
                        end_time=end_time
                    )
                    seg_caption = cap_res.get("caption", "")
                    segment_captions.append({
                        "segment_idx": idx + 1,
                        "start": start_time,
                        "end": end_time,
                        "caption": seg_caption
                    })
                    print(f"    描述: {seg_caption[:80]}...\n")
                except Exception as e:
                    print(f"    警告: 片段 {idx+1} 描述失败: {e}\n")
                    segment_captions.append({
                        "segment_idx": idx + 1,
                        "start": start_time,
                        "end": end_time,
                        "caption": f"[描述失败: {str(e)}]"
                    })
        else:
            # 如果没有 segments，对整个视频进行 caption
            print("=== 对整个视频进行描述 ===\n")
            cap_res = caption_op.execute(video_path, target_path=upload_target_path)
            caption = cap_res.get("caption")

    # ========== 5) 组装 prompt，跑 LLM（默认可跑） ==========
    # 说明：LLM 本身只接收文本。这里我们把“问题 + 选项 + (可选)caption/segments摘要”拼成一个 prompt。
    lines = []
    lines.append("你是一个视频问答助手。请根据给定信息回答选择题。")
    lines.append("输出格式要求：只输出一个大写字母选项（例如：A），不要输出额外解释。")
    lines.append("")
    lines.append(f"问题：{question}")
    lines.append("选项：")
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {opt}")

    # 如果有片段级别的 caption，优先使用（已包含时间信息）
    if segment_captions:
        lines.append("")
        lines.append("辅助信息（视频片段描述，按时间顺序）：")
        # 限制显示前10个片段，避免 prompt 过长
        for seg_cap in segment_captions[:10]:
            lines.append(f"片段 {seg_cap['segment_idx']} ({seg_cap['start']:.1f}s - {seg_cap['end']:.1f}s): {seg_cap['caption']}")
        if len(segment_captions) > 10:
            lines.append(f"... (还有 {len(segment_captions) - 10} 个片段未显示)")
    elif caption:
        # 如果没有片段级别的 caption，使用整个视频的 caption
        lines.append("")
        lines.append("辅助信息（视频描述/caption）：")
        lines.append(caption)
        
        # 如果有 segments 但没有片段级别的 caption，显示时间戳
        if segments:
            lines.append("")
            lines.append("辅助信息（分割片段时间戳，单位秒，最多显示前10段）：")
            for s in segments[:10]:
                lines.append(f"- start={s.get('start')} end={s.get('end')}")

    prompt = "\n".join(lines)

    # 如果没有 OPENAI_API_KEY（且你又选了 openai 的 pid），就不实际请求 API，避免直接报错
    if llm_op.provider == "openai":
        # 1) SDK 未安装：直接提示安装，并输出 prompt 供你手动测试
        try:
            from openai import OpenAI as _OpenAI  # noqa: F401
        except Exception:
            print("检测到你选择了 OpenAI，但 Python 包 `openai` 未安装。")
            print("请先安装：pip install openai\n")
            print("=== Prompt ===")
            print(prompt)
            return

        # 2) Key 未设置：不请求 API，避免报错
        if not os.getenv("OPENAI_API_KEY"):
            print("检测到你选择了 OpenAI，但环境变量 OPENAI_API_KEY 未设置。")
            print("我先把 prompt 打印出来（你设置好 key 后再运行即可）。\n")
            print("=== Prompt ===")
            print(prompt)
            return

    llm_res = llm_op.execute(prompt, temperature=0.2, max_tokens=64)
    text = (llm_res or {}).get("response") or ""

    # 解析模型输出的选项字母
    m = re.search(r"\b([A-Z])\b", text.strip())
    pred_letter = m.group(1) if m else None
    pred_idx = (ord(pred_letter) - ord("A")) if pred_letter else None

    print("\n=== LLM 输出 ===")
    print(text)
    print("\n=== 解析结果 ===")
    print(f"pred_letter: {pred_letter} | pred_idx: {pred_idx}")

    # 与 GT 简单对比（如果数据里带 answer_idx）
    if isinstance(answer_idx, int) and pred_idx is not None:
        print(f"gt_idx: {answer_idx} | correct: {pred_idx == answer_idx}")
    elif answer is not None and pred_idx is not None and pred_idx < len(options):
        print(f"pred_answer: {options[pred_idx]} | gt_answer: {answer}")
    print()


if __name__ == "__main__":
    # 打印可用的 (provider, region, model) 组合
    list_available_ops()

    # 运行一个示例调用（按需修改 pid 和输入）
    run_demo()