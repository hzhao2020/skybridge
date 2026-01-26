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
from core.video_qa_workflow import VideoQAWorkflow

# 配置代理（如果环境变量未设置，则使用默认值）
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


# def run_demo():
#     """
#     一个“可直接运行”的 EgoSchema demo：拿 train[0] 的一个样本跑一次 QA workflow。

#     你需要在哪里选择 provider / region / model？
#     - 就在下面的 `segment_pid` / `caption_pid` / `llm_pid` 这三个变量里选择。
#     - 也可以先运行 `list_available_ops()`，从输出的 pid 列表里挑你想要的组合。
    
#     pid 命名规则：
#         - 视频分割: seg_{provider}_{region简称}
#             例: seg_google_us, seg_aws_eu
        
#         - 视觉描述: cap_{provider}_{model简称}_{region简称}
#             例: cap_google_flash_us, cap_aws_nova_lite_sg
        
#         - LLM查询: llm_{provider}_{model简称}_{region简称}
#             例: llm_google_pro_us, llm_aws_sonnet_eu, llm_openai_gpt4o
    
#     可用的 provider/region/model 组合:
#         Google:  us (us-west1), eu (europe-west1), sg (asia-southeast1)
#         Amazon:  us (us-west-2), eu (eu-central-1), sg (ap-southeast-1)
#         OpenAI:  global (无区域)
#     """
    
#     # ========== 1) 选择你的 provider / region / model（改这里即可） ==========
#     # 默认仅跑 LLM（最容易跑通）；如果你已配置好 GCP/AWS 凭证，可打开视频相关步骤。
    
#     # 视频分割 (Video Segmentation)
#     # 可选: seg_google_us, seg_google_eu, seg_google_tw, seg_aws_us, seg_aws_eu, seg_aws_sg
#     # segment_pid = "seg_google_tw"
#     segment_pid = "seg_aws_sg"
    
#     # 视频物理切割 (Video Split / Cutting)
#     # 可选: split_google_us, split_google_eu, split_google_sg, split_aws_us, split_aws_eu, split_aws_sg
#     # 注意：Google split 的 service_url 可由 registry 根据 gcloud 项目号自动推导，或通过 GCP_VIDEOSPLIT_SERVICE_URL 覆盖；AWS split 默认用 Lambda 函数名 video-splitter
#     # split_pid = "split_google_sg"
#     split_pid = "split_aws_sg"
    
#     # 视觉描述 (Visual Captioning)  
#     # Google: cap_google_flash_lite_us, cap_google_flash_us, cap_google_flash_lite_eu, ...
#     # Amazon: cap_aws_nova_lite_us, cap_aws_nova_pro_us, cap_aws_nova_lite_eu, ...
#     # caption_pid = "cap_google_flash_sg"
#     caption_pid = "cap_aws_nova_lite_sg"
    
#     # LLM 查询
#     # Google: llm_google_flash_us, llm_google_pro_us, llm_google_flash_eu, ...
#     # Amazon: llm_aws_haiku_us, llm_aws_sonnet_us, llm_aws_haiku_eu, ...
#     # OpenAI: llm_openai_gpt4o_mini, llm_openai_gpt4o
#     llm_pid = "llm_openai_gpt4o_mini"

#     # 是否执行视频相关云服务（需要你本机已配置好对应云的凭证）
#     enable_video_segment = True
#     enable_video_split = True
#     enable_video_caption = True
    
#     # 分段数上限（防止 segment 过多导致 split/caption 太慢/太贵）
#     max_segments = 12
    
#     # Google Cloud Function VideoSplit 服务 URL（split_google_* 可选覆盖）
#     # 未设置时使用 registry 推导的按 region 的 URL；单 URL 覆盖可用：
#     # - GCP_VIDEOSPLIT_SERVICE_URL 或 VIDEOSPLIT_SERVICE_URL
#     google_videosplit_service_url = os.getenv("GCP_VIDEOSPLIT_SERVICE_URL") or os.getenv("VIDEOSPLIT_SERVICE_URL")
    
#     # AWS Lambda VideoSplit 函数名（可选覆盖）
#     aws_videosplit_function_name = os.getenv("AWS_VIDEOSPLIT_FUNCTION_NAME") or os.getenv("AWS_VIDEOSPLIT_LAMBDA_FUNCTION_NAME")

#     # ========== 2) 取 EgoSchema 第 1 条样本（train[0]） ==========
#     sample = build_dataset("EgoSchema", "train")[0]
#     qid = sample.get("qid") or "egoschema_train_0"
#     question = sample.get("question") or ""
#     options = sample.get("options") or []
#     answer = sample.get("answer")
#     answer_idx = sample.get("answer_idx")
#     video_path = sample.get("video_path")

#     print("=== EgoSchema 样本 ===")
#     print(f"qid: {qid}")
#     print(f"video_path: {video_path}")
#     print(f"question: {question}")
#     if options:
#         for i, opt in enumerate(options):
#             letter = chr(ord("A") + i)
#             print(f"  {letter}. {opt}")
#     if answer is not None or answer_idx is not None:
#         print(f"gt(answer): {answer} | gt(answer_idx): {answer_idx}")
#     print()

#     # 上传路径说明：我们会把视频上传到 bucket 下的这个目录（避免散落在根目录）
#     # 例如 GCP: gs://video_us/inputs/egoschema/<qid>/<filename>
#     # 例如 AWS: s3://sky-video-us/inputs/egoschema/<qid>/<filename>
#     upload_target_path = f"videos/egoschema/"

#     # ========== 3) 获取操作实例（按 pid 选择 provider/region/model） ==========
#     segment_op = get_operation(segment_pid) if enable_video_segment else None
#     split_op = get_operation(split_pid) if enable_video_split else None
#     caption_op = get_operation(caption_pid) if enable_video_caption else None
#     llm_op = get_operation(llm_pid)
    
#     print("=== Workflow 配置 ===")
#     if segment_op is not None:
#         print(f"视频分割: {segment_pid} -> provider={segment_op.provider}, region={segment_op.region}")
#     else:
#         print("视频分割: (disabled)")
    
#     if split_op is not None:
#         print(f"视频切割: {split_pid} -> provider={split_op.provider}, region={split_op.region}")
#         if split_op.provider == "google":
#             default_url = getattr(split_op, "service_url", None)
#             print(f"  Cloud Function service_url: {google_videosplit_service_url or default_url or '(missing)'}")
#         elif split_op.provider == "amazon":
#             print(f"  Lambda function_name: {aws_videosplit_function_name or '(default)'}")
#     else:
#         print("视频切割: (disabled)")

#     if caption_op is not None:
#         print(f"视觉描述: {caption_pid} -> provider={caption_op.provider}, region={caption_op.region}, model={caption_op.model_name}")
#     else:
#         print("视觉描述: (disabled)")

#     print(f"LLM查询:  {llm_pid} -> provider={llm_op.provider}, model={llm_op.model_name}")
#     print(f"视频上传目录: {upload_target_path}")
#     print()
    
#     # ========== 4) Workflow: Segment → Split → Caption → LLM ==========
#     segments = None
#     segment_video_uris = None  # split 后每个片段对应的视频 URI（云端）
#     all_captions = []  # 存储所有片段的 caption，用于后续合并

#     if (enable_video_segment or enable_video_split or enable_video_caption) and (not video_path or not os.path.exists(video_path)):
#         raise FileNotFoundError(
#             f"找不到视频文件：{video_path}\n"
#             f"请确认 datasets/EgoSchema/videos_sampled 下存在对应视频，或先关闭 enable_video_segment/enable_video_split/enable_video_caption"
#         )

#     # Step 1: 视频分割 (Video Segmentation)
#     if enable_video_segment and segment_op is not None:
#         print("\n" + "="*60)
#         print("Step 1: 视频分割 (Video Segmentation)")
#         print("="*60)
#         seg_res = segment_op.execute(video_path, target_path=upload_target_path)
#         segments = seg_res.get("segments")
#         # 防御：segments 可能为 None
#         segments = segments or []
        
#         # 过滤掉非法片段（end<=start），并做上限截断
#         segments = [s for s in segments if float(s.get("end", 0) or 0) > float(s.get("start", 0) or 0)]
#         if max_segments and len(segments) > max_segments:
#             segments = segments[:max_segments]
#         print(f"✓ 视频分割完成，共 {len(segments) if segments else 0} 个片段\n")
        
#         if segments:
#             print("片段列表：")
#             for idx, seg in enumerate(segments[:5]):  # 只显示前5个
#                 print(f"  片段 {idx+1}: {seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s")
#             if len(segments) > 5:
#                 print(f"  ... (还有 {len(segments) - 5} 个片段)")
#             print()
#     else:
#         print("\n" + "="*60)
#         print("Step 1: 视频分割 (跳过)")
#         print("="*60 + "\n")

#     # Step 2: 视频物理切割 (Video Split)
#     if enable_video_split and split_op is not None and segments and len(segments) > 0:
#         print("="*60)
#         print("Step 2: 视频切割 (Video Split)")
#         print("="*60)
        
#         split_kwargs = {"target_path": upload_target_path}
#         if split_op.provider == "google":
#             service_url = google_videosplit_service_url or getattr(split_op, "service_url", None)
#             if not service_url:
#                 raise ValueError(
#                     "你开启了视频切割并选择了 Google split，但未提供 Cloud Function service_url。\n"
#                     "请设置环境变量 GCP_VIDEOSPLIT_SERVICE_URL（或 VIDEOSPLIT_SERVICE_URL），或确保 gcloud 已配置当前项目（以自动推导 URL）；"
#                     "也可设置 GCP_PROJECT_NUMBER 或 GCP_VIDEOSPLIT_SERVICE_URLS（JSON）。"
#                 )
#             split_kwargs["service_url"] = service_url
#         elif split_op.provider == "amazon":
#             if aws_videosplit_function_name:
#                 split_kwargs["function_name"] = aws_videosplit_function_name
        
#         split_res = split_op.execute(video_path, segments=segments, **split_kwargs)
#         segment_video_uris = split_res.get("output_uris") or []
        
#         print(f"✓ 视频切割完成，共输出 {len(segment_video_uris)} 个片段文件\n")
#         if segment_video_uris:
#             print("分段视频 URI（前5个）：")
#             for u in segment_video_uris[:5]:
#                 print(f"  - {u}")
#             if len(segment_video_uris) > 5:
#                 print(f"  ... (还有 {len(segment_video_uris) - 5} 个片段)")
#             print()
#     else:
#         print("="*60)
#         print("Step 2: 视频切割 (跳过)")
#         print("="*60 + "\n")

#     # Step 3: 视频描述 (Video Captioning)
#     if enable_video_caption and caption_op is not None:
#         print("="*60)
#         print("Step 3: 视频描述 (Video Captioning)")
#         print("="*60)

#         # 目前只需要对“完整视频”生成一次 caption（不再依赖 start/end time）
#         try:
#             cap_res = caption_op.execute(video_path, target_path=upload_target_path)
#             full_caption = cap_res.get("caption", "")
#             all_captions.append({
#                 "segment_idx": 0,
#                 "caption": full_caption
#             })
#             print(f"✓ 视频描述完成: {full_caption[:100]}...\n")
#         except Exception as e:
#             print(f"✗ 视频描述失败: {e}\n")
#     else:
#         print("="*60)
#         print("Step 3: 视频片段描述 (跳过)")
#         print("="*60 + "\n")

#     # Step 4: 合并所有 Captions (Concentrate Captions)
#     print("="*60)
#     print("Step 4: 合并视频描述 (Concentrate Captions)")
#     print("="*60)
    
#     if all_captions:
#         # 将所有 captions 合并成一个字符串
#         caption_parts = []
#         for cap_info in all_captions:
#             seg_idx = int(cap_info.get("segment_idx", 0) or 0)
#             if seg_idx > 0:
#                 caption_parts.append(f"[片段 {seg_idx}] {cap_info.get('caption', '')}")
#             else:
#                 caption_parts.append(f"[完整视频] {cap_info.get('caption', '')}")
        
#         concentrated_captions = "\n\n".join(caption_parts)
#         print(f"✓ 已合并 {len(all_captions)} 个片段的描述")
#         print(f"  总长度: {len(concentrated_captions)} 字符\n")
#     else:
#         concentrated_captions = ""
#         print("⚠ 没有可用的视频描述\n")

#     # Step 5: 组装 Prompt 并提交给 LLM
#     print("="*60)
#     print("Step 5: LLM 查询 (LLM Query)")
#     print("="*60)
    
#     # ========== 5) 组装 prompt，跑 LLM（默认可跑） ==========
#     # 说明：LLM 本身只接收文本。这里我们把“问题 + 选项 + (可选)caption/segments摘要”拼成一个 prompt。
#     lines = []
#     lines.append("你是一个视频问答助手。请根据给定信息回答选择题。")
#     lines.append("输出格式要求：只输出一个大写字母选项（例如：A），不要输出额外解释。")
#     lines.append("")
#     lines.append(f"问题：{question}")
#     lines.append("选项：")
#     for i, opt in enumerate(options):
#         letter = chr(ord("A") + i)
#         lines.append(f"{letter}. {opt}")

#     # 添加合并后的 captions
#     if concentrated_captions:
#         lines.append("")
#         lines.append("="*50)
#         lines.append("视频内容描述：")
#         lines.append("="*50)
#         # 如果 captions 太长，截取前 N 个字符（保留完整片段）
#         max_length = 4000  # 限制 prompt 长度，避免超出模型限制
#         if len(concentrated_captions) > max_length:
#             # 尝试按片段截取
#             truncated_parts = []
#             current_length = 0
#             for cap_info in all_captions:
#                 seg_idx = int(cap_info.get("segment_idx", 0) or 0)
#                 if seg_idx > 0:
#                     cap_text = f"[片段 {seg_idx}] {cap_info.get('caption', '')}"
#                 else:
#                     cap_text = f"[完整视频] {cap_info.get('caption', '')}"
#                 if current_length + len(cap_text) > max_length:
#                     break
#                 truncated_parts.append(cap_text)
#                 current_length += len(cap_text) + 2  # +2 for "\n\n"
            
#             if truncated_parts:
#                 lines.append("\n\n".join(truncated_parts))
#                 lines.append(f"\n... (还有 {len(all_captions) - len(truncated_parts)} 个片段未显示)")
#             else:
#                 # 如果第一个片段就太长，直接截取
#                 lines.append(concentrated_captions[:max_length] + "...")
#         else:
#             lines.append(concentrated_captions)

#     prompt = "\n".join(lines)
    
#     print(f"Prompt 长度: {len(prompt)} 字符")
#     if concentrated_captions:
#         print(f"包含 {len(all_captions)} 个片段的描述")
#     print()

#     # 如果没有 OPENAI_API_KEY（且你又选了 openai 的 pid），就不实际请求 API，避免直接报错
#     if llm_op.provider == "openai":
#         # 1) SDK 未安装：直接提示安装，并输出 prompt 供你手动测试
#         try:
#             from openai import OpenAI as _OpenAI  # noqa: F401
#         except Exception:
#             print("检测到你选择了 OpenAI，但 Python 包 `openai` 未安装。")
#             print("请先安装：pip install openai\n")
#             print("=== Prompt ===")
#             print(prompt)
#             return

#         # 2) Key 未设置：不请求 API，避免报错
#         if not os.getenv("OPENAI_API_KEY"):
#             print("检测到你选择了 OpenAI，但环境变量 OPENAI_API_KEY 未设置。")
#             print("我先把 prompt 打印出来（你设置好 key 后再运行即可）。\n")
#             print("如需使用第三方兼容平台，请同时设置：OPENAI_BASE_URL（例如 https://api.openai-proxy.org/v1）")
#             print("=== Prompt ===")
#             print(prompt)
#             return

#     # 执行 LLM 查询
#     print("正在提交查询到 LLM...")
#     llm_res = llm_op.execute(prompt, temperature=0.2, max_tokens=64)
#     text = (llm_res or {}).get("response") or ""

#     # Step 5: 解析结果
#     print("="*60)
#     print("Step 6: 解析结果 (Parse Result)")
#     print("="*60)
    
#     # 解析模型输出的选项字母
#     m = re.search(r"\b([A-Z])\b", text.strip())
#     pred_letter = m.group(1) if m else None
#     pred_idx = (ord(pred_letter) - ord("A")) if pred_letter else None

#     print("\n=== LLM 原始输出 ===")
#     print(text)
#     print("\n=== 解析结果 ===")
#     print(f"预测选项: {pred_letter} (索引: {pred_idx})")

#     # 与 GT 对比
#     if isinstance(answer_idx, int) and pred_idx is not None:
#         is_correct = pred_idx == answer_idx
#         print(f"正确答案: {answer_idx} ({chr(ord('A') + answer_idx)})")
#         print(f"预测正确: {'✓ 是' if is_correct else '✗ 否'}")
#     elif answer is not None and pred_idx is not None and pred_idx < len(options):
#         pred_answer = options[pred_idx]
#         print(f"预测答案: {pred_answer}")
#         print(f"正确答案: {answer}")
#         print(f"预测正确: {'✓ 是' if pred_answer == answer else '✗ 否'}")
    
#     print("\n" + "="*60)
#     print("Workflow 完成！")
#     print("="*60 + "\n")
    
#     # 返回结果供后续使用
#     return {
#         "qid": qid,
#         "question": question,
#         "segments": segments,
#         "segment_video_uris": segment_video_uris,
#         "captions": all_captions,
#         "concentrated_captions": concentrated_captions,
#         "llm_response": text,
#         "pred_letter": pred_letter,
#         "pred_idx": pred_idx,
#         "answer": answer,
#         "answer_idx": answer_idx,
#         "correct": pred_idx == answer_idx if isinstance(answer_idx, int) else None
#     }


def run_workflow_demo():
    """
    使用抽象的Workflow框架运行demo
    
    这个函数展示了如何使用VideoQAWorkflow来执行workflow。
    你可以通过配置来指定每个步骤使用的operation（通过pid）。
    """
    # ========== 1) 创建workflow实例 ==========
    workflow = VideoQAWorkflow()
    
    # ========== 2) 配置workflow（指定每个步骤使用的operation） ==========
    # 你可以通过修改这个配置来使用不同的provider/region/model组合
    config = {
        "segment": {
            "operation_pid": "seg_google_tw",  # 可以改为 seg_google_us, seg_aws_eu 等
            "enabled": True,
        },
        "split": {
            "operation_pid": "split_google_sg",  # 可以改为 split_google_us, split_aws_eu 等
            "enabled": True,
        },
        "caption": {
            "operation_pid": "cap_google_flash_sg",  # 可以改为 cap_google_flash_us 等
            "enabled": True,
        },
        "llm_query": {
            "operation_pid": "llm_google_flash_sg",  # 可以改为 llm_google_pro_us, llm_aws_sonnet_eu 等
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
    
    # ========== 5) 处理结果 ==========
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
    # 打印最后的 caption
    # if result and "captions" in result and result["captions"]:
    #     print("\n最后的caption：")
    #     # captions 可能是一个list，通常是各个片段的caption
    #     if isinstance(result["captions"], list):
    #         print(result["captions"][-1])
    #     else:
    #         print(result["captions"])
    # else:
    #     print("无caption结果")

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
    # 运行一个示例调用（按需修改 pid 和输入）
    # run_demo()  # 原始实现
    
    # 使用新的Workflow框架（推荐）
    # run_workflow_demo()

    # print(res)