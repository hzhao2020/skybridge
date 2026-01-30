"""
Murakkab: 视频问答 workflow 的具体实现

Workflow 步骤：
1. segment: 视频分割（检测场景切换）
2. split: 视频物理切割
3. audio_to_text: 音频转文本（Speech-to-Text）
4. frame_extraction: 帧提取
5. object_detection: 目标检测（在提取的帧上）
6. llm_query: LLM 查询（使用文本、标注帧和问题）
7. parse_result: 解析结果
"""
import os
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from core.workflow import Workflow, WorkflowStep
from ops.registry import get_operation


def _get_video_name(video_path: str) -> str:
    """从视频路径中提取视频名称（不含扩展名）"""
    if not video_path:
        return "unknown_video"

    # 处理云存储URI
    if video_path.startswith("s3://") or video_path.startswith("gs://"):
        parsed = urlparse(video_path)
        video_filename = os.path.basename(parsed.path)
    else:
        # 本地路径
        video_filename = os.path.basename(video_path)

    # 移除扩展名
    video_name = os.path.splitext(video_filename)[0]
    return video_name if video_name else "unknown_video"


def _build_operation_target_path(operation_name: str, video_path: str) -> str:
    """构建operation的输出路径：results/{operation_name}/{video_name}/"""
    video_name = _get_video_name(video_path)
    return f"results/{operation_name}/{video_name}/"


class MurakkabVideoQA(Workflow):
    """
    基于 Murakkab 逻辑的异构视频问答工作流
    
    输入数据格式：
    {
        "video_path": str,  # 视频文件路径
        "question": str,    # 问题
        "options": List[str],  # 选项列表
        "answer": Optional[str],  # 正确答案（可选）
        "answer_idx": Optional[int],  # 正确答案索引（可选）
        "qid": Optional[str],  # 问题ID（可选）
        "max_segments": Optional[int],  # 最大片段数（可选，默认12）
    }
    """

    def __init__(self):
        super().__init__(
            name="Murakkab",
            description="基于 Murakkab 逻辑的异构视频问答工作流：Segment → Split → Audio-to-Text & Frame Extraction → Object Detection → LLM Query → Parse Result",
        )

        # 添加所有步骤
        self._setup_steps()


    def _setup_steps(self):
        """设置所有workflow步骤"""

        # Step 1: 视频分割
        self.add_step(
            name="segment",
            description="视频分割 (Video Segmentation)",
            dependencies=[],
            execute_func=self._execute_segment,
            input_keys=["video_path", "max_segments"],
            output_keys=["segments", "video_uri"],
        )

        # Step 2: 视频物理切割
        self.add_step(
            name="split",
            description="视频切割 (Video Split)",
            dependencies=["segment"],
            execute_func=self._execute_split,
            input_keys=["video_uri", "video_path", "segments"],
            output_keys=["segment_video_uris"],
        )

        self.add_step(
            name="audio_to_text",
            description="音频提取",
            dependencies=["split"],
            execute_func=self._execute_audio_to_text,
            input_keys=["segment_video_uris", "video_path"],
            output_keys=["video_text"],
        )

        self.add_step(
            name="frame_extraction",
            description="帧抽取",
            dependencies=["split"],
            execute_func=self._execute_frame_extraction,
            input_keys=["segment_video_uris", "video_path"],
            output_keys=["frame_uris"],
        )

        self.add_step(
            name="object_detection",
            description="物品检测",
            dependencies=["frame_extraction"],
            execute_func=self._execute_object_detection,
            input_keys=["frame_uris", "video_path"],
            output_keys=["annotated_frames"],
        )

        self.add_step(
            name="llm_query",
            description="LLM 查询 (LLM Query)",
            dependencies=["object_detection", "audio_to_text"],
            execute_func=self._execute_llm_query,
            input_keys=["question", "options", "annotated_frames", "video_text"],
            output_keys=["llm_response", "prompt"],
        )

        # Step 6: 解析结果
        self.add_step(
            name="parse_result",
            description="解析结果 (Parse Result)",
            dependencies=["llm_query"],
            execute_func=self._execute_parse_result,
            input_keys=["llm_response", "options", "answer", "answer_idx"],
            output_keys=["pred_letter", "pred_idx", "correct"],
        )

    def get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "segment": {
                "operation_pid": "seg_aws_sg",
                "enabled": True,
            },
            "split": {
                "operation_pid": "split_aws_sg",
                "enabled": True,
            },
            "audio_to_text": {
                "operation_pid": "stt_aws_sg",  # 假设存在
                "enabled": True,
            },
            "frame_extraction": {
                "operation_pid": "frame_extract_aws_sg",  # 假设存在
                "enabled": True,
            },
            "object_detection": {
                "operation_pid": "obj_detect_aws_sg",
                "enabled": True,
            },
            "llm_query": {
                "operation_pid": "llm_openai_gpt4o_mini",
                "enabled": True,
            },
        }

    # --- 执行逻辑实现 (内部调用 operation) ---

    def _execute_segment(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行视频分割步骤"""
        from ops.registry import get_operation
        import time

        video_path = workflow.context.get("video_path")
        max_segments = workflow.context.get("max_segments", 12)

        # 检查视频文件是否存在
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(
                f"找不到视频文件：{video_path}\n"
                f"请确认视频文件存在，或禁用segment步骤"
            )

        # 构建输出路径：results/segment/{video_name}/
        target_path = _build_operation_target_path("segment", video_path)

        # 获取operation
        segment_op = get_operation(step.operation_pid)

        # 执行分割（传输时间在operation内部，operation会记录实际开始时间）
        seg_res = segment_op.execute(video_path, target_path=target_path)
        segments = seg_res.get("segments") or []

        # 获取云存储URI（segment operation已将视频上传到云存储）
        video_uri = seg_res.get("source_used") or seg_res.get("input_video") or video_path

        # 过滤非法片段并截断
        segments = [
            s
            for s in segments
            if float(s.get("end", 0) or 0) > float(s.get("start", 0) or 0)
        ]
        if max_segments and len(segments) > max_segments:
            segments = segments[:max_segments]

        print(f"✓ 视频分割完成，共 {len(segments)} 个片段")
        if segments:
            print("片段列表（前5个）：")
            for idx, seg in enumerate(segments[:5]):
                print(f"  片段 {idx+1}: {seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s")
            if len(segments) > 5:
                print(f"  ... (还有 {len(segments) - 5} 个片段)")
        print(f"  视频已上传到云存储: {video_uri}")

        return {"segments": segments, "video_uri": video_uri}

    def _execute_split(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行视频切割步骤"""
        from ops.registry import get_operation

        # 优先使用segment步骤返回的云存储URI，如果没有则使用原始路径
        video_uri = workflow.context.get("video_uri") or workflow.context.get("video_path")
        video_path = workflow.context.get("video_path")
        segments = workflow.context.get("segments")

        # 如果没有segments，跳过
        if not segments or len(segments) == 0:
            print("⚠ 没有segments，跳过视频切割")
            return {"segment_video_uris": []}

        # 构建输出路径：results/split/{video_name}/
        target_path = _build_operation_target_path("split", video_path)

        # 获取operation
        split_op = get_operation(step.operation_pid)

        # 准备参数
        split_kwargs = {"target_path": target_path}

        # 执行切割（使用云存储URI，避免重复上传）
        split_res = split_op.execute(video_uri, segments=segments, **split_kwargs)
        segment_video_uris = split_res.get("output_uris") or []

        print(f"✓ 视频切割完成，共输出 {len(segment_video_uris)} 个片段文件")
        if segment_video_uris:
            print("分段视频 URI（前5个）：")
            for u in segment_video_uris[:5]:
                print(f"  - {u}")
            if len(segment_video_uris) > 5:
                print(f"  ... (还有 {len(segment_video_uris) - 5} 个片段)")

        return {"segment_video_uris": segment_video_uris}

    def _execute_audio_to_text(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行音频转文本步骤：对每个分段视频提取音频并转文本"""
        from ops.registry import get_operation
        import time

        segment_video_uris = workflow.context.get("segment_video_uris") or []
        video_path = workflow.context.get("video_path")

        if not segment_video_uris:
            print("⚠ 没有分段视频（segment_video_uris 为空），跳过音频转文本")
            return {"video_text": ""}

        # 构建输出路径：results/audio_to_text/{video_name}/
        target_path = _build_operation_target_path("audio_to_text", video_path)

        stt_op = get_operation(step.operation_pid)
        all_texts = []

        # 临时移除operation_pid，避免workflow.execute_step重复记录整个步骤的时间
        original_operation_pid = step.operation_pid
        step.operation_pid = None

        try:
            for i, video_uri in enumerate(segment_video_uris):
                seg_idx = i + 1  # 1-based
                try:
                    # 记录每个片段STT的时间
                    seg_start_time = time.time()
                    stt_res = stt_op.execute(video_uri, target_path=target_path)
                    seg_end_time = time.time()

                    # 记录每个片段的时间
                    try:
                        from utils.timing import TimingRecorder

                        recorder = TimingRecorder()
                        recorder.record_operation(
                            f"audio_to_text_segment_{seg_idx}",
                            original_operation_pid,
                            seg_start_time,
                            seg_end_time,
                        )
                    except Exception:
                        pass  # 如果记录失败，不影响主流程

                    text = (stt_res or {}).get("text", "") or (stt_res or {}).get("transcript", "") or ""
                    all_texts.append({"segment_idx": seg_idx, "text": text})
                    preview = text[:80] + "..." if len(text) > 80 else text
                    print(f"✓ 片段 {seg_idx}/{len(segment_video_uris)} 音频转文本完成: {preview}")
                except Exception as e:
                    print(f"✗ 片段 {seg_idx}/{len(segment_video_uris)} 音频转文本失败: {e}")
                    all_texts.append({"segment_idx": seg_idx, "text": ""})
        finally:
            # 恢复operation_pid
            step.operation_pid = original_operation_pid

        # 合并所有文本
        text_parts = []
        for text_info in all_texts:
            text_content = text_info.get("text", "").strip()
            if text_content:
                seg_idx = int(text_info.get("segment_idx", 0) or 0)
                if seg_idx > 0:
                    text_parts.append(f"[片段 {seg_idx}] {text_content}")
                else:
                    text_parts.append(f"[完整视频] {text_content}")

        video_text = "\n\n".join(text_parts) if text_parts else ""
        print(f"✓ 音频转文本完成，共 {len(text_parts)} 个有内容的片段（共 {len(all_texts)} 个片段）")
        print(f"  总长度: {len(video_text)} 字符")

        return {"video_text": video_text}

    def _execute_frame_extraction(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行帧提取步骤：从每个分段视频中提取关键帧"""
        from ops.registry import get_operation
        import time

        segment_video_uris = workflow.context.get("segment_video_uris") or []
        video_path = workflow.context.get("video_path")

        if not segment_video_uris:
            print("⚠ 没有分段视频（segment_video_uris 为空），跳过帧提取")
            return {"frame_uris": []}

        # 构建输出路径：results/frame_extraction/{video_name}/
        target_path = _build_operation_target_path("frame_extraction", video_path)

        frame_extract_op = get_operation(step.operation_pid)
        all_frame_uris = []

        # 临时移除operation_pid，避免workflow.execute_step重复记录整个步骤的时间
        original_operation_pid = step.operation_pid
        step.operation_pid = None

        try:
            for i, video_uri in enumerate(segment_video_uris):
                seg_idx = i + 1  # 1-based
                try:
                    # 记录每个片段帧提取的时间
                    seg_start_time = time.time()
                    extract_res = frame_extract_op.execute(video_uri, target_path=target_path)
                    seg_end_time = time.time()

                    # 记录每个片段的时间
                    try:
                        from utils.timing import TimingRecorder

                        recorder = TimingRecorder()
                        recorder.record_operation(
                            f"frame_extraction_segment_{seg_idx}",
                            original_operation_pid,
                            seg_start_time,
                            seg_end_time,
                        )
                    except Exception:
                        pass  # 如果记录失败，不影响主流程

                    frame_uris = extract_res.get("frame_uris") or extract_res.get("frames") or []
                    all_frame_uris.extend(frame_uris)
                    print(f"✓ 片段 {seg_idx}/{len(segment_video_uris)} 帧提取完成: {len(frame_uris)} 帧")
                except Exception as e:
                    print(f"✗ 片段 {seg_idx}/{len(segment_video_uris)} 帧提取失败: {e}")
        finally:
            # 恢复operation_pid
            step.operation_pid = original_operation_pid

        print(f"✓ 帧提取完成，共提取 {len(all_frame_uris)} 帧")
        return {"frame_uris": all_frame_uris}

    def _execute_object_detection(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行目标检测步骤：在提取的帧上进行目标检测"""
        from ops.registry import get_operation

        frame_uris = workflow.context.get("frame_uris") or []
        video_path = workflow.context.get("video_path")

        if not frame_uris:
            print("⚠ 没有提取的帧（frame_uris 为空），跳过目标检测")
            return {"annotated_frames": []}

        # 构建输出路径：results/object_detection/{video_name}/
        target_path = _build_operation_target_path("object_detection", video_path)

        obj_detect_op = get_operation(step.operation_pid)
        all_annotated_frames = []

        # 临时移除operation_pid，避免workflow.execute_step重复记录整个步骤的时间
        original_operation_pid = step.operation_pid
        step.operation_pid = None

        try:
            for i, frame_uri in enumerate(frame_uris):
                frame_idx = i + 1  # 1-based
                try:
                    import time
                    frame_start_time = time.time()
                    detect_res = obj_detect_op.execute(frame_uri, target_path=target_path)
                    frame_end_time = time.time()

                    # 记录每个帧的时间
                    try:
                        from utils.timing import TimingRecorder

                        recorder = TimingRecorder()
                        recorder.record_operation(
                            f"object_detection_frame_{frame_idx}",
                            original_operation_pid,
                            frame_start_time,
                            frame_end_time,
                        )
                    except Exception:
                        pass  # 如果记录失败，不影响主流程

                    # 假设operation返回包含检测结果的字典
                    # 格式可能是：{"detected_objects": [...], "annotated_frame_uri": "..."}
                    annotated_frame_uri = detect_res.get("annotated_frame_uri") or frame_uri
                    detected_objects = detect_res.get("detected_objects") or []
                    
                    all_annotated_frames.append({
                        "frame_uri": frame_uri,
                        "annotated_frame_uri": annotated_frame_uri,
                        "detected_objects": detected_objects,
                    })
                    
                    print(f"✓ 帧 {frame_idx}/{len(frame_uris)} 目标检测完成: {len(detected_objects)} 个物体")
                except Exception as e:
                    print(f"✗ 帧 {frame_idx}/{len(frame_uris)} 目标检测失败: {e}")
                    all_annotated_frames.append({
                        "frame_uri": frame_uri,
                        "annotated_frame_uri": frame_uri,
                        "detected_objects": [],
                    })
        finally:
            # 恢复operation_pid
            step.operation_pid = original_operation_pid

        print(f"✓ 目标检测完成，共处理 {len(all_annotated_frames)} 帧")
        return {"annotated_frames": all_annotated_frames}

    def _execute_llm_query(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行LLM查询步骤：使用文本、标注帧和问题进行多模态推理"""
        from ops.registry import get_operation

        question = workflow.context.get("question", "")
        options = workflow.context.get("options", [])
        video_text = workflow.context.get("video_text", "")
        annotated_frames = workflow.context.get("annotated_frames", [])

        # 获取operation
        llm_op = get_operation(step.operation_pid)

        # 组装prompt
        lines = []
        lines.append("你是一个视频问答助手。请根据给定的视频信息（包括音频转文本和视觉目标检测结果）回答选择题。")
        lines.append("输出格式要求：只输出一个大写字母选项（例如：A），不要输出额外解释。")
        lines.append("")
        lines.append(f"问题：{question}")
        lines.append("选项：")
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}. {opt}")

        # 添加音频转文本内容
        if video_text:
            lines.append("")
            lines.append("=" * 50)
            lines.append("音频转文本内容：")
            lines.append("=" * 50)
            lines.append(video_text)

        # 添加目标检测结果
        if annotated_frames:
            lines.append("")
            lines.append("=" * 50)
            lines.append("视觉目标检测结果：")
            lines.append("=" * 50)
            for i, frame_info in enumerate(annotated_frames, 1):
                detected_objects = frame_info.get("detected_objects", [])
                if detected_objects:
                    lines.append(f"\n[帧 {i}]")
                    for obj in detected_objects:
                        obj_name = obj.get("name", "未知物体")
                        confidence = obj.get("confidence", 0)
                        lines.append(f"  - {obj_name} (置信度: {confidence:.2f})")
                    # 如果有标注后的帧URI，可以添加到prompt中（如果LLM支持图像输入）
                    annotated_frame_uri = frame_info.get("annotated_frame_uri")
                    if annotated_frame_uri:
                        lines.append(f"  标注帧URI: {annotated_frame_uri}")

        prompt = "\n".join(lines)

        print(f"Prompt 长度: {len(prompt)} 字符")
        if video_text:
            print(f"包含音频转文本内容，长度: {len(video_text)} 字符")
        if annotated_frames:
            total_objects = sum(len(f.get("detected_objects", [])) for f in annotated_frames)
            print(f"包含 {len(annotated_frames)} 帧的目标检测结果，共 {total_objects} 个检测到的物体")
        if not video_text and not annotated_frames:
            print("⚠ 警告：没有视频内容信息，LLM将仅基于问题和选项回答")

        # 检查OpenAI配置
        if llm_op.provider == "openai":
            try:
                from openai import OpenAI as _OpenAI  # noqa: F401
            except Exception:
                print("检测到你选择了 OpenAI，但 Python 包 `openai` 未安装。")
                print("请先安装：pip install openai\n")
                # 如果OpenAI未配置，打印prompt以便调试
                print("\n" + "=" * 80)
                print("=== [Murakkab Workflow] Full Prompt to LLM ===")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")
                return {"llm_response": "", "prompt": prompt}

            if not os.getenv("OPENAI_API_KEY"):
                print("检测到你选择了 OpenAI，但环境变量 OPENAI_API_KEY 未设置。")
                print("我先把 prompt 打印出来（你设置好 key 后再运行即可）。\n")
                print("如需使用第三方兼容平台，请同时设置：OPENAI_BASE_URL（例如 https://api.openai-proxy.org/v1）")
                # 如果OpenAI未配置，打印prompt以便调试
                print("\n" + "=" * 80)
                print("=== [Murakkab Workflow] Full Prompt to LLM ===")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")
                return {"llm_response": "", "prompt": prompt}

        # 执行LLM查询，强制设置temperature=0以确保确定性输出
        print("正在提交查询到 LLM (temperature=0)...")
        llm_res = llm_op.execute(prompt, temperature=0, max_tokens=64)
        text = (llm_res or {}).get("response") or ""

        return {"llm_response": text, "prompt": prompt}

    def _execute_parse_result(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行解析结果步骤"""
        text = workflow.context.get("llm_response", "")
        options = workflow.context.get("options", [])
        answer = workflow.context.get("answer")
        answer_idx = workflow.context.get("answer_idx")

        # 解析模型输出的选项字母
        m = re.search(r"\b([A-Z])\b", text.strip())
        pred_letter = m.group(1) if m else None
        pred_idx = (ord(pred_letter) - ord("A")) if pred_letter else None

        print("\n=== LLM 原始输出 ===")
        print(text)
        print("\n=== 解析结果 ===")
        print(f"预测选项: {pred_letter} (索引: {pred_idx})")

        # 与GT对比
        correct = None
        if isinstance(answer_idx, int) and pred_idx is not None:
            correct = pred_idx == answer_idx
            print(f"正确答案: {answer_idx} ({chr(ord('A') + answer_idx)})")
            print(f"预测正确: {'✓ 是' if correct else '✗ 否'}")
        elif answer is not None and pred_idx is not None and pred_idx < len(options):
            pred_answer = options[pred_idx]
            correct = pred_answer == answer
            print(f"预测答案: {pred_answer}")
            print(f"正确答案: {answer}")
            print(f"预测正确: {'✓ 是' if correct else '✗ 否'}")

        return {
            "pred_letter": pred_letter,
            "pred_idx": pred_idx,
            "correct": correct,
        }