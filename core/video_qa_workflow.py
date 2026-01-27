# core/video_qa_workflow.py
"""
VideoQAWorkflow: 视频问答workflow的具体实现

Workflow步骤：
1. segment: 视频分割（检测场景切换）
2. split: 视频物理切割
3. caption: 在 segment、split 之后，对每个分段视频分别 caption，再合并
4. concentrate_captions: 合并所有描述
5. llm_query: LLM查询
6. parse_result: 解析结果
"""
import os
import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from core.workflow import Workflow, WorkflowStep
from ops.registry import get_operation


def _get_video_name(video_path: str) -> str:
    """从视频路径中提取视频名称（不含扩展名）"""
    if not video_path:
        return "unknown_video"
    
    # 处理云存储URI
    if video_path.startswith('s3://') or video_path.startswith('gs://'):
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


class VideoQAWorkflow(Workflow):
    """
    视频问答Workflow
    
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
            name="VideoQAWorkflow",
            description="视频问答workflow：Segment → Split → Caption → LLM Query → Parse Result"
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
        
        # Step 3: 视频描述（在 segment、split 之后，对每个分段视频分别 caption，再合并）
        self.add_step(
            name="caption",
            description="视频描述 (Video Captioning)",
            dependencies=["segment", "split"],
            execute_func=self._execute_caption,
            input_keys=["segment_video_uris", "video_path"],
            output_keys=["captions"],
        )
        
        # Step 4: 合并描述
        self.add_step(
            name="concentrate_captions",
            description="合并视频描述 (Concentrate Captions)",
            dependencies=["caption"],
            execute_func=self._execute_concentrate_captions,
            input_keys=["captions"],
            output_keys=["concentrated_captions"],
        )
        
        # Step 5: LLM查询
        self.add_step(
            name="llm_query",
            description="LLM 查询 (LLM Query)",
            dependencies=["concentrate_captions"],
            execute_func=self._execute_llm_query,
            input_keys=["question", "options", "concentrated_captions"],
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
            "caption": {
                "operation_pid": "cap_aws_nova_lite_sg",
                "enabled": True,
            },
            "llm_query": {
                "operation_pid": "llm_openai_gpt4o_mini",
                "enabled": True,
            },
        }
    
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
        
        # 记录operation时间（如果operation没有记录，这里作为fallback）
        # 注意：operation应该在execute方法中记录时间（排除传输时间）
        # 这里只检查是否已记录，如果没有则记录整个步骤时间
        try:
            from utils.timing import TimingRecorder
            recorder = TimingRecorder()
            timing = recorder.get_timing()
            if timing:
                recorded = any(op.operation_name == step.name for op in timing.operations)
                if not recorded:
                    # 如果没有记录，说明operation没有在execute中记录时间
                    # 这种情况下，我们无法区分传输时间和operation时间，所以不记录
                    # 或者记录整个步骤时间作为fallback
                    pass  # 暂时不记录，等待operation自己记录
        except Exception:
            pass
        
        # 获取云存储URI（segment operation已将视频上传到云存储）
        video_uri = seg_res.get("source_used") or seg_res.get("input_video") or video_path
        
        # 过滤非法片段并截断
        segments = [s for s in segments if float(s.get("end", 0) or 0) > float(s.get("start", 0) or 0)]
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
        
        # Google Cloud Function：registry已根据region设置service_url，直接使用即可
        # AWS Lambda：registry已设置默认函数名"video-splitter"，直接使用即可
        # 如需覆盖，请在registry层面配置（Google通过GCP_VIDEOSPLIT_SERVICE_URLS，AWS通过修改registry注册代码）
        
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
    
    def _execute_caption(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行视频描述步骤：在 segment、split 之后，对每个分段视频分别 caption，再合并。"""
        from ops.registry import get_operation
        import time
        
        segment_video_uris = workflow.context.get("segment_video_uris") or []
        video_path = workflow.context.get("video_path")
        
        if not segment_video_uris:
            print("⚠ 没有分段视频（segment_video_uris 为空），跳过 caption")
            return {"captions": []}
        
        # 构建输出路径：results/caption/{video_name}/
        target_path = _build_operation_target_path("caption", video_path)
        
        caption_op = get_operation(step.operation_pid)
        all_captions = []
        
        # 临时移除operation_pid，避免workflow.execute_step重复记录整个步骤的时间
        # （因为我们会为每个片段单独记录时间）
        original_operation_pid = step.operation_pid
        step.operation_pid = None
        
        try:
            for i, video_uri in enumerate(segment_video_uris):
                seg_idx = i + 1  # 1-based，与 concentrate_captions 中 [片段 N] 一致
                try:
                    # 记录每个片段caption的时间
                    seg_start_time = time.time()
                    cap_res = caption_op.execute(video_uri, target_path=target_path)
                    seg_end_time = time.time()
                    
                    # 记录每个片段的时间
                    try:
                        from utils.timing import TimingRecorder
                        recorder = TimingRecorder()
                        recorder.record_operation(
                            f"caption_segment_{seg_idx}",
                            original_operation_pid,
                            seg_start_time,
                            seg_end_time
                        )
                    except Exception:
                        pass  # 如果记录失败，不影响主流程
                    
                    cap_text = (cap_res or {}).get("caption", "") or ""
                    all_captions.append({"segment_idx": seg_idx, "caption": cap_text})
                    preview = cap_text[:80] + "..." if len(cap_text) > 80 else cap_text
                    print(f"✓ 片段 {seg_idx}/{len(segment_video_uris)} 描述完成: {preview}")
                except Exception as e:
                    print(f"✗ 片段 {seg_idx}/{len(segment_video_uris)} 描述失败: {e}")
                    all_captions.append({"segment_idx": seg_idx, "caption": ""})
        finally:
            # 恢复operation_pid
            step.operation_pid = original_operation_pid
        
        print(f"✓ 视频描述完成，共 {len(all_captions)} 个片段")
        return {"captions": all_captions}
    
    def _execute_concentrate_captions(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行合并描述步骤"""
        all_captions = workflow.context.get("captions", [])
        
        if not all_captions:
            print("⚠ 没有可用的视频描述")
            return {"concentrated_captions": ""}
        
        # 合并所有captions，但只包含有实际内容的caption
        caption_parts = []
        for cap_info in all_captions:
            caption_text = cap_info.get('caption', '').strip()
            # 只有当caption有实际内容时才添加
            if caption_text:
                seg_idx = int(cap_info.get("segment_idx", 0) or 0)
                if seg_idx > 0:
                    caption_parts.append(f"[片段 {seg_idx}] {caption_text}")
                else:
                    caption_parts.append(f"[完整视频] {caption_text}")
        
        concentrated_captions = "\n\n".join(caption_parts) if caption_parts else ""
        print(f"✓ 已合并 {len(caption_parts)} 个有内容的片段描述（共 {len(all_captions)} 个片段）")
        print(f"  总长度: {len(concentrated_captions)} 字符")
        
        return {"concentrated_captions": concentrated_captions}
    
    def _execute_llm_query(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行LLM查询步骤"""
        from ops.registry import get_operation
        
        question = workflow.context.get("question", "")
        options = workflow.context.get("options", [])
        concentrated_captions = workflow.context.get("concentrated_captions", "")
        all_captions = workflow.context.get("captions", [])
        
        # 获取operation
        llm_op = get_operation(step.operation_pid)
        
        # 组装prompt
        lines = []
        lines.append("你是一个视频问答助手。请根据给定信息回答选择题。")
        lines.append("输出格式要求：只输出一个大写字母选项（例如：A），不要输出额外解释。")
        lines.append("")
        lines.append(f"问题：{question}")
        lines.append("选项：")
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            lines.append(f"{letter}. {opt}")
        
        # 添加合并后的captions
        # 构建完整的caption内容（不截取，确保完整打印）
        caption_content_lines = []
        if concentrated_captions:
            caption_content_lines.append("")
            caption_content_lines.append("="*50)
            caption_content_lines.append("视频内容描述：")
            caption_content_lines.append("="*50)
            
            # 使用all_captions构建完整的caption列表，不截取
            if all_captions:
                caption_parts = []
                for cap_info in all_captions:
                    seg_idx = int(cap_info.get("segment_idx", 0) or 0)
                    caption_text = cap_info.get('caption', '')
                    if seg_idx > 0:
                        cap_text = f"[片段 {seg_idx}] {caption_text}"
                    else:
                        cap_text = f"[完整视频] {caption_text}"
                    caption_parts.append(cap_text)
                caption_content_lines.append("\n\n".join(caption_parts))
            else:
                # 如果没有all_captions，使用concentrated_captions
                caption_content_lines.append(concentrated_captions)
        
        # 将caption内容添加到prompt中
        lines.extend(caption_content_lines)
        
        prompt = "\n".join(lines)
        
        print(f"Prompt 长度: {len(prompt)} 字符")
        if concentrated_captions:
            print(f"包含 {len(all_captions)} 个片段的描述")
            # 打印caption的详细信息
            for i, cap_info in enumerate(all_captions, 1):
                seg_idx = cap_info.get("segment_idx", 0)
                caption_text = cap_info.get('caption', '')
                caption_len = len(caption_text)
                print(f"  片段 {seg_idx} caption长度: {caption_len} 字符")
        else:
            print("⚠ 警告：没有视频描述内容，LLM将仅基于问题和选项回答")
        
        # 打印prompt摘要（详细内容由operation内部打印）
        print(f"Prompt总长度: {len(prompt)} 字符")
        print(f"包含 {len(all_captions)} 个片段的描述")
        
        # 检查OpenAI配置
        if llm_op.provider == "openai":
            try:
                from openai import OpenAI as _OpenAI  # noqa: F401
            except Exception:
                print("检测到你选择了 OpenAI，但 Python 包 `openai` 未安装。")
                print("请先安装：pip install openai\n")
                # 如果OpenAI未配置，打印prompt以便调试
                print("\n" + "=" * 80)
                print("=== [VideoQA Workflow] Full Prompt to LLM ===")
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
                print("=== [VideoQA Workflow] Full Prompt to LLM ===")
                print("=" * 80)
                print(prompt)
                print("=" * 80 + "\n")
                return {"llm_response": "", "prompt": prompt}
        
        # 执行LLM查询，强制设置temperature=0以确保确定性输出
        # 注意：prompt和response的详细打印由operation内部处理，这里不再重复打印
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
            "correct": correct
        }
