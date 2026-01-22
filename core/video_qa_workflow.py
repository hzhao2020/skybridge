# core/video_qa_workflow.py
"""
VideoQAWorkflow: 视频问答workflow的具体实现

Workflow步骤：
1. segment: 视频分割（检测场景切换）
2. split: 视频物理切割
3. caption: 视频描述生成
4. concentrate_captions: 合并所有描述
5. llm_query: LLM查询
6. parse_result: 解析结果
"""
import os
import re
from typing import Dict, Any, Optional
from core.workflow import Workflow, WorkflowStep
from ops.registry import get_operation


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
        "upload_target_path": Optional[str],  # 上传路径（可选，默认"videos/egoschema/"）
        "max_segments": Optional[int],  # 最大片段数（可选，默认12）
        "google_videosplit_service_url": Optional[str],  # Google Cloud Function VideoSplit 服务 URL（可选）
        "aws_videosplit_function_name": Optional[str],  # AWS Lambda函数名（可选）
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
            input_keys=["video_path", "upload_target_path", "max_segments"],
            output_keys=["segments"],
        )
        
        # Step 2: 视频物理切割
        self.add_step(
            name="split",
            description="视频切割 (Video Split)",
            dependencies=["segment"],
            execute_func=self._execute_split,
            input_keys=["video_path", "segments", "upload_target_path"],
            output_keys=["segment_video_uris"],
        )
        
        # Step 3: 视频描述
        self.add_step(
            name="caption",
            description="视频描述 (Video Captioning)",
            dependencies=[],
            execute_func=self._execute_caption,
            input_keys=["video_path", "upload_target_path"],
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
        
        video_path = workflow.context.get("video_path")
        upload_target_path = workflow.context.get("upload_target_path", "videos/egoschema/")
        max_segments = workflow.context.get("max_segments", 12)
        
        # 检查视频文件是否存在
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(
                f"找不到视频文件：{video_path}\n"
                f"请确认视频文件存在，或禁用segment步骤"
            )
        
        # 获取operation
        segment_op = get_operation(step.operation_pid)
        
        # 执行分割
        seg_res = segment_op.execute(video_path, target_path=upload_target_path)
        segments = seg_res.get("segments") or []
        
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
        
        return {"segments": segments}
    
    def _execute_split(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行视频切割步骤"""
        from ops.registry import get_operation
        
        video_path = workflow.context.get("video_path")
        segments = workflow.context.get("segments")
        upload_target_path = workflow.context.get("upload_target_path", "videos/egoschema/")
        
        # 如果没有segments，跳过
        if not segments or len(segments) == 0:
            print("⚠ 没有segments，跳过视频切割")
            return {"segment_video_uris": []}
        
        # 获取operation
        split_op = get_operation(step.operation_pid)
        
        # 准备参数
        split_kwargs = {"target_path": upload_target_path}
        
        # Google Cloud Function：优先用配置/环境变量，否则用 registry 推导的 service_url
        if split_op.provider == "google":
            service_url = (
                workflow.context.get("google_videosplit_service_url") or
                os.getenv("GCP_VIDEOSPLIT_SERVICE_URL") or
                os.getenv("VIDEOSPLIT_SERVICE_URL") or
                getattr(split_op, "service_url", None)
            )
            if not service_url:
                raise ValueError(
                    "你开启了视频切割并选择了 Google split，但未提供 Cloud Function service_url。\n"
                    "请设置环境变量 GCP_VIDEOSPLIT_SERVICE_URL（或 VIDEOSPLIT_SERVICE_URL）、配置 google_videosplit_service_url，"
                    "或确保 gcloud 已配置当前项目以自动推导。"
                )
            split_kwargs["service_url"] = service_url
        
        # AWS Lambda可以指定函数名
        elif split_op.provider == "amazon":
            function_name = (
                workflow.context.get("aws_videosplit_function_name") or
                os.getenv("AWS_VIDEOSPLIT_FUNCTION_NAME") or
                os.getenv("AWS_VIDEOSPLIT_LAMBDA_FUNCTION_NAME")
            )
            if function_name:
                split_kwargs["function_name"] = function_name
        
        # 执行切割
        split_res = split_op.execute(video_path, segments=segments, **split_kwargs)
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
        """执行视频描述步骤"""
        from ops.registry import get_operation
        
        video_path = workflow.context.get("video_path")
        upload_target_path = workflow.context.get("upload_target_path", "videos/egoschema/")
        
        # 检查视频文件是否存在
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(
                f"找不到视频文件：{video_path}\n"
                f"请确认视频文件存在，或禁用caption步骤"
            )
        
        # 获取operation
        caption_op = get_operation(step.operation_pid)
        
        # 执行描述生成
        try:
            cap_res = caption_op.execute(video_path, target_path=upload_target_path)
            full_caption = cap_res.get("caption", "")
            
            all_captions = [{
                "segment_idx": 0,
                "caption": full_caption
            }]
            
            print(f"✓ 视频描述完成: {full_caption[:100]}...")
            return {"captions": all_captions}
        except Exception as e:
            print(f"✗ 视频描述失败: {e}")
            return {"captions": []}
    
    def _execute_concentrate_captions(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """执行合并描述步骤"""
        all_captions = workflow.context.get("captions", [])
        
        if not all_captions:
            print("⚠ 没有可用的视频描述")
            return {"concentrated_captions": ""}
        
        # 合并所有captions
        caption_parts = []
        for cap_info in all_captions:
            seg_idx = int(cap_info.get("segment_idx", 0) or 0)
            if seg_idx > 0:
                caption_parts.append(f"[片段 {seg_idx}] {cap_info.get('caption', '')}")
            else:
                caption_parts.append(f"[完整视频] {cap_info.get('caption', '')}")
        
        concentrated_captions = "\n\n".join(caption_parts)
        print(f"✓ 已合并 {len(all_captions)} 个片段的描述")
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
        if concentrated_captions:
            lines.append("")
            lines.append("="*50)
            lines.append("视频内容描述：")
            lines.append("="*50)
            
            # 如果captions太长，截取
            max_length = 4000
            if len(concentrated_captions) > max_length:
                truncated_parts = []
                current_length = 0
                for cap_info in all_captions:
                    seg_idx = int(cap_info.get("segment_idx", 0) or 0)
                    if seg_idx > 0:
                        cap_text = f"[片段 {seg_idx}] {cap_info.get('caption', '')}"
                    else:
                        cap_text = f"[完整视频] {cap_info.get('caption', '')}"
                    if current_length + len(cap_text) > max_length:
                        break
                    truncated_parts.append(cap_text)
                    current_length += len(cap_text) + 2
                
                if truncated_parts:
                    lines.append("\n\n".join(truncated_parts))
                    lines.append(f"\n... (还有 {len(all_captions) - len(truncated_parts)} 个片段未显示)")
                else:
                    lines.append(concentrated_captions[:max_length] + "...")
            else:
                lines.append(concentrated_captions)
        
        prompt = "\n".join(lines)
        
        print(f"Prompt 长度: {len(prompt)} 字符")
        if concentrated_captions:
            print(f"包含 {len(all_captions)} 个片段的描述")
        
        # 检查OpenAI配置
        if llm_op.provider == "openai":
            try:
                from openai import OpenAI as _OpenAI  # noqa: F401
            except Exception:
                print("检测到你选择了 OpenAI，但 Python 包 `openai` 未安装。")
                print("请先安装：pip install openai\n")
                print("=== Prompt ===")
                print(prompt)
                return {"llm_response": "", "prompt": prompt}
            
            if not os.getenv("OPENAI_API_KEY"):
                print("检测到你选择了 OpenAI，但环境变量 OPENAI_API_KEY 未设置。")
                print("我先把 prompt 打印出来（你设置好 key 后再运行即可）。\n")
                print("如需使用第三方兼容平台，请同时设置：OPENAI_BASE_URL（例如 https://api.openai-proxy.org/v1）")
                print("=== Prompt ===")
                print(prompt)
                return {"llm_response": "", "prompt": prompt}
        
        # 执行LLM查询
        print("正在提交查询到 LLM...")
        llm_res = llm_op.execute(prompt, temperature=0.2, max_tokens=64)
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
