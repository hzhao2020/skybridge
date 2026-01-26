#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 caption 模块的独立脚本
"""
import os
import sys

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

from ops.registry import get_operation

def test_caption():
    """
    测试 caption 模块
    """
    # 视频 URL
    video_uri = "gs://video_sg/results/split/0a8109fe-15b9-4f5c-b5f2-993013cb216b//split_segments/0a8109fe-15b9-4f5c-b5f2-993013cb216b_segment_1_0_179.mp4"
    
    # 选择 caption 操作（使用 Google Vertex AI，新加坡区域）
    # 可用的选项：
    # - cap_google_flash_sg: 使用 gemini-2.5-flash 模型
    # - cap_google_flash_lite_sg: 使用 gemini-2.5-flash 模型（lite版本）
    caption_pid = "cap_google_flash_sg"
    
    print(f"正在获取 caption 操作实例: {caption_pid}")
    try:
        caption_op = get_operation(caption_pid)
        print(f"✓ 成功获取 caption 操作实例")
        print(f"  Provider: {caption_op.provider}")
        print(f"  Region: {caption_op.region}")
        print(f"  Model: {caption_op.model_name}")
        print(f"  Storage Bucket: {caption_op.storage_bucket}")
    except Exception as e:
        print(f"✗ 获取 caption 操作实例失败: {e}")
        return
    
    print(f"\n正在为视频生成 caption...")
    print(f"视频 URI: {video_uri}")
    
    try:
        # 调用 caption 操作
        result = caption_op.execute(video_uri)
        
        print(f"\n✓ Caption 生成成功！")
        print(f"\n结果详情:")
        print(f"  Provider: {result.get('provider')}")
        print(f"  Model: {result.get('model')}")
        print(f"  Caption: {result.get('caption')}")
        print(f"  Source Used: {result.get('source_used')}")
        if 'start_time' in result:
            print(f"  Start Time: {result.get('start_time')}")
        if 'end_time' in result:
            print(f"  End Time: {result.get('end_time')}")
        
        # 打印完整的 result 字典（用于调试）
        print(f"\n完整结果字典:")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n✗ Caption 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_caption()
