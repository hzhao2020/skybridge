# ops/impl/openai_ops.py
import os
from typing import Dict, Any, Optional
from ops.base import LLMQuery

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI library not installed. Install with: pip install openai")

class OpenAILLMImpl(LLMQuery):
    def __init__(self, model_name: str):
        # OpenAI 是 SaaS，没有 Region Bucket 概念，传 None
        super().__init__(provider="openai", region="global", storage_bucket=None, model_name=model_name)
        self._openai_client = None
    
    @property
    def openai_client(self):
        """懒加载 OpenAI 客户端"""
        if self._openai_client is None:
            if "OpenAI" not in globals():
                raise ImportError("缺少依赖：openai（请先 pip install openai）")

            # 从 config.py 读取配置
            try:
                import config
                api_key = getattr(config, 'OPENAI_API_KEY', None)
                base_url = getattr(config, 'OPENAI_BASE_URL', None)
            except ImportError:
                raise RuntimeError("config.py 未找到，无法读取 OpenAI 配置。")
            
            if not api_key:
                raise RuntimeError("config.py 中未找到 OPENAI_API_KEY 配置。")

            # 使用 config.py 中的 base_url（如果设置了）
            # base_url 应该包含 /v1 后缀，例如: "https://api.openai-proxy.org/v1"
            if base_url:
                self._openai_client = OpenAI(base_url=base_url, api_key=api_key)
            else:
                # 如果没有设置 base_url，使用官方 OpenAI API
                self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [OpenAI] Model: {self.model_name} ---")
        
        # 强制设置temperature为0以确保确定性输出
        temperature = 0
        
        # 打印完整的prompt
        print("\n" + "=" * 80)
        print("=== [OpenAI] Full Prompt ===")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")
        
        # 调用 OpenAI API
        print(f"    Sending prompt to {self.model_name} (temperature={temperature})...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            
            answer = response.choices[0].message.content
            
            # 打印完整的响应
            print("\n" + "=" * 80)
            print("=== [OpenAI] Full Response ===")
            print("=" * 80)
            print(answer)
            print("=" * 80 + "\n")
            
            return {
                "provider": "openai",
                "model": self.model_name,
                "response": answer
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")