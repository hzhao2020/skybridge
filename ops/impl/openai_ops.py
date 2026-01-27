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
        # 兼容第三方 OpenAI API 代理/兼容平台：
        # - OPENAI_BASE_URL: 例如 https://api.openai-proxy.org/v1 （注意要带 /v1）
        self.base_url = os.getenv("OPENAI_BASE_URL")
    
    @property
    def openai_client(self):
        """懒加载 OpenAI 客户端"""
        if self._openai_client is None:
            if "OpenAI" not in globals():
                raise ImportError("缺少依赖：openai（请先 pip install openai）")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")

            # 如果设置了 OPENAI_BASE_URL，则按第三方平台调用；否则走官方默认
            if self.base_url:
                self._openai_client = OpenAI(base_url=self.base_url, api_key=api_key)
            else:
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