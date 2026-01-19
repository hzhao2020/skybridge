# ops/impl/openai_ops.py
from typing import Dict, Any
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
            # 从环境变量读取 API key，如果没有设置会抛出错误
            self._openai_client = OpenAI()
        return self._openai_client

    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [OpenAI] Model: {self.model_name} ---")
        
        # 调用 OpenAI API
        print(f"    Sending prompt to {self.model_name}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            
            answer = response.choices[0].message.content
            print(f"    Response received: {answer[:100]}...")
            
            return {
                "provider": "openai",
                "model": self.model_name,
                "response": answer
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")