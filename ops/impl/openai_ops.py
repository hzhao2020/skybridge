# ops/impl/openai_ops.py
from typing import Dict, Any
from ops.base import LLMQuery

class OpenAILLMImpl(LLMQuery):
    def __init__(self, model_name: str):
        # OpenAI 是 Global 服务，通常没有 Region 概念
        super().__init__(provider="openai", region="global", model_name=model_name)

    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        print(f"[OpenAI] Model: {self.model_name} | Prompt: {prompt[:20]}...")
        # 模拟 API 调用: openai.chat.completions.create
        return {
            "provider": "openai",
            "model": self.model_name,
            "response": f"ChatGPT ({self.model_name}) answer"
        }