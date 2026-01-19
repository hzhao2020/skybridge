from typing import Dict, Any
from ops.base import LLMQuery

class OpenAILLMImpl(LLMQuery):
    def __init__(self, model_name: str):
        # OpenAI 不需要 storage_bucket，这里传 None 或 dummy
        super().__init__(provider="openai", region="global", storage_bucket="none", model_name=model_name)

    def execute(self, prompt: str, **kwargs) -> Dict[str, Any]:
        print(f"--- [OpenAI] Model: {self.model_name} ---")
        return {
            "provider": "openai",
            "model": self.model_name,
            "response": f"ChatGPT response to: {prompt[:10]}..."
        }