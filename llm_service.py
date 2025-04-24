import os
from vllm import LLM, SamplingParams

os.environ["KV_CACHE_PREFETCH"] = "1"

class LLMService:
    def __init__(self):
        self.llm = LLM(
            model="mistral-7b",
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_prefetch_tokens=24
        )

    def generate(self, prompt: str):
        params = SamplingParams(
            temperature=0.7,
            max_tokens=256,
            kv_cache_config={"prefetch": True, "prefill_tokens": 24}
        )
        return self.llm.generate(prompt, params)
