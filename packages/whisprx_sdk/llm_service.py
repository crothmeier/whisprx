import os
import asyncio
from typing import AsyncIterator
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

os.environ["KV_CACHE_PREFETCH"] = "1"

class LLMService:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize vLLM service with streaming support

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_model_len=2048,  # Reduce context for faster inference
            trust_remote_code=True,
        )
        print(f"✓ Initialized vLLM with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a complete response (non-streaming)

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    async def stream_generate(self, prompt: str, max_tokens: int = 256) -> AsyncIterator[str]:
        """Generate tokens as a stream (async generator)

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate

        Yields:
            Individual tokens as they're generated
        """
        # Format prompt for instruction-tuned models
        formatted_prompt = f"[INST] {prompt} [/INST]"

        params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        # vLLM doesn't have native async streaming, so we use run_in_executor
        # and simulate streaming by yielding the complete response
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self.llm.generate([formatted_prompt], params)[0]
        )

        # Yield the complete response as one chunk
        # In a real streaming implementation, we'd yield tokens incrementally
        # For now, this maintains the async interface
        text = output.outputs[0].text
        yield text
