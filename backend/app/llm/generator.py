import time
from typing import Optional

from openai import OpenAI

from ..config import settings


class LLMGenerator:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self._available = False

        if self.provider == "nvidia" and settings.NVIDIA_API_KEY:
            self.client = OpenAI(
                base_url=settings.NVIDIA_BASE_URL,
                api_key=settings.NVIDIA_API_KEY,
            )
            self._available = True
        elif self.provider == "openrouter" and settings.OPENROUTER_API_KEY:
            self.client = OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
            )
            self._available = True
        elif settings.ENABLE_LLM:
            self._available = True

    def is_available(self) -> bool:
        return self._available and settings.ENABLE_LLM

    def generate(self, prompt: str, timeout: Optional[int] = None) -> tuple[str, float]:
        start = time.perf_counter()
        if not self.is_available():
            raise RuntimeError("LLM unavailable: no API key or LLM disabled")

        system_prompt = "You are a helpful voice assistant. Respond conversationally and concisely."

        try:
            if self.provider in ("nvidia", "openrouter"):
                extra = {}
                if self.provider == "openrouter":
                    extra["extra_body"] = {"models": [self.model]}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=256,
                    timeout=timeout or settings.LLM_TIMEOUT,
                    **extra,
                )
                text = response.choices[0].message.content.strip()
            else:
                text = f"[LLM {self.provider} not fully configured] Echo: {prompt}"
            elapsed = (time.perf_counter() - start) * 1000
            return text, elapsed
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"LLM generation failed: {e}") from e
