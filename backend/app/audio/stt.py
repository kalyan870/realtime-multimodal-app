import io
import time
from typing import Optional

from huggingface_hub import InferenceClient

from ..config import settings


class SpeechToText:
    def __init__(self):
        self.client = InferenceClient(token=settings.HF_TOKEN) if settings.HF_TOKEN else None
        self.model = settings.STT_MODEL
        self._available = bool(settings.HF_TOKEN)

    def is_available(self) -> bool:
        return self._available and settings.ENABLE_STT

    def transcribe(self, audio_bytes: bytes, timeout: Optional[int] = None) -> tuple[str, float]:
        start = time.perf_counter()
        if not self.is_available():
            raise RuntimeError("STT unavailable: no HF token or STT disabled")
        try:
            result = self.client.automatic_speech_recognition(
                audio_bytes,
                model=self.model,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return result.text, elapsed
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"STT failed: {e}") from e
