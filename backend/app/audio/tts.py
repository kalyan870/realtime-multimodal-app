import io
import time
from typing import Optional

from gtts import gTTS

from ..config import settings


class TextToSpeech:
    def __init__(self):
        self._available = settings.ENABLE_TTS

    def is_available(self) -> bool:
        return self._available

    def synthesize(self, text: str, timeout: Optional[int] = None) -> tuple[bytes, float]:
        start = time.perf_counter()
        if not self.is_available():
            raise RuntimeError("TTS unavailable: TTS disabled")
        try:
            buf = io.BytesIO()
            tts = gTTS(text=text, lang="en", slow=False)
            tts.write_to_fp(buf)
            buf.seek(0)
            elapsed = (time.perf_counter() - start) * 1000
            return buf.read(), elapsed
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"TTS failed: {e}") from e
