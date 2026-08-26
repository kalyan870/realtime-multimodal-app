import asyncio
import io
import time
import wave
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..config import settings
from ..audio.stt import SpeechToText
from ..audio.tts import TextToSpeech
from ..llm.generator import LLMGenerator
from .latency_budget import tracker as latency_tracker, STAGE_BUDGETS

_executor = ThreadPoolExecutor(max_workers=2)
stt_engine = SpeechToText()
tts_engine = TextToSpeech()
llm_engine = LLMGenerator()


class DegradationStatus:
    def __init__(self):
        self.stt_available: bool = True
        self.llm_available: bool = True
        self.tts_available: bool = True
        self.stt_failures: int = 0
        self.llm_failures: int = 0
        self.tts_failures: int = 0
        self.max_failures: int = 3

    def mark_stt_failure(self):
        self.stt_failures += 1
        if self.stt_failures >= self.max_failures:
            self.stt_available = False

    def mark_llm_failure(self):
        self.llm_failures += 1
        if self.llm_failures >= self.max_failures:
            self.llm_available = False

    def mark_tts_failure(self):
        self.tts_failures += 1
        if self.tts_failures >= self.max_failures:
            self.tts_available = False

    def mark_stt_success(self):
        self.stt_failures = max(0, self.stt_failures - 1)
        if self.stt_failures < self.max_failures:
            self.stt_available = True

    def mark_llm_success(self):
        self.llm_failures = max(0, self.llm_failures - 1)
        if self.llm_failures < self.max_failures:
            self.llm_available = True

    def mark_tts_success(self):
        self.tts_failures = max(0, self.tts_failures - 1)
        if self.tts_failures < self.max_failures:
            self.tts_available = True

    def degraded_modes(self) -> list[str]:
        modes = []
        if not self.stt_available:
            modes.append("STT degraded — using text input fallback")
        if not self.llm_available:
            modes.append("LLM degraded — using cached responses")
        if not self.tts_available:
            modes.append("TTS degraded — returning text only")
        return modes

    def to_dict(self) -> dict:
        return {
            "stt_available": self.stt_available and settings.ENABLE_STT,
            "llm_available": self.llm_available and settings.ENABLE_LLM,
            "tts_available": self.tts_available and settings.ENABLE_TTS,
            "stt_failures": self.stt_failures,
            "llm_failures": self.llm_failures,
            "tts_failures": self.tts_failures,
            "degraded_modes": self.degraded_modes(),
            "max_failures_before_degradation": self.max_failures,
        }


degradation = DegradationStatus()


def preprocess_audio(audio_bytes: bytes) -> bytes:
    start = time.perf_counter()
    try:
        if audio_bytes[:4] == b"RIFF":
            with io.BytesIO(audio_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    params = wf.getparams()
                    if params.sampwidth != 2 or params.framerate not in (8000, 16000, 22050, 44100, 48000):
                        pass
        elapsed = (time.perf_counter() - start) * 1000
        latency_tracker.record("audio_preprocess", elapsed)
        return audio_bytes
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        latency_tracker.record("audio_preprocess", elapsed)
        raise RuntimeError(f"Audio preprocessing failed: {e}") from e


def run_with_timeout(fn, timeout: float, *args, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as e:
        future.cancel()
        raise TimeoutError(f"Timed out after {timeout}s") from e


def process_pipeline(audio_bytes: Optional[bytes] = None, text_input: Optional[str] = None) -> dict:
    overall_start = time.perf_counter()
    modes = degradation.degraded_modes()
    stages_completed = []
    errors = []

    transcription = text_input
    audio_response = None
    text_response = None
    llm_prompt = text_input or ""

    if audio_bytes and not transcription:
        try:
            processed = preprocess_audio(audio_bytes)
            stages_completed.append("audio_preprocess")

            if stt_engine.is_available() and degradation.stt_available:
                raw_text, stt_latency = run_with_timeout(
                    stt_engine.transcribe, settings.STT_TIMEOUT, processed
                )
                latency_tracker.record("speech_to_text", stt_latency)
                stages_completed.append("speech_to_text")
                transcription = raw_text
                llm_prompt = raw_text
                degradation.mark_stt_success()
            else:
                raise RuntimeError("STT unavailable after degradation")
        except Exception as e:
            errors.append(f"STT: {e}")
            latency_tracker.record("speech_to_text", 0)
            degradation.mark_stt_failure()
            return {
                "status": "degraded",
                "error": f"Speech recognition failed: {e}. Please use text input.",
                "text_response": None,
                "audio_base64": None,
                "stages_completed": stages_completed,
                "latency_ms": {},
                "total_latency_ms": round((time.perf_counter() - overall_start) * 1000, 1),
                "degraded_modes": degradation.degraded_modes(),
                "errors": errors,
            }

    if not llm_prompt:
        return {
            "status": "error",
            "error": "No input provided. Provide audio or text.",
            "text_response": None,
            "audio_base64": None,
            "stages_completed": stages_completed,
            "latency_ms": {},
            "total_latency_ms": round((time.perf_counter() - overall_start) * 1000, 1),
            "degraded_modes": degradation.degraded_modes(),
            "errors": errors,
        }

    try:
        if llm_engine.is_available() and degradation.llm_available:
            text_response, llm_latency = run_with_timeout(
                llm_engine.generate, settings.LLM_TIMEOUT, llm_prompt
            )
            latency_tracker.record("llm_inference", llm_latency)
            stages_completed.append("llm_inference")
            degradation.mark_llm_success()
        else:
            text_response = "[CACHE] Real-time voice assistant is in limited mode. Please check configuration."
            latency_tracker.record("llm_inference", 0)
            stages_completed.append("llm_inference")
    except Exception as e:
        errors.append(f"LLM: {e}")
        latency_tracker.record("llm_inference", 0)
        degradation.mark_llm_failure()
        text_response = "[DEGRADED] Language model temporarily unavailable. Using cached response."
        stages_completed.append("llm_inference")

    try:
        if tts_engine.is_available() and degradation.tts_available and text_response:
            audio_bytes_out, tts_latency = run_with_timeout(
                tts_engine.synthesize, settings.TTS_TIMEOUT, text_response
            )
            latency_tracker.record("text_to_speech", tts_latency)
            stages_completed.append("text_to_speech")
            import base64
            audio_response = base64.b64encode(audio_bytes_out).decode("utf-8")
            degradation.mark_tts_success()
        else:
            latency_tracker.record("text_to_speech", 0)
    except Exception as e:
        errors.append(f"TTS: {e}")
        latency_tracker.record("text_to_speech", 0)
        degradation.mark_tts_failure()

    encode_start = time.perf_counter()
    total_latency = (encode_start - overall_start) * 1000
    latest_latencies = latency_tracker.latest()
    response = {
        "status": "success" if not errors else "partial",
        "text_response": text_response,
        "audio_base64": audio_response,
        "transcription": transcription,
        "stages_completed": stages_completed,
        "latency_ms": latest_latencies,
        "total_latency_ms": round(total_latency, 1),
        "degraded_modes": degradation.degraded_modes(),
        "errors": errors if errors else None,
    }

    encode_elapsed = (time.perf_counter() - encode_start) * 1000
    latency_tracker.record("response_encode", encode_elapsed)
    return response


def get_latency_report() -> dict:
    return {
        "budgets": STAGE_BUDGETS,
        "stages": list(STAGE_BUDGETS.keys()),
        "measurements": latency_tracker.report(),
    }


def get_degradation_status() -> dict:
    return degradation.to_dict()
