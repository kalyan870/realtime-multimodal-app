STAGE_BUDGETS = {
    "audio_preprocess": {"p50_ms": 50, "p95_ms": 150, "p99_ms": 300, "description": "WAV conversion, resampling, format check"},
    "speech_to_text":   {"p50_ms": 800, "p95_ms": 2500, "p99_ms": 5000, "description": "Whisper inference on HF Inference API"},
    "llm_inference":    {"p50_ms": 1500, "p95_ms": 5000, "p99_ms": 10000, "description": "NVIDIA Llama 3.1 70B inference"},
    "text_to_speech":   {"p50_ms": 1000, "p95_ms": 3000, "p99_ms": 6000, "description": "gTTS HTTP request + MP3 encoding"},
    "response_encode":  {"p50_ms": 20, "p95_ms": 50, "p99_ms": 100, "description": "Base64 encode + JSON serialize"},
}

STAGE_LABELS = {
    "audio_preprocess": "Audio Preprocessing",
    "speech_to_text": "Speech-to-Text (Whisper)",
    "llm_inference": "LLM Inference (NVIDIA Llama)",
    "text_to_speech": "Text-to-Speech (gTTS)",
    "response_encode": "Response Encoding",
}

STAGE_ORDER = ["audio_preprocess", "speech_to_text", "llm_inference", "text_to_speech", "response_encode"]

TOTAL_BUDGET = {
    "p50_ms": sum(s["p50_ms"] for s in STAGE_BUDGETS.values()),
    "p95_ms": sum(s["p95_ms"] for s in STAGE_BUDGETS.values()),
    "p99_ms": sum(s["p99_ms"] for s in STAGE_BUDGETS.values()),
}


class LatencyTracker:
    def __init__(self):
        self.measurements: dict[str, list[float]] = {k: [] for k in STAGE_BUDGETS}

    def record(self, stage: str, elapsed_ms: float):
        if stage in self.measurements:
            self.measurements[stage].append(elapsed_ms)
            if len(self.measurements[stage]) > 1000:
                self.measurements[stage].pop(0)

    def report(self) -> dict:
        result = {}
        for stage, times in self.measurements.items():
            if not times:
                result[stage] = {"count": 0, "last_ms": None, "avg_ms": None, "min_ms": None, "max_ms": None}
                continue
            sorted_t = sorted(times)
            n = len(sorted_t)
            result[stage] = {
                "count": n,
                "last_ms": round(sorted_t[-1], 1),
                "avg_ms": round(sum(sorted_t) / n, 1),
                "min_ms": round(sorted_t[0], 1),
                "max_ms": round(sorted_t[-1], 1),
                "p50_ms": round(sorted_t[n // 2], 1),
                "p95_ms": round(sorted_t[int(n * 0.95)], 1),
                "p99_ms": round(sorted_t[int(n * 0.99)], 1),
            }
        return result

    def latest(self) -> dict[str, float]:
        return {s: t[-1] if t else None for s, t in self.measurements.items()}


tracker = LatencyTracker()
