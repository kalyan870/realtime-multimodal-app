import base64
import io
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .pipeline.orchestrator import process_pipeline, get_latency_report, get_degradation_status
from .pipeline.latency_budget import STAGE_BUDGETS, STAGE_LABELS, TOTAL_BUDGET

app = FastAPI(title="Real-Time Multimodal App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "Real-Time Multimodal App",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "stt_enabled": settings.ENABLE_STT,
        "tts_enabled": settings.ENABLE_TTS,
    }


@app.get("/latency-budget")
def latency_budget():
    stages = []
    for key in STAGE_BUDGETS:
        stages.append({
            "id": key,
            "label": STAGE_LABELS.get(key, key),
            **STAGE_BUDGETS[key],
        })
    return {"stages": stages, "total": TOTAL_BUDGET}


@app.get("/latency-report")
def latency_report():
    return get_latency_report()


@app.get("/degradation")
def degradation_status():
    status = get_degradation_status()
    status["config"] = {
        "stt_timeout_s": settings.STT_TIMEOUT,
        "llm_timeout_s": settings.LLM_TIMEOUT,
        "tts_timeout_s": settings.TTS_TIMEOUT,
        "pipeline_timeout_s": settings.PIPELINE_TIMEOUT,
        "max_retries": settings.MAX_RETRIES,
    }
    return status


@app.post("/process")
async def process(file: UploadFile = File(None), text: str = Form(None)):
    audio_bytes = None
    if file and file.filename:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(status_code=413, content={"error": "File too large (max 10MB)"})
        audio_bytes = contents

    result = process_pipeline(audio_bytes=audio_bytes, text_input=text)
    return result
