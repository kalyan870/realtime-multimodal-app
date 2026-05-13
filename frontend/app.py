import json
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:7860")

st.set_page_config(page_title="Multimodal Voice Assistant", page_icon="🎤", layout="wide")

st.title("🎤 Real-Time Multimodal Voice Assistant")
st.markdown("Speak or type — the pipeline transcribes, thinks, and speaks back.")

tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Voice / Text", "⏱️ Latency Budget", "📊 Live Metrics", "⚠️ Degradation"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        input_mode = st.radio("Input mode", ["Upload Audio", "Type Text"], horizontal=True)

        text_input = None
        audio_file = None

        if input_mode == "Upload Audio":
            audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])
            if audio_file:
                st.audio(audio_file, format="audio/wav")
        else:
            text_input = st.text_area("Type your message", placeholder="Ask anything...", height=100)

        submit = st.button("🚀 Process", type="primary", use_container_width=True)

    with col2:
        st.subheader("Response")
        response_placeholder = st.empty()
        audio_placeholder = st.empty()
        latency_placeholder = st.empty()

    if submit:
        if not audio_file and not text_input:
            st.warning("Provide audio or text input.")
        else:
            response_placeholder.info("Processing...")
            try:
                files = None
                data = None
                if audio_file:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                if text_input:
                    data = {"text": text_input}

                resp = requests.post(f"{API_URL}/process", files=files, data=data, timeout=45)
                result = resp.json()

                if result["status"] in ("success", "partial"):
                    response_placeholder.success(result.get("text_response", "No response"))

                    if result.get("audio_base64"):
                        import base64
                        audio_bytes = base64.b64decode(result["audio_base64"])
                        audio_placeholder.audio(audio_bytes, format="audio/mp3")

                    lat = result.get("latency_ms", {})
                    total = result.get("total_latency_ms", 0)
                    latency_placeholder.metric("Total Latency", f"{total:.0f} ms")

                    with st.expander("Latency Breakdown"):
                        for stage, label in [
                            ("audio_preprocess", "Audio Preprocessing"),
                            ("speech_to_text", "Speech-to-Text"),
                            ("llm_inference", "LLM Inference"),
                            ("text_to_speech", "Text-to-Speech"),
                            ("response_encode", "Response Encoding"),
                        ]:
                            val = lat.get(stage)
                            if val is not None:
                                st.metric(label, f"{val:.0f} ms" if val > 0 else "N/A")

                    if result.get("degraded_modes"):
                        for mode in result["degraded_modes"]:
                            st.warning(f"⚠️ {mode}")

                    if result.get("errors"):
                        for err in result["errors"]:
                            st.error(f"❌ {err}")
                else:
                    response_placeholder.error(result.get("error", "Unknown error"))
                    if result.get("degraded_modes"):
                        for mode in result["degraded_modes"]:
                            st.warning(f"⚠️ {mode}")
            except requests.exceptions.Timeout:
                response_placeholder.error("⏱️ Pipeline timed out. Check component timeouts.")
            except Exception as e:
                response_placeholder.error(f"Error: {e}")

with tab2:
    st.subheader("End-to-End Latency Budget")
    st.markdown("""
    Expected latency per stage under **P50 / P95 / P99** conditions.
    Total budget: **~3.4s P50 / ~10.7s P95 / ~21.4s P99**
    """)

    try:
        resp = requests.get(f"{API_URL}/latency-budget", timeout=5)
        budget_data = resp.json()
        stages = budget_data.get("stages", [])
        total = budget_data.get("total", {})

        cols = st.columns([3, 1, 1, 1, 3])
        cols[0].markdown("**Stage**")
        cols[1].markdown("**P50**")
        cols[2].markdown("**P95**")
        cols[3].markdown("**P99**")
        cols[4].markdown("**Description**")

        for s in stages:
            cols = st.columns([3, 1, 1, 1, 3])
            cols[0].markdown(s["label"])
            cols[1].markdown(f"{s['p50_ms']} ms")
            cols[2].markdown(f"{s['p95_ms']} ms")
            cols[3].markdown(f"{s['p99_ms']} ms")
            cols[4].markdown(s["description"])

        st.divider()
        tc = st.columns([3, 1, 1, 1, 3])
        tc[0].markdown("**Total**")
        tc[1].markdown(f"**{total['p50_ms']} ms**")
        tc[2].markdown(f"**{total['p95_ms']} ms**")
        tc[3].markdown(f"**{total['p99_ms']} ms**")
        tc[4].markdown("**End-to-end pipeline**")
    except Exception as e:
        st.warning(f"Cannot fetch budget (is backend running?): {e}")

    with st.expander("📐 Latency Budget Design Notes"):
        st.markdown("""
        ### Design Decisions
        - **Audio Preprocessing (≤50ms P50)**: WAV header validation, format check — CPU-only, near-zero
        - **Speech-to-Text (800ms P50)**: HF Inference API call to `whisper-large-v3-turbo` — network latency dominates
        - **LLM Inference (1.5s P50)**: NVIDIA Llama 3.1 70B via API — first-token latency + generation
        - **Text-to-Speech (1s P50)**: gTTS makes HTTP request to Google TTS — network + encoding
        - **Response Encoding (20ms P50)**: Base64 + JSON serialization — trivial

        ### Degradation Strategy
        | Failure | Fallback |
        |---------|----------|
        | STT fails | Text input mode |
        | LLM fails | Cached/generic response |
        | TTS fails | Text-only response |

        ### Timeout Handling
        - Per-component timeouts: STT 15s, LLM 20s, TTS 15s
        - Hard pipeline timeout: 30s
        - Circuit breaker: 3 consecutive failures disables component
        - Auto-recovery: 1 success reduces failure count
        """)

with tab3:
    st.subheader("Live Latency Metrics")
    try:
        report = requests.get(f"{API_URL}/latency-report", timeout=5).json()
        measurements = report.get("measurements", {})

        for stage, metrics in measurements.items():
            if metrics.get("count", 0) > 0:
                with st.container():
                    sc = st.columns(6)
                    sc[0].metric("Stage", stage)
                    sc[1].metric("Count", metrics["count"])
                    sc[2].metric("Avg", f'{metrics["avg_ms"]:.0f} ms')
                    sc[3].metric("P50", f'{metrics["p50_ms"]:.0f} ms')
                    sc[4].metric("P95", f'{metrics["p95_ms"]:.0f} ms')
                    sc[5].metric("P99", f'{metrics["p99_ms"]:.0f} ms')
    except Exception as e:
        st.warning(f"Cannot fetch metrics: {e}")

    if st.button("🔄 Refresh Metrics"):
        st.rerun()

with tab4:
    st.subheader("Graceful Degradation & Timeout Config")
    try:
        deg = requests.get(f"{API_URL}/degradation", timeout=5).json()
        sc = st.columns(4)
        sc[0].metric("STT Available", "✅" if deg.get("stt_available") else "❌")
        sc[1].metric("LLM Available", "✅" if deg.get("llm_available") else "❌")
        sc[2].metric("TTS Available", "✅" if deg.get("tts_available") else "❌")
        sc[3].metric("Max Failures", deg.get("max_failures_before_degradation", "?"))

        if deg.get("degraded_modes"):
            for mode in deg["degraded_modes"]:
                st.warning(f"⚠️ {mode}")
        else:
            st.success("✅ All components operational")

        st.divider()
        st.subheader("Timeout Configuration")
        cfg = deg.get("config", {})
        tc = st.columns(5)
        tc[0].metric("STT Timeout", f'{cfg.get("stt_timeout_s", "?")}s')
        tc[1].metric("LLM Timeout", f'{cfg.get("llm_timeout_s", "?")}s')
        tc[2].metric("TTS Timeout", f'{cfg.get("tts_timeout_s", "?")}s')
        tc[3].metric("Pipeline Timeout", f'{cfg.get("pipeline_timeout_s", "?")}s')
        tc[4].metric("Max Retries", cfg.get("max_retries", "?"))
    except Exception as e:
        st.warning(f"Cannot fetch degradation status: {e}")
