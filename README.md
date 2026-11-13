# Real-Time Multimodal Voice Assistant

An interactive voice-and-text assistant built with a Streamlit UI and FastAPI backend. It can use Hugging Face speech recognition, an OpenRouter or NVIDIA language model, and gTTS speech synthesis.

## Overview

The current implementation accepts text or an uploaded audio file, optionally transcribes audio, sends the resulting prompt to a configured language model, and can return generated speech. Requests use a synchronous request/response flow; token streaming and retrieval are not implemented in the current pipeline.

## Core capabilities

- Text input and uploaded-audio interaction
- Optional speech-to-text through Hugging Face Inference
- LLM integration through OpenRouter or NVIDIA
- Optional speech output through gTTS
- Containerized local development with Docker Compose
- Latency reports and component-degradation status endpoints

## System workflow

1. Submit text or an audio file in Streamlit
2. Transcribe audio when speech recognition is enabled
3. Send the prompt to the configured language model
4. Optionally synthesize speech from the response
5. Return response text and optional generated audio

## Architecture

![Concept diagram for a retrieval-augmented voice assistant](rag%20for%20multimodel%20assistant%20voice.png)

![Voice assistant architecture](rag%20for%20multimodel%20voice%20assistant.png)

## Getting started

Clone the repository, configure the required model or service environment variables, and start the application with Docker Compose. Set `CORS_ALLOW_ORIGINS` to a comma-separated list of trusted browser origins when the UI is hosted away from the local Compose defaults (`http://localhost:8501,http://127.0.0.1:8501`). The Streamlit server calls the API server-side, so browser CORS is not required for that path. CORS is not authentication; do not expose the API publicly without an appropriate access-control layer.

Clone URL: https://github.com/kalyan870/realtime-multimodal-app.git
Run: docker compose up --build

Review the project configuration files before running the assistant and keep API keys out of source control.

## Tech focus

Voice Interfaces • Speech Recognition • OpenRouter and NVIDIA LLM APIs • Docker Compose • FastAPI

## Project status

This project is actively evolving as I improve response quality, component reliability, and the overall voice experience.

## Author

Built by [Kalyan](https://github.com/kalyan870).
