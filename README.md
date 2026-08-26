# Real-Time Multimodal Voice Assistant

A real-time multimodal assistant that combines voice interaction, retrieval-augmented generation, and OpenRouter-powered language models in a containerized application.

## Overview

This project explores a practical voice-first AI workflow: capture user input, process speech, retrieve relevant context, generate an answer, and stream the interaction back to the user.

## Core capabilities

- Real-time voice assistant interaction
- Multimodal input and response workflow
- LLM integration through OpenRouter
- Retrieval-Augmented Generation architecture
- Containerized local development with Docker Compose
- Streaming conversation experience

## System workflow

1. Capture user voice input
2. Convert and process the request
3. Retrieve relevant context
4. Send the enriched prompt to the language model
5. Stream the response back to the user
6. Continue the conversation with context

## Architecture

![RAG architecture](rag%20for%20multimodel%20assistant%20voice.png)

![Voice assistant architecture](rag%20for%20multimodel%20voice%20assistant.png)

## Getting started

Clone the repository, configure the required model or service environment variables, and start the application with Docker Compose.

Clone URL: https://github.com/kalyan870/realtime-multimodal-app.git
Run: docker compose up --build

Review the project configuration files before running the assistant and keep API keys out of source control.

## Tech focus

Multimodal AI • Voice Interfaces • OpenRouter LLMs • RAG • Docker Compose • Real-Time Applications

## Project status

This project is actively evolving as I improve response quality, retrieval, streaming behavior, and the overall voice experience.

## Author

Built by [Kalyan](https://github.com/kalyan870).
