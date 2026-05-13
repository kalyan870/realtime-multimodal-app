import os


class Settings:
    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
    NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")
    OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openrouter")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    STT_MODEL: str = os.environ.get("STT_MODEL", "openai/whisper-large-v3-turbo")
    TTS_MODEL: str = os.environ.get("TTS_MODEL", "espnet/kan-bayashi_ljspeech_vits")

    STT_TIMEOUT: int = int(os.environ.get("STT_TIMEOUT", "15"))
    LLM_TIMEOUT: int = int(os.environ.get("LLM_TIMEOUT", "20"))
    TTS_TIMEOUT: int = int(os.environ.get("TTS_TIMEOUT", "15"))
    PIPELINE_TIMEOUT: int = int(os.environ.get("PIPELINE_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "2"))

    ENABLE_STT: bool = os.environ.get("ENABLE_STT", "true").lower() == "true"
    ENABLE_TTS: bool = os.environ.get("ENABLE_TTS", "true").lower() == "true"
    ENABLE_LLM: bool = os.environ.get("ENABLE_LLM", "true").lower() == "true"


settings = Settings()
