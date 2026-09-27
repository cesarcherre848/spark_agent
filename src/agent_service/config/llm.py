import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env.qa")
load_dotenv(".env.dev")


@dataclass
class LLMSettings:
    """Configuración centralizada para modelos de lenguaje (LLM)."""
    provider: str = os.getenv("LLM_PROVIDER", "google")
    model_name: str = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
    request_timeout: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "60.0"))
    max_output_tokens: Optional[int] = (
        int(os.getenv("LLM_MAX_OUTPUT_TOKENS"))
        if os.getenv("LLM_MAX_OUTPUT_TOKENS")
        else None
    )
    api_key: Optional[str] = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )


def get_llm_settings() -> LLMSettings:
    """Obtiene la configuración actual de LLM desde las variables de entorno."""
    return LLMSettings()
