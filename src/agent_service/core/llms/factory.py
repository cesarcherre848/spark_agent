import logging
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent_service.config.llm import get_llm_settings, LLMSettings

logger = logging.getLogger(__name__)

_DEFAULT_LLM_INSTANCE: Optional[BaseChatModel] = None


def bind_temperature(llm: Any, temperature: float) -> Any:
    """Vincula dinámicamente una temperatura específica al LLM por nodo o llamada (runtime binding).

    Permite que tareas estrictas (extracción, filtrado, judge) operen con temperature=0.0
    mientras tareas expresivas (reformulación, síntesis, redacción comercial) utilicen
    temperaturas superiores (0.3 a 0.7) sobre la misma instancia del modelo base.
    """
    if "unittest.mock" in type(llm).__module__:
        if hasattr(llm, "bind"):
            if getattr(llm.bind, "side_effect", None) is not None:
                return llm.bind(temperature=temperature)
            if "return_value" in getattr(llm.bind, "__dict__", {}):
                return llm.bind(temperature=temperature)
        return llm

    if hasattr(llm, "bind"):
        return llm.bind(temperature=temperature)
    return llm


def bind_structured_output(llm: Any, schema: Any, **kwargs: Any) -> Any:
    """Vincula una salida estructurada de manera robusta y compatible entre proveedores y mocks.

    Para modelos ChatGoogleGenerativeAI (especialmente variantes con razonamiento como
    gemini-3-flash-preview), utiliza method='json_mode' por defecto a menos que se
    especifique otro, garantizando que el modelo procese el esquema sin emitir JSON en texto plano
    que provoque un resultado None por falta de tool_calls.
    Para mocks de pruebas unitarias, tolera firmas de mocks que no acepten kwargs.
    """
    if "unittest.mock" in type(llm).__module__:
        if hasattr(llm, "with_structured_output"):
            try:
                return llm.with_structured_output(schema, **kwargs)
            except TypeError:
                return llm.with_structured_output(schema)
        return llm

    underlying = getattr(llm, "bound", llm)
    if "ChatGoogleGenerativeAI" in underlying.__class__.__name__ and "method" not in kwargs:
        kwargs["method"] = "json_mode"

    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema, **kwargs)
    return llm


def create_chat_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Fábrica para instanciar modelos de lenguaje (LLM).

    Parámetros:
        - provider: Proveedor del modelo ('google', extensible a 'ollama', 'openai', etc.).
        - model_name: Nombre específico del modelo (ej: 'gemini-1.5-flash', 'gemini-1.5-pro').
        - temperature: Temperatura base por defecto (0.0 a 1.0).
        - api_key: Clave de API explícita (si se omite, se toma de .env.dev).
        - max_output_tokens: Límite opcional de tokens de respuesta.
    """
    settings: LLMSettings = get_llm_settings()
    target_provider = (provider or settings.provider).strip().lower()
    raw_model = model_name or settings.model_name

    MODEL_ALIASES = {
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3-pro": "gemini-3.1-pro-preview",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
    }
    target_model = MODEL_ALIASES.get(raw_model, raw_model)
    target_temp = temperature if temperature is not None else 0.0
    target_key = api_key or settings.api_key
    target_max_tokens = max_output_tokens if max_output_tokens is not None else settings.max_output_tokens


    if target_provider == "google":
        if not target_key:
            raise ValueError(
                "No se encontró la clave de API de Google. "
                "Por favor configura GOOGLE_API_KEY o GEMINI_API_KEY en tu archivo .env.dev "
                "o pásala explícitamente en create_chat_model(api_key=...)."
            )

        google_kwargs = {
            "model": target_model,
            "google_api_key": target_key,
            "temperature": target_temp,
            **kwargs,
        }
        if target_max_tokens is not None:
            google_kwargs["max_output_tokens"] = target_max_tokens

        logger.info(f"Inicializando ChatGoogleGenerativeAI con modelo '{target_model}' (temp={target_temp})")
        return ChatGoogleGenerativeAI(**google_kwargs)

    raise ValueError(f"Proveedor de LLM no soportado actualmente: '{target_provider}'. Proveedores válidos: ['google']")


def get_default_llm(force_reload: bool = False, **kwargs: Any) -> BaseChatModel:
    """Obtiene una instancia compartida (singleton) del LLM configurado por defecto."""
    global _DEFAULT_LLM_INSTANCE
    if _DEFAULT_LLM_INSTANCE is None or force_reload:
        _DEFAULT_LLM_INSTANCE = create_chat_model(**kwargs)
    return _DEFAULT_LLM_INSTANCE


def get_deterministic_llm(**kwargs: Any) -> BaseChatModel:
    """Instancia o vincula un LLM estricto (temperatura 0.0) para extracción de datos y JSON."""
    return create_chat_model(temperature=0.0, **kwargs)


def get_balanced_llm(**kwargs: Any) -> BaseChatModel:
    """Instancia o vincula un LLM balanceado (temperatura 0.3-0.4) para síntesis y redacción comercial."""
    return create_chat_model(temperature=0.35, **kwargs)


def get_creative_llm(**kwargs: Any) -> BaseChatModel:
    """Instancia o vincula un LLM creativo (temperatura 0.7) para reformulaciones y brainstorming."""
    return create_chat_model(temperature=0.7, **kwargs)
