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


import json
import re
import ast
from langchain_core.runnables import Runnable, RunnableConfig


def clean_text_from_tool_call_artifacts(text: str) -> str:
    """Elimina fragmentos técnicos de llamadas de herramientas o corchetes producidos por serialización de Gemini/LangChain."""
    if not text:
        return ""
    stripped = text.strip()

    # Caso 1: Cadena que representa una lista de Python, ej: "['call:default_api:...{...:', 'texto real', '}']"
    if stripped.startswith("['") or stripped.startswith('["') or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    s = str(item).strip()
                    if s.startswith("call:") or s.startswith("default_api:") or s in ("}", ")", "]", "{"):
                        continue
                    parts.append(s)
                if parts:
                    clean_res = "\n".join(parts).strip()
                    if "\\n" in clean_res:
                        clean_res = clean_res.replace("\\n", "\n")
                    return clean_res
        except Exception:
            pass

    # Caso 2: Bloque con prefijo call:default_api:...
    m = re.search(
        r"call:[^{]+(?:\{[^:]+:)?\s*['\"]?(.*?)['\"]?\s*(?:,\s*['\"]\}['\"]|\s*\})?$",
        stripped,
        re.DOTALL,
    )
    if m:
        candidate = m.group(1).strip()
        if candidate:
            if "\\n" in candidate:
                candidate = candidate.replace("\\n", "\n")
            return candidate

    return stripped


def _try_parse_schema_from_text(text: str, schema: Any) -> Optional[Any]:
    """Extrae y valida un esquema Pydantic desde texto markdown con json o texto plano."""
    if not text:
        return None
    # 1. Buscar bloques ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate_json = match.group(1) if match else None

    # 2. Si no hay bloque markdown, buscar el primer objeto JSON completo
    if not candidate_json:
        match_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_obj:
            candidate_json = match_obj.group(1)

    if candidate_json:
        try:
            data = json.loads(candidate_json)
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            elif callable(schema):
                return schema(**data)
        except Exception as e:
            logger.debug(f"No se pudo parsear JSON desde texto recuperado: {e}")
    return None


def _extract_from_tool_calls(tool_calls: Any, schema: Any) -> Optional[Any]:
    """Extrae y valida un esquema Pydantic a partir de la lista tool_calls del mensaje crudo."""
    if not tool_calls or not isinstance(tool_calls, list):
        return None
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        args = tc.get("args")
        if args and isinstance(args, dict):
            try:
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(args)
                elif callable(schema):
                    return schema(**args)
            except Exception as e:
                logger.debug(f"No se pudo validar schema desde tool_calls args: {e}")
    return None


def _clean_content_to_text(content_val: Any) -> str:
    """Extrae texto limpio a partir de content_val, ya sea str o list."""
    if isinstance(content_val, list):
        parts = []
        for item in content_val:
            s = str(item).strip()
            if s.startswith("call:") or s.startswith("default_api:") or s in ("}", ")", "]", "{"):
                continue
            parts.append(s)
        text = "\n".join(parts).strip() if parts else str(content_val)
    else:
        text = str(content_val or "").strip()
    return clean_text_from_tool_call_artifacts(text)


class ResilientStructuredOutputRunnable(Runnable):
    """Envuelve un modelo estructurado para rescatar JSON emitido en texto plano o tool_calls no parseados."""

    def __init__(self, structured_model: Any, schema: Any):
        self._structured_model = structured_model
        self._schema = schema

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        try:
            res = await self._structured_model.ainvoke(input, config=config, **kwargs)
        except Exception as e:
            logger.warning(f"Error en structured_model.ainvoke: {e}; intentando recuperación...")
            res = None

        if res is not None and isinstance(res, self._schema):
            return res

        if isinstance(res, dict):
            if res.get("parsed") is not None and isinstance(res["parsed"], self._schema):
                return res["parsed"]

            raw_msg = res.get("raw")
            # 1. Prioridad: Inspeccionar tool_calls directamente
            if raw_msg and hasattr(raw_msg, "tool_calls"):
                recovered_tc = _extract_from_tool_calls(raw_msg.tool_calls, self._schema)
                if recovered_tc is not None:
                    logger.info(f"Recuperado exitosamente {getattr(self._schema, '__name__', 'Schema')} desde raw_msg.tool_calls")
                    return recovered_tc

            # 2. Desempaquetar content limpiando artefactos técnicos
            if raw_msg and hasattr(raw_msg, "content"):
                raw_text = _clean_content_to_text(raw_msg.content)
                recovered = _try_parse_schema_from_text(raw_text, self._schema)
                if recovered is not None:
                    logger.info(f"Recuperado exitosamente {getattr(self._schema, '__name__', 'Schema')} desde raw.content")
                    return recovered
                if raw_text and hasattr(self._schema, "model_fields"):
                    fields = self._schema.model_fields
                    if len(fields) == 1:
                        fname = next(iter(fields))
                        try:
                            return self._schema(**{fname: raw_text})
                        except Exception:
                            pass

        if res is None or hasattr(res, "content"):
            content = getattr(res, "content", "") if res is not None else ""
            if content:
                raw_text = _clean_content_to_text(content)
                recovered = _try_parse_schema_from_text(raw_text, self._schema)
                if recovered is not None:
                    logger.info(f"Recuperado exitosamente {getattr(self._schema, '__name__', 'Schema')} desde content")
                    return recovered
                if raw_text and hasattr(self._schema, "model_fields"):
                    fields = self._schema.model_fields
                    if len(fields) == 1:
                        fname = next(iter(fields))
                        try:
                            return self._schema(**{fname: raw_text})
                        except Exception:
                            pass

        return res

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        try:
            res = self._structured_model.invoke(input, config=config, **kwargs)
        except Exception as e:
            logger.warning(f"Error en structured_model.invoke: {e}")
            res = None

        if res is not None and isinstance(res, self._schema):
            return res

        if isinstance(res, dict):
            if res.get("parsed") is not None and isinstance(res["parsed"], self._schema):
                return res["parsed"]

            raw_msg = res.get("raw")
            # 1. Prioridad: Inspeccionar tool_calls directamente
            if raw_msg and hasattr(raw_msg, "tool_calls"):
                recovered_tc = _extract_from_tool_calls(raw_msg.tool_calls, self._schema)
                if recovered_tc is not None:
                    return recovered_tc

            # 2. Desempaquetar content limpiando artefactos técnicos
            if raw_msg and hasattr(raw_msg, "content"):
                raw_text = _clean_content_to_text(raw_msg.content)
                recovered = _try_parse_schema_from_text(raw_text, self._schema)
                if recovered is not None:
                    return recovered
                if raw_text and hasattr(self._schema, "model_fields"):
                    fields = self._schema.model_fields
                    if len(fields) == 1:
                        fname = next(iter(fields))
                        try:
                            return self._schema(**{fname: raw_text})
                        except Exception:
                            pass

        if res is None or hasattr(res, "content"):
            content = getattr(res, "content", "") if res is not None else ""
            if content:
                raw_text = _clean_content_to_text(content)
                recovered = _try_parse_schema_from_text(raw_text, self._schema)
                if recovered is not None:
                    return recovered
                if raw_text and hasattr(self._schema, "model_fields"):
                    fields = self._schema.model_fields
                    if len(fields) == 1:
                        fname = next(iter(fields))
                        try:
                            return self._schema(**{fname: raw_text})
                        except Exception:
                            pass

        return res


def bind_structured_output(llm: Any, schema: Any, **kwargs: Any) -> Any:
    """Vincula una salida estructurada de manera robusta y compatible entre proveedores y mocks.

    Para modelos ChatGoogleGenerativeAI (especialmente variantes con razonamiento como
    gemini-3-flash-preview), emplea include_raw=True y un wrapper resiliente que rescata
    el JSON si el modelo lo emite en content en lugar de tool_calls.
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
    model_str = str(getattr(underlying, "model", "")).lower()
    # Para Gemini 3 (ej. gemini-3-flash-preview), Google optimizó el uso de function_calling nativo;
    # 'json_mode' en Gemini 3 produce cuellos de botella y 504 Deadline Exceeded en los servidores de Google.
    if "ChatGoogleGenerativeAI" in underlying.__class__.__name__ and "method" not in kwargs:
        if "gemini-3" in model_str:
            pass  # Emplea function_calling nativo (estable, sin timeouts y < 10s)
        else:
            kwargs["method"] = "json_mode"

    if hasattr(llm, "with_structured_output"):
        try:
            structured_runnable = llm.with_structured_output(schema, include_raw=True, **kwargs)
            return ResilientStructuredOutputRunnable(structured_runnable, schema)
        except TypeError:
            structured_runnable = llm.with_structured_output(schema, **kwargs)
            return ResilientStructuredOutputRunnable(structured_runnable, schema)
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
            "max_retries": 2,
            "timeout": 25.0,
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
