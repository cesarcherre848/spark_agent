import pytest
from unittest.mock import patch
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent_service.core.llms.factory import create_chat_model, get_default_llm
from src.agent_service.config.llm import LLMSettings


def test_create_google_model_with_explicit_key():
    """Verifica que la fábrica instancie ChatGoogleGenerativeAI con parámetros correctos."""
    model = create_chat_model(
        provider="google",
        model_name="gemini-1.5-flash",
        temperature=0.2,
        api_key="test_api_key_12345",
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "models/gemini-1.5-flash"
    assert model.temperature == 0.2


def test_create_google_model_custom_tokens():
    """Verifica personalización de modelo y tokens máximos."""
    model = create_chat_model(
        provider="google",
        model_name="gemini-1.5-pro",
        temperature=0.0,
        max_output_tokens=1000,
        api_key="test_api_key_12345",
    )

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "models/gemini-1.5-pro"
    assert model.max_output_tokens == 1000


def test_create_google_model_missing_key_raises_value_error():
    """Verifica que se lance un ValueError claro si no hay API key configurada."""
    with patch("src.agent_service.core.llms.factory.get_llm_settings") as mock_settings:
        mock_settings.return_value = LLMSettings(
            provider="google",
            model_name="gemini-1.5-flash",
            api_key=None,
        )

        with pytest.raises(ValueError, match="No se encontró la clave de API de Google"):
            create_chat_model(provider="google", api_key=None)


def test_create_model_unsupported_provider_raises_value_error():
    """Verifica que proveedores no soportados lancen ValueError."""
    with pytest.raises(ValueError, match="Proveedor de LLM no soportado"):
        create_chat_model(provider="unsupported_xyz", api_key="fake")


def test_get_default_llm_singleton():
    """Verifica el comportamiento singleton de get_default_llm."""
    llm1 = get_default_llm(api_key="test_key_singleton", force_reload=True)
    llm2 = get_default_llm()

    assert llm1 is llm2


def test_bind_temperature_per_call():
    """Verifica la vinculación dinámica de temperatura por llamada o nodo."""
    from src.agent_service.core.llms.factory import (
        bind_temperature,
        get_deterministic_llm,
        get_balanced_llm,
        get_creative_llm,
    )

    base_llm = create_chat_model(provider="google", api_key="test_key", temperature=0.0)

    # Vinculación por llamada (bind)
    bound_strict = bind_temperature(base_llm, 0.0)
    bound_creative = bind_temperature(base_llm, 0.7)

    assert bound_strict.kwargs["temperature"] == 0.0
    assert bound_creative.kwargs["temperature"] == 0.7

    # Presets
    det_llm = get_deterministic_llm(api_key="test_key")
    assert det_llm.temperature == 0.0

    bal_llm = get_balanced_llm(api_key="test_key")
    assert bal_llm.temperature == 0.35

    cre_llm = get_creative_llm(api_key="test_key")
    assert cre_llm.temperature == 0.7


def test_bind_structured_output():
    """Verifica que bind_structured_output configure method='json_mode' para Google y tolere mocks."""
    from pydantic import BaseModel
    from unittest.mock import MagicMock
    from src.agent_service.core.llms.factory import bind_structured_output

    class SampleSchema(BaseModel):
        field: str

    # 1. Comportamiento con ChatGoogleGenerativeAI
    base_llm = create_chat_model(provider="google", api_key="test_key")
    bound_structured = bind_structured_output(base_llm, SampleSchema)
    assert bound_structured is not None

    # 2. Tolerancia con mocks de unittest
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = "mock_output"
    res = bind_structured_output(mock_llm, SampleSchema)
    assert res == "mock_output"
    mock_llm.with_structured_output.assert_called_once_with(SampleSchema)


def test_clean_text_from_tool_call_artifacts():
    from src.agent_service.core.llms.factory import clean_text_from_tool_call_artifacts

    # Caso 1: Cadena representando lista de Gemini con call:default_api
    raw_sample = (
        "['call:default_api:RecommendationSynthesisResponse{response_text:', "
        "'He seleccionado para ti las opciones más versátiles.\\n\\n* [5441] Rosa', '}'"
        "]"
    )
    cleaned = clean_text_from_tool_call_artifacts(raw_sample)
    assert "call:default_api" not in cleaned
    assert "['" not in cleaned
    assert "He seleccionado para ti las opciones más versátiles.\n\n* [5441] Rosa" == cleaned

    # Caso 2: Texto normal no se ve alterado
    normal_text = "Hola, ¿cómo estás? Tenemos productos disponibles."
    assert clean_text_from_tool_call_artifacts(normal_text) == normal_text

    # Caso 3: Texto vacío o None
    assert clean_text_from_tool_call_artifacts("") == ""
    assert clean_text_from_tool_call_artifacts(None) == ""


@pytest.mark.asyncio
async def test_resilient_structured_output_recovers_from_tool_calls():
    from pydantic import BaseModel
    from unittest.mock import AsyncMock, MagicMock
    from src.agent_service.core.llms.factory import ResilientStructuredOutputRunnable

    class SynthSchema(BaseModel):
        response_text: str

    mock_runnable = AsyncMock()
    # Simula la respuesta de LangChain cuando Gemini emite tool_calls
    mock_msg = MagicMock()
    mock_msg.tool_calls = [
        {
            "name": "default_api:SynthSchema",
            "args": {"response_text": "Texto recuperado de tool_calls exitosamente."},
        }
    ]
    mock_msg.content = "['call:default_api:SynthSchema{response_text:', 'Texto recuperado', '}']"

    mock_runnable.ainvoke.return_value = {
        "raw": mock_msg,
        "parsed": None,
    }

    resilient = ResilientStructuredOutputRunnable(mock_runnable, SynthSchema)
    result = await resilient.ainvoke("input")

    assert isinstance(result, SynthSchema)
    assert result.response_text == "Texto recuperado de tool_calls exitosamente."


