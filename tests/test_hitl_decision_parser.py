import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.hitl.schemas import HITLBinaryDecision
from src.agent_service.core.hitl.parser import parse_hitl_binary_decision


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_accept(mock_llm):
    """Prueba que una confirmación afirmativa del LLM retorne True."""
    mock_llm.ainvoke.return_value = HITLBinaryDecision(
        decision="accept",
        reasoning="El usuario confirma y autoriza la operación.",
    )

    result = await parse_hitl_binary_decision(
        user_input="sí por favor, procede con la actualización",
        context_question="¿Deseas actualizar el cliente?",
        llm=mock_llm,
    )

    assert result is True
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_reject(mock_llm):
    """Prueba que una respuesta negativa del LLM retorne False."""
    mock_llm.ainvoke.return_value = HITLBinaryDecision(
        decision="reject",
        reasoning="El usuario prefiere no actualizar y cancelar la acción.",
    )

    result = await parse_hitl_binary_decision(
        user_input="para nada, mejor no hagas nada",
        context_question="¿Deseas actualizar el cliente?",
        llm=mock_llm,
    )

    assert result is False
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_dict_return(mock_llm):
    """Prueba compatibilidad cuando el LLM retorna un diccionario con la clave decision."""
    mock_llm.ainvoke.return_value = {"decision": "accept", "reasoning": "Confirmado"}

    result = await parse_hitl_binary_decision(
        user_input="dale, adelante",
        context_question="¿Confirmas?",
        llm=mock_llm,
    )

    assert result is True


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_empty_input(mock_llm):
    """Prueba que si el input del usuario está vacío o compuesto sólo de espacios, retorne False sin llamar al LLM."""
    result = await parse_hitl_binary_decision(
        user_input="   ",
        context_question="¿Confirmas?",
        llm=mock_llm,
    )

    assert result is False
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_none_input(mock_llm):
    """Prueba que si el input es None, retorne False de forma segura."""
    result = await parse_hitl_binary_decision(
        user_input=None,  # type: ignore[arg-type]
        context_question="¿Confirmas?",
        llm=mock_llm,
    )

    assert result is False
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_llm_exception_fallback(mock_llm):
    """Prueba que ante un error en la llamada al LLM, aplique fallback seguro (False)."""
    mock_llm.ainvoke.side_effect = RuntimeError("Connection timeout to LLM")

    result = await parse_hitl_binary_decision(
        user_input="sí",
        context_question="¿Confirmas?",
        llm=mock_llm,
    )

    assert result is False


@pytest.mark.asyncio
async def test_parse_hitl_binary_decision_unexpected_return_fallback(mock_llm):
    """Prueba que si el LLM retorna None o un objeto sin decision, retorne False."""
    mock_llm.ainvoke.return_value = None

    result = await parse_hitl_binary_decision(
        user_input="algo confuso",
        context_question="¿Confirmas?",
        llm=mock_llm,
    )

    assert result is False


@pytest.mark.asyncio
async def test_parse_hitl_entity_selection_select(mock_llm):
    from src.agent_service.core.hitl.parser import parse_hitl_entity_selection
    from src.agent_service.core.hitl.schemas import HITLEntitySelection

    mock_llm.ainvoke.return_value = HITLEntitySelection(
        action="select",
        selected_index=0,
        selected_entity_name="Carlos Pérez",
        reasoning="El usuario eligió la primera opción.",
    )

    result = await parse_hitl_entity_selection(
        user_input="al primero que me dijiste",
        options=["Carlos Pérez (Tel: 987654321)", "Carlos Perú SAC (Tel: 912345678)"],
        llm=mock_llm,
    )

    assert result.action == "select"
    assert result.selected_index == 0
    assert result.selected_entity_name == "Carlos Pérez"


@pytest.mark.asyncio
async def test_parse_hitl_order_choice_selected(mock_llm):
    from src.agent_service.core.hitl.parser import parse_hitl_order_choice
    from src.agent_service.core.hitl.schemas import HITLOrderChoice

    mock_llm.ainvoke.return_value = HITLOrderChoice(
        choice="selected",
        selected_order_name="SO002",
        reasoning="El usuario prefiere actualizar la orden SO002 existente.",
    )

    result = await parse_hitl_order_choice(
        user_input="actualiza la orden que teníamos abierta SO002",
        candidate_orders=["SO002 - Cotización por $500", "SO005 - Cotización por $120"],
        llm=mock_llm,
    )

    assert result.choice == "selected"
    assert result.selected_order_name == "SO002"


@pytest.mark.asyncio
async def test_parse_hitl_order_choice_new(mock_llm):
    from src.agent_service.core.hitl.parser import parse_hitl_order_choice
    from src.agent_service.core.hitl.schemas import HITLOrderChoice

    mock_llm.ainvoke.return_value = HITLOrderChoice(
        choice="new",
        selected_order_name=None,
        reasoning="El usuario desea abrir una nueva orden desde cero.",
    )

    result = await parse_hitl_order_choice(
        user_input="prefiero abrir una nueva orden desde cero",
        candidate_orders=["SO002 - Cotización por $500"],
        llm=mock_llm,
    )

    assert result.choice == "new"
