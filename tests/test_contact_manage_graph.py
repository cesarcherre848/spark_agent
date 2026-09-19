import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.agent_service.core.hitl.schemas import HITLBinaryDecision
from src.agent_service.graph.sub_graphs.contact_manage.graph import build_contact_manage_graph
from src.agent_service.graph.sub_graphs.contact_manage.schemas import (
    CustomerExtractionResult,
    DuplicateCheckResult,
    CustomerSynthesizeResponse,
)


@pytest.fixture
def mock_get_customers_tool():
    tool = AsyncMock()
    tool.return_value = [
        {"id": 14, "name": "Empresa Alfa", "phone": "987654321", "user_id": [5, "Cesar Cherre"]},
        {"id": 25, "name": "Distribuidora Beta", "phone": "912345678, 014567890", "user_id": [5, "Cesar Cherre"]},
    ]
    return tool


@pytest.fixture
def mock_list_current_customers_tool():
    tool = AsyncMock()
    tool.return_value = []
    return tool


@pytest.fixture
def mock_upsert_customer_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "created",
        "id": 88,
        "name": "Nuevo Cliente SAC",
        "phone": "999888777",
        "user_id": 5,
    }
    return tool


@pytest.fixture
def mock_remove_customer_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "archived",
        "id": 14,
        "name": "Empresa Alfa",
    }
    return tool


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_contact_manage_list_branch(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba la rama de listado de cartera del vendedor."""
    # 1. Extractor detecta list
    # 2. Synthesizer genera respuesta
    mock_llm.ainvoke.side_effect = [
        CustomerExtractionResult(action="list", name=None, phones=[]),
        CustomerSynthesizeResponse(response_text="Tienes 2 clientes en tu cartera: Empresa Alfa y Distribuidora Beta."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "muéstrame mis clientes",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_list"}}

    final_state = await graph.ainvoke(state, config=config)

    assert final_state["contact_action"] == "list"
    assert len(final_state["customers_list"]) == 2
    assert "Empresa Alfa" in final_state["final_response"]
    mock_get_customers_tool.assert_awaited_once_with(user_id=5, query=None, limit=15)


@pytest.mark.asyncio
async def test_contact_manage_upsert_branch_no_duplicates(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba el registro de un nuevo cliente cuando no existen duplicados en Odoo."""
    # list_current_customers no devuelve coincidencias
    mock_list_current_customers_tool.return_value = []

    mock_llm.ainvoke.side_effect = [
        # Extracción
        CustomerExtractionResult(action="upsert", name="Nuevo Cliente SAC", phones=["999888777"]),
        # Síntesis
        CustomerSynthesizeResponse(response_text="Cliente Nuevo Cliente SAC registrado con éxito."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "agrega al cliente Nuevo Cliente SAC con teléfono 999888777",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_upsert_no_dup"}}

    final_state = await graph.ainvoke(state, config=config)

    assert final_state["contact_action"] == "upsert"
    assert final_state["extracted_name"] == "Nuevo Cliente SAC"
    assert final_state["operation_result"]["success"] is True
    assert final_state["operation_result"]["id"] == 88
    mock_upsert_customer_tool.assert_awaited_once_with(
        user_id=5,
        name="Nuevo Cliente SAC",
        phones=["999888777"],
        contact_id=None,
    )


@pytest.mark.asyncio
async def test_contact_manage_upsert_branch_with_duplicates_hitl_accept(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba la rama upsert cuando se detecta un posible duplicado y el usuario confirma en lenguaje natural."""
    mock_list_current_customers_tool.return_value = [
        {"id": 42, "name": "Juan Perez", "phone": "987000111"}
    ]

    mock_llm.ainvoke.side_effect = [
        # Extracción
        CustomerExtractionResult(action="upsert", name="Juan Perez Garcia", phones=["987000111"]),
        # LLM as Judge confirma duplicado
        DuplicateCheckResult(has_duplicates=True, matched_customer_id=42, reasoning="Mismo teléfono y nombre similar."),
        # HITL Decision Parser interpreta respuesta en lenguaje natural como afirmativa
        HITLBinaryDecision(decision="accept", reasoning="El usuario confirma actualizar el registro."),
        # Síntesis
        CustomerSynthesizeResponse(response_text="Cliente Juan Perez (ID 42) actualizado con éxito."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "actualiza el teléfono de Juan Perez Garcia a 987000111",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_upsert_hitl_accept"}}

    # 1. Primera ejecución: debe pausar en feedback_user_duplicate_node
    paused_state = await graph.ainvoke(state, config=config)

    # Verificar interrupción HITL
    snapshot = await graph.aget_state(config)
    assert len(snapshot.tasks) > 0
    assert len(snapshot.tasks[0].interrupts) > 0
    interrupt_data = snapshot.tasks[0].interrupts[0].value
    assert interrupt_data["type"] == "duplicate_customer_conflict"
    assert interrupt_data["candidates"][0]["id"] == 42

    # 2. Reanudar con confirmación en lenguaje natural
    resumed_state = await graph.ainvoke(Command(resume="sí por favor, actualiza los datos"), config=config)

    assert resumed_state["upsert_confirmed"] is True
    mock_upsert_customer_tool.assert_awaited_once_with(
        user_id=5,
        name="Juan Perez Garcia",
        phones=["987000111"],
        contact_id=42,
    )


@pytest.mark.asyncio
async def test_contact_manage_upsert_branch_with_duplicates_hitl_reject(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba la rama upsert cuando se detecta duplicado y el usuario cancela la operación en lenguaje natural."""
    mock_list_current_customers_tool.return_value = [
        {"id": 42, "name": "Juan Perez", "phone": "987000111"}
    ]

    mock_llm.ainvoke.side_effect = [
        # Extracción
        CustomerExtractionResult(action="upsert", name="Juan Perez Garcia", phones=["987000111"]),
        # LLM as Judge confirma duplicado
        DuplicateCheckResult(has_duplicates=True, matched_customer_id=42, reasoning="Mismo teléfono y nombre similar."),
        # HITL Decision Parser interpreta respuesta en lenguaje natural como negativa
        HITLBinaryDecision(decision="reject", reasoning="El usuario declina la actualización."),
        # Síntesis
        CustomerSynthesizeResponse(response_text="Operación cancelada. No se modificó el cliente."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "actualiza a Juan Perez",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_upsert_hitl_reject"}}

    await graph.ainvoke(state, config=config)

    # Reanudar con negativa coloquial
    resumed_state = await graph.ainvoke(Command(resume="no gracias, mejor déjalo ahí"), config=config)

    assert resumed_state["upsert_confirmed"] is False
    # No se debió ejecutar el upsert
    mock_upsert_customer_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_contact_manage_remove_branch_hitl_confirm(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba la rama remove con confirmación HITL en lenguaje natural."""
    mock_llm.ainvoke.side_effect = [
        # Extracción
        CustomerExtractionResult(action="remove", name="Empresa Alfa", contact_id=14),
        # HITL Decision Parser interpreta confirmación
        HITLBinaryDecision(decision="accept", reasoning="Usuario confirma eliminación"),
        # Síntesis tras confirmación
        CustomerSynthesizeResponse(response_text="El cliente Empresa Alfa (ID 14) ha sido archivado correctamente."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "elimina a Empresa Alfa de mis clientes",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_remove_hitl_confirm"}}

    # 1. Primera ejecución: debe pausar en feedback_user_remove_node
    await graph.ainvoke(state, config=config)
    snapshot = await graph.aget_state(config)
    assert len(snapshot.tasks[0].interrupts) > 0
    assert snapshot.tasks[0].interrupts[0].value["type"] == "remove_customer_confirmation"

    # 2. Reanudar con confirmación coloquial
    resumed_state = await graph.ainvoke(Command(resume="sí, dale procede a eliminarlo"), config=config)

    assert resumed_state["remove_confirmed"] is True
    mock_remove_customer_tool.assert_awaited_once_with(
        contact_id=14,
        user_id=5,
    )
    assert "archivado correctamente" in resumed_state["final_response"]


@pytest.mark.asyncio
async def test_contact_manage_remove_branch_hitl_reject(
    mock_llm,
    mock_get_customers_tool,
    mock_list_current_customers_tool,
    mock_upsert_customer_tool,
    mock_remove_customer_tool,
):
    """Prueba la rama remove cuando el usuario cancela la eliminación en lenguaje natural."""
    mock_llm.ainvoke.side_effect = [
        # Extracción
        CustomerExtractionResult(action="remove", name="Empresa Alfa", contact_id=14),
        # HITL Decision Parser interpreta rechazo
        HITLBinaryDecision(decision="reject", reasoning="Usuario declina eliminación"),
        # Síntesis tras cancelación
        CustomerSynthesizeResponse(response_text="Operación cancelada. El cliente no fue eliminado."),
    ]

    graph = build_contact_manage_graph(
        llm=mock_llm,
        get_customers_tool=mock_get_customers_tool,
        list_current_customers_tool=mock_list_current_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
        remove_customer_tool=mock_remove_customer_tool,
        checkpointer=MemorySaver(),
    )

    state = {
        "raw_query": "elimina a Empresa Alfa",
        "user_id": 5,
    }
    config = {"configurable": {"thread_id": "thread_test_remove_hitl_reject"}}

    # 1. Primera ejecución: pausar
    await graph.ainvoke(state, config=config)

    # 2. Reanudar con negativa
    resumed_state = await graph.ainvoke(Command(resume="no, para nada, no lo borres"), config=config)

    assert resumed_state["remove_confirmed"] is False
    mock_remove_customer_tool.assert_not_awaited()

