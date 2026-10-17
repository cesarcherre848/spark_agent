"""
tests/test_context_judge.py - Pruebas unitarias para el Juez de Contexto (ContextJudgeNode) y TTL de 24h
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

from src.agent_service.graph.main_graph import (
    create_context_judge_node,
    create_router_node,
    ContextJudgement,
    RouterDecision,
    MainGraphState,
)
from src.agent_service.api.whatsapp.service import WhatsAppService
from src.agent_service.api.whatsapp.config import WhatsAppSettings


@pytest.mark.asyncio
async def test_context_judge_fastpath_no_history():
    """Si no hay historial previo (o solo la consulta actual), debe marcarse autónomo de inmediato."""
    mock_llm = MagicMock()
    judge_node = create_context_judge_node(mock_llm)

    state: MainGraphState = {
        "raw_query": "Agregar Cliente a Valeria Osores",
        "messages": [HumanMessage(content="Agregar Cliente a Valeria Osores")],
    }

    result = await judge_node(state)

    assert result["is_standalone"] is True
    assert result["context_usefulness"] == "irrelevant_or_obsolete"
    assert result["relevant_context_summary"] is None
    # No debió invocar el LLM (ahorro de latencia y costo)
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_context_judge_topic_change_standalone():
    """Cuando hay un cambio de tema o cliente ('Renzo' -> 'Agregar Cliente Valeria'), el Juez lo detecta como autónomo."""
    mock_llm = MagicMock()

    expected_judgement = ContextJudgement(
        is_standalone=True,
        context_usefulness="irrelevant_or_obsolete",
        relevant_context_summary=None,
        suggested_intent="contact",
    )

    with patch("src.agent_service.graph.main_graph.bind_structured_output") as mock_bind:
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected_judgement)
        mock_bind.return_value = mock_structured

        judge_node = create_context_judge_node(mock_llm)

        state: MainGraphState = {
            "raw_query": "Agregar Cliente a Valeria Osores",
            "messages": [
                HumanMessage(content="Hola, ¿cómo va el pedido de Renzo?"),
                AIMessage(content="El pedido SO001 de Renzo está confirmado."),
                HumanMessage(content="Agregar Cliente a Valeria Osores"),
            ],
        }

        result = await judge_node(state)

        assert result["is_standalone"] is True
        assert result["context_usefulness"] == "irrelevant_or_obsolete"
        assert result["relevant_context_summary"] is None


@pytest.mark.asyncio
async def test_context_judge_follow_up_dependent():
    """Preguntas de seguimiento ('¿y el segundo?') deben clasificarse como dependientes de contexto."""
    mock_llm = MagicMock()

    expected_judgement = ContextJudgement(
        is_standalone=False,
        context_usefulness="fully_useful",
        relevant_context_summary="El usuario pregunta por el precio del segundo perfume mostrado (Mithyka).",
        suggested_intent="advisor",
    )

    with patch("src.agent_service.graph.main_graph.bind_structured_output") as mock_bind:
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=expected_judgement)
        mock_bind.return_value = mock_structured

        judge_node = create_context_judge_node(mock_llm)

        state: MainGraphState = {
            "raw_query": "¿cuánto cuesta el segundo?",
            "messages": [
                HumanMessage(content="recomiéndame perfumes de L'Bel"),
                AIMessage(content="1. Live Intense\n2. Mithyka\n3. Bleu Intense"),
                HumanMessage(content="¿cuánto cuesta el segundo?"),
            ],
        }

        result = await judge_node(state)

        assert result["is_standalone"] is False
        assert result["context_usefulness"] == "fully_useful"
        assert "Mithyka" in (result["relevant_context_summary"] or "")


@pytest.mark.asyncio
async def test_context_judge_llm_exception_fallback():
    """Ante cualquier error inesperado del LLM, el Juez debe hacer fallback seguro a standalone."""
    mock_llm = MagicMock()

    with patch("src.agent_service.graph.main_graph.bind_structured_output") as mock_bind:
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("API timeout"))
        mock_bind.return_value = mock_structured

        judge_node = create_context_judge_node(mock_llm)

        state: MainGraphState = {
            "raw_query": "planes de incentivos",
            "messages": [
                HumanMessage(content="hola"),
                AIMessage(content="¡Hola!"),
                HumanMessage(content="planes de incentivos"),
            ],
        }

        result = await judge_node(state)

        assert result["is_standalone"] is True
        assert result["context_usefulness"] == "irrelevant_or_obsolete"
        assert result["relevant_context_summary"] is None


@pytest.mark.asyncio
async def test_router_node_uses_context_judge_verdict():
    """El router debe clasificar 'Agregar Cliente a Valeria Osores' como contact sin contaminarse con historial previo."""
    mock_llm = MagicMock()
    router_decision = RouterDecision(
        intent="contact",
        reasoning="Gestión de cartera: solicitud de agregar un nuevo cliente comercial.",
    )

    with patch("src.agent_service.graph.main_graph.bind_structured_output") as mock_bind:
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=router_decision)
        mock_bind.return_value = mock_structured

        router_node = create_router_node(mock_llm)

        state: MainGraphState = {
            "raw_query": "Agregar Cliente a Valeria Osores",
            "is_standalone": True,
            "context_usefulness": "irrelevant_or_obsolete",
            "relevant_context_summary": None,
            "messages": [
                HumanMessage(content="¿cómo va el pedido de Renzo?"),
                AIMessage(content="Pedido de Renzo pendiente"),
                HumanMessage(content="Agregar Cliente a Valeria Osores"),
            ],
        }

        result = await router_node(state)

        assert result["intent"] == "contact"
        assert result["router_model"] == "llm"
        # Verificar que la llamada al structured router no incluyó historial previo
        call_args = mock_structured.ainvoke.call_args[0][0]
        # Debe contener solo SystemMessage y HumanMessage (sin los mensajes de Renzo)
        assert len(call_args) == 2


@pytest.mark.asyncio
async def test_whatsapp_session_ttl_inactivity_purges_messages():
    """Verifica que si la sesión de WhatsApp superó 24h de inactividad, se purguen los mensajes con RemoveMessage."""
    mock_client = MagicMock()
    mock_client.mark_message_as_read = AsyncMock()
    mock_client.send_typing_indicator = AsyncMock()

    mock_resolver = MagicMock()
    mock_resolver.resolve_user_id = AsyncMock(return_value=10)

    mock_graph = MagicMock()
    # Simular un snapshot de más de 24 horas (ej: 2 días atrás)
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    mock_state_snapshot = MagicMock()
    mock_state_snapshot.created_at = old_timestamp
    mock_state_snapshot.tasks = []
    m1 = HumanMessage(content="mensaje viejo de hace 2 días", id="msg-1")
    mock_state_snapshot.values = {
        "messages": [m1],
        "active_customer": "Renzo",
    }

    mock_graph.aget_state = AsyncMock(return_value=mock_state_snapshot)
    mock_graph.aupdate_state = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={"final_response": "Respuesta fresca", "messages": []})

    settings = WhatsAppSettings(
        auto_reply=False,
        session_ttl_seconds=86400.0,
        delay_warning_seconds=0.0,
    )

    service = WhatsAppService(
        client=mock_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=settings,
    )

    await service.handle_incoming_message(
        sender_phone="51953320946",
        text="Agregar Cliente a Valeria Osores",
        message_id="wa_msg_123",
    )

    # Debe haber llamado a aupdate_state para purgar el mensaje viejo
    mock_graph.aupdate_state.assert_called_once()
    call_args = mock_graph.aupdate_state.call_args
    reset_dict = call_args[0][1]

    # Verificar que se envió RemoveMessage para msg-1
    assert "messages" in reset_dict
    assert len(reset_dict["messages"]) == 1
    assert isinstance(reset_dict["messages"][0], RemoveMessage)
    assert reset_dict["messages"][0].id == "msg-1"
    assert reset_dict["active_customer"] is None
