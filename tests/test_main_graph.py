import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent_service.graph.main_graph import (
    MainGraphState,
    RouterDecision,
    create_router_node,
    create_general_chat_node,
    _route_after_router,
    build_main_graph,
)
from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore


# ==============================================================================
# 1. FIXTURES & MOCKS
# ==============================================================================
@pytest.fixture
def mock_llm():
    """Mock de LLM para pruebas unitarias del Router y Chat General."""
    llm = MagicMock(spec=BaseChatModel)
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_memory_store():
    """Mock de UserMemoryStore para simular persistencia y recuperación."""
    store = MagicMock(spec=UserMemoryStore)
    store.search_memory = AsyncMock(return_value=[
        {"content": "Cotizó 5 unidades de SKU 1", "similarity": 0.82}
    ])
    store.add_memory = AsyncMock(return_value=101)
    return store


# ==============================================================================
# 2. TESTS DEL ROUTER DECISION SCHEMA Y ENRUTAMIENTO
# ==============================================================================
def test_router_decision_schema_valid():
    decision_general = RouterDecision(intent="general", reasoning="Es un saludo de cortesía")
    assert decision_general.intent == "general"

    decision_rag = RouterDecision(intent="rag", reasoning="Busca recomendaciones de catálogo")
    assert decision_rag.intent == "rag"

    decision_resolver = RouterDecision(intent="resolver", reasoning="Solicita cotización de SKU 1")
    assert decision_resolver.intent == "resolver"

    decision_contact = RouterDecision(intent="contact", reasoning="Pide ver su cartera de clientes")
    assert decision_contact.intent == "contact"

    decision_sales = RouterDecision(intent="sales", reasoning="Pide consultar o gestionar órdenes de venta")
    assert decision_sales.intent == "sales"

    decision_recommender = RouterDecision(intent="recommender", reasoning="Pide recomendaciones cruzadas o complementarias")
    assert decision_recommender.intent == "recommender"


def test_route_after_router_helper():
    assert _route_after_router({"intent": "rag"}) == "product_rag"
    assert _route_after_router({"intent": "recommender"}) == "product_recomender"
    assert _route_after_router({"intent": "resolver"}) == "product_resolver"
    assert _route_after_router({"intent": "contact"}) == "contact_manage"
    assert _route_after_router({"intent": "sales"}) == "sales_manage"
    assert _route_after_router({"intent": "general"}) == "general_chat"
    assert _route_after_router({}) == "general_chat"


@pytest.mark.asyncio
async def test_create_router_node_general(mock_llm):
    mock_llm.ainvoke.return_value = RouterDecision(
        intent="general",
        reasoning="El usuario saluda amablemente",
    )
    router_fn = create_router_node(mock_llm)

    state = {
        "raw_query": "¡Hola buenos días!",
        "messages": [HumanMessage(content="¡Hola buenos días!")],
        "user_id": 5,
    }

    result = await router_fn(state)
    assert result["intent"] == "general"
    assert "amablemente" in result["intent_reasoning"]
    assert result["raw_query"] == "¡Hola buenos días!"


@pytest.mark.asyncio
async def test_create_router_node_rag(mock_llm):
    mock_llm.ainvoke.return_value = RouterDecision(
        intent="rag",
        reasoning="El cliente busca recomendaciones de productos sin códigos específicos",
    )
    router_fn = create_router_node(mock_llm)

    state = {
        "raw_query": "¿Qué cremas faciales hidratantes tienen?",
        "messages": [HumanMessage(content="¿Qué cremas faciales hidratantes tienen?")],
        "user_id": 5,
    }

    result = await router_fn(state)
    assert result["intent"] == "rag"
    assert result["raw_query"] == "¿Qué cremas faciales hidratantes tienen?"


@pytest.mark.asyncio
async def test_create_router_node_resolver(mock_llm):
    mock_llm.ainvoke.return_value = RouterDecision(
        intent="resolver",
        reasoning="El cliente solicita cotización directa especificando SKU 1",
    )
    router_fn = create_router_node(mock_llm)

    state = {
        "raw_query": "Cotízame 5 unidades del SKU 1",
        "messages": [HumanMessage(content="Cotízame 5 unidades del SKU 1")],
        "user_id": 5,
    }

    result = await router_fn(state)
    assert result["intent"] == "resolver"


@pytest.mark.asyncio
async def test_create_router_node_contact(mock_llm):
    mock_llm.ainvoke.return_value = RouterDecision(
        intent="contact",
        reasoning="El vendedor solicita listar los clientes asignados a su cartera",
    )
    router_fn = create_router_node(mock_llm)

    state = {
        "raw_query": "muéstrame mis clientes en Odoo",
        "messages": [HumanMessage(content="muéstrame mis clientes en Odoo")],
        "user_id": 5,
    }

    result = await router_fn(state)
    assert result["intent"] == "contact"
    assert "cartera" in result["intent_reasoning"]
    assert result["raw_query"] == "muéstrame mis clientes en Odoo"


# ==============================================================================
# 3. TESTS DE GENERAL CHAT NODE
# ==============================================================================
@pytest.mark.asyncio
async def test_create_general_chat_node(mock_llm):
    mock_llm.ainvoke.return_value = AIMessage(
        content="¡Hola! Soy Spark Agent, tu asistente comercial de ERP. ¿En qué puedo ayudarte hoy?"
    )
    chat_fn = create_general_chat_node(mock_llm)

    state = {
        "raw_query": "Hola",
        "messages": [HumanMessage(content="Hola")],
        "user_context": "<antecedentes_usuario>- Prefiere Unique S.A.</antecedentes_usuario>",
    }

    result = await chat_fn(state)
    assert "Spark Agent" in result["final_response"]
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


# ==============================================================================
# 4. TESTS END-TO-END DEL GRAFO PRINCIPAL UNIFICADO
# ==============================================================================
@pytest.mark.asyncio
async def test_build_main_graph_flow_general(mock_memory_store):
    # Mock Router respondiendo "general"
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.bind = MagicMock(return_value=router_mock)
    router_mock.with_structured_output = MagicMock(return_value=router_mock)
    router_mock.ainvoke = AsyncMock(return_value=RouterDecision(
        intent="general",
        reasoning="Saludo general",
    ))

    # Mock Chat General respondiendo texto
    chat_mock = MagicMock(spec=BaseChatModel)
    chat_mock.bind = MagicMock(return_value=chat_mock)
    chat_mock.ainvoke = AsyncMock(return_value=AIMessage(
        content="¡Hola! ¿En qué puedo ayudarte hoy?"
    ))

    # Construimos subgrafos dummy para RAG y Resolver
    builder_rag = StateGraph(MainGraphState)
    builder_rag.add_node("rag_dummy", lambda s: {"final_response": "Respuesta RAG"})
    builder_rag.add_edge(START, "rag_dummy")
    builder_rag.add_edge("rag_dummy", END)
    dummy_rag = builder_rag.compile()

    builder_res = StateGraph(MainGraphState)
    builder_res.add_node("res_dummy", lambda s: {"final_response": "Respuesta Resolver", "memory_to_save": "Cotizó SKU 1"})
    builder_res.add_edge(START, "res_dummy")
    builder_res.add_edge("res_dummy", END)
    dummy_resolver = builder_res.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock if kw.get("temperature") == 0.0 else chat_mock)

    app = build_main_graph(
        llm=combined_llm,
        resolver_graph=dummy_resolver,
        rag_graph=dummy_rag,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-thread-general"}}
    result = await app.ainvoke(
        {
            "raw_query": "Hola, ¿qué tal?",
            "user_id": 5,
        },
        config=config,
    )

    # Verificaciones
    assert result["intent"] == "general"
    assert "¡Hola! ¿En qué puedo ayudarte hoy?" in result["final_response"]
    # Memoria recuperada presente
    assert result["user_context"] is not None
    mock_memory_store.search_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_main_graph_flow_rag(mock_memory_store):
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.bind = MagicMock(return_value=router_mock)
    router_mock.with_structured_output = MagicMock(return_value=router_mock)
    router_mock.ainvoke = AsyncMock(return_value=RouterDecision(
        intent="rag",
        reasoning="Búsqueda exploratoria en catálogo",
    ))

    builder_rag = StateGraph(MainGraphState)
    builder_rag.add_node(
        "rag_exec",
        lambda s: {
            "final_response": "Te recomiendo la Loción de Seda y el Desmaquillador Doble Fase.",
            "matched_skus": ["1", "11"],
        },
    )
    builder_rag.add_edge(START, "rag_exec")
    builder_rag.add_edge("rag_exec", END)
    dummy_rag = builder_rag.compile()

    dummy_res_builder = StateGraph(MainGraphState)
    dummy_res_builder.add_node("noop_res", lambda s: {})
    dummy_res_builder.add_edge(START, "noop_res")
    dummy_res_builder.add_edge("noop_res", END)
    dummy_resolver = dummy_res_builder.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock)

    app = build_main_graph(
        llm=combined_llm,
        resolver_graph=dummy_resolver,
        rag_graph=dummy_rag,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-thread-rag"}}
    result = await app.ainvoke(
        {
            "raw_query": "¿Qué cremas tienen?",
            "user_id": 5,
        },
        config=config,
    )

    assert result["intent"] == "rag"
    assert "Loción de Seda" in result["final_response"]
    assert result["matched_skus"] == ["1", "11"]


@pytest.mark.asyncio
async def test_build_main_graph_flow_resolver_and_save_memory(mock_memory_store):
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.bind = MagicMock(return_value=router_mock)
    router_mock.with_structured_output = MagicMock(return_value=router_mock)
    router_mock.ainvoke = AsyncMock(return_value=RouterDecision(
        intent="resolver",
        reasoning="Cotización directa de SKU",
    ))

    builder_res = StateGraph(MainGraphState)
    builder_res.add_node(
        "res_exec",
        lambda s: {
            "final_response": "Cotización: 2 unidades de SKU 1 con Unique S.A. Total: S/ 420.00",
            "memory_to_save": "Cotización realizada: 2 unidades del SKU 1 (Loción de Seda) con Unique S.A.",
        },
    )
    builder_res.add_edge(START, "res_exec")
    builder_res.add_edge("res_exec", END)
    dummy_resolver = builder_res.compile()

    dummy_rag_builder = StateGraph(MainGraphState)
    dummy_rag_builder.add_node("noop_rag", lambda s: {})
    dummy_rag_builder.add_edge(START, "noop_rag")
    dummy_rag_builder.add_edge("noop_rag", END)
    dummy_rag = dummy_rag_builder.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock)

    app = build_main_graph(
        llm=combined_llm,
        resolver_graph=dummy_resolver,
        rag_graph=dummy_rag,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-thread-resolver"}}
    result = await app.ainvoke(
        {
            "raw_query": "Cotízame 2 unidades del SKU 1",
            "user_id": 5,
            "session_id": "test-thread-resolver",
        },
        config=config,
    )

    assert result["intent"] == "resolver"
    assert "Total: S/ 420.00" in result["final_response"]
    # Verifica que save_memory persistió en el store
    assert result.get("saved_memory_id") == 101
    mock_memory_store.add_memory.assert_awaited_once_with(
        user_id=5,
        content="Cotización realizada: 2 unidades del SKU 1 (Loción de Seda) con Unique S.A.",
        session_id="test-thread-resolver",
    )


@pytest.mark.asyncio
async def test_build_main_graph_flow_contextual_pricing(mock_memory_store):
    """Verifica que consultas como '¿qué precio tienen?' tras una recomendación RAG
    se enruten a resolver conservando matched_skus e historial de mensajes."""
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.bind = MagicMock(return_value=router_mock)
    router_mock.with_structured_output = MagicMock(return_value=router_mock)
    router_mock.ainvoke = AsyncMock(return_value=RouterDecision(
        intent="resolver",
        reasoning="Consulta de precios de productos recomendados",
    ))

    builder_res = StateGraph(MainGraphState)
    builder_res.add_node(
        "res_exec",
        lambda s: {
            "final_response": f"Precios en Odoo ERP para SKUs {s.get('matched_skus')}: S/ 210.00",
            "items": {"1": {"attributes": {"cantidad": 1}}},
        },
    )
    builder_res.add_edge(START, "res_exec")
    builder_res.add_edge("res_exec", END)
    dummy_resolver = builder_res.compile()

    dummy_rag_builder = StateGraph(MainGraphState)
    dummy_rag_builder.add_node("noop_rag", lambda s: {})
    dummy_rag_builder.add_edge(START, "noop_rag")
    dummy_rag_builder.add_edge("noop_rag", END)
    dummy_rag = dummy_rag_builder.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock)

    app = build_main_graph(
        llm=combined_llm,
        resolver_graph=dummy_resolver,
        rag_graph=dummy_rag,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-contextual-pricing"}}
    result = await app.ainvoke(
        {
            "raw_query": "me parecen interesantes, ¿qué precio tienen?",
            "user_id": 5,
            "matched_skus": ["1", "11"],
            "messages": [
                HumanMessage(content="¿Qué cremas tienen?"),
                AIMessage(content="Te recomiendo la Loción de Seda (SKU: 1)."),
                HumanMessage(content="me parecen interesantes, ¿qué precio tienen?"),
            ],
        },
        config=config,
    )

    assert result["intent"] == "resolver"
    assert "Precios en Odoo ERP para SKUs ['1', '11']" in result["final_response"]


@pytest.mark.asyncio
async def test_build_main_graph_flow_contact(mock_memory_store):
    """Verifica que consultas relacionadas con la cartera de clientes se enruten al subgrafo contact_manage."""
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.bind = MagicMock(return_value=router_mock)
    router_mock.with_structured_output = MagicMock(return_value=router_mock)
    router_mock.ainvoke = AsyncMock(return_value=RouterDecision(
        intent="contact",
        reasoning="Petición para listar clientes asignados",
    ))

    builder_contact = StateGraph(MainGraphState)
    builder_contact.add_node(
        "contact_exec",
        lambda s: {
            "final_response": "Tienes 3 clientes registrados en tu cartera de Odoo.",
            "contact_action": "list",
        },
    )
    builder_contact.add_edge(START, "contact_exec")
    builder_contact.add_edge("contact_exec", END)
    dummy_contact = builder_contact.compile()

    dummy_resolver_builder = StateGraph(MainGraphState)
    dummy_resolver_builder.add_node("noop_res", lambda s: {})
    dummy_resolver_builder.add_edge(START, "noop_res")
    dummy_resolver_builder.add_edge("noop_res", END)
    dummy_resolver = dummy_resolver_builder.compile()

    dummy_rag_builder = StateGraph(MainGraphState)
    dummy_rag_builder.add_node("noop_rag", lambda s: {})
    dummy_rag_builder.add_edge(START, "noop_rag")
    dummy_rag_builder.add_edge("noop_rag", END)
    dummy_rag = dummy_rag_builder.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock)

    app = build_main_graph(
        llm=combined_llm,
        resolver_graph=dummy_resolver,
        rag_graph=dummy_rag,
        contact_graph=dummy_contact,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-contact-routing"}}
    result = await app.ainvoke(
        {
            "raw_query": "muéstrame mis clientes",
            "user_id": 5,
            "messages": [HumanMessage(content="muéstrame mis clientes")],
        },
        config=config,
    )

    assert result["intent"] == "contact"
    assert "Tienes 3 clientes registrados" in result["final_response"]


@pytest.mark.asyncio
async def test_main_graph_routes_to_sales_manage(mock_memory_store):
    """Verifica que el Router derive a sales_manage cuando la intención es 'sales'."""
    router_mock = MagicMock(spec=BaseChatModel)
    router_mock.with_structured_output = MagicMock(
        return_value=AsyncMock(
            ainvoke=AsyncMock(
                return_value=RouterDecision(
                    intent="sales",
                    reasoning="El usuario solicita ver sus órdenes de venta",
                )
            )
        )
    )

    builder_sales = StateGraph(MainGraphState)
    builder_sales.add_node(
        "sales_exec",
        lambda s: {
            "final_response": "Tienes 2 cotizaciones activas en Odoo.",
            "sales_action": "list",
        },
    )
    builder_sales.add_edge(START, "sales_exec")
    builder_sales.add_edge("sales_exec", END)
    dummy_sales = builder_sales.compile()

    combined_llm = MagicMock(spec=BaseChatModel)
    combined_llm.bind = MagicMock(side_effect=lambda **kw: router_mock)

    app = build_main_graph(
        llm=combined_llm,
        sales_graph=dummy_sales,
        memory_store=mock_memory_store,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-sales-routing"}}
    result = await app.ainvoke(
        {
            "raw_query": "muéstrame mis pedidos y cotizaciones",
            "user_id": 5,
            "messages": [HumanMessage(content="muéstrame mis pedidos y cotizaciones")],
        },
        config=config,
    )

    assert result["intent"] == "sales"
    assert "Tienes 2 cotizaciones activas" in result["final_response"]

