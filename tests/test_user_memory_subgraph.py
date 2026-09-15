import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore
from src.agent_service.graph.sub_graphs.user_memory.nodes import UserMemoryNodes
from src.agent_service.graph.sub_graphs.user_memory.graph import build_user_memory_graph
from src.agent_service.config.database import get_db_pool, close_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service


@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=Embeddings)
    # Vector dummy de dimensión 1024
    service.aembed_query = AsyncMock(return_value=[0.1] * 1024)
    return service


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = MagicMock()
    cur = AsyncMock()

    # Context managers asíncronos para pool.connection()
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pool.connection.return_value = conn_cm

    # Context manager asíncrono para conn.cursor()
    cur_cm = MagicMock()
    cur_cm.__aenter__ = AsyncMock(return_value=cur)
    cur_cm.__aexit__ = AsyncMock(return_value=None)
    conn.cursor.return_value = cur_cm

    return pool, cur


@pytest.mark.asyncio
async def test_user_memory_store_add_memory(mock_pool, mock_embedding_service):
    pool, cur = mock_pool
    cur.fetchone.return_value = (42,)

    store = UserMemoryStore(pool=pool, embedding_service=mock_embedding_service)
    mem_id = await store.add_memory(
        user_id=5,
        content="Cotizó 3 unidades de SKU 1 con Unique S.A.",
        session_id="session-123",
    )

    assert mem_id == 42
    mock_embedding_service.aembed_query.assert_awaited_once_with(
        "Cotizó 3 unidades de SKU 1 con Unique S.A."
    )
    cur.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_user_memory_store_search_memory_threshold_filtering(mock_pool, mock_embedding_service):
    pool, cur = mock_pool
    # Dos filas: una con similitud 0.85 (pasa) y otra con 0.40 (filtrada por threshold=0.55)
    cur.fetchall.return_value = [
        (1, "Cotizó 3 unidades de SKU 1", 0.85, "2026-09-14 10:00:00"),
        (2, "Preguntó por el clima", 0.40, "2026-09-14 09:00:00"),
    ]

    store = UserMemoryStore(pool=pool, embedding_service=mock_embedding_service)
    results = await store.search_memory(
        user_id=5,
        query="¿Qué cotizamos?",
        limit=5,
        threshold=0.55,
    )

    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["content"] == "Cotizó 3 unidades de SKU 1"
    assert results[0]["similarity"] == 0.85


@pytest.mark.asyncio
async def test_user_memory_nodes_retrieve():
    mock_store = MagicMock(spec=UserMemoryStore)
    mock_store.search_memory = AsyncMock(
        return_value=[
            {"id": 1, "content": "Prefiere envíos con Siderperu", "similarity": 0.80}
        ]
    )

    nodes = UserMemoryNodes(store=mock_store)
    state = {
        "user_id": 5,
        "raw_query": "¿Cuál es mi proveedor preferido?",
    }

    result = await nodes.retrieve_memory_node(state)
    assert result["user_context"] is not None
    assert "<antecedentes_usuario>" in result["user_context"]
    assert "Prefiere envíos con Siderperu" in result["user_context"]
    assert len(result["retrieved_memories"]) == 1


@pytest.mark.asyncio
async def test_user_memory_nodes_retrieve_empty_when_no_match():
    mock_store = MagicMock(spec=UserMemoryStore)
    mock_store.search_memory = AsyncMock(return_value=[])

    nodes = UserMemoryNodes(store=mock_store)
    state = {
        "user_id": 5,
        "raw_query": "Hola, buenos días",
    }

    result = await nodes.retrieve_memory_node(state)
    assert result["user_context"] is None
    assert result["retrieved_memories"] == []


@pytest.mark.asyncio
async def test_user_memory_nodes_save():
    mock_store = MagicMock(spec=UserMemoryStore)
    mock_store.add_memory = AsyncMock(return_value=99)

    nodes = UserMemoryNodes(store=mock_store)
    state = {
        "user_id": 5,
        "memory_to_save": "Cotización generada para 5 unidades SKU 2",
        "session_id": "thread-abc",
    }

    result = await nodes.save_memory_node(state)
    assert result["saved_memory_id"] == 99
    mock_store.add_memory.assert_awaited_once_with(
        user_id=5,
        content="Cotización generada para 5 unidades SKU 2",
        session_id="thread-abc",
    )


@pytest.mark.asyncio
async def test_user_memory_graph_execution():
    mock_store = MagicMock(spec=UserMemoryStore)
    mock_store.search_memory = AsyncMock(
        return_value=[{"id": 10, "content": "Factura a 30 días", "similarity": 0.9}]
    )
    mock_store.add_memory = AsyncMock(return_value=101)

    app = build_user_memory_graph(store=mock_store)

    # Caso 1: Consulta normal -> enruta a retrieve_memory
    res1 = await app.ainvoke({"user_id": 5, "raw_query": "condiciones de pago"})
    assert res1["user_context"] is not None
    assert "Factura a 30 días" in res1["user_context"]

    # Caso 2: Memoria a guardar -> enruta a save_memory
    res2 = await app.ainvoke({"user_id": 5, "memory_to_save": "Nuevo acuerdo de crédito"})
    assert res2["saved_memory_id"] == 101


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_memory_real_db_and_isolation():
    """Prueba de integración end-to-end con PostgreSQL real, pgvector y BAAI/bge-m3."""
    pool = get_db_pool()
    if pool.closed:
        await pool.open()

    embedding_service = get_embedding_service()
    store = UserMemoryStore(pool=pool, embedding_service=embedding_service)

    test_user_id = 5
    other_user_id = 2
    session_id = "test-session-int-01"
    content = "El cliente acordó cotizar 10 unidades de Loción de Seda con Unique S.A."

    inserted_id = None
    try:
        # 1. Guardar memoria real con embedding
        inserted_id = await store.add_memory(
            user_id=test_user_id,
            content=content,
            session_id=session_id,
        )
        assert inserted_id is not None

        # 2. Búsqueda semántica con consulta afín
        results = await store.search_memory(
            user_id=test_user_id,
            query="¿Cuántas unidades de loción acordamos con Unique?",
            limit=2,
            threshold=0.50,
        )
        assert len(results) >= 1
        assert "Loción de Seda" in results[0]["content"]
        assert results[0]["similarity"] >= 0.50

        # 3. Verificación de Aislamiento Multi-Usuario
        isolated_results = await store.search_memory(
            user_id=other_user_id,
            query="¿Cuántas unidades de loción acordamos con Unique?",
            limit=2,
            threshold=0.50,
        )
        # El user_id=2 no debe ver las memorias del user_id=5
        assert not any(r["id"] == inserted_id for r in isolated_results)

    finally:
        # Limpiar registro de prueba
        if inserted_id:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM user_memory WHERE id = %s;", (inserted_id,))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_checkpointer_integration():
    """Prueba de persistencia transaccional real usando AsyncPostgresSaver en PostgreSQL."""
    from src.agent_service.graph.sub_graphs.user_memory.checkpointer import get_postgres_checkpointer
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict

    class SimpleState(TypedDict):
        val: str

    pool = get_db_pool()
    if pool.closed:
        await pool.open()

    checkpointer = await get_postgres_checkpointer(pool=pool)
    assert checkpointer is not None

    workflow = StateGraph(SimpleState)
    workflow.add_node("step1", lambda s: {"val": s["val"] + " -> step1"})
    workflow.add_edge(START, "step1")
    workflow.add_edge("step1", END)
    app = workflow.compile(checkpointer=checkpointer)

    thread_id = "test-checkpointer-thread-01"
    config = {"configurable": {"thread_id": thread_id}}
    res = await app.ainvoke({"val": "init"}, config=config)
    assert res["val"] == "init -> step1"

    # Verificar que el checkpoint se guardó en PostgreSQL
    state = await app.aget_state(config)
    assert state.values["val"] == "init -> step1"
