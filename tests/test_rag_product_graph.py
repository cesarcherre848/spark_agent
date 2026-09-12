import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.graph.sub_graphs.rag_product.graph import build_rag_product_graph
from src.agent_service.graph.sub_graphs.rag_product.schemas import (
    NormalizedQuery,
    EvaluationResult,
    QueryRefinementResult,
    FinalAnswer,
    format_candidates_for_prompt,
)


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=ProductVectorStore)
    store.ahybrid_search = AsyncMock()
    return store


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Esmalte de uñas en gel color rojo rubí de larga duración.",
            metadata={
                "product_id": 1,
                "sku": "ESM-ROJO-001",
                "name": "Esmalte Gel Rojo Rubí",
                "rrf_score": 0.95,
            },
        ),
        Document(
            page_content="Esmalte para uñas secado rápido acabado mate color nude.",
            metadata={
                "product_id": 2,
                "sku": "ESM-NUDE-002",
                "name": "Esmalte Mate Nude",
                "rrf_score": 0.82,
            },
        ),
    ]


def test_format_candidates_for_prompt(sample_documents):
    """Verifica que el helper de candidatos formatee de manera limpia sin filtrar SKUs al LLM."""
    result = format_candidates_for_prompt(sample_documents, max_desc_len=50)
    
    assert "[1] Esmalte Gel Rojo Rubí" in result
    assert "[2] Esmalte Mate Nude" in result
    assert "ESM-ROJO-001" not in result
    assert "ESM-NUDE-002" not in result


def test_format_candidates_for_prompt_empty():
    assert "No se encontraron" in format_candidates_for_prompt([])


@pytest.mark.asyncio
async def test_rag_product_graph_happy_path(mock_vector_store, sample_documents):
    """Test 1: Búsqueda exitosa a la primera (Happy Path).
    START -> normalize -> retrieve -> judge (is_sufficient=True) -> synthesize -> END
    """
    mock_vector_store.ahybrid_search.return_value = sample_documents

    # Mocks para structured outputs
    mock_normalizer = AsyncMock()
    mock_normalizer.ainvoke.return_value = NormalizedQuery(search_query="esmalte de uñas rojo")

    mock_judge = AsyncMock()
    mock_judge.ainvoke.return_value = EvaluationResult(
        is_sufficient=True,
        selected_indices=[1],
        critique="El producto [1] coincide exactamente con el esmalte rojo buscado.",
    )

    mock_refiner = AsyncMock()

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="Te recomendamos el Esmalte Gel Rojo Rubí, disponible con excelente cobertura."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == NormalizedQuery:
            return mock_normalizer
        elif schema == EvaluationResult:
            return mock_judge
        elif schema == QueryRefinementResult:
            return mock_refiner
        elif schema == FinalAnswer:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_rag_product_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        top_k=5,
        default_max_iterations=2,
    )

    initial_state = {
        "raw_query": "hola, busco algún esmalte rojo para uñas",
        "user_id": 10,
    }

    final_state = await app.ainvoke(initial_state)

    # Verificaciones
    assert final_state["refined_query"] == "esmalte de uñas rojo"
    assert final_state["is_sufficient"] is True
    assert final_state["iteration_count"] == 1
    # Extracción determinista de SKU en Python a partir del índice [1]
    assert final_state["matched_skus"] == ["ESM-ROJO-001"]
    assert "Esmalte Gel Rojo Rubí" in final_state["final_response"]

    mock_normalizer.ainvoke.assert_awaited_once()
    mock_judge.ainvoke.assert_awaited_once()
    mock_refiner.ainvoke.assert_not_awaited()
    mock_synthesizer.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_product_graph_reflection_loop(mock_vector_store, sample_documents):
    """Test 2: Ciclo de auto-reflexión (Loop 1).
    1a iteración: judge dice is_sufficient=False -> reflection_refined -> 2a iteración retrieve -> judge dice is_sufficient=True -> synthesize -> END
    """
    mock_vector_store.ahybrid_search.return_value = sample_documents

    mock_normalizer = AsyncMock()
    mock_normalizer.ainvoke.return_value = NormalizedQuery(search_query="algo para pintar")

    # El juez falla en la primera vuelta y aprueba en la segunda
    mock_judge = AsyncMock()
    mock_judge.ainvoke.side_effect = [
        EvaluationResult(
            is_sufficient=False,
            selected_indices=[],
            critique="La consulta 'algo para pintar' es muy ambigua.",
        ),
        EvaluationResult(
            is_sufficient=True,
            selected_indices=[2],
            critique="El producto [2] coincide con esmalte mate.",
        ),
    ]

    mock_refiner = AsyncMock()
    mock_refiner.ainvoke.return_value = QueryRefinementResult(
        refined_query="esmalte de uñas secado rápido mate"
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="Tenemos disponible el Esmalte Mate Nude de secado rápido."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == NormalizedQuery:
            return mock_normalizer
        elif schema == EvaluationResult:
            return mock_judge
        elif schema == QueryRefinementResult:
            return mock_refiner
        elif schema == FinalAnswer:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_rag_product_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        top_k=5,
        default_max_iterations=2,
    )

    initial_state = {
        "raw_query": "quiero algo para pintar",
        "user_id": 10,
    }

    final_state = await app.ainvoke(initial_state)

    # Verificaciones
    assert final_state["iteration_count"] == 2
    assert final_state["is_sufficient"] is True
    # Extraído deterministamente para el índice [2]
    assert final_state["matched_skus"] == ["ESM-NUDE-002"]
    assert "Esmalte Mate Nude" in final_state["final_response"]

    assert mock_vector_store.ahybrid_search.await_count == 2
    mock_refiner.ainvoke.assert_awaited_once()
    mock_synthesizer.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_product_graph_max_iterations_fallback(mock_vector_store, sample_documents):
    """Test 3: Límite de iteraciones seguras (evita bucle infinito).
    Si el juez siempre retorna is_sufficient=False, al llegar a max_iterations (2)
    se enruta a synthesize para generar una disculpa cordial al cliente.
    """
    mock_vector_store.ahybrid_search.return_value = sample_documents

    mock_normalizer = AsyncMock()
    mock_normalizer.ainvoke.return_value = NormalizedQuery(search_query="taladro percutor industrial")

    # El juez rechaza en ambas oportunidades
    mock_judge = AsyncMock()
    mock_judge.ainvoke.return_value = EvaluationResult(
        is_sufficient=False,
        selected_indices=[],
        critique="No vendemos herramientas de ferretería en el catálogo de cosméticos.",
    )

    mock_refiner = AsyncMock()
    mock_refiner.ainvoke.return_value = QueryRefinementResult(
        refined_query="herramientas industriales"
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="Lamentablemente en este momento no contamos con taladros ni herramientas en nuestro catálogo."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == NormalizedQuery:
            return mock_normalizer
        elif schema == EvaluationResult:
            return mock_judge
        elif schema == QueryRefinementResult:
            return mock_refiner
        elif schema == FinalAnswer:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_rag_product_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        top_k=5,
        default_max_iterations=2,
    )

    initial_state = {
        "raw_query": "necesito un taladro percutor",
        "user_id": 10,
    }

    final_state = await app.ainvoke(initial_state)

    # Verificaciones
    assert final_state["iteration_count"] == 2
    assert final_state["is_sufficient"] is False
    assert final_state["matched_skus"] == []
    assert "no contamos con taladros" in final_state["final_response"]
