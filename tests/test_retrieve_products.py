import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.schemas import ProductCatalogFilter
from src.agent_service.graph.sub_graphs.product_rag.nodes import ProductRagNodes


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    llm.with_structured_output.return_value = MagicMock()
    return llm


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=ProductVectorStore)
    store.ahybrid_search = AsyncMock()
    return store


@pytest.mark.asyncio
async def test_retrieve_products_uses_raw_query_when_no_refined_query(
    mock_llm, mock_vector_store
):
    # Given sample mock documents
    sample_docs = [
        Document(
            page_content="Laptop HP Pavilion 15.6",
            metadata={"product_id": 101, "sku": "HP-15-PAV", "name": "Laptop HP", "rrf_score": 0.95},
        ),
        Document(
            page_content="Laptop Lenovo ThinkPad E14",
            metadata={"product_id": 102, "sku": "THINK-E14", "name": "Laptop Lenovo", "rrf_score": 0.88},
        ),
    ]
    mock_vector_store.ahybrid_search.return_value = sample_docs

    nodes = ProductRagNodes(llm=mock_llm, vector_store=mock_vector_store, top_k=5)

    state = {
        "raw_query": "busco una laptop para oficina",
        "refined_query": None,
        "iteration_count": 0,
        "user_id": 5,
    }

    # When
    result = await nodes.retrieve_products(state)

    # Then
    assert "retrieved_products" in result
    assert len(result["retrieved_products"]) == 2
    assert result["retrieved_products"][0].metadata["sku"] == "HP-15-PAV"
    assert result["iteration_count"] == 1

    # Verify ahybrid_search was invoked with raw_query
    mock_vector_store.ahybrid_search.assert_awaited_once()
    called_kwargs = mock_vector_store.ahybrid_search.await_args.kwargs
    assert called_kwargs["query"] == "busco una laptop para oficina"
    assert called_kwargs["k"] == 5
    assert called_kwargs["alpha"] == 0.5
    assert isinstance(called_kwargs["filters"], ProductCatalogFilter)
    assert called_kwargs["filters"].user_id == 5


@pytest.mark.asyncio
async def test_retrieve_products_prioritizes_refined_query(
    mock_llm, mock_vector_store
):
    mock_vector_store.ahybrid_search.return_value = []
    nodes = ProductRagNodes(llm=mock_llm, vector_store=mock_vector_store, top_k=3)

    state = {
        "raw_query": "laptop barata",
        "refined_query": "laptop gama de entrada intel core i3 8gb ram",
        "iteration_count": 1,
    }

    # When
    result = await nodes.retrieve_products(state)

    # Then
    assert result["iteration_count"] == 2
    mock_vector_store.ahybrid_search.assert_awaited_once()
    called_kwargs = mock_vector_store.ahybrid_search.await_args.kwargs
    assert called_kwargs["query"] == "laptop gama de entrada intel core i3 8gb ram"
    assert called_kwargs["k"] == 3


@pytest.mark.asyncio
async def test_vector_store_ahybrid_search_parses_db_rows_correctly():
    """Verify that ProductVectorStore correctly builds Documents from SQL tuples."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = AsyncMock()

    # Configure async context managers for pool and connection
    mock_pool.connection.return_value.__aenter__.return_value = mock_conn
    mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

    # Simulated PostgreSQL return records from fused_candidates
    # Columns: id, product_id, product_tmpl_id, name, source_text, vendor_id, sku, rrf_score
    mock_cursor.fetchall.return_value = [
        (
            1,
            500,
            300,
            "Monitor Dell 27 Pulgadas",
            "Producto: Monitor Dell 27 Pulgadas IPS 75Hz",
            10,
            "DELL-M27",
            0.825,
        ),
    ]

    mock_embedding_service = AsyncMock()
    mock_embedding_service.aembed_query.return_value = [0.1, 0.2, 0.3]

    store = ProductVectorStore(
        pool=mock_pool,
        embedding_service=mock_embedding_service,
        table_name="product_catalog",
        embedding_column="embedding_1536",
    )

    filters = ProductCatalogFilter(user_id=5)
    results = await store.ahybrid_search(
        query="monitor para trabajo",
        k=2,
        alpha=0.7,
        filters=filters,
    )

    assert len(results) == 1
    doc = results[0]
    assert doc.page_content == "Producto: Monitor Dell 27 Pulgadas IPS 75Hz"
    assert doc.metadata["product_id"] == 500
    assert doc.metadata["sku"] == "DELL-M27"
    assert doc.metadata["vendor_id"] == 10
    assert doc.metadata["rrf_score"] == 0.825
