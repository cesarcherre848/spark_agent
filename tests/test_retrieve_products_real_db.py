import os
import pytest
import torch
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from langchain_huggingface import HuggingFaceEmbeddings
from unittest.mock import MagicMock
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.graph.sub_graphs.product_rag.nodes import ProductRagNodes

load_dotenv(".env.dev")


@pytest.fixture(scope="module")
def db_conninfo():
    host = os.getenv("PG_HOST") or os.getenv("ODOO_PG_HOST")
    user = os.getenv("PG_USER") or os.getenv("ODOO_PG_USER")
    password = os.getenv("PG_PASSWORD") or os.getenv("ODOO_PG_PASSWORD")
    dbname = os.getenv("PG_DATABASE") or os.getenv("ODOO_PG_DATABASE")
    port = os.getenv("PG_PORT") or os.getenv("ODOO_PG_PORT", "5432")

    if not all([host, user, password, dbname]):
        pytest.skip("Faltan variables de base de datos en .env.dev para el test real.")

    return f"host={host} port={port} user={user} password={password} dbname={dbname} connect_timeout=5"


@pytest.fixture(scope="module")
def embedding_service():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_products_with_real_database(db_conninfo, embedding_service):
    """Prueba de integración end-to-end contra PostgreSQL (pgvector + RRF) y BAAI/bge-m3."""
    pool = AsyncConnectionPool(conninfo=db_conninfo, min_size=1, max_size=2, open=False)
    await pool.open()

    try:
        vector_store = ProductVectorStore(
            pool=pool,
            embedding_service=embedding_service,
            table_name="product_vector_embedding",
            embedding_column="embedding_1024",
        )

        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.with_structured_output.return_value = MagicMock()

        nodes = ProductRagNodes(llm=mock_llm, vector_store=vector_store, top_k=20)

        state = {
            "raw_query": "Perfumes con aroma a rosas",
            "refined_query": None,
            "iteration_count": 0,
        }

        # Ejecutar retrieve_products contra la base de datos real
        result = await nodes.retrieve_products(state)

        print(result)

        # Verificaciones
        assert "retrieved_products" in result, "El resultado debe contener la clave 'retrieved_products'"
        docs = result["retrieved_products"]
        assert len(docs) > 0, "Se esperaba encontrar productos en la base de datos"
        assert result["iteration_count"] == 1, "iteration_count debe incrementarse a 1"

        # Verificar estructura de los documentos retornados
        for doc in docs:
            assert doc.page_content, "El documento debe contener contenido (page_content)"
            assert "sku" in doc.metadata, "La metadata debe contener el SKU del producto"
            assert "product_id" in doc.metadata, "La metadata debe contener el product_id"
            assert "rrf_score" in doc.metadata, "La metadata debe contener el puntaje RRF"
            assert doc.metadata["rrf_score"] > 0, "El score RRF debe ser mayor a 0"

    finally:
        await pool.close()
