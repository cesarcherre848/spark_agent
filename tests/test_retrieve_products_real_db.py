import os
import pytest
import torch
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from langchain_huggingface import HuggingFaceEmbeddings

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.schemas import ProductCatalogFilter

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

        # Ejecutar búsqueda híbrida contra la base de datos real
        docs = await vector_store.ahybrid_search(
            query="Perfumes con aroma a rosas",
            k=20,
        )

        # Verificaciones
        assert len(docs) > 0, "Se esperaba encontrar productos en la base de datos"

        # Verificar estructura de los documentos retornados
        for doc in docs:
            assert doc.page_content, "El documento debe contener contenido (page_content)"
            assert "sku" in doc.metadata, "La metadata debe contener el SKU del producto"
            assert "product_id" in doc.metadata, "La metadata debe contener el product_id"
            assert "rrf_score" in doc.metadata, "La metadata debe contener el puntaje RRF"
            assert doc.metadata["rrf_score"] > 0, "El score RRF debe ser mayor a 0"

    finally:
        await pool.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_products_with_metadata_tags_real_db(db_conninfo, embedding_service):
    """Prueba de integración real filtrando por marca Yanbal y edición C10 en la vista de metadatos."""
    pool = AsyncConnectionPool(conninfo=db_conninfo, min_size=1, max_size=2, open=False)
    await pool.open()

    try:
        vector_store = ProductVectorStore(
            pool=pool,
            embedding_service=embedding_service,
            table_name="product_vector_embedding",
            embedding_column="embedding_1024",
        )

        catalog_filter = ProductCatalogFilter(
            marca="Yanbal",
            edicion="C10",
        )

        docs = await vector_store.ahybrid_search(
            query="labial",
            k=5,
            filters=catalog_filter,
        )
        assert len(docs) > 0, "Se esperaba encontrar productos de Yanbal C10"
        for doc in docs:
            assert doc.metadata.get("marca") == "Yanbal"
            assert doc.metadata.get("edicion") == "C10"
            assert "tags" in doc.metadata
            assert doc.metadata["tags"]["marca"] == "Yanbal"
            assert doc.metadata["tags"]["edicion"] == "C10"
    finally:
        await pool.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_products_esika_brand_variations_real_db(db_conninfo, embedding_service):
    """Prueba que la búsqueda de Ésika funciona con tilde, sin tilde ('esika') y con doble s ('essika')."""
    pool = AsyncConnectionPool(conninfo=db_conninfo, min_size=1, max_size=2, open=False)
    await pool.open()

    try:
        vector_store = ProductVectorStore(
            pool=pool,
            embedding_service=embedding_service,
            table_name="product_vector_embedding",
            embedding_column="embedding_1024",
        )

        for variant in ["Ésika", "esika", "essika"]:
            catalog_filter = ProductCatalogFilter(
                marca=variant,
                user_id=5,
            )
            docs = await vector_store.ahybrid_search(
                query="labiales mate",
                k=5,
                filters=catalog_filter,
            )
            assert len(docs) > 0, f"Se esperaba encontrar labiales de Ésika usando variante: '{variant}'"
            for doc in docs:
                assert doc.metadata.get("marca") == "Ésika"
                assert "tags" in doc.metadata
                assert doc.metadata["tags"]["marca"] == "Ésika"
            assert any("labial" in doc.page_content.lower() or "mate" in doc.page_content.lower() for doc in docs)

            # Confirmar que el SKU 06780 de Ésika está presente entre los primeros candidatos
            skus = [d.metadata.get("sku") for d in docs]
            assert "06780" in skus, f"El SKU 06780 (COLORFIX LABIAL) debe encontrarse para variante '{variant}'"
    finally:
        await pool.close()

