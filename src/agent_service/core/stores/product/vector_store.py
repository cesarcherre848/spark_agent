import re
from typing import Any, Dict, List, Optional
from psycopg_pool import AsyncConnectionPool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.agent_service.core.stores.product.schemas import ProductCatalogFilter


def normalize_brand(brand: Optional[str]) -> Optional[str]:
    """Normaliza variantes comunes, acentos y errores ortográficos a las marcas canónicas del ERP."""
    if not brand:
        return None
    b = brand.strip().lower()
    if re.search(r"\b(es+ika|ésika)\b", b):
        return "Ésika"
    if re.search(r"\byanbal\b", b):
        return "Yanbal"
    if re.search(r"\b(belcorp|cyzone|l'?bel)\b", b):
        return b.title()
    return brand.strip()


_HYBRID_QUERY_SQL = """
    WITH semantic_search AS (
        SELECT 
            emb.id,
            ROW_NUMBER() OVER (ORDER BY emb.{embedding_column} <=> %(query_embedding)s::vector) AS rank_sem
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        WHERE emb.active = TRUE
          AND (%(user_id)s::integer IS NULL OR vuap.user_id IS NOT NULL)
          AND (%(pagina)s::integer IS NULL OR vuap.pagina = %(pagina)s::integer)
          AND (%(edicion)s::text IS NULL OR lower(vuap.edicion) = lower(%(edicion)s::text))
          AND (
              %(marca)s::text IS NULL 
              OR lower(vuap.marca) = lower(%(marca)s::text)
              OR position(translate(lower(%(marca)s::text), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou') in translate(lower(vuap.marca), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou')) > 0
              OR (%(marca)s::text ~* '^es+ika' AND vuap.marca ~* '^es+ika|^ésika')
          )
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
        ORDER BY emb.{embedding_column} <=> %(query_embedding)s::vector
        LIMIT %(k_candidates)s
    ),
    lexical_search AS (
        SELECT 
            emb.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(
                    to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')),
                    websearch_to_tsquery('spanish', %(query_text)s)
                ) DESC
            ) AS rank_lex
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        WHERE emb.active = TRUE
          AND (%(user_id)s::integer IS NULL OR vuap.user_id IS NOT NULL)
          AND (%(pagina)s::integer IS NULL OR vuap.pagina = %(pagina)s::integer)
          AND (%(edicion)s::text IS NULL OR lower(vuap.edicion) = lower(%(edicion)s::text))
          AND (
              %(marca)s::text IS NULL 
              OR lower(vuap.marca) = lower(%(marca)s::text)
              OR position(translate(lower(%(marca)s::text), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou') in translate(lower(vuap.marca), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou')) > 0
              OR (%(marca)s::text ~* '^es+ika' AND vuap.marca ~* '^es+ika|^ésika')
          )
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
          AND to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')) 
              @@ websearch_to_tsquery('spanish', %(query_text)s)
        LIMIT %(k_candidates)s
    ),
    fused_candidates AS (
        SELECT DISTINCT ON (emb.id)
            emb.id,
            emb.product_id,
            emb.product_tmpl_id,
            emb.name,
            emb.source_text,
            vuap.vendor_id,
            vuap.sku,
            vuap.pagina,
            vuap.edicion,
            vuap.marca,
            (
                COALESCE(%(alpha)s / (60.0 + sem.rank_sem), 0.0) +
                COALESCE((1.0 - %(alpha)s) / (60.0 + lex.rank_lex), 0.0)
            ) AS rrf_score
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        LEFT JOIN semantic_search sem ON sem.id = emb.id
        LEFT JOIN lexical_search lex ON lex.id = emb.id
        WHERE (sem.id IS NOT NULL OR lex.id IS NOT NULL)
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
        ORDER BY emb.id
    )
    SELECT 
        id,
        product_id,
        product_tmpl_id,
        name,
        source_text,
        vendor_id,
        sku,
        rrf_score,
        pagina,
        edicion,
        marca
    FROM fused_candidates
    ORDER BY rrf_score DESC
    LIMIT %(k)s;
"""


class ProductVectorStore(VectorStore):
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_service: Embeddings,
        table_name: str = "product_vector_embedding",
        embedding_column: str = "embedding_1024",
        ai_provider: Optional[str] = "huggingface",
        ai_model: Optional[str] = "BAAI/bge-m3",
    ):
        self._pool = pool
        self._embedding_service = embedding_service
        self._table_name = table_name
        self._embedding_column = embedding_column
        self._ai_provider = ai_provider
        self._ai_model = ai_model

        # Extraer dimensión de la columna (ej. '1536' de 'embedding_1536')
        dim_str = "".join(filter(str.isdigit, embedding_column))
        self._embedding_dim = int(dim_str) if dim_str else 1024

        # Compilación segura usando la plantilla definida
        self._compiled_query = _HYBRID_QUERY_SQL.format(
            table_name=self._table_name,
            embedding_column=self._embedding_column,
        )

    async def ahybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7,
        filters: Optional[ProductCatalogFilter] = None,
    ) -> List[Document]:
        # Normalizar consulta a minúsculas y espacios limpios para evitar fragmentación de sub-tokens
        clean_query = " ".join((query or "").lower().split())

        # Generar embedding o crear vector dummy con dimensión exacta si alpha == 0.0
        if alpha > 0.0:
            query_embedding = await self._embedding_service.aembed_query(clean_query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
        else:
            embedding_str = f"[{','.join(['0.0'] * self._embedding_dim)}]"

        target_user_id = filters.user_id if filters else None
        target_pagina = (
            filters.pagina
            if (filters and filters.pagina is not None)
            else (filters.metadata.pagina if (filters and filters.metadata) else None)
        )
        target_edicion = (
            filters.edicion
            if (filters and filters.edicion is not None)
            else (filters.metadata.edicion if (filters and filters.metadata) else None)
        )
        raw_marca = (
            filters.marca
            if (filters and filters.marca is not None)
            else (filters.metadata.marca if (filters and filters.metadata) else None)
        )
        target_marca = normalize_brand(raw_marca)

        params: Dict[str, Any] = {
            "query_embedding": embedding_str,
            "query_text": clean_query,
            "user_id": target_user_id,
            "pagina": target_pagina,
            "edicion": target_edicion,
            "marca": target_marca,
            "alpha": alpha,
            "k_candidates": max(k * 4, 20),
            "k": k,
            "ai_provider": self._ai_provider,
            "ai_model": self._ai_model,
        }

        if getattr(self._pool, "closed", False) is True:
            await self._pool.open()

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self._compiled_query, params)
                records = await cur.fetchall()

        results: List[Document] = []
        for row in records:
            pagina_val = row[8] if len(row) > 8 else None
            edicion_val = row[9] if len(row) > 9 else None
            marca_val = row[10] if len(row) > 10 else None
            results.append(
                Document(
                    page_content=row[4] or row[3] or "",
                    metadata={
                        "embedding_id": row[0],
                        "product_id": row[1],
                        "product_tmpl_id": row[2],
                        "name": row[3],
                        "vendor_id": row[5],
                        "sku": row[6],
                        "rrf_score": float(row[7]),
                        "pagina": pagina_val,
                        "edicion": edicion_val,
                        "marca": marca_val,
                        "tags": {
                            "pagina": pagina_val,
                            "edicion": edicion_val,
                            "marca": marca_val,
                        },
                    },
                )
            )

        return results

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return await self.ahybrid_search(
            query=query, 
            k=k, 
            alpha=1.0, 
            filters=kwargs.get("filters")
        )

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        raise NotImplementedError("Usar métodos asíncronos (ahybrid_search o asimilarity_search).")

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, **kwargs: Any):
        raise NotImplementedError("Poblado mediante pipeline dedicado.")