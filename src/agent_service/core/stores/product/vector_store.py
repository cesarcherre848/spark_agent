from typing import Any, Dict, List, Optional
from psycopg_pool import AsyncConnectionPool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.agent_service.core.stores.product.schemas import ProductCatalogFilter


_HYBRID_QUERY_SQL = """
    WITH semantic_search AS (
        SELECT 
            emb.id,
            ROW_NUMBER() OVER (ORDER BY emb.{embedding_column} <=> %(query_embedding)s::vector) AS rank_sem
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products vuap 
          ON vuap.product_id = emb.product_id 
         AND vuap.user_id = %(user_id)s
        WHERE emb.active = TRUE
          AND (%(user_id)s IS NULL OR vuap.user_id IS NOT NULL)
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
        LEFT JOIN view_user_authorized_products vuap 
          ON vuap.product_id = emb.product_id 
         AND vuap.user_id = %(user_id)s
        WHERE emb.active = TRUE
          AND (%(user_id)s IS NULL OR vuap.user_id IS NOT NULL)
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
            (
                COALESCE(%(alpha)s / (60.0 + sem.rank_sem), 0.0) +
                COALESCE((1.0 - %(alpha)s) / (60.0 + lex.rank_lex), 0.0)
            ) AS rrf_score
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products vuap 
          ON vuap.product_id = emb.product_id 
         AND vuap.user_id = %(user_id)s
        LEFT JOIN semantic_search sem ON sem.id = emb.id
        LEFT JOIN lexical_search lex ON lex.id = emb.id
        WHERE sem.id IS NOT NULL OR lex.id IS NOT NULL
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
        rrf_score
    FROM fused_candidates
    ORDER BY rrf_score DESC
    LIMIT %(k)s;
"""


class ProductVectorStore(VectorStore):
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_service: Embeddings,
        table_name: str = "product_catalog",
        embedding_column: str = "embedding_1536",
    ):
        self._pool = pool
        self._embedding_service = embedding_service
        self._table_name = table_name
        self._embedding_column = embedding_column

        # Extraer dimensión de la columna (ej. '1536' de 'embedding_1536')
        dim_str = "".join(filter(str.isdigit, embedding_column))
        self._embedding_dim = int(dim_str) if dim_str else 1536

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
        # Generar embedding o crear vector dummy con dimensión exacta si alpha == 0.0
        if alpha > 0.0:
            query_embedding = await self._embedding_service.aembed_query(query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
        else:
            embedding_str = f"[{','.join(['0.0'] * self._embedding_dim)}]"

        target_user_id = filters.user_id if filters else None

        params: Dict[str, Any] = {
            "query_embedding": embedding_str,
            "query_text": query,
            "user_id": target_user_id,
            "alpha": alpha,
            "k_candidates": max(k * 4, 20),
            "k": k,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self._compiled_query, params)
                records = await cur.fetchall()

        results: List[Document] = []
        for row in records:
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