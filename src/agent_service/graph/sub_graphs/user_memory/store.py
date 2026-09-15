import logging
from typing import Optional, List, Dict, Any
from psycopg_pool import AsyncConnectionPool
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class UserMemoryStore:
    """Almacén de memoria a largo plazo con búsqueda semántica en PostgreSQL (pgvector)."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_service: Embeddings,
        table_name: str = "user_memory",
        embedding_dim: int = 1024,
    ):
        self._pool = pool
        self._embedding_service = embedding_service
        self._table_name = table_name
        self._embedding_dim = embedding_dim

    async def _ensure_open(self) -> None:
        if getattr(self._pool, "closed", False) is True:
            await self._pool.open()

    async def add_memory(
        self,
        user_id: int,
        content: str,
        session_id: Optional[str] = None,
    ) -> int:
        """Genera embedding y persiste un hecho o resumen en la memoria del usuario."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("El contenido de la memoria no puede estar vacío.")

        await self._ensure_open()

        embedding = await self._embedding_service.aembed_query(clean_content)
        emb_str = f"[{','.join(map(str, embedding))}]"

        insert_sql = f"""
            INSERT INTO {self._table_name} (user_id, session_id, content, embedding_1024)
            VALUES (%(user_id)s, %(session_id)s, %(content)s, %(emb)s::vector)
            RETURNING id;
        """

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    insert_sql,
                    {
                        "user_id": int(user_id),
                        "session_id": session_id,
                        "content": clean_content,
                        "emb": emb_str,
                    },
                )
                row = await cur.fetchone()
                memory_id = row[0]
                logger.info(f"Memoria registrada [id={memory_id}] para user_id={user_id}")
                return memory_id

    async def search_memory(
        self,
        user_id: int,
        query: str,
        limit: int = 2,
        threshold: float = 0.55,
    ) -> List[Dict[str, Any]]:
        """Búsqueda semántica por similitud de coseno en la memoria del usuario especificado."""
        clean_query = query.strip()
        if not clean_query:
            return []

        query_emb = await self._embedding_service.aembed_query(clean_query)
        emb_str = f"[{','.join(map(str, query_emb))}]"

        search_sql = f"""
            SELECT id, content, (1.0 - (embedding_1024 <=> %(query_emb)s::vector)) AS similarity, created_at
            FROM {self._table_name}
            WHERE user_id = %(user_id)s
            ORDER BY embedding_1024 <=> %(query_emb)s::vector
            LIMIT %(limit)s;
        """

        await self._ensure_open()

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    search_sql,
                    {
                        "user_id": int(user_id),
                        "query_emb": emb_str,
                        "limit": max(limit, 1),
                    },
                )
                records = await cur.fetchall()

        results = []
        for row in records:
            sim = float(row[2])
            if sim >= threshold:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "similarity": sim,
                    "created_at": str(row[3]),
                })

        return results

    async def get_user_memories(
        self,
        user_id: int,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Obtiene las memorias más recientes de un usuario para inspección o comandos CLI."""
        sql = f"""
            SELECT id, content, session_id, created_at
            FROM {self._table_name}
            WHERE user_id = %(user_id)s
            ORDER BY created_at DESC
            LIMIT %(limit)s;
        """

        await self._ensure_open()

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, {"user_id": int(user_id), "limit": limit})
                records = await cur.fetchall()

        return [
            {
                "id": r[0],
                "content": r[1],
                "session_id": r[2],
                "created_at": str(r[3]),
            }
            for r in records
        ]
