import logging
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage

from src.agent_service.graph.sub_graphs.user_memory.state import UserMemoryState
from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore

logger = logging.getLogger(__name__)


class UserMemoryNodes:
    """Nodos ejecutores para el subgrafo de memoria a largo plazo."""

    def __init__(
        self,
        store: UserMemoryStore,
        default_limit: int = 2,
        default_threshold: float = 0.55,
    ):
        self._store = store
        self._limit = default_limit
        self._threshold = default_threshold

    async def retrieve_memory_node(self, state: UserMemoryState) -> dict:
        """Busca antecedentes semánticos relevantes en PostgreSQL para el usuario actual."""
        user_id_raw = state.get("user_id", 5)
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            user_id = 5

        # Obtener la consulta desde raw_query o desde el último HumanMessage
        query = state.get("raw_query")
        if not query:
            messages = state.get("messages", [])
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    query = str(m.content)
                    break

        if not query or not query.strip():
            return {"user_context": None, "retrieved_memories": []}

        memories = await self._store.search_memory(
            user_id=user_id,
            query=query,
            limit=self._limit,
            threshold=self._threshold,
        )

        if not memories:
            return {"user_context": None, "retrieved_memories": []}

        # Formatear el bloque de antecedentes para inyección limpia
        bullet_points = "\n".join(f"- {m['content']}" for m in memories)
        user_context = (
            "<antecedentes_usuario>\n"
            f"{bullet_points}\n"
            "</antecedentes_usuario>"
        )

        logger.info(
            f"Se recuperaron {len(memories)} memorias semánticas relevantes para user_id={user_id}"
        )
        return {
            "user_context": user_context,
            "retrieved_memories": memories,
        }

    async def save_memory_node(self, state: UserMemoryState) -> dict:
        """Persiste una nueva memoria en PostgreSQL si existe un hecho o resumen generado."""
        user_id_raw = state.get("user_id", 5)
        try:
            user_id = int(user_id_raw)
        except (ValueError, TypeError):
            user_id = 5

        memory_text = state.get("memory_to_save")
        session_id = state.get("session_id")

        if not memory_text or not memory_text.strip():
            return {}

        memory_id = await self._store.add_memory(
            user_id=user_id,
            content=memory_text.strip(),
            session_id=session_id,
        )

        return {"saved_memory_id": memory_id}
