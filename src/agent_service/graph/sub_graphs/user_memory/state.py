from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserMemoryState(TypedDict, total=False):
    """Estado del subgrafo de memoria a largo plazo (user_memory)."""

    # Historial de mensajes conversacionales
    messages: Annotated[List[BaseMessage], add_messages]

    # Contexto del usuario comercial y sesión
    user_id: int
    raw_query: Optional[str]
    session_id: Optional[str]

    # Contexto histórico recuperado mediante búsqueda semántica
    user_context: Optional[str]
    retrieved_memories: List[Dict[str, Any]]

    # Campo para almacenar un nuevo recuerdo tras cotización o hito
    memory_to_save: Optional[str]
    saved_memory_id: Optional[int]
