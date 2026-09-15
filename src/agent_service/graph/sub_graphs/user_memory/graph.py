from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END

from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.graph.sub_graphs.user_memory.state import UserMemoryState
from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore
from src.agent_service.graph.sub_graphs.user_memory.nodes import UserMemoryNodes


def _route_memory_action(state: UserMemoryState) -> Literal["save_memory", "retrieve_memory"]:
    """Enruta hacia guardado si hay un hecho explícito pendiente de persistir, o a recuperación."""
    if state.get("memory_to_save"):
        return "save_memory"
    return "retrieve_memory"


def build_user_memory_graph(
    store: UserMemoryStore,
    default_limit: int = 2,
    default_threshold: float = 0.55,
):
    """Construye y compila el subgrafo modular de memoria (user_memory)."""
    nodes = UserMemoryNodes(
        store=store,
        default_limit=default_limit,
        default_threshold=default_threshold,
    )

    workflow = StateGraph(state_schema=UserMemoryState)

    workflow.add_node("retrieve_memory", nodes.retrieve_memory_node)
    workflow.add_node("save_memory", nodes.save_memory_node)

    workflow.add_conditional_edges(
        START,
        _route_memory_action,
        {
            "retrieve_memory": "retrieve_memory",
            "save_memory": "save_memory",
        },
    )

    workflow.add_edge("retrieve_memory", END)
    workflow.add_edge("save_memory", END)

    return workflow.compile()


def get_user_memory_graph():
    """Fábrica sin argumentos para inspección visual en LangGraph Studio."""
    pool = get_db_pool()
    embeddings = get_embedding_service()
    store = UserMemoryStore(pool=pool, embedding_service=embeddings)
    return build_user_memory_graph(store=store)
