"""
src/agent_service/graph/sub_graphs/sales_incentives/graph.py - Ensamble del subgrafo unificado sales_incentives.
"""

import logging
from typing import Literal, Optional, Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.graph.sub_graphs.sales_incentives.state import SalesIncentivesState
from src.agent_service.graph.sub_graphs.sales_incentives.nodes import SalesIncentivesNodes
from src.agent_service.tools.incentive_tools import (
    odoo_get_active_incentives,
)
from src.agent_service.tools.sales_tools import odoo_list_sales_orders
from src.agent_service.tools.contact_tools import odoo_list_current_customers

logger = logging.getLogger(__name__)


def _route_after_critic(
    state: SalesIncentivesState,
) -> Literal["synthesize_coaching", "actor_planner"]:
    """Enrutador condicional tras la evaluación del Critic:
    - Si el plan fue aprobado o se alcanzó el límite de iteraciones -> synthesize_coaching
    - Si el plan fue rechazado y restan iteraciones -> actor_planner (loop de mejora con feedback)
    """
    is_approved = state.get("is_approved", False)
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    if is_approved or iteration_count >= max_iterations:
        logger.info(
            f"[SalesIncentives] Plan APROBADO o límite alcanzado (iteración {iteration_count}/{max_iterations}). "
            "Avanzando a redacción de coaching."
        )
        return "synthesize_coaching"

    logger.info(
        f"[SalesIncentives] Plan RECHAZADO por el Critic (iteración {iteration_count}/{max_iterations}). "
        "Re-planificando con feedback correctivo."
    )
    return "actor_planner"


def build_sales_incentives_graph(
    llm: BaseChatModel,
    list_orders_tool: Any = odoo_list_sales_orders,
    get_incentives_tool: Any = odoo_get_active_incentives,
    list_customers_tool: Any = odoo_list_current_customers,
    catalog_search_tool: Optional[Any] = None,
    default_max_iterations: int = 3,
    default_epsilon: float = 0.25,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el subgrafo sales_incentives con patrón Actor-Critic y política ε-greedy."""
    nodes = SalesIncentivesNodes(
        llm=llm,
        list_orders_tool=list_orders_tool,
        get_incentives_tool=get_incentives_tool,
        list_customers_tool=list_customers_tool,
        catalog_search_tool=catalog_search_tool,
        default_max_iterations=default_max_iterations,
        default_epsilon=default_epsilon,
    )

    workflow = StateGraph(state_schema=SalesIncentivesState)

    # Registro de nodos
    workflow.add_node("gather_context_and_trends", nodes.gather_context_and_trends_node)
    workflow.add_node("actor_planner", nodes.actor_planner_node)
    workflow.add_node("critic_evaluator", nodes.critic_evaluator_node)
    workflow.add_node("synthesize_coaching", nodes.synthesize_coaching_node)

    # Conexiones
    workflow.add_edge(START, "gather_context_and_trends")
    workflow.add_edge("gather_context_and_trends", "actor_planner")
    workflow.add_edge("actor_planner", "critic_evaluator")

    workflow.add_conditional_edges(
        "critic_evaluator",
        _route_after_critic,
        {
            "synthesize_coaching": "synthesize_coaching",
            "actor_planner": "actor_planner",
        },
    )

    workflow.add_edge("synthesize_coaching", END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def get_sales_incentives_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Fábrica oficial sin argumentos para LangGraph Studio y compilación del grafo principal."""
    llm = get_default_llm()

    catalog_tool = None
    try:
        from src.agent_service.tools.product_tools import create_search_product_catalog_tool
        from src.agent_service.config.database import get_db_pool
        from src.agent_service.core.embeddings.factory import get_embedding_service
        from src.agent_service.core.stores.product.vector_store import ProductVectorStore

        pool = get_db_pool()
        embeddings = get_embedding_service()
        vector_store = ProductVectorStore(pool=pool, embedding_service=embeddings)
        catalog_tool = create_search_product_catalog_tool(vector_store)
    except Exception as exc:
        logger.warning(f"[SalesIncentivesGraph] No se pudo inicializar búsqueda de catálogo: {exc}")

    return build_sales_incentives_graph(
        llm=llm,
        catalog_search_tool=catalog_tool,
        checkpointer=checkpointer,
    )
