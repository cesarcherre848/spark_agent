"""
src/agent_service/graph/sub_graphs/product_advisor/graph.py - Ensamble del subgrafo product_advisor.

Implementa el flujo unificado con:
1. Nodo Planificador: Genera el plan estructurado de herramientas según la intención.
2. Nodo Ejecutor: Ejecuta de forma asíncrona las herramientas de catálogo y Odoo ERP.
3. Nodo Sintetizador: Redacta la respuesta comercial adecuada al canal y normas whitelabel.
4. Nodo Juez de Rúbrica: Evalúa calidad, relevancia, exactitud factual y respeto a restricciones.
5. Loop de Reflexión: Si la rúbrica no se cumple, retroalimenta al planificador (máximo 3 iteraciones).
6. Nodo Finalizador: Estandariza divisas y emite la respuesta final.
"""

import logging
from typing import Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes

logger = logging.getLogger(__name__)


def _route_after_rubric(
    state: ProductAdvisorState,
) -> Literal["finalize_response", "plan_and_select_tools"]:
    """Enrutador condicional tras la evaluación por rúbrica:
    - Si cumple la rúbrica -> finalize_response
    - Si iteration_count >= max_iterations (default 3) -> finalize_response (corte forzoso)
    - Si no cumple e iteration_count < max_iterations -> plan_and_select_tools (reintento con feedback)
    """
    meets_rubric = state.get("meets_rubric", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if meets_rubric:
        logger.info(
            f"product_advisor: Rúbrica APROBADA (iteración {iteration_count}). Avanzando a finalización."
        )
        return "finalize_response"

    if iteration_count >= max_iterations:
        logger.warning(
            f"product_advisor: Rúbrica NO alcanzada tras {iteration_count} iteraciones. "
            f"Alcanzado límite máximo ({max_iterations}); finalizando con mejor respuesta disponible."
        )
        return "finalize_response"

    logger.info(
        f"product_advisor: Rúbrica RECHAZADA (iteración {iteration_count}/{max_iterations}). "
        "Re-planificando con feedback correctivo de rúbrica."
    )
    return "plan_and_select_tools"


def build_product_advisor_graph(
    llm: BaseChatModel,
    vector_store: Optional[ProductVectorStore] = None,
    odoo_client: Optional[OdooClient] = None,
    default_max_iterations: int = 3,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    tools_registry: Optional[Dict[str, Any]] = None,
):
    """Construye y compila el subgrafo unificado product_advisor con patrón Planner-Executor y reflexión."""
    nodes = ProductAdvisorNodes(
        llm=llm,
        vector_store=vector_store,
        odoo_client=odoo_client,
        default_max_iterations=default_max_iterations,
        tools_registry=tools_registry,
    )

    workflow = StateGraph(state_schema=ProductAdvisorState)

    # Registro de nodos
    workflow.add_node("plan_and_select_tools", nodes.plan_and_select_tools)
    workflow.add_node("execute_tools", nodes.execute_tools)
    workflow.add_node("synthesize_draft", nodes.synthesize_draft)
    workflow.add_node("rubric_evaluator_judge", nodes.rubric_evaluator_judge)
    workflow.add_node("finalize_response", nodes.finalize_response)

    # Conexiones
    workflow.add_edge(START, "plan_and_select_tools")
    workflow.add_edge("plan_and_select_tools", "execute_tools")
    workflow.add_edge("execute_tools", "synthesize_draft")
    workflow.add_edge("synthesize_draft", "rubric_evaluator_judge")

    workflow.add_conditional_edges(
        "rubric_evaluator_judge",
        _route_after_rubric,
        {
            "finalize_response": "finalize_response",
            "plan_and_select_tools": "plan_and_select_tools",
        },
    )

    workflow.add_edge("finalize_response", END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def get_product_advisor_graph():
    """Fábrica sin argumentos para LangGraph Studio y punto de entrada oficial."""
    from src.agent_service.core.llms.factory import get_default_llm
    from src.agent_service.config.database import get_db_pool
    from src.agent_service.core.embeddings.factory import get_embedding_service
    from src.agent_service.tools.sales_tools import get_shared_odoo_client

    llm = get_default_llm()
    pool = get_db_pool()
    embeddings = get_embedding_service()
    vector_store = ProductVectorStore(pool=pool, embedding_service=embeddings)
    odoo_client = get_shared_odoo_client()

    return build_product_advisor_graph(
        llm=llm,
        vector_store=vector_store,
        odoo_client=odoo_client,
        default_max_iterations=3,
    )
