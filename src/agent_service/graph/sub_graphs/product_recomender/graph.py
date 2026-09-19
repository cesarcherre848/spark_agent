"""
src/agent_service/graph/sub_graphs/product_recomender/graph.py - Ensamble del subgrafo product_recomender

Implementa el flujo del subgrafo con:
1. Extracción de intención de recomendación, top-k dinámico y restricciones.
2. Recuperación híbrida con sobre-muestreo (retrieval_k >= max(top_k * 2, 6)).
3. Enriquecimiento oficial de precios, categorías y unidades desde Odoo ERP.
4. Filtrado determinista por reglas de negocio y restricciones del usuario.
5. Evaluación por Rúbrica multi-criterio (Relevancia, Filtros, Diversidad, Integridad de Datos).
6. Auto-reflexión y reformulación adaptativa (Evaluator-Optimizer Loop) si no cumple la rúbrica.
7. Síntesis ejecutiva comercial para el vendedor/cliente.
"""

import logging
from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.graph.sub_graphs.product_recomender.state import ProductRecomenderState
from src.agent_service.graph.sub_graphs.product_recomender.nodes import ProductRecomenderNodes

logger = logging.getLogger(__name__)


def _route_after_rubric(
    state: ProductRecomenderState,
) -> Literal["synthesize_recommendations", "reflection_optimizer"]:
    """Enrutador condicional según el veredicto de la Rúbrica de Calidad."""
    meets_rubric = state.get("meets_rubric", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 2)

    if meets_rubric:
        logger.info(
            f"product_recomender: Rúbrica APROBADA (iteración {iteration_count}). "
            "Avanzando a síntesis de recomendaciones."
        )
        return "synthesize_recommendations"

    if iteration_count >= max_iterations:
        logger.warning(
            f"product_recomender: Rúbrica NO alcanzada tras {iteration_count} iteraciones. "
            "Alcanzado límite de reintentos; avanzando a síntesis con mejores candidatos disponibles."
        )
        return "synthesize_recommendations"

    logger.info(
        f"product_recomender: Rúbrica RECHAZADA (iteración {iteration_count}/{max_iterations}). "
        "Activando ciclo de reflexión y optimización de búsqueda."
    )
    return "reflection_optimizer"


def build_product_recomender_graph(
    llm: BaseChatModel,
    vector_store: Optional[ProductVectorStore] = None,
    odoo_client: Optional[OdooClient] = None,
    default_top_k: int = 15,
    default_max_iterations: int = 2,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el subgrafo product_recomender con patrón Evaluator-Optimizer."""
    nodes = ProductRecomenderNodes(
        llm=llm,
        vector_store=vector_store,
        odoo_client=odoo_client,
        default_top_k=default_top_k,
        default_max_iterations=default_max_iterations,
    )

    workflow = StateGraph(state_schema=ProductRecomenderState)

    # Registro de nodos
    workflow.add_node("extract_recommendation_intent", nodes.extract_recommendation_intent)
    workflow.add_node("retrieve_candidate_products", nodes.retrieve_candidate_products)
    workflow.add_node("enrich_product_details", nodes.enrich_product_details)
    workflow.add_node("apply_user_filters", nodes.apply_user_filters)
    workflow.add_node("rubric_evaluator_judge", nodes.rubric_evaluator_judge)
    workflow.add_node("reflection_optimizer", nodes.reflection_optimizer)
    workflow.add_node("synthesize_recommendations", nodes.synthesize_recommendations)

    # Flujo secuencial inicial
    workflow.add_edge(START, "extract_recommendation_intent")
    workflow.add_edge("extract_recommendation_intent", "retrieve_candidate_products")
    workflow.add_edge("retrieve_candidate_products", "enrich_product_details")
    workflow.add_edge("enrich_product_details", "apply_user_filters")
    workflow.add_edge("apply_user_filters", "rubric_evaluator_judge")

    # Bifurcación condicional basada en la Rúbrica
    workflow.add_conditional_edges(
        "rubric_evaluator_judge",
        _route_after_rubric,
        {
            "synthesize_recommendations": "synthesize_recommendations",
            "reflection_optimizer": "reflection_optimizer",
        },
    )

    # Ciclo de auto-reflexión y reformulación (Loop Evaluator-Optimizer)
    workflow.add_edge("reflection_optimizer", "retrieve_candidate_products")

    # Salida terminal
    workflow.add_edge("synthesize_recommendations", END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def get_product_recomender_graph():
    """Fábrica sin argumentos para inspección en LangGraph Studio o ejecución directa."""
    from src.agent_service.core.llms.factory import get_default_llm
    from src.agent_service.config.database import get_db_pool
    from src.agent_service.core.embeddings.factory import get_embedding_service
    from src.agent_service.tools.sales_tools import get_shared_odoo_client

    llm = get_default_llm()
    pool = get_db_pool()
    embeddings = get_embedding_service()
    vector_store = ProductVectorStore(pool=pool, embedding_service=embeddings)
    odoo_client = get_shared_odoo_client()

    return build_product_recomender_graph(
        llm=llm,
        vector_store=vector_store,
        odoo_client=odoo_client,
        checkpointer=MemorySaver(),
    )
