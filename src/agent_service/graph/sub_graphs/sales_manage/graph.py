"""
src/agent_service/graph/sub_graphs/sales_manage/graph.py - Ensamble del subgrafo unificado sales_manage.

Implementa el patrón Planner - Executor - Synthesizer con Evaluación por Rúbrica y Reflexión:
1. Nodo Planificador (plan_and_select_tools): Analiza intención, contexto multi-turno y genera plan estructurado.
2. Soporte HITL condicional: Confirmación humana para operaciones sensibles (confirmación, anulación, desbloqueo).
3. Nodo Ejecutor (execute_tools): Ejecuta herramientas de Odoo de forma asíncrona.
4. Nodo Sintetizador (synthesize_draft): Redacta borrador comercial completo atendiendo directrices de detalle de productos.
5. Nodo Juez de Rúbrica (rubric_evaluator_judge): Evalúa calidad factual, relevancia y completitud de líneas.
6. Loop de Reflexión:
   - Fast Reflection Loop -> Regresa a 'synthesize_draft' si faltó claridad o se omitieron productos ya obtenidos.
   - Slow Reflection Loop -> Regresa a 'plan_and_select_tools' si faltan datos de herramientas.
7. Nodo Finalizador (finalize_response): Estandariza divisas y formatea nativo para WhatsApp/Web.
"""

import logging
from typing import Literal, Optional, Any, Callable, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.graph.sub_graphs.sales_manage.state import SalesManageState
from src.agent_service.graph.sub_graphs.sales_manage.nodes import SalesManageNodes
from src.agent_service.tools.sales_tools import (
    odoo_list_sales_orders,
    odoo_list_current_sales_orders,
    odoo_create_quotation,
    odoo_update_quotation,
    odoo_view_quotation,
    odoo_confirm_order,
    odoo_unlock_order,
    odoo_update_order,
    odoo_lock_order,
    odoo_remove_sale_order,
    get_shared_odoo_client,
)
from src.agent_service.tools.contact_tools import (
    odoo_list_current_customers,
    odoo_upsert_customer,
)

logger = logging.getLogger(__name__)


def _route_after_planning(
    state: SalesManageState,
) -> Literal[
    "execute_tools",
    "feedback_intent_clarification_node",
    "feedback_ambiguous_customer_node",
    "feedback_duplicate_sales_node",
    "feedback_remove_order_node",
    "guardrail_unlock_node",
    "synthesize_draft",
]:
    """Enrutador condicional tras la fase de planificación:
    - Si se canceló la operación -> synthesize_draft
    - Si requiere HITL de cancelación -> feedback_remove_order_node
    - Si requiere HITL de desbloqueo -> guardrail_unlock_node
    - Si requiere resolución de cliente ambiguo -> feedback_ambiguous_customer_node
    - Si requiere resolución de duplicados -> feedback_duplicate_sales_node
    - Si requiere aclaración de intención / conflicto -> feedback_intent_clarification_node
    - En otro caso -> execute_tools directamente
    """
    if state.get("cancellation_reason"):
        return "synthesize_draft"

    requires_hitl = state.get("requires_hitl", False)
    hitl_type = state.get("hitl_type")

    if requires_hitl:
        if hitl_type == "remove_order" and not state.get("remove_confirmed"):
            return "feedback_remove_order_node"
        elif hitl_type == "unlock_order" and not state.get("unlock_confirmed"):
            return "guardrail_unlock_node"
        elif hitl_type == "ambiguous_customer":
            return "feedback_ambiguous_customer_node"
        elif hitl_type == "duplicate":
            return "feedback_duplicate_sales_node"
        elif hitl_type == "clarification":
            return "feedback_intent_clarification_node"

    return "execute_tools"


def _route_after_ambiguous_customer(
    state: SalesManageState,
) -> Literal["feedback_duplicate_sales_node", "execute_tools", "synthesize_draft"]:
    if state.get("cancellation_reason"):
        return "synthesize_draft"
    if state.get("requires_hitl") and state.get("hitl_type") == "duplicate":
        return "feedback_duplicate_sales_node"
    return "execute_tools"


def _route_after_duplicate_sales(
    state: SalesManageState,
) -> Literal["execute_tools", "synthesize_draft"]:
    if state.get("cancellation_reason"):
        return "synthesize_draft"
    return "execute_tools"


def _route_after_intent_clarification(
    state: SalesManageState,
) -> Literal["execute_tools", "synthesize_draft"]:
    if state.get("cancellation_reason"):
        return "synthesize_draft"
    return "execute_tools"


def _route_after_remove_order(
    state: SalesManageState,
) -> Literal["execute_tools", "synthesize_draft"]:
    if state.get("cancellation_reason"):
        return "synthesize_draft"
    return "execute_tools"


def _route_after_guardrail_unlock(
    state: SalesManageState,
) -> Literal["execute_tools", "synthesize_draft"]:
    if state.get("cancellation_reason"):
        return "synthesize_draft"
    return "execute_tools"


def _route_after_rubric(
    state: SalesManageState,
) -> Literal["finalize_response", "synthesize_draft", "plan_and_select_tools"]:
    """Enrutador condicional tras la evaluación por rúbrica:
    - Si cumple la rúbrica -> finalize_response
    - Si iteration_count >= max_iterations (default 3) -> finalize_response (corte forzoso)
    - Si no cumple e iteration_count < max_iterations:
      * Si reflection_action == "refine_synthesis" -> synthesize_draft (Fast Reflection Loop)
      * En otro caso -> plan_and_select_tools (Slow Reflection Loop)
    """
    meets_rubric = state.get("meets_rubric", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    reflection_action = state.get("reflection_action", "replan_tools")

    if meets_rubric or reflection_action == "approve":
        logger.info(
            f"sales_manage: Rúbrica APROBADA (iteración {iteration_count}). Avanzando a finalización."
        )
        return "finalize_response"

    if iteration_count >= max_iterations:
        logger.warning(
            f"sales_manage: Rúbrica NO alcanzada tras {iteration_count} iteraciones. "
            f"Alcanzado límite máximo ({max_iterations}); finalizando con mejor respuesta disponible."
        )
        return "finalize_response"

    if reflection_action == "refine_synthesis":
        logger.info(
            f"sales_manage: Rúbrica RECHAZADA por claridad o detalle omitido (iteración {iteration_count}/{max_iterations}). "
            "Ejecutando Fast Reflection Loop hacia 'synthesize_draft' para refinar redacción sin re-ejecutar herramientas."
        )
        return "synthesize_draft"

    logger.info(
        f"sales_manage: Rúbrica RECHAZADA por datos insuficientes (iteración {iteration_count}/{max_iterations}). "
        "Re-planificando con herramientas en 'plan_and_select_tools'."
    )
    return "plan_and_select_tools"


def build_sales_manage_graph(
    llm: BaseChatModel,
    list_orders_tool: Any = odoo_list_sales_orders,
    list_current_orders_tool: Any = odoo_list_current_sales_orders,
    create_quotation_tool: Any = odoo_create_quotation,
    update_quotation_tool: Any = odoo_update_quotation,
    view_quotation_tool: Any = odoo_view_quotation,
    confirm_order_tool: Any = odoo_confirm_order,
    unlock_order_tool: Any = odoo_unlock_order,
    update_order_tool: Any = odoo_update_order,
    lock_order_tool: Any = odoo_lock_order,
    remove_order_tool: Any = odoo_remove_sale_order,
    list_customers_tool: Any = odoo_list_current_customers,
    upsert_customer_tool: Any = odoo_upsert_customer,
    catalog_search_tool: Optional[Any] = None,
    default_max_iterations: int = 3,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    tools_registry: Optional[Dict[str, Any]] = None,
):
    """Construye y compila el subgrafo unificado sales_manage con patrón Planner-Executor y reflexión."""
    nodes = SalesManageNodes(
        llm=llm,
        list_orders_tool=list_orders_tool,
        list_current_orders_tool=list_current_orders_tool,
        create_quotation_tool=create_quotation_tool,
        update_quotation_tool=update_quotation_tool,
        view_quotation_tool=view_quotation_tool,
        confirm_order_tool=confirm_order_tool,
        unlock_order_tool=unlock_order_tool,
        update_order_tool=update_order_tool,
        lock_order_tool=lock_order_tool,
        remove_order_tool=remove_order_tool,
        list_customers_tool=list_customers_tool,
        upsert_customer_tool=upsert_customer_tool,
        catalog_search_tool=catalog_search_tool,
        default_max_iterations=default_max_iterations,
        tools_registry=tools_registry,
    )

    workflow = StateGraph(state_schema=SalesManageState)

    # 1. Registro de nodos principales del patrón
    workflow.add_node("plan_and_select_tools", nodes.plan_and_select_tools)
    workflow.add_node("execute_tools", nodes.execute_tools)
    workflow.add_node("synthesize_draft", nodes.synthesize_draft)
    workflow.add_node("rubric_evaluator_judge", nodes.rubric_evaluator_judge)
    workflow.add_node("finalize_response", nodes.finalize_response)

    # Nodos HITL para confirmaciones y desambiguación
    workflow.add_node("feedback_intent_clarification_node", nodes.feedback_intent_clarification_node)
    workflow.add_node("feedback_ambiguous_customer_node", nodes.feedback_ambiguous_customer_node)
    workflow.add_node("feedback_duplicate_sales_node", nodes.feedback_duplicate_sales_node)
    workflow.add_node("feedback_remove_order_node", nodes.feedback_remove_order_node)
    workflow.add_node("guardrail_unlock_node", nodes.guardrail_unlock_node)

    # 2. Conexiones
    workflow.add_edge(START, "plan_and_select_tools")

    workflow.add_conditional_edges(
        "plan_and_select_tools",
        _route_after_planning,
        {
            "execute_tools": "execute_tools",
            "feedback_intent_clarification_node": "feedback_intent_clarification_node",
            "feedback_ambiguous_customer_node": "feedback_ambiguous_customer_node",
            "feedback_duplicate_sales_node": "feedback_duplicate_sales_node",
            "feedback_remove_order_node": "feedback_remove_order_node",
            "guardrail_unlock_node": "guardrail_unlock_node",
            "synthesize_draft": "synthesize_draft",
        },
    )

    # Conexiones condicionales de salidas HITL
    workflow.add_conditional_edges(
        "feedback_ambiguous_customer_node",
        _route_after_ambiguous_customer,
        {
            "feedback_duplicate_sales_node": "feedback_duplicate_sales_node",
            "execute_tools": "execute_tools",
            "synthesize_draft": "synthesize_draft",
        },
    )
    workflow.add_conditional_edges(
        "feedback_duplicate_sales_node",
        _route_after_duplicate_sales,
        {
            "execute_tools": "execute_tools",
            "synthesize_draft": "synthesize_draft",
        },
    )
    workflow.add_conditional_edges(
        "feedback_intent_clarification_node",
        _route_after_intent_clarification,
        {
            "execute_tools": "execute_tools",
            "synthesize_draft": "synthesize_draft",
        },
    )
    workflow.add_conditional_edges(
        "feedback_remove_order_node",
        _route_after_remove_order,
        {
            "execute_tools": "execute_tools",
            "synthesize_draft": "synthesize_draft",
        },
    )
    workflow.add_conditional_edges(
        "guardrail_unlock_node",
        _route_after_guardrail_unlock,
        {
            "execute_tools": "execute_tools",
            "synthesize_draft": "synthesize_draft",
        },
    )

    workflow.add_edge("execute_tools", "synthesize_draft")
    workflow.add_edge("synthesize_draft", "rubric_evaluator_judge")

    # Bifurcación condicional tras evaluación por rúbrica
    workflow.add_conditional_edges(
        "rubric_evaluator_judge",
        _route_after_rubric,
        {
            "finalize_response": "finalize_response",
            "synthesize_draft": "synthesize_draft",
            "plan_and_select_tools": "plan_and_select_tools",
        },
    )

    workflow.add_edge("finalize_response", END)

    from langgraph.checkpoint.memory import MemorySaver
    cp = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=cp)


def get_sales_manage_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Fábrica oficial para instanciar y compilar sales_manage con herramientas conectadas."""
    llm = get_default_llm()

    # Intentar cargar herramienta de búsqueda semántica de catálogo si el entorno lo permite
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
        logger.warning(f"[SalesManageGraph] No se pudo inicializar búsqueda semántica para órdenes: {exc}")

    return build_sales_manage_graph(
        llm=llm,
        catalog_search_tool=catalog_tool,
        default_max_iterations=3,
        checkpointer=checkpointer,
    )
