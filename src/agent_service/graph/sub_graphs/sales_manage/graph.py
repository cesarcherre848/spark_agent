"""
src/agent_service/graph/sub_graphs/sales_manage/graph.py - Ensamble del subgrafo sales_manage
"""

from typing import Literal, Optional, Any, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.graph.sub_graphs.sales_manage.state import SalesManageState
from src.agent_service.graph.sub_graphs.sales_manage.nodes import SalesManageNodes
from src.agent_service.tools.sales_tools import (
    extract_order_code,
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
)

from src.agent_service.tools.contact_tools import (
    odoo_list_current_customers,
    odoo_upsert_customer,
)


def _route_after_extraction(
    state: SalesManageState,
) -> Literal[
    "view_quotation_node",
    "update_quotation_node",
    "confirm_order_node",
    "list_sales_orders_node",
    "feedback_remove_order_node",
    "guardrail_unlock_node",
    "resolve_customer_node",
]:
    """Enrutador posterior a la extracción: aplica patrón híbrido (Fast-Path vs Slow-Path)."""
    action = state.get("sales_action", "list")
    raw_query = state.get("raw_query") or ""
    has_explicit_order = bool(extract_order_code(raw_query))
    has_target = bool(state.get("target_order_name") or state.get("target_order_id"))
    has_customer = bool(state.get("customer_name"))

    # 1. Fast-Path: Ver cotización u orden específica
    if action == "view" or (action == "list" and has_target):
        return "view_quotation_node"

    # 2. Fast-Path: Modificación o adición de líneas a una orden existente
    if action in ("remove_items", "add_items") and has_target:
        return "update_quotation_node"

    if action == "upsert" and has_target:
        return "update_quotation_node"


    # 3. Confirmación directa si ya se tiene la orden identificada
    if action == "confirm" and has_target:
        return "confirm_order_node"

    # 4. Listado general sin orden específica
    if action == "list":
        return "list_sales_orders_node"

    # 5. Guardrail HITL: Cancelación / anulación
    if action == "remove":
        return "feedback_remove_order_node"

    # 6. Guardrail HITL: Edición de pedido confirmado / bloqueado
    if action == "edit_order":
        return "guardrail_unlock_node"

    # 7. Slow-Path: Nueva cotización o gestión que requiere resolución de cliente
    return "resolve_customer_node"


def _route_after_reflection(
    state: SalesManageState,
) -> Literal[
    "feedback_intent_clarification_node",
    "view_quotation_node",
    "update_quotation_node",
    "confirm_order_node",
    "list_sales_orders_node",
    "feedback_remove_order_node",
    "guardrail_unlock_node",
    "resolve_customer_node",
]:
    """Enrutador posterior a la reflexión: si la intención es ambigua, escala a HITL; si no, avanza."""
    if state.get("is_intent_clear") is False:
        return "feedback_intent_clarification_node"
    return _route_after_extraction(state)


def _route_after_intent_clarification(
    state: SalesManageState,
) -> Literal[
    "view_quotation_node",
    "update_quotation_node",
    "confirm_order_node",
    "list_sales_orders_node",
    "feedback_remove_order_node",
    "guardrail_unlock_node",
    "resolve_customer_node",
    "synthesize_sales_response",
]:
    """Enrutador tras aclaración HITL de intención."""
    if state.get("cancellation_reason"):
        return "synthesize_sales_response"
    return _route_after_extraction(state)



def _route_after_customer_resolution(
    state: SalesManageState,
) -> Literal["feedback_ambiguous_customer_node", "feedback_unknown_customer_node", "list_current_sales_orders_node"]:
    """Enrutador de resolución de clientes: detecta ambigüedad, inexistencia o coincidencia inequívoca."""
    if state.get("candidate_partners") and len(state["candidate_partners"]) > 1:
        return "feedback_ambiguous_customer_node"
    if state.get("customer_not_found"):
        return "feedback_unknown_customer_node"
    return "list_current_sales_orders_node"


def _route_after_ambiguous_feedback(
    state: SalesManageState,
) -> Literal["list_current_sales_orders_node", "feedback_unknown_customer_node", "synthesize_sales_response"]:
    """Enrutador tras aclaración de ambigüedad de clientes."""
    if state.get("customer_resolved"):
        return "list_current_sales_orders_node"
    if state.get("customer_not_found"):
        return "feedback_unknown_customer_node"
    return "synthesize_sales_response"


def _route_after_unknown_feedback(
    state: SalesManageState,
) -> Literal["list_current_sales_orders_node", "synthesize_sales_response"]:
    """Enrutador tras ofrecimiento de alta de cliente."""
    if state.get("customer_resolved"):
        return "list_current_sales_orders_node"
    return "synthesize_sales_response"


def _route_action_branch(
    state: SalesManageState,
) -> Literal["update_quotation_node", "create_quotation_node", "confirm_order_node", "guardrail_unlock_node", "synthesize_sales_response"]:
    """Bifurca hacia la acción específica (upsert, confirm, edit_order)."""
    action = state.get("sales_action", "upsert")
    choice = state.get("duplicate_choice", "new")
    has_target = bool(state.get("target_order_id") or state.get("target_order_name"))

    if action in ("upsert", "add_items", "remove_items"):
        if choice == "selected" or (has_target and choice != "new"):
            return "update_quotation_node"
        return "create_quotation_node"


    if action == "confirm":
        if has_target:
            return "confirm_order_node"
        return "create_quotation_node"

    if action == "edit_order":
        return "guardrail_unlock_node"

    return "synthesize_sales_response"


def _route_after_duplicate_judge(
    state: SalesManageState,
) -> Literal["feedback_duplicate_sales_node", "update_quotation_node", "create_quotation_node", "confirm_order_node", "guardrail_unlock_node", "synthesize_sales_response"]:
    """Enrutador tras evaluación de duplicados por el LLM."""
    if state.get("has_possible_duplicates", False):
        return "feedback_duplicate_sales_node"
    return _route_action_branch(state)


def _route_after_duplicate_feedback(
    state: SalesManageState,
) -> Literal["update_quotation_node", "create_quotation_node", "confirm_order_node", "guardrail_unlock_node", "synthesize_sales_response"]:
    """Enrutador tras respuesta HITL de duplicados."""
    if state.get("duplicate_choice") == "cancel":
        return "synthesize_sales_response"
    return _route_action_branch(state)


def _route_after_create_quotation(
    state: SalesManageState,
) -> Literal["confirm_order_node", "view_quotation_node"]:
    """Enrutador tras crear cotización: si la acción era confirmación directa, avanza a confirmar."""
    if state.get("sales_action") == "confirm":
        return "confirm_order_node"
    return "view_quotation_node"


def _route_after_guardrail_unlock(
    state: SalesManageState,
) -> Literal["unlock_order_node", "synthesize_sales_response"]:
    """Enrutador tras decisión HITL del guardrail de desbloqueo."""
    if state.get("unlock_confirmed", False):
        return "unlock_order_node"
    return "synthesize_sales_response"


def _route_after_remove_confirm(
    state: SalesManageState,
) -> Literal["remove_order_node", "synthesize_sales_response"]:
    """Enrutador tras confirmación HITL de anulación de orden."""
    if state.get("remove_confirmed", False):
        return "remove_order_node"
    return "synthesize_sales_response"


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
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el subgrafo sales_manage con soporte de Human-in-the-Loop y herramientas desacopladas."""
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
    )

    workflow = StateGraph(state_schema=SalesManageState)

    # 1. Registro de nodos
    workflow.add_node("extract_sales_info", nodes.extract_sales_info)
    workflow.add_node("reflect_intent_node", nodes.reflect_intent_node)
    workflow.add_node("feedback_intent_clarification_node", nodes.feedback_intent_clarification_node)
    workflow.add_node("list_sales_orders_node", nodes.list_sales_orders_node)
    workflow.add_node("resolve_customer_node", nodes.resolve_customer_node)
    workflow.add_node("feedback_ambiguous_customer_node", nodes.feedback_ambiguous_customer_node)
    workflow.add_node("feedback_unknown_customer_node", nodes.feedback_unknown_customer_node)
    workflow.add_node("list_current_sales_orders_node", nodes.list_current_sales_orders_node)
    workflow.add_node("duplicate_candidates_judge_node", nodes.duplicate_candidates_judge_node)
    workflow.add_node("feedback_duplicate_sales_node", nodes.feedback_duplicate_sales_node)
    workflow.add_node("create_quotation_node", nodes.create_quotation_node)
    workflow.add_node("update_quotation_node", nodes.update_quotation_node)
    workflow.add_node("view_quotation_node", nodes.view_quotation_node)
    workflow.add_node("confirm_order_node", nodes.confirm_order_node)
    workflow.add_node("guardrail_unlock_node", nodes.guardrail_unlock_node)
    workflow.add_node("unlock_order_node", nodes.unlock_order_node)
    workflow.add_node("update_order_node", nodes.update_order_node)
    workflow.add_node("lock_order_node", nodes.lock_order_node)
    workflow.add_node("feedback_remove_order_node", nodes.feedback_remove_order_node)
    workflow.add_node("remove_order_node", nodes.remove_order_node)
    workflow.add_node("synthesize_sales_response", nodes.synthesize_sales_response)

    # 2. Conexiones
    workflow.add_edge(START, "extract_sales_info")
    workflow.add_edge("extract_sales_info", "reflect_intent_node")

    # Bifurcación tras reflexión crítica (escala a HITL si hay duda o enruta directamente)
    workflow.add_conditional_edges(
        "reflect_intent_node",
        _route_after_reflection,
        {
            "feedback_intent_clarification_node": "feedback_intent_clarification_node",
            "view_quotation_node": "view_quotation_node",
            "update_quotation_node": "update_quotation_node",
            "confirm_order_node": "confirm_order_node",
            "list_sales_orders_node": "list_sales_orders_node",
            "feedback_remove_order_node": "feedback_remove_order_node",
            "guardrail_unlock_node": "guardrail_unlock_node",
            "resolve_customer_node": "resolve_customer_node",
        },
    )

    # Bifurcación tras aclaración HITL de intención
    workflow.add_conditional_edges(
        "feedback_intent_clarification_node",
        _route_after_intent_clarification,
        {
            "view_quotation_node": "view_quotation_node",
            "update_quotation_node": "update_quotation_node",
            "confirm_order_node": "confirm_order_node",
            "list_sales_orders_node": "list_sales_orders_node",
            "feedback_remove_order_node": "feedback_remove_order_node",
            "guardrail_unlock_node": "guardrail_unlock_node",
            "resolve_customer_node": "resolve_customer_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )


    # Rama List
    workflow.add_edge("list_sales_orders_node", "synthesize_sales_response")

    # Resolución de clientes
    workflow.add_conditional_edges(
        "resolve_customer_node",
        _route_after_customer_resolution,
        {
            "feedback_ambiguous_customer_node": "feedback_ambiguous_customer_node",
            "feedback_unknown_customer_node": "feedback_unknown_customer_node",
            "list_current_sales_orders_node": "list_current_sales_orders_node",
        },
    )
    workflow.add_conditional_edges(
        "feedback_ambiguous_customer_node",
        _route_after_ambiguous_feedback,
        {
            "list_current_sales_orders_node": "list_current_sales_orders_node",
            "feedback_unknown_customer_node": "feedback_unknown_customer_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )
    workflow.add_conditional_edges(
        "feedback_unknown_customer_node",
        _route_after_unknown_feedback,
        {
            "list_current_sales_orders_node": "list_current_sales_orders_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )

    # Subflujo de duplicados
    workflow.add_edge("list_current_sales_orders_node", "duplicate_candidates_judge_node")
    workflow.add_conditional_edges(
        "duplicate_candidates_judge_node",
        _route_after_duplicate_judge,
        {
            "feedback_duplicate_sales_node": "feedback_duplicate_sales_node",
            "update_quotation_node": "update_quotation_node",
            "create_quotation_node": "create_quotation_node",
            "confirm_order_node": "confirm_order_node",
            "guardrail_unlock_node": "guardrail_unlock_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )
    workflow.add_conditional_edges(
        "feedback_duplicate_sales_node",
        _route_after_duplicate_feedback,
        {
            "update_quotation_node": "update_quotation_node",
            "create_quotation_node": "create_quotation_node",
            "confirm_order_node": "confirm_order_node",
            "guardrail_unlock_node": "guardrail_unlock_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )

    # Creación y actualización de cotización
    workflow.add_conditional_edges(
        "create_quotation_node",
        _route_after_create_quotation,
        {
            "confirm_order_node": "confirm_order_node",
            "view_quotation_node": "view_quotation_node",
        },
    )
    workflow.add_edge("update_quotation_node", "view_quotation_node")
    workflow.add_edge("view_quotation_node", "synthesize_sales_response")

    # Confirmación
    workflow.add_edge("confirm_order_node", "synthesize_sales_response")

    # Rama Edit Order (desbloqueo ➔ actualización ➔ re-bloqueo)
    workflow.add_conditional_edges(
        "guardrail_unlock_node",
        _route_after_guardrail_unlock,
        {
            "unlock_order_node": "unlock_order_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )
    workflow.add_edge("unlock_order_node", "update_order_node")
    workflow.add_edge("update_order_node", "lock_order_node")
    workflow.add_edge("lock_order_node", "synthesize_sales_response")

    # Rama Remove
    workflow.add_conditional_edges(
        "feedback_remove_order_node",
        _route_after_remove_confirm,
        {
            "remove_order_node": "remove_order_node",
            "synthesize_sales_response": "synthesize_sales_response",
        },
    )
    workflow.add_edge("remove_order_node", "synthesize_sales_response")

    # Salida
    workflow.add_edge("synthesize_sales_response", END)

    cp = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=cp)


def get_sales_manage_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Factory helper para instanciar y compilar sales_manage con el LLM por defecto."""
    llm = get_default_llm()
    return build_sales_manage_graph(llm=llm, checkpointer=checkpointer)
