"""
src/agent_service/graph/sub_graphs/sales_manage/state.py - Definición del estado de LangGraph para sales_manage
"""

from typing import Optional, List, Dict, Any, Literal, Annotated, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SalesManageState(TypedDict, total=False):
    """Estado del subgrafo de gestión de órdenes de venta, cotizaciones y pedidos en Odoo."""
    messages: Annotated[List[BaseMessage], add_messages]
    raw_query: Optional[str]
    user_id: Optional[Union[int, str]]
    session_id: Optional[str]

    # Extracción de intención y atributos
    sales_action: Optional[Literal["list", "view", "upsert", "add_items", "remove_items", "confirm", "edit_order", "remove"]]
    customer_name: Optional[str]
    items: List[Dict[str, Any]]
    status_filter: Optional[str]
    period_filter: Optional[str]
    target_order_name: Optional[str]  # Código visible (ej: SO001)

    # Patrón de Reflexión y Aclaración de Intención
    is_intent_clear: Optional[bool]
    reflection_reasoning: Optional[str]
    clarification_question: Optional[str]
    clarification_options: Optional[List[str]]


    # Resolución de cliente en cartera
    partner_id: Optional[int]  # ID numérico interno en Odoo (nunca mostrado al usuario)
    candidate_partners: Optional[List[Dict[str, Any]]]  # Clientes similares para desambiguar
    customer_resolved: bool
    customer_not_found: bool

    # Rama List (órdenes agrupadas por estado)
    orders_list: Optional[Dict[str, List[Dict[str, Any]]]]

    # Subflujo Duplicados
    current_candidates: Optional[List[Dict[str, Any]]]
    has_possible_duplicates: bool
    duplicate_rationale: Optional[str]
    duplicate_choice: Optional[Literal["new", "selected", "cancel"]]
    target_order_id: Optional[int]  # ID numérico interno de la orden en Odoo

    # Guardrails y confirmaciones HITL
    unlock_confirmed: bool
    remove_confirmed: bool

    # Resultado y Síntesis
    order_view_data: Optional[Dict[str, Any]]
    operation_result: Optional[Dict[str, Any]]
    cancellation_reason: Optional[str]
    final_response: Optional[str]

