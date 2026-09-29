"""
src/agent_service/graph/sub_graphs/sales_incentives/state.py - Estado tipado del subgrafo sales_incentives con soporte multimarca.
"""

from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage


class SalesIncentivesState(TypedDict, total=False):
    """Estado compartido a lo largo del flujo Actor-Critic en sales_incentives."""
    # Contexto conversacional y del usuario
    messages: List[BaseMessage]
    raw_query: Optional[str]
    user_id: Optional[int]
    partner_id: Optional[int]
    customer_name: Optional[str]
    supplier_id: Optional[int]
    supplier_name: Optional[str]
    target_supplier: Optional[Dict[str, Any]]
    is_multibrand: Optional[bool]
    channel: Optional[str]
    user_context: Optional[str]

    # Datos recopilados de Odoo ERP
    orders_data: Optional[Dict[str, Any]]
    active_incentives: Optional[List[Dict[str, Any]]]
    sales_trends: Optional[List[Dict[str, Any]]]
    gap_info: Optional[Dict[str, Any]]
    multibrand_gaps: Optional[Dict[int, Dict[str, Any]]]
    multibrand_metrics: Optional[Dict[int, Dict[str, Any]]]
    current_date_str: Optional[str]

    # Planificación Actor-Critic con política ε-greedy
    epsilon: Optional[float]
    candidate_plan: Optional[Dict[str, Any]]
    critic_score: Optional[float]
    critic_evaluation: Optional[Dict[str, Any]]
    critique: Optional[str]
    suggested_improvements: Optional[List[str]]
    is_approved: Optional[bool]
    iteration_count: Optional[int]
    max_iterations: Optional[int]

    # Respuestas comerciales
    draft_response: Optional[str]
    final_response: Optional[str]
