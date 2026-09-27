"""
src/agent_service/graph/sub_graphs/product_advisor/state.py - Estado de LangGraph para product_advisor.
"""

from typing import Optional, List, Dict, Any, Union, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ProductAdvisorState(TypedDict, total=False):
    """Estado del subgrafo product_advisor unificando búsqueda semántica híbrida,
    recomendaciones, llamadas a herramientas en Odoo y ciclo de reflexión con rúbrica."""

    # Mensajes y sesión
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: Optional[Union[int, str]]
    partner_id: Optional[int]
    customer_name: Optional[str]
    session_id: Optional[str]
    channel: Optional[str]

    # Consultas y planeamiento
    raw_query: str
    refined_query: Optional[str]
    plan_rationale: Optional[str]
    planned_tools: Optional[List[Dict[str, Any]]]
    tool_observations: Optional[List[Dict[str, Any]]]

    # Candidatos y productos
    candidate_products: Optional[List[Dict[str, Any]]]
    retrieved_products: Optional[List[Any]]
    enriched_products: Optional[List[Dict[str, Any]]]
    filtered_products: Optional[List[Dict[str, Any]]]
    matched_skus: Optional[List[str]]

    # Historial y preferencias de cliente
    customer_purchase_history: Optional[Dict[str, Any]]

    # Filtros y restricciones de catálogo
    metadata_filters: Optional[Dict[str, Any]]
    budget_constraints: Optional[Dict[str, Any]]

    # Síntesis y borrador
    draft_response: Optional[str]
    final_response: Optional[str]

    # Evaluación por Rúbrica y Reflexión
    meets_rubric: bool
    rubric_scores: Optional[Dict[str, float]]
    critique: Optional[str]
    suggested_improvements: Optional[List[str]]
    iteration_count: int
    max_iterations: int

    # Compatibilidad con subgrafos previos
    is_sufficient: bool
    top_k: Optional[int]
    base_product: Optional[str]
    relation_type: Optional[str]
