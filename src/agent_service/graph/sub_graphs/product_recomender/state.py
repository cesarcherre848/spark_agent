"""
src/agent_service/graph/sub_graphs/product_recomender/state.py - Estado de LangGraph para product_recomender
"""

from typing import Optional, List, Dict, Any, Union, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ProductRecomenderState(TypedDict, total=False):
    """Estado del subgrafo de recomendación de productos con patrón de reflexión y top-k dinámico."""
    # Mensajes y sesión
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: Optional[Union[int, str]]
    raw_query: Optional[str]
    session_id: Optional[str]

    # Extracción de intención y parámetros
    base_product: Optional[str]               # SKU o nombre del producto base (ej: '6189', 'Sexy Glam')
    relation_type: Optional[str]              # 'cross_sell', 'up_sell', 'substitute', 'general_recommendation'
    top_k: int                                # Cantidad final solicitada por el usuario (dinámica, default 15)
    sort_by: Optional[str]                    # 'price_asc', 'price_desc', 'relevance'
    filters: Dict[str, Any]                   # min_price, max_price, category, required_attributes
    search_query: str                         # Términos semánticos optimizados para vector store

    # Candidatos y enriquecimiento
    candidate_documents: List[Any]            # Documentos recuperados con regla 3x
    enriched_products: List[Dict[str, Any]]   # Candidatos cruzados con Odoo ERP (precios, stock, ficha)
    filtered_products: List[Dict[str, Any]]   # Candidatos tras filtros duros y exclusión de auto-recomendación
    final_recommended_products: List[Dict[str, Any]] # Los top-k seleccionados tras evaluación

    # Ciclo de reflexión y rúbrica
    meets_rubric: bool                        # Veredicto del Juez Evaluador
    rubric_scores: Dict[str, float]           # Puntuaciones numéricas por criterio
    critique: Optional[str]                   # Diagnóstico analítico de fallos
    suggested_refinement: Optional[str]       # Guía de ajuste de búsqueda para el optimizador
    iteration_count: int                      # Contador de iteraciones del bucle de reflexión
    max_iterations: int                       # Límite máximo de iteraciones permitidas (default 2)

    # Respuesta final
    final_response: Optional[str]             # Texto comercial generado para el usuario
