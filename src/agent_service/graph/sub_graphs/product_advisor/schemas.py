"""
src/agent_service/graph/sub_graphs/product_advisor/schemas.py - Modelos y esquemas Pydantic para product_advisor.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class AdvisorToolCall(BaseModel):
    """Definición estructurada de una invocación de herramienta decidida por el planificador."""
    tool_name: Literal[
        "search_product_catalog",
        "get_product_odoo_details",
        "get_cross_sell_recommendations",
        "filter_and_sort_products",
        "get_customer_purchase_history",
    ] = Field(description="Nombre exacto de la herramienta a invocar.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Argumentos pasados a la herramienta.",
    )
    purpose: str = Field(description="Razón y objetivo por el cual se invoca esta herramienta.")


class AdvisorPlan(BaseModel):
    """Plan de acción generado por el planificador para resolver la consulta del usuario."""
    reasoning: str = Field(
        description="Análisis estructurado de la intención, requerimientos y restricciones del usuario."
    )
    strategy: Literal[
        "direct_search",
        "personalized_recommendation",
        "cross_sell",
        "budget_filtering",
    ] = Field(
        default="direct_search",
        description="Estrategia general adoptada para responder a la necesidad.",
    )
    tool_calls: List[AdvisorToolCall] = Field(
        default_factory=list,
        description="Lista ordenada de herramientas a ejecutar para obtener los datos necesarios.",
    )
    extracted_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadatos comerciales detectados en la consulta (ej: {'marca': 'NombreMarca', 'pagina': 12, 'edicion': 'C10'}).",
    )


class QualityRubricEvaluation(BaseModel):
    """Rúbrica de evaluación multi-criterio para evaluar la calidad y fidelidad de la respuesta."""
    relevance_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿La respuesta satisface directamente la consulta o recomendación pedida?",
    )
    grounding_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿Los SKUs, precios y especificaciones provienen 100% de las herramientas sin alucinación?",
    )
    constraints_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿Se respetaron los filtros solicitados por el usuario (marca, página, edición, presupuesto, público objetivo / género)?",
    )
    presentation_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿El tono es comercial, conciso y adecuado al canal (WhatsApp / Web)?",
    )
    is_approved: bool = Field(
        ...,
        description="True si todos los criterios son >= 7.0 y el promedio es >= 8.0.",
    )
    critique: Optional[str] = Field(
        default=None,
        description="Explicación detallada de los puntos débiles o fallos si la rúbrica es rechazada.",
    )
    remedy_suggestions: Optional[List[str]] = Field(
        default_factory=list,
        description="Acciones correctivas sugeridas para que el planificador ajuste el plan en la siguiente iteración.",
    )


def format_products_for_advisor_prompt(products: List[Any], max_desc_len: int = 250) -> str:
    """Formatea una lista de productos (Document o dict) en texto conciso para inyectar en prompts,
    delegando en el sintetizador padre BaseSynthesizerNode."""
    from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
    return BaseSynthesizerNode.format_products_context(products, max_desc_len=max_desc_len)

