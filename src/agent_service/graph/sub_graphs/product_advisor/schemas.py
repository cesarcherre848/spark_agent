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
        "rerank_and_filter_candidates",
        "get_customer_purchase_history",
        "get_active_incentive_targets",
    ] = Field(description="Nombre exacto de la herramienta a invocar.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Argumentos pasados a la herramienta.",
    )
    purpose: str = Field(description="Razón y objetivo por el cual se invoca esta herramienta.")


class CustomerIntentSpec(BaseModel):
    """Especificación semántica estructurada de intención para modular búsqueda y reranking dinámico."""
    target_gender: Optional[Literal["female", "male", "unisex", "kids"]] = Field(
        default=None,
        description="Género inferido del destinatario ('female', 'male', 'unisex', 'kids').",
    )
    target_age: Optional[int] = Field(
        default=None,
        description="Edad numérica del cliente o destinatario si fue mencionada (ej: 50, 32, 18, etc.).",
    )
    target_age_group: Optional[Literal["kids", "teen_young", "adult", "mature"]] = Field(
        default=None,
        description="Etapa de vida inferida a partir de la edad o contexto: 'kids' (<13), 'teen_young' (13-24), 'adult' (25-44), 'mature' (>=45).",
    )
    anatomical_focus: Optional[Literal["facial", "body", "hair", "fragrance", "makeup"]] = Field(
        default=None,
        description="Enfoque de producto: 'facial' (rostro), 'body' (cuerpo), 'hair' (cabello), 'fragrance' (colonia/perfume), 'makeup' (maquillaje).",
    )
    brand_preference: Optional[str] = Field(
        default=None,
        description="Marca preferida si fue indicada (ej: 'Yanbal', 'Ésika').",
    )
    product_profile: Optional[str] = Field(
        default=None,
        description="Perfil o estilo buscado (ej: 'fresca_diario', 'citrica', 'antiarrugas', 'hidratante').",
    )
    skin_type: Optional[str] = Field(
        default=None,
        description="Tipo de piel (ej: 'mixta', 'grasa', 'seca', 'sensible').",
    )
    excluded_targets: List[str] = Field(
        default_factory=list,
        description="Perfiles incompatibles a excluir (ej: ['juvenil_teen', 'masculino', 'corporal']).",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Tope de presupuesto si el usuario indicó un límite.",
    )


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
    intent_spec: Optional[CustomerIntentSpec] = Field(
        default=None,
        description="Perfil de intención semántica estructurada extraído para guiar búsqueda y reranking.",
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
    clarity_and_unambiguity_score: float = Field(
        default=8.0,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿La respuesta es 100% clara, sin afirmaciones ambiguas ni mezcla errónea de marcas/campañas ante referencias de seguimiento ('esto a qué catálogo pertenece')?",
    )
    intent_fulfillment_score: float = Field(
        default=8.0,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿La respuesta cumple fielmente con el cambio o preferencia solicitada en este turno (ej. menor precio que el turno previo o cobertura de categorías solicitadas)?",
    )
    criteria_clarity_score: float = Field(
        default=8.0,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿El criterio de la consulta está 100% claro? Si el criterio es ambiguo o relativo con disyuntivas marcadas en catálogo, califica < 7.0.",
    )
    is_approved: bool = Field(
        ...,
        description="True si todos los criterios son >= 7.0 y el promedio es >= 8.0.",
    )
    reflection_action: Literal[
        "approve",
        "refine_synthesis",
        "re_rerank_with_feedback",
        "replan_tools",
        "ask_clarification",
    ] = Field(
        default="approve",
        description="Acción de reflexión: 'approve' si pasa; 'refine_synthesis' si solo requiere ajuste de redacción/formato; 're_rerank_with_feedback' para re-ejecutar el reranker con feedback semántico (Medium Loop); 'replan_tools' si faltan datos en catálogo (Slow Loop); 'ask_clarification' si el criterio es ambiguo o hay conflicto de catálogo.",
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


class FinalAnswer(BaseModel):
    """Borrador o respuesta final comercial generada por el sintetizador."""
    response_text: str = Field(
        description="Respuesta comercial y técnica orientada al cliente explicando las opciones recomendadas o disculpándose cordialmente si no hay disponibilidad."
    )


