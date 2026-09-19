"""
src/agent_service/graph/sub_graphs/product_recomender/schemas.py - Esquemas Pydantic para product_recomender

Define modelos estructurados para:
1. Extracción de intención, filtros y top-k dinámico.
2. Fichas comerciales de productos enriquecidos con Odoo.
3. Rúbrica de evaluación objetiva en 4 criterios y auto-reflexión.
4. Síntesis ejecutiva final.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


RecommendationRelationType = Literal[
    "cross_sell",            # Productos complementarios para combinar (ej: lápiz + fijador)
    "up_sell",               # Alternativas de gama superior o mayor valor agregado
    "substitute",            # Alternativas directas o reemplazos equivalentes
    "general_recommendation" # Recomendación basada en necesidades o beneficios solicitados
]


SortByOption = Literal["price_asc", "price_desc", "relevance"]


class RecommendationIntentExtraction(BaseModel):
    """Extracción estructurada de la intención de recomendación del usuario."""
    base_product: Optional[str] = Field(
        default=None,
        description="Código SKU o nombre comercial del producto de referencia si el usuario lo indicó (ej: '6189', 'Sexy Glam').",
    )
    relation_type: RecommendationRelationType = Field(
        default="general_recommendation",
        description="Tipo de relación comercial solicitada ('cross_sell', 'up_sell', 'substitute', 'general_recommendation').",
    )
    top_k: int = Field(
        default=15,
        description="Cantidad de recomendaciones solicitadas por el usuario (ej: 2, 5, 15). Por defecto 15 si no se especificó número exacto.",
    )
    min_price: Optional[float] = Field(
        default=None,
        description="Precio mínimo indicado en la consulta si se especificó.",
    )
    max_price: Optional[float] = Field(
        default=None,
        description="Precio máximo o presupuesto límite indicado por el usuario (ej: 'menos de 100 soles' -> 100.0).",
    )
    category: Optional[str] = Field(
        default=None,
        description="Categoría o tipo de producto solicitado (ej: 'labiales', 'delineadores', 'cuidado facial').",
    )
    sort_by: SortByOption = Field(
        default="relevance",
        description=(
            "Criterio de ordenamiento: 'price_asc' si el usuario pide los más baratos o económicos; "
            "'price_desc' si pide opciones caras, premium o de alta gama; 'relevance' para orden por afinidad estándar."
        ),
    )
    required_attributes: List[str] = Field(
        default_factory=list,
        description="Atributos o beneficios explícitamente requeridos (ej: ['a prueba de agua', 'larga duración', 'mate']).",
    )
    search_query: str = Field(
        description="Consulta semántica depurada optimizada para búsqueda vectorial híbrida en el catálogo de productos.",
    )


class EnrichedRecommendedProduct(BaseModel):
    """Producto enriquecido con ficha técnica y datos comerciales oficiales de Odoo."""
    product_id: Optional[int] = Field(default=None, description="ID interno de Odoo (product.product).")
    sku: str = Field(description="Código comercial o SKU (default_code).")
    name: str = Field(description="Nombre comercial del producto.")
    price: Optional[float] = Field(default=None, description="Precio unitario de lista oficial en Odoo.")
    currency: str = Field(default="PEN", description="Moneda comercial oficial (ej: 'PEN', 'USD').")
    category: Optional[str] = Field(default=None, description="Categoría asignada en el catálogo de Odoo.")
    uom: Optional[str] = Field(default="Units", description="Unidad de medida comercial.")
    description: str = Field(default="", description="Descripción comercial de venta y características.")
    relation_type: str = Field(default="general_recommendation", description="Tipo de relación comercial asociada.")
    recommendation_reason: Optional[str] = Field(
        default=None,
        description="Motivo o argumento comercial por el cual este producto se recomienda al cliente.",
    )


class RubricCriteriaScores(BaseModel):
    """Calificaciones numéricas (1.0 a 5.0) para cada dimensión de la rúbrica de recomendación."""
    relevance: float = Field(
        ge=1.0,
        le=5.0,
        description="Relevancia comercial y afinidad con la necesidad o producto base indicado.",
    )
    filter_compliance: float = Field(
        ge=1.0,
        le=5.0,
        description="Cumplimiento estricto de filtros de usuario (presupuesto, categoría, características).",
    )
    diversity: float = Field(
        ge=1.0,
        le=5.0,
        description="Variedad y diferenciación de las opciones propuestas (sin redundancia ni opciones repetidas).",
    )
    data_completeness: float = Field(
        ge=1.0,
        le=5.0,
        description="Completitud de datos comerciales esenciales (precio válido, ficha técnica y stock).",
    )


class RubricEvaluationResult(BaseModel):
    """Evaluación objetiva del Juez LLM contra la Rúbrica de Calidad y Consistencia."""
    scores: RubricCriteriaScores = Field(
        description="Puntuación en cada uno de los 4 criterios de la rúbrica.",
    )
    meets_rubric: bool = Field(
        description=(
            "True si el conjunto de productos cumple la rúbrica de calidad: promedio ponderado >= 4.0, "
            "sin violaciones de precio máximo/mínimo ni filtros excluyentes. False si requiere refinamiento."
        ),
    )
    selected_skus: List[str] = Field(
        default_factory=list,
        description="Lista de SKUs de los mejores productos recomendados que satisfacen la rúbrica, truncada exactamente al top_k solicitado.",
    )
    critique: str = Field(
        description="Diagnóstico analítico de la calidad de las recomendaciones y áreas de mejora identificadas.",
    )
    suggested_refinement: Optional[str] = Field(
        default=None,
        description="Sugerencias de reformulación de búsqueda para el optimizador en caso de no cumplir la rúbrica.",
    )


class RecommendationSynthesisResponse(BaseModel):
    """Respuesta comercial estructurada dirigida al usuario final."""
    response_text: str = Field(
        description="Respuesta profesional, persuasiva y clara presentando las recomendaciones con sus precios y justificación.",
    )
