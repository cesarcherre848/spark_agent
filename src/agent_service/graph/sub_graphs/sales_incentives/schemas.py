"""
src/agent_service/graph/sub_graphs/sales_incentives/schemas.py - Modelos Pydantic para Actor-Critic en sales_incentives con soporte de Planificación Inteligente.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SuggestedProductCombo(BaseModel):
    """Propuesta de combinación de productos para venta sugerida."""
    combo_name: str = Field(description="Nombre comercial atractivo del combo o paquete.")
    skus: List[str] = Field(description="Lista de códigos SKU incluidos en la recomendación.")
    supplier_name: Optional[str] = Field(None, description="Marca o fabricante del combo (ej. 'Yanbal', 'Belcorp').")
    target_customer: Optional[str] = Field(None, description="Cliente sugerido de la cartera para ofrecer este combo.")
    estimated_amount_pen: float = Field(description="Monto total estimado de venta en PEN (S/.).")
    strategy_type: Literal["incentive_bonus", "organic_trend", "hybrid"] = Field(
        description="Estrategia: 'incentive_bonus' (explotación de bono), 'organic_trend' (exploración de tendencia) o 'hybrid'."
    )
    rationale: str = Field(description="Justificación comercial y beneficio que aporta a las metas.")


class IncentiveMilestone(BaseModel):
    """Objetivo o hito comercial a alcanzar dentro de la campaña."""
    milestone_name: str = Field(description="Nombre del hito (ej: 'Salto a 30% de Descuento', 'Colonia Floral de Regalo', 'Tarjeta Visa S/ 700').")
    supplier_name: Optional[str] = Field(None, description="Marca o proveedor asociado (ej. 'Unique S.A.', 'CETCO S.A.').")
    target_type: Literal["discount_tier", "gift_product", "cash_card", "rappel"] = Field(
        description="Tipo de beneficio de incentivo."
    )
    gap_pen: Optional[float] = Field(None, description="Monto en soles faltante para alcanzar el hito.")
    gap_units: Optional[int] = Field(None, description="Unidades faltantes de productos calificados.")
    reward_description: str = Field(description="Detalle del premio, descuento o bono ganado.")
    action_steps: List[str] = Field(description="Pasos concretos e inmediatos para cerrar la brecha.")


class SupplierIncentivePlan(BaseModel):
    """Sub-plan de incentivos y metas desglosado para un proveedor o marca específica."""
    supplier_id: int = Field(description="ID del proveedor en Odoo (ej. 9 para Unique S.A., 17 para CETCO S.A.).")
    supplier_name: str = Field(description="Razón social del proveedor (ej. 'Unique S.A.', 'CETCO S.A.').")
    brand_display_name: str = Field(description="Nombre comercial de la marca (ej. 'Yanbal (Unique)', 'Belcorp (Ésika / Cyzone / L\'Bel)').")
    confirmed_total_pen: float = Field(default=0.0, description="Ventas confirmadas en PEN de esta marca.")
    draft_total_pen: float = Field(default=0.0, description="Cotizaciones en borrador en PEN de esta marca.")
    current_status_summary: str = Field(description="Resumen de la escala o estado actual de esta marca.")
    target_summary: str = Field(description="Próxima meta o escala a conquistar en esta marca.")
    gap_summary: str = Field(description="Brecha faltante en soles (S/.) o unidades para esta marca.")
    milestones: List[IncentiveMilestone] = Field(default_factory=list, description="Hitos comerciales específicos de esta marca.")
    suggested_combos: List[SuggestedProductCombo] = Field(default_factory=list, description="Combos de productos recomendados de esta marca.")


class IncentiveActionPlan(BaseModel):
    """Estrategia comercial generada por el Actor (Planificador Estratégico)."""
    detected_user_intent: str = Field(
        default="Coaching de ventas e incentivos",
        description="Intención específica detectada del usuario (ej: 'Consulta sobre el mayor incentivo de Yanbal', 'Diagnóstico de ventas y brechas', 'Consulta de avance multimarca').",
    )
    target_brand_detected: str = Field(
        default="Yanbal (Unique S.A.)",
        description="Marca o proveedor identificado en la consulta (ej. 'Yanbal (Unique S.A.)', 'Belcorp (CETCO S.A.)' o 'Multimarca / Todas las marcas').",
    )
    direct_answer: str = Field(
        default="",
        description="Respuesta directa, completa y contundente a la pregunta específica del usuario (ej. si preguntó por el mayor incentivo de Yanbal, describe el Nivel 4 de GanaMás con Tarjeta Visa de S/ 700 y la Escala de 35%).",
    )
    summary: str = Field(description="Resumen motivador de coaching para el vendedor.")
    current_status_overview: str = Field(default="", description="Diagnóstico del volumen actual vendido y escalas alcanzadas si es relevante.")
    is_multibrand: bool = Field(default=False, description="True solo si el usuario consultó de forma general por todas las marcas.")
    target_supplier_name: Optional[str] = Field(None, description="Nombre de la marca si la consulta fue específica (ej. 'Unique S.A.', 'CETCO S.A.').")
    featured_promotions: List[str] = Field(
        default_factory=list,
        description="Lista de incentivos o promociones activas en Odoo que responden o sustentan la estrategia.",
    )
    supplier_plans: List[SupplierIncentivePlan] = Field(default_factory=list, description="Planes desglosados por proveedor/marca.")
    milestones: List[IncentiveMilestone] = Field(default_factory=list, description="Hitos globales o priorizados a conquistar.")
    suggested_combos: List[SuggestedProductCombo] = Field(default_factory=list, description="Combos o productos recomendados según política ε-greedy.")
    projected_additional_earnings_pen: float = Field(default=0.0, description="Ganancia o valor de premios adicional proyectado.")
    action_call: str = Field(
        default="¿Deseas que preparemos un pedido para avanzar hacia esta meta?",
        description="Llamado a la acción motivador para el vendedor.",
    )
    epsilon_balance_rationale: str = Field(
        default="",
        description="Explicación del balance entre productos de incentivo (75%) y tendencias de pedidos reales (25%)."
    )


class CriticPlanEvaluation(BaseModel):
    """Evaluación técnica y de negocio realizada por el Critic."""
    is_approved: bool = Field(description="True si el plan es viable, realista y maximiza el retorno del vendedor (score >= 8.0).")
    score: float = Field(description="Puntuación integral asignada al plan (escala de 0.0 a 10.0).")
    alignment_score: float = Field(default=9.0, description="Calificación de respuesta directa a lo que preguntó el usuario (0 a 10).")
    temporal_feasibility_score: float = Field(description="Calificación de viabilidad en el tiempo restante de campaña (0 a 10).")
    constraint_adherence_score: float = Field(description="Respeto de restricciones de presupuestos, mínimos y catálogo (0 a 10).")
    brand_integrity_score: float = Field(default=8.5, description="Calificación de aislamiento estricto de marca si se pidió una en específico (0 a 10).")
    odoo_veracity_score: float = Field(default=9.0, description="Calificación de uso exclusivo de promociones reales activas en Odoo (0 a 10).")
    roi_score: float = Field(description="Puntuación de optimización del beneficio neto para el vendedor (0 a 10).")
    critiques: List[str] = Field(default_factory=list, description="Observaciones o debilidades identificadas en el plan.")
    suggested_adjustments: List[str] = Field(default_factory=list, description="Ajustes obligatorios requeridos para el Actor.")
