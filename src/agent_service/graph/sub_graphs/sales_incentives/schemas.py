"""
src/agent_service/graph/sub_graphs/sales_incentives/schemas.py - Modelos Pydantic para Actor-Critic en sales_incentives.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SuggestedProductCombo(BaseModel):
    """Propuesta de combinación de productos para venta sugerida."""
    combo_name: str = Field(description="Nombre comercial atractivo del combo o paquete.")
    skus: List[str] = Field(description="Lista de códigos SKU incluidos en la recomendación.")
    target_customer: Optional[str] = Field(None, description="Cliente sugerido de la cartera para ofrecer este combo.")
    estimated_amount_pen: float = Field(description="Monto total estimado de venta en PEN (S/.).")
    strategy_type: Literal["incentive_bonus", "organic_trend", "hybrid"] = Field(
        description="Estrategia: 'incentive_bonus' (explotación de bono), 'organic_trend' (exploración de tendencia) o 'hybrid'."
    )
    rationale: str = Field(description="Justificación comercial y beneficio que aporta a las metas.")


class IncentiveMilestone(BaseModel):
    """Objetivo o hito comercial a alcanzar dentro de la campaña."""
    milestone_name: str = Field(description="Nombre del hito (ej: 'Salto a 30% de Descuento', 'Colonia Soy Sexy Gratis').")
    target_type: Literal["discount_tier", "gift_product", "cash_card", "rappel"] = Field(
        description="Tipo de beneficio de incentivo."
    )
    gap_pen: Optional[float] = Field(None, description="Monto en soles faltante para alcanzar el hito.")
    gap_units: Optional[int] = Field(None, description="Unidades faltantes de productos calificados.")
    reward_description: str = Field(description="Detalle del premio, descuento o bono ganado.")
    action_steps: List[str] = Field(description="Pasos concretos e inmediatos para cerrar la brecha.")


class IncentiveActionPlan(BaseModel):
    """Estrategia comercial generada por el Actor."""
    summary: str = Field(description="Resumen ejecutivo del plan motivador para el vendedor.")
    current_status_overview: str = Field(description="Diagnóstico del volumen actual vendido y escala alcanzada.")
    milestones: List[IncentiveMilestone] = Field(description="Hitos priorizados a conquistar.")
    suggested_combos: List[SuggestedProductCombo] = Field(description="Combos o productos recomendados según política ε-greedy.")
    projected_additional_earnings_pen: float = Field(description="Ganancia o valor de premios adicional proyectado.")
    epsilon_balance_rationale: str = Field(
        description="Explicación del balance entre productos de incentivo (75%) y tendencias de pedidos reales (25%)."
    )


class CriticPlanEvaluation(BaseModel):
    """Evaluación técnica y de negocio realizada por el Critic."""
    is_approved: bool = Field(description="True si el plan es viable, realista y maximiza el retorno del vendedor (score >= 8.0).")
    score: float = Field(description="Puntuación integral asignada al plan (escala de 0.0 a 10.0).")
    temporal_feasibility_score: float = Field(description="Calificación de viabilidad en el tiempo restante de campaña (0 a 10).")
    constraint_adherence_score: float = Field(description="Respeto de restricciones de presupuestos, mínimos y catálogo (0 a 10).")
    roi_score: float = Field(description="Puntuación de optimización del beneficio neto para el vendedor (0 a 10).")
    critiques: List[str] = Field(default_factory=list, description="Observaciones o debilidades identificadas en el plan.")
    suggested_adjustments: List[str] = Field(default_factory=list, description="Ajustes obligatorios requeridos para el Actor.")
