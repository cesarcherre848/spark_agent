"""
src/agent_service/core/guardrails/schemas.py - Modelos de datos para el sistema de Guardrails
"""

from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field


class GuardrailAction(str, Enum):
    """Acciones operativas resultantes de la evaluación de guardrails."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class ViolationCategory(str, Enum):
    """Categorías de violación de Hard Constraints en Spark Agent."""
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_LEAK = "system_leak"
    SQL_INJECTION = "sql_injection"
    OUT_OF_SCOPE = "out_of_scope"
    HARMFUL_CONTENT = "harmful_content"
    NONE = "none"


class GuardrailResult(BaseModel):
    """Resultado de la evaluación de un Guardrail sobre una consulta de usuario."""
    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="Acción decidida por las reglas de guardrail: allow, warn, block."
    )
    is_blocked: bool = Field(
        default=False,
        description="Indica si la consulta fue bloqueada por infringir una restricción dura."
    )
    is_warning: bool = Field(
        default=False,
        description="Indica si la consulta generó una advertencia comercial no bloqueante."
    )
    category: ViolationCategory = Field(
        default=ViolationCategory.NONE,
        description="Categoría de transgresión detectada."
    )
    reason: str = Field(
        default="",
        description="Explicación técnica concisa del motivo de la decisión."
    )
    warning_message: Optional[str] = Field(
        default=None,
        description="Mensaje educado de advertencia comercial para orientar al usuario sin interrumpir."
    )
    refusal_message: Optional[str] = Field(
        default=None,
        description="Mensaje educado con redirección comercial para devolver al usuario en caso de bloqueo."
    )
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Diccionario con los scores y probabilidades numéricas calculadas (noul/score)."
    )
