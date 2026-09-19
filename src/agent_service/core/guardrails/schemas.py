"""
src/agent_service/core/guardrails/schemas.py - Modelos de datos para el sistema de Guardrails
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
    is_blocked: bool = Field(
        default=False,
        description="Indica si la consulta fue bloqueada por infringir una restricción dura."
    )
    category: ViolationCategory = Field(
        default=ViolationCategory.NONE,
        description="Categoría de transgresión detectada."
    )
    reason: str = Field(
        default="",
        description="Explicación técnica concisa del motivo del bloqueo."
    )
    refusal_message: Optional[str] = Field(
        default=None,
        description="Mensaje educado con redirección comercial para devolver al usuario."
    )
