"""
src/agent_service/core/guardrails - Módulo central de Guardrails y Hard Constraints para Spark Agent
"""

from src.agent_service.core.guardrails.schemas import (
    ViolationCategory,
    GuardrailResult,
)
from src.agent_service.core.guardrails.rules import (
    HARD_CONSTRAINTS_DESCRIPTION,
    DEFAULT_REFUSAL_MESSAGE,
    REFUSAL_BY_CATEGORY,
    format_guardrail_refusal,
)
from src.agent_service.core.guardrails.evaluator import (
    evaluate_input_guardrail,
)

__all__ = [
    "ViolationCategory",
    "GuardrailResult",
    "HARD_CONSTRAINTS_DESCRIPTION",
    "DEFAULT_REFUSAL_MESSAGE",
    "REFUSAL_BY_CATEGORY",
    "format_guardrail_refusal",
    "evaluate_input_guardrail",
]
