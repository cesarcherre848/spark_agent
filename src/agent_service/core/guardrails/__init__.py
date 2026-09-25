"""
src/agent_service/core/guardrails - Módulo central de Guardrails y Hard Constraints para Spark Agent
"""

from src.agent_service.core.guardrails.schemas import (
    GuardrailAction,
    ViolationCategory,
    GuardrailResult,
)
from src.agent_service.core.guardrails.rules import (
    HARD_CONSTRAINTS_DESCRIPTION,
    DEFAULT_REFUSAL_MESSAGE,
    DEFAULT_WARNING_MESSAGE,
    REFUSAL_BY_CATEGORY,
    format_guardrail_refusal,
    format_guardrail_warning,
    LAYA_GUARD_QUESTIONS,
    evaluate_laya_scores,
)
from src.agent_service.core.guardrails.evaluator import (
    evaluate_input_guardrail,
)

__all__ = [
    "GuardrailAction",
    "ViolationCategory",
    "GuardrailResult",
    "HARD_CONSTRAINTS_DESCRIPTION",
    "DEFAULT_REFUSAL_MESSAGE",
    "DEFAULT_WARNING_MESSAGE",
    "REFUSAL_BY_CATEGORY",
    "format_guardrail_refusal",
    "format_guardrail_warning",
    "LAYA_GUARD_QUESTIONS",
    "evaluate_laya_scores",
    "evaluate_input_guardrail",
]
