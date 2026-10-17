"""
src/agent_service/core/guardrails/evaluator.py - Evaluador Híbrido de Guardrails (Capa 1 Fast-Path + Capa 2 LLM Guardrail)
"""

import logging
import unicodedata
import re
from typing import Optional

from src.agent_service.core.guardrails.schemas import (
    ViolationCategory,
    GuardrailAction,
    GuardrailResult,
)
from src.agent_service.core.guardrails.rules import (
    PROMPT_INJECTION_PATTERNS,
    SYSTEM_LEAK_PATTERNS,
    SQL_INJECTION_PATTERNS,
    OBVIOUS_OUT_OF_SCOPE_PATTERNS,
    HARMFUL_PATTERNS,
    format_guardrail_refusal,
)
from src.agent_service.core.guardrails.llm_evaluator import (
    LLMGuardrailEvaluator,
    get_llm_guardrail_evaluator,
)

logger = logging.getLogger(__name__)

# Stub de compatibilidad hacia atrás para monkeypatches de pruebas existentes
run_laya_evaluation = None


def _strip_accents(text: str) -> str:
    """Elimina acentos y tildes para robustecer la comparación regex."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _clean_mobile_space_typos(text: str) -> str:
    """Repara saltos de línea continuos, letras sueltas y espaciado de mensajería móvil."""
    if not text:
        return text
    cleaned = re.sub(r"[\r\n]+", " ", text)
    cleaned = re.sub(r"\b([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})\s+([sS])\b", r"\1\2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_explicit_commercial_inquiry(text: str) -> bool:
    """Compatibilidad retrospectiva."""
    return True


def evaluate_fast_path_regex(raw_query: Optional[str]) -> Optional[GuardrailResult]:
    """Evalúa la Capa 1 determinista mediante regex compiladas (< 1ms).
    Retorna GuardrailResult si se detecta una violación crítica, o None si pasa limpio.
    """
    if not raw_query or not raw_query.strip():
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            is_blocked=False,
            is_warning=False,
            category=ViolationCategory.NONE,
            reason="Consulta vacía o sin texto.",
        )

    query = raw_query.strip()
    query_unaccented = _strip_accents(query)
    text_variants = [query, query_unaccented]

    # 1. Comprobación de Prompt Injection / Jailbreak directo
    for pattern, reason in PROMPT_INJECTION_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.warning(f"Guardrail Capa 1 Bloqueado [PROMPT_INJECTION]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    is_blocked=True,
                    is_warning=False,
                    category=ViolationCategory.PROMPT_INJECTION,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.PROMPT_INJECTION),
                )

    # 2. Comprobación de Extracción de System Prompts y Credenciales
    for pattern, reason in SYSTEM_LEAK_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.warning(f"Guardrail Capa 1 Bloqueado [SYSTEM_LEAK]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    is_blocked=True,
                    is_warning=False,
                    category=ViolationCategory.SYSTEM_LEAK,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.SYSTEM_LEAK),
                )

    # 3. Comprobación de Inyecciones SQL y comandos destructivos
    for pattern, reason in SQL_INJECTION_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.warning(f"Guardrail Capa 1 Bloqueado [SQL_INJECTION]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    is_blocked=True,
                    is_warning=False,
                    category=ViolationCategory.SQL_INJECTION,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.SQL_INJECTION),
                )

    # 4. Comprobación de Contenido Dañino
    for pattern, reason in HARMFUL_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.warning(f"Guardrail Capa 1 Bloqueado [HARMFUL_CONTENT]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    is_blocked=True,
                    is_warning=False,
                    category=ViolationCategory.HARMFUL_CONTENT,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.HARMFUL_CONTENT),
                )

    # 5. Comprobación de Solicitudes Obvias Fuera de Ámbito (Out-of-Scope)
    for pattern, reason in OBVIOUS_OUT_OF_SCOPE_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.info(f"Guardrail Capa 1 Desviado [OUT_OF_SCOPE]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    is_blocked=True,
                    is_warning=False,
                    category=ViolationCategory.OUT_OF_SCOPE,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.OUT_OF_SCOPE),
                )

    return None


def evaluate_input_guardrail(raw_query: Optional[str]) -> GuardrailResult:
    """Evaluación síncrona rápida de Capa 1 Regex. Si pasa la Capa 1, devuelve ALLOW."""
    fast_path_result = evaluate_fast_path_regex(raw_query)
    if fast_path_result is not None:
        return fast_path_result

    return GuardrailResult(
        action=GuardrailAction.ALLOW,
        is_blocked=False,
        is_warning=False,
        category=ViolationCategory.NONE,
        reason="Consulta segura: validada por Capa 1 determinista.",
    )


async def aevaluate_input_guardrail(
    raw_query: Optional[str],
    llm_evaluator: Optional[LLMGuardrailEvaluator] = None,
) -> GuardrailResult:
    """Evaluación híbrida completa:
    1. Capa 1 Fast-Path Regex (< 1ms). Si bloquea, retorna de inmediato.
    2. Capa 2 LLM Guardrail: Evalúa comprensión profunda del dominio comercial sin falsos positivos.
    """
    fast_path_result = evaluate_fast_path_regex(raw_query)
    if fast_path_result is not None:
        return fast_path_result

    query_cleaned = _clean_mobile_space_typos(raw_query or "")
    evaluator = llm_evaluator or get_llm_guardrail_evaluator()
    return await evaluator.aevaluate(query_cleaned)
