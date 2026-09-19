"""
src/agent_service/core/guardrails/evaluator.py - Evaluador Determinista de Guardrail (Capa 1 Fast-Path)
"""

import logging
import unicodedata
from typing import Optional

from src.agent_service.core.guardrails.schemas import ViolationCategory, GuardrailResult
from src.agent_service.core.guardrails.rules import (
    PROMPT_INJECTION_PATTERNS,
    SYSTEM_LEAK_PATTERNS,
    SQL_INJECTION_PATTERNS,
    OBVIOUS_OUT_OF_SCOPE_PATTERNS,
    HARMFUL_PATTERNS,
    format_guardrail_refusal,
)

logger = logging.getLogger(__name__)


def _strip_accents(text: str) -> str:
    """Elimina acentos y tildes para robustecer la comparación regex."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def evaluate_input_guardrail(raw_query: Optional[str]) -> GuardrailResult:
    """Evalúa de forma determinista y ultrarrápida (< 1ms) las restricciones duras de entrada.

    Comprueba en orden de criticidad:
    1. Intentos de Prompt Injection y Jailbreaking.
    2. Intentos de filtración de System Prompts o credenciales.
    3. Inyecciones SQL o comandos destructivos de sistema.
    4. Actividades peligrosas o dañinas.
    5. Solicitudes explícitas fuera de contexto comercial (código, tareas, redacción de ficción).

    Retorna:
        GuardrailResult indicando si la consulta debe ser bloqueada inmediatamente.
    """
    if not raw_query or not raw_query.strip():
        return GuardrailResult(
            is_blocked=False,
            category=ViolationCategory.NONE,
            reason="Consulta vacía o sin texto.",
        )

    query = raw_query.strip()
    query_unaccented = _strip_accents(query)
    # Lista de variantes de texto para cotejar (original y sin acentos)
    text_variants = [query, query_unaccented]

    # 1. Comprobación de Prompt Injection / Jailbreak
    for pattern, reason in PROMPT_INJECTION_PATTERNS:
        for variant in text_variants:
            if pattern.search(variant):
                logger.warning(f"Guardrail Capa 1 Bloqueado [PROMPT_INJECTION]: {reason} - Query: '{raw_query}'")
                return GuardrailResult(
                    is_blocked=True,
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
                    is_blocked=True,
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
                    is_blocked=True,
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
                    is_blocked=True,
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
                    is_blocked=True,
                    category=ViolationCategory.OUT_OF_SCOPE,
                    reason=reason,
                    refusal_message=format_guardrail_refusal(ViolationCategory.OUT_OF_SCOPE),
                )

    # Consulta segura para continuar el flujo comercial
    return GuardrailResult(
        is_blocked=False,
        category=ViolationCategory.NONE,
        reason="Consulta segura: no transgrede restricciones duras de Capa 1.",
    )
