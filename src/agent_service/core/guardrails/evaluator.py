"""
src/agent_service/core/guardrails/evaluator.py - Evaluador Híbrido de Guardrails (Capa 1 Fast-Path + Capa 2 Laya System 1)
"""

import logging
import unicodedata
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
    LAYA_GUARD_QUESTIONS,
    format_guardrail_refusal,
    evaluate_laya_scores,
)
from src.agent_service.core.guardrails.laya_client import run_laya_evaluation

import re

logger = logging.getLogger(__name__)

COMMERCIAL_CATALOG_KEYWORDS = {
    "perfume", "perfumes", "colonia", "colonias", "fragancia", "fragancias",
    "labial", "labiales", "crema", "cremas", "maquillaje", "rimel", "máscara",
    "shampoo", "desodorante", "bloqueador", "protector", "reloj", "aretes",
    "yanbal", "yambal", "esika", "ésika", "essika", "cyzone", "lbel", "l'bel",
    "catalogo", "catálogo", "precio", "precios", "cuanto", "cuánto", "costo",
    "hombre", "hombres", "mujer", "mujeres", "niño", "niños", "niña", "niñas",
    "comprar", "cotizar", "pedido", "producto", "productos", "opcion", "opciones",
    "gama", "linea", "línea", "marca", "marcas", "ofertas", "descuentos"
}


def _strip_accents(text: str) -> str:
    """Elimina acentos y tildes para robustecer la comparación regex."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _clean_mobile_space_typos(text: str) -> str:
    """Repara letras sueltas y errores de espaciado comunes en móviles (ej: 'hombre s de' -> 'hombres de')
    para evitar que letras huérfanas activen falsos positivos de token splitting en Laya."""
    if not text:
        return text
    cleaned = re.sub(r"\b([a-zA-ZáéíóúÁÉÍÓÚñÑ]{2,})\s+([sS])\b", r"\1\2", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_explicit_commercial_inquiry(text: str) -> bool:
    """Verifica si la consulta contiene términos comerciales inequívocos de catálogo."""
    if not text:
        return False
    words = {w.strip("?,.:;!¡¿\"'()").lower() for w in text.split()}
    return bool(words.intersection(COMMERCIAL_CATALOG_KEYWORDS))


def evaluate_input_guardrail(raw_query: Optional[str]) -> GuardrailResult:
    """Evalúa la consulta de usuario usando una arquitectura híbrida de dos niveles:
    
    1. Capa 1 (Fast-Path Determinista < 1ms):
       Expresiones regulares precompiladas para interceptar ataques destructivos críticos:
       - Inyecciones SQL (DROP, TRUNCATE, DELETE FROM).
       - Comandos destructivos de SO o ejecución de código.
       - Extracción directa de credenciales o system prompts.
       - Jailbreaks directos y solicitudes fuera de ámbito explícitas.
    
    2. Capa 2 (Laya Decision Engine ~30ms):
       Inferencia no-autorregresiva en Hugging Face con primitivas 'score' y 'noul':
       - Detección semántica de jailbreaks y manipulación.
       - Exfiltración sutil de información.
       - Desvío de contexto comercial (out of scope).
       - Nivel de daño/severidad (score 0..3).
       Aplica reglas para decidir si continuar limpio (ALLOW), advertir amistosamente (WARN),
       o bloquear (BLOCK).

    Retorna:
        GuardrailResult con acción, flags, categoría, razones y mensajes correspondientes.
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

    # =========================================================================
    # CAPA 1: FAST-PATH DETERMINISTA REGEX (< 1ms)
    # =========================================================================

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

    # =========================================================================
    # CAPA 2: LAYA SYSTEM 1 DECISION ENGINE (~30ms)
    # =========================================================================
    try:
        query_cleaned = _clean_mobile_space_typos(query)
        laya_result = run_laya_evaluation(query_cleaned, LAYA_GUARD_QUESTIONS)
    except Exception as exc:
        logger.warning(f"Excepción durante la ejecución de Laya ('{exc}'). Fallback a Capa 1 determinista.")
        laya_result = None

    if laya_result is not None:
        answers = laya_result.get("answers", {})
        action, category, reason, warning_msg, refusal_msg, scores = evaluate_laya_scores(answers)

        # Salvaguarda de Dominio Comercial: si Capa 1 no detectó transgresiones reales, la consulta es
        # explícitamente de catálogo y Laya determinó que no es severamente dañina (< 2.0) ni fuera de ámbito (<= 0.30):
        is_comm = is_explicit_commercial_inquiry(query) or is_explicit_commercial_inquiry(query_cleaned)
        p_out_scope = scores.get("out_of_scope", 0.0)
        score_harm = scores.get("harm_severity", 0.0)
        if is_comm and p_out_scope <= 0.30 and score_harm < 2.0:
            if action != GuardrailAction.ALLOW:
                logger.info(
                    f"Guardrail Capa 2: Desactivando falso {action.value} de Laya ({category.value}) para consulta comercial legítima: '{raw_query}'"
                )
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                is_blocked=False,
                is_warning=False,
                category=ViolationCategory.NONE,
                reason="Consulta comercial legítima protegida contra falsos positivos de Laya.",
                scores=scores,
            )

        if action == GuardrailAction.BLOCK:

            logger.warning(
                f"Guardrail Capa 2 (Laya) Bloqueado [{category.value}]: {reason} - Scores: {scores} - Query: '{raw_query}'"
            )
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                is_blocked=True,
                is_warning=False,
                category=category,
                reason=reason,
                refusal_message=refusal_msg or format_guardrail_refusal(category),
                scores=scores,
            )

        if action == GuardrailAction.WARN:
            logger.info(
                f"Guardrail Capa 2 (Laya) Advertencia [{category.value}]: {reason} - Scores: {scores} - Query: '{raw_query}'"
            )
            return GuardrailResult(
                action=GuardrailAction.WARN,
                is_blocked=False,
                is_warning=True,
                category=category,
                reason=reason,
                warning_message=warning_msg,
                scores=scores,
            )

        # ALLOW
        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            is_blocked=False,
            is_warning=False,
            category=ViolationCategory.NONE,
            reason=reason,
            scores=scores,
        )

    # Fallback seguro si Laya no está disponible o falla durante la ejecución
    return GuardrailResult(
        action=GuardrailAction.ALLOW,
        is_blocked=False,
        is_warning=False,
        category=ViolationCategory.NONE,
        reason="Consulta segura: validada por Capa 1 determinista.",
    )
