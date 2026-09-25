"""
src/agent_service/core/guardrails/laya_client.py - Cliente singleton con carga perezosa (lazy load) para el modelo Laya (Hugging Face)
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_LAYA_AGENT = None


def get_laya_agent(force_reload: bool = False):
    """Retorna la instancia singleton de Laya Agent cargada de forma perezosa.

    El modelo por defecto es 'convaiinnovations/laya-multilingual' (compatible con español y 100+ idiomas),
    y puede sobreescribirse mediante la variable de entorno LAYA_GUARDRAIL_MODEL.
    """
    global _LAYA_AGENT
    if _LAYA_AGENT is None or force_reload:
        try:
            from laya import load
            model_name = os.getenv("LAYA_GUARDRAIL_MODEL", "convaiinnovations/laya-multilingual")
            logger.info(f"Cargando modelo Laya para Guardrails desde Hugging Face: '{model_name}'...")
            _LAYA_AGENT = load(model_name)
            logger.info(f"Modelo Laya '{model_name}' cargado exitosamente.")
        except Exception as exc:
            logger.error(f"Fallo al inicializar el modelo Laya para Guardrails: {exc}", exc_info=True)
            _LAYA_AGENT = None
            raise exc

    return _LAYA_AGENT


def run_laya_evaluation(
    query: str,
    questions: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ejecuta la inferencia de preguntas tipadas sobre la consulta utilizando Laya.

    Argumentos:
        query: Texto de la consulta del usuario a evaluar.
        questions: Diccionario de preguntas tipadas con primitivas 'score' y 'noul'.

    Retorna:
        Diccionario con las respuestas generadas por Laya o None si ocurre una excepción.
    """
    if not query or not query.strip():
        return None

    try:
        agent = get_laya_agent()
        if agent is None:
            return None

        # agent.predict ejecuta el forward pass no-autorregresivo
        result = agent.predict(query.strip(), questions)
        return result
    except Exception as exc:
        logger.warning(
            f"Error durante la inferencia de Guardrail con Laya ('{exc}'). "
            f"Fallback automático a Capa 1 determinista."
        )
        return None


def reset_laya_agent() -> None:
    """Restablece la instancia en memoria (utilizado para pruebas unitarias)."""
    global _LAYA_AGENT
    _LAYA_AGENT = None
