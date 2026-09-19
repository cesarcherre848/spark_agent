"""
src/agent_service/soul/prompts.py - Ensamblador Moderado e Inyector Global de SOUL

Diseñado para un óptimo balance entre riqueza de comportamiento, consumo de tokens y latencia.
"""

from enum import Enum
from typing import Optional, Union
import textwrap

from src.agent_service.soul.persona import SoulPersona
from src.agent_service.soul.tone_and_style import SoulToneAndStyle
from src.agent_service.soul.subtlety import SoulSubtlety


class SoulRole(str, Enum):
    """Especializaciones operativas de rol conservando el mismo núcleo de SOUL."""
    GENERAL = "general"
    CATALOG_RAG = "catalog_rag"
    RECOMMENDER = "recommender"
    QUOTATION = "quotation"
    SALES_ORDERS = "sales_orders"
    CONTACTS = "contacts"


_ROLE_SPECIALIZATIONS = {
    SoulRole.GENERAL: "Rol: Orientadora y Asistente Comercial General (orientación en catálogo, pedidos y cartera).",
    SoulRole.CATALOG_RAG: "Rol: Consultora Experta en Catálogo (presentación técnica de productos, beneficios y SKUs oficiales).",
    SoulRole.RECOMMENDER: "Rol: Estratega Consultiva de Recomendación (cross-selling y alternativas afines con justificación).",
    SoulRole.QUOTATION: "Rol: Asesora de Consolidación de Cotizaciones (desglose por proveedor, cantidades y subtotales).",
    SoulRole.SALES_ORDERS: "Rol: Copiloto Ejecutiva de Gestión de Ventas y Pedidos Multimarca (seguimiento, confirmación y control de órdenes).",
    SoulRole.CONTACTS: "Rol: Gestora Diplomática de Cartera Comercial (altas, consultas y prevención de clientes duplicados).",
}


def get_soul_prompt(role: Optional[Union[SoulRole, str]] = None) -> str:
    """Genera el bloque de prompt maestro de SOUL balanceado en tokens y latencia.
    
    Args:
        role: Rol operativo específico opcional para contextualizar el arquetipo.

    Returns:
        String con el prompt constitucional de SOUL conciso y de alta densidad.
    """
    resolved_role: Optional[SoulRole] = None
    if role:
        try:
            resolved_role = SoulRole(role) if isinstance(role, str) else role
        except (ValueError, KeyError):
            resolved_role = None

    role_line = f"\n{_ROLE_SPECIALIZATIONS[resolved_role]}" if resolved_role and resolved_role in _ROLE_SPECIALIZATIONS else ""

    sections = [
        f"Eres **{SoulPersona.NAME}** ({SoulPersona.ROLE}). {SoulPersona.CORE_MISSION}{role_line}",
        "### DIRECTRICES DE IDENTIDAD E INTERACCIÓN:\n" + SoulPersona.INTERACTION_PRINCIPLES,
        "### TONO Y ESTILO:\n" + SoulToneAndStyle.TONE_GUIDELINES,
        "### FORMATO Y PRIVACIDAD:\n" + SoulToneAndStyle.WRITING_STYLE_GUIDELINES,
        "### SUTILEZA Y VENTA CONSULTIVA:\n" + SoulSubtlety.SUBTLETY_GUIDELINES,
    ]

    return "\n\n".join(sections)


def inject_soul(
    task_prompt: str,
    role: Optional[Union[SoulRole, str]] = None,
    extra_context: Optional[str] = None,
) -> str:
    """Inyecta la identidad global de SOUL en las directrices de una tarea o nodo específico.

    Args:
        task_prompt: Instrucciones particulares del nodo o subgrafo.
        role: Especialización de rol para este nodo.
        extra_context: Bloque de contexto adicional o restricciones locales (opcional).

    Returns:
        System prompt unificado y moderado listo para SystemMessage(content=...).
    """
    soul_core = get_soul_prompt(role=role)
    task_clean = textwrap.dedent(task_prompt).strip()

    parts = [
        soul_core,
        "\n### TAREA ESPECÍFICA:\n" + task_clean,
    ]

    if extra_context and extra_context.strip():
        parts.append("\n### CONTEXTO COMPLEMENTARIO:\n" + textwrap.dedent(extra_context).strip())

    return "\n".join(parts)
