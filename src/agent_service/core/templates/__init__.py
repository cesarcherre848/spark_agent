"""
src/agent_service/core/templates/__init__.py - Módulo de Plantillas de Diálogo y Textos Centralizados de MIA
"""

from src.agent_service.core.templates.dialogs import (
    HITLDialogTemplates,
    WhitelabelSanitizer,
    FORBIDDEN_BACKEND_TERMS,
)

__all__ = [
    "HITLDialogTemplates",
    "WhitelabelSanitizer",
    "FORBIDDEN_BACKEND_TERMS",
]
