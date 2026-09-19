"""
src/agent_service/soul/__init__.py - API Pública del Módulo Global SOUL de Spark Agent
"""

from src.agent_service.soul.persona import SoulPersona
from src.agent_service.soul.tone_and_style import SoulToneAndStyle
from src.agent_service.soul.subtlety import SoulSubtlety
from src.agent_service.soul.prompts import (
    SoulRole,
    get_soul_prompt,
    inject_soul,
)

__all__ = [
    "SoulPersona",
    "SoulToneAndStyle",
    "SoulSubtlety",
    "SoulRole",
    "get_soul_prompt",
    "inject_soul",
]
