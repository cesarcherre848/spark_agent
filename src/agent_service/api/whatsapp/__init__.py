"""
src/agent_service/api/whatsapp - Módulo Exclusivo para WhatsApp Cloud API (Meta)
"""

from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client
from src.agent_service.api.whatsapp.service import WhatsAppService, get_whatsapp_service
from src.agent_service.api.whatsapp.router import router

__all__ = [
    "WhatsAppSettings",
    "get_whatsapp_settings",
    "WhatsAppClient",
    "get_whatsapp_client",
    "WhatsAppService",
    "get_whatsapp_service",
    "router",
]
