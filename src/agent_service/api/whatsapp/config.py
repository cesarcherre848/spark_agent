"""
src/agent_service/api/whatsapp/config.py - Configuración de WhatsApp Cloud API
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env.dev")


@dataclass
class WhatsAppSettings:
    """Configuración para la integración con Meta WhatsApp Cloud API."""
    access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    business_account_id: str = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
    verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "spark_verify_token_default")
    api_version: str = os.getenv("WHATSAPP_API_VERSION", "v21.0")
    api_base_url: str = os.getenv("WHATSAPP_API_BASE_URL", "https://graph.facebook.com")
    auto_reply: bool = os.getenv("WHATSAPP_AUTO_REPLY", "true").lower() in ("true", "1", "yes")


def get_whatsapp_settings() -> WhatsAppSettings:
    """Obtiene la configuración actual de WhatsApp."""
    return WhatsAppSettings()
