"""
src/agent_service/api/whatsapp/config.py - Configuración de WhatsApp Cloud API
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(".env.dev", override=True)
load_dotenv(".env.qa", override=True)


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

    # Indicador de estado escribiendo ("typing_indicator")
    typing_indicator_enabled: bool = os.getenv("WHATSAPP_TYPING_INDICATOR_ENABLED", "true").lower() in ("true", "1", "yes")
    typing_refresh_interval: float = float(os.getenv("WHATSAPP_TYPING_REFRESH_INTERVAL", "20.0"))

    # Timer para mensaje explicativo creativo en demoras prolongadas
    delay_warning_seconds: float = float(os.getenv("WHATSAPP_DELAY_WARNING_SECONDS", "60.0"))
    delay_warning_temperature: float = float(os.getenv("WHATSAPP_DELAY_WARNING_TEMPERATURE", "0.85"))


def get_whatsapp_settings() -> WhatsAppSettings:
    """Obtiene la configuración actual de WhatsApp."""
    return WhatsAppSettings()
