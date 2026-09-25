"""
src/agent_service/api/whatsapp/client.py - Cliente HTTP Asíncrono para WhatsApp Cloud API de Meta
"""

import logging
from typing import Optional, Dict, Any
import httpx

from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings

logger = logging.getLogger(__name__)


class WhatsAppClient:
    """Cliente HTTP asíncrono para interactuar con Meta WhatsApp Cloud API."""

    def __init__(
        self,
        settings: Optional[WhatsAppSettings] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._settings = settings or get_whatsapp_settings()
        self._client = client

    def _get_url(self) -> str:
        """Construye la URL base del endpoint de mensajes de WhatsApp Cloud API."""
        return (
            f"{self._settings.api_base_url.rstrip('/')}/"
            f"{self._settings.api_version}/"
            f"{self._settings.phone_number_id}/messages"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Retorna las cabeceras de autorización Bearer para Meta Graph API."""
        return {
            "Authorization": f"Bearer {self._settings.access_token}",
            "Content-Type": "application/json",
        }

    async def send_text_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False,
    ) -> Dict[str, Any]:
        """Envía un mensaje de texto plano a un usuario de WhatsApp.

        Args:
            to: Número telefónico destino en formato internacional (solo dígitos, ej: 51999999999).
            text: Mensaje de texto a enviar al usuario.
            preview_url: Si se habilita vista previa de URLs en el mensaje.

        Returns:
            Diccionario con la respuesta de Meta Graph API conteniendo los IDs de mensaje generados.
        """
        clean_to = "".join(filter(str.isdigit, str(to)))
        if not clean_to:
            raise ValueError("El destinatario 'to' no contiene un número telefónico válido.")

        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("El mensaje de texto no puede estar vacío.")

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": clean_text,
            },
        }

        url = self._get_url()
        headers = self._get_headers()

        logger.info(f"[WhatsAppClient] Enviando mensaje a {clean_to} (Longitud: {len(clean_text)} chars)...")

        # Usar cliente inyectado o crear sesión temporal
        if self._client is not None:
            response = await self._client.post(url, headers=headers, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error(
                    f"[WhatsAppClient] Error al enviar mensaje a Meta (HTTP {response.status_code}): {response.text}"
                )
            response.raise_for_status()
            return response.json()

    async def mark_message_as_read(self, message_id: str) -> bool:
        """Marca un mensaje entrante como leído en WhatsApp (doble check azul)."""
        if not message_id:
            return False

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        url = self._get_url()
        headers = self._get_headers()

        try:
            if self._client is not None:
                resp = await self._client.post(url, headers=headers, json=payload, timeout=5.0)
                return resp.status_code == 200

            async with httpx.AsyncClient(timeout=5.0) as http_client:
                resp = await http_client.post(url, headers=headers, json=payload)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"[WhatsAppClient] No fue posible marcar mensaje {message_id} como leído: {exc}")
            return False


_GLOBAL_WHATSAPP_CLIENT: Optional[WhatsAppClient] = None


def get_whatsapp_client() -> WhatsAppClient:
    """Proveedor singleton/dependency injection de WhatsAppClient."""
    global _GLOBAL_WHATSAPP_CLIENT
    if _GLOBAL_WHATSAPP_CLIENT is None:
        _GLOBAL_WHATSAPP_CLIENT = WhatsAppClient()
    return _GLOBAL_WHATSAPP_CLIENT
