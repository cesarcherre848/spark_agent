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

    def _get_url(self, phone_number_id: Optional[str] = None) -> str:
        """Construye la URL base del endpoint de mensajes de WhatsApp Cloud API."""
        pid = phone_number_id or self._settings.phone_number_id
        return (
            f"{self._settings.api_base_url.rstrip('/')}/"
            f"{self._settings.api_version}/"
            f"{pid}/messages"
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
        phone_number_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Envía un mensaje de texto plano a un usuario de WhatsApp.

        Args:
            to: Número telefónico destino en formato internacional (solo dígitos, ej: 51999999999).
            text: Mensaje de texto a enviar al usuario.
            preview_url: Si se habilita vista previa de URLs en el mensaje.
            phone_number_id: Identificador opcional del número emisor en WhatsApp Cloud API.

        Returns:
            Diccionario con la respuesta de Meta Graph API conteniendo los IDs de mensaje generados.
        """
        from src.agent_service.api.service import is_bsuid
        s_to = str(to or "").strip()
        if is_bsuid(s_to):
            clean_to = s_to
        else:
            clean_to = "".join(filter(str.isdigit, s_to))

        if not clean_to:
            raise ValueError("El destinatario 'to' no contiene un número telefónico o BSUID válido.")

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

        url = self._get_url(phone_number_id=phone_number_id)
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

    async def mark_message_as_read(
        self,
        message_id: str,
        phone_number_id: Optional[str] = None,
    ) -> bool:
        """Marca un mensaje entrante como leído en WhatsApp (doble check azul)."""
        if not message_id:
            return False

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        url = self._get_url(phone_number_id=phone_number_id)
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

    async def send_typing_indicator(
        self,
        message_id: str,
        phone_number_id: Optional[str] = None,
    ) -> bool:
        """Envía el estado 'escribiendo...' a WhatsApp en respuesta al mensaje indicado.

        Utiliza la especificación oficial de Meta Cloud API: enviar status='read'
        junto a typing_indicator={'type': 'text'} asociado al message_id entrante.
        Meta expira automáticamente el indicador tras ~25 segundos o al enviar un mensaje.
        """
        if not message_id:
            return False

        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {
                "type": "text",
            },
        }

        url = self._get_url(phone_number_id=phone_number_id)
        headers = self._get_headers()

        try:
            if self._client is not None:
                resp = await self._client.post(url, headers=headers, json=payload, timeout=5.0)
                return resp.status_code == 200

            async with httpx.AsyncClient(timeout=5.0) as http_client:
                resp = await http_client.post(url, headers=headers, json=payload)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"[WhatsAppClient] No fue posible enviar typing indicator para {message_id}: {exc}")
            return False

    async def download_audio(self, media_id: str) -> tuple[bytes, str]:
        """Descarga el contenido binario de un audio desde Meta Cloud API.

        Paso 1: Consulta los metadatos del objeto multimedia para obtener su URL temporal.
        Paso 2: Descarga los bytes con autorización Bearer.

        Returns:
            Tupla (audio_bytes, mime_type)
        """
        if not media_id:
            raise ValueError("El media_id es obligatorio para descargar el audio.")

        meta_url = (
            f"{self._settings.api_base_url.rstrip('/')}/"
            f"{self._settings.api_version}/{media_id}"
        )
        headers = self._get_headers()

        logger.info(f"[WhatsAppClient] Consultando URL de descarga para media_id={media_id}...")

        if self._client is not None:
            resp = await self._client.get(meta_url, headers=headers, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            download_url = data.get("url")
            mime_type = data.get("mime_type", "audio/ogg")
            if not download_url:
                raise ValueError(f"Meta no devolvió 'url' para media_id={media_id}")

            audio_resp = await self._client.get(download_url, headers={"Authorization": headers["Authorization"]}, timeout=60.0)
            audio_resp.raise_for_status()
            return audio_resp.content, mime_type

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resp = await http_client.get(meta_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            download_url = data.get("url")
            mime_type = data.get("mime_type", "audio/ogg")
            if not download_url:
                raise ValueError(f"Meta no devolvió 'url' para media_id={media_id}")

            audio_resp = await http_client.get(download_url, headers={"Authorization": headers["Authorization"]})
            audio_resp.raise_for_status()
            return audio_resp.content, mime_type


_GLOBAL_WHATSAPP_CLIENT: Optional[WhatsAppClient] = None


def get_whatsapp_client() -> WhatsAppClient:
    """Proveedor singleton/dependency injection de WhatsAppClient."""
    global _GLOBAL_WHATSAPP_CLIENT
    if _GLOBAL_WHATSAPP_CLIENT is None:
        _GLOBAL_WHATSAPP_CLIENT = WhatsAppClient()
    return _GLOBAL_WHATSAPP_CLIENT
