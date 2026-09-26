"""
src/agent_service/api/whatsapp/responder.py - Módulo de Despacho y Formateo de Respuestas para WhatsApp Cloud API
"""

import logging
from typing import Optional, Dict, Any, List

from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client
from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.core.llms.factory import extract_clean_text

logger = logging.getLogger(__name__)

# Límite seguro de caracteres por mensaje para WhatsApp Cloud API (máximo 4096 caracteres)
MAX_WHATSAPP_MESSAGE_LENGTH = 4000


class WhatsAppResponder:
    """Módulo responsable de formatear, sanear, fragmentar y enviar respuestas salientes a WhatsApp."""

    def __init__(
        self,
        client: Optional[WhatsAppClient] = None,
        settings: Optional[WhatsAppSettings] = None,
    ):
        self._client = client or get_whatsapp_client()
        self._settings = settings or get_whatsapp_settings()

    @staticmethod
    def format_response(text: str) -> str:
        """Sanitiza y limpia el texto generado por el agente para WhatsApp, eliminando artefactos internos."""
        clean = extract_clean_text(str(text or ""))
        return clean.strip()

    @staticmethod
    def chunk_message(text: str, max_length: int = MAX_WHATSAPP_MESSAGE_LENGTH) -> List[str]:
        """Divide mensajes extensos en fragmentos respetando saltos de línea y límites de WhatsApp."""
        if not text:
            return []
        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining = text
        while len(remaining) > max_length:
            split_idx = remaining.rfind("\n", 0, max_length)
            if split_idx == -1:
                split_idx = remaining.rfind(" ", 0, max_length)
            if split_idx == -1:
                split_idx = max_length

            chunk = remaining[:split_idx].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_idx:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    async def send_response(
        self,
        to: str,
        text: str,
        phone_number_id: Optional[str] = None,
        preview_url: bool = False,
    ) -> List[Dict[str, Any]]:
        """Envía una respuesta conversacional al usuario destinatario vía Meta WhatsApp Cloud API.

        Args:
            to: Número telefónico destino en formato internacional.
            text: Mensaje o respuesta conversacional generada por el agente.
            phone_number_id: Identificador opcional del número de teléfono en Meta.
            preview_url: Si se habilita la previsualización de URLs en el mensaje.

        Returns:
            Lista de respuestas de Meta Graph API por cada fragmento enviado.
        """
        if not self._settings.auto_reply:
            logger.info(f"[WhatsAppResponder] auto_reply está inactivo; mensaje a {to} omitido.")
            return []

        formatted_text = self.format_response(text)
        if not formatted_text:
            formatted_text = "Disculpa, no pude generar una respuesta en este momento."

        chunks = self.chunk_message(formatted_text)
        results: List[Dict[str, Any]] = []

        for chunk in chunks:
            try:
                kwargs: Dict[str, Any] = {"to": to, "text": chunk}
                if preview_url:
                    kwargs["preview_url"] = preview_url
                if phone_number_id:
                    kwargs["phone_number_id"] = phone_number_id

                res = await self._client.send_text_message(**kwargs)
                results.append(res)
                logger.info(f"[WhatsAppResponder] Fragmento de respuesta despachado exitosamente a {to}")
            except Exception as exc:
                logger.error(f"[WhatsAppResponder] Error al despachar respuesta a {to}: {exc}")
                raise

        return results


_GLOBAL_WHATSAPP_RESPONDER: Optional[WhatsAppResponder] = None


def get_whatsapp_responder() -> WhatsAppResponder:
    """Proveedor singleton/dependency injection de WhatsAppResponder."""
    global _GLOBAL_WHATSAPP_RESPONDER
    if _GLOBAL_WHATSAPP_RESPONDER is None:
        _GLOBAL_WHATSAPP_RESPONDER = WhatsAppResponder()
    return _GLOBAL_WHATSAPP_RESPONDER
