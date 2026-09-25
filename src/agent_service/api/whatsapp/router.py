"""
src/agent_service/api/whatsapp/router.py - Endpoints FastAPI para Meta WhatsApp Cloud API
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, BackgroundTasks, HTTPException, status
from fastapi.responses import PlainTextResponse

from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.api.whatsapp.service import WhatsAppService, get_whatsapp_service
from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WhatsApp"])


@router.get("/webhook", summary="Verificación del Webhook de Meta")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    settings: WhatsAppSettings = Depends(get_whatsapp_settings),
):
    """Handshake de suscripción requerido por Meta para validar la propiedad del Webhook.

    Meta envía:
    - hub.mode = 'subscribe'
    - hub.verify_token = token secreto configurado en el panel de Meta
    - hub.challenge = cadena o entero aleatorio generado por Meta

    Retorna hub.challenge en texto plano (Content-Type: text/plain) si el token coincide.
    """
    logger.info(f"[WhatsApp] Solicitud de verificación recibida: mode={hub_mode}")

    if hub_mode == "subscribe" and hub_verify_token == settings.verify_token:
        logger.info("[WhatsApp] Verificación de Webhook exitosa.")
        return PlainTextResponse(content=hub_challenge or "", status_code=status.HTTP_200_OK)

    logger.warning("[WhatsApp] Verificación de Webhook rechazada: token incorrecto o modo no soportado.")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Verificación de token fallida o modo no reconocido.",
    )


@router.post("/webhook", summary="Recepción de Eventos de WhatsApp Cloud API")
async def receive_webhook(
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    service: WhatsAppService = Depends(get_whatsapp_service),
):
    """Punto de recepción de notificaciones push de WhatsApp Cloud API.

    Para cumplir con el SLA estricto de Meta (<3s):
    1. Responde de inmediato con HTTP 200 {"status": "ok"}.
    2. Encola el procesamiento del mensaje, resolución en Odoo y respuesta en BackgroundTasks.
    """
    # Si es una notificación de entrega/lectura (statuses) sin mensajes, no encolar tarea
    has_messages = False
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            if "messages" in change.get("value", {}):
                has_messages = True
                break

    if has_messages:
        background_tasks.add_task(service.process_webhook_payload, payload)

    return {"status": "ok"}


@router.post("/send", summary="Envío Manual/Directo de Mensaje de WhatsApp")
async def send_message_endpoint(
    payload: Dict[str, Any],
    client: WhatsAppClient = Depends(get_whatsapp_client),
):
    """Endpoint auxiliar para enviar mensajes de WhatsApp directamente vía Meta Graph API."""
    to = payload.get("to") or payload.get("phone")
    message = payload.get("message") or payload.get("text")
    preview_url = bool(payload.get("preview_url", False))

    if not to or not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Se requieren los campos 'to' (teléfono) y 'message' (texto).",
        )

    try:
        resp = await client.send_text_message(to=str(to), text=str(message), preview_url=preview_url)
        return {"status": "sent", "meta_response": resp}
    except Exception as exc:
        logger.error(f"[WhatsApp] Error enviando mensaje manual a {to}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error al comunicar con Meta Graph API: {exc}",
        )
