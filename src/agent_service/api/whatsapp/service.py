"""
src/agent_service/api/whatsapp/service.py - Orquestador de Mensajería para WhatsApp Cloud API
"""

import logging
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage

from src.agent_service.api.service import PhoneUserResolver, get_phone_user_resolver, normalize_phone
from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client
from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.core.llms.factory import extract_clean_text

logger = logging.getLogger(__name__)


class WhatsAppService:
    """Servicio orquestador para procesar eventos entrantes de WhatsApp y despachar respuestas."""

    def __init__(
        self,
        client: Optional[WhatsAppClient] = None,
        resolver: Optional[PhoneUserResolver] = None,
        graph_app: Optional[Any] = None,
        settings: Optional[WhatsAppSettings] = None,
    ):
        self._client = client or get_whatsapp_client()
        self._resolver = resolver or get_phone_user_resolver()
        self._graph_app = graph_app
        self._settings = settings or get_whatsapp_settings()

    def _get_graph_app(self) -> Any:
        """Obtiene de forma perezosa la instancia del grafo principal si no fue inyectada."""
        if self._graph_app is None:
            from src.agent_service.graph.main_graph import get_main_graph
            self._graph_app = get_main_graph()
        return self._graph_app

    @staticmethod
    def extract_text_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae los mensajes entrantes de tipo 'text' desde la estructura del payload de Meta."""
        extracted: List[Dict[str, Any]] = []
        if not payload or not isinstance(payload, dict):
            return extracted

        for entry in payload.get("entry", []):
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes", []):
                if not isinstance(change, dict):
                    continue
                value = change.get("value", {})
                if not isinstance(value, dict):
                    continue

                messages = value.get("messages", [])
                if not isinstance(messages, list):
                    continue

                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "text" and "text" in msg:
                        text_dict = msg.get("text", {})
                        body = text_dict.get("body", "").strip() if isinstance(text_dict, dict) else ""
                        sender = msg.get("from", "").strip()
                        msg_id = msg.get("id", "")
                        if body and sender:
                            extracted.append({
                                "from": sender,
                                "body": body,
                                "id": msg_id,
                                "timestamp": msg.get("timestamp", ""),
                            })

        return extracted

    async def handle_incoming_message(
        self,
        sender_phone: str,
        text: str,
        message_id: Optional[str] = None,
    ) -> Optional[str]:
        """Procesa un mensaje de usuario: resuelve vendedor en Odoo, ejecuta el grafo y envía respuesta."""
        normalized_phone = normalize_phone(sender_phone)
        if not normalized_phone:
            logger.warning(f"[WhatsAppService] Número de teléfono inválido o vacío: '{sender_phone}'")
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        # 1. Marcar como leído en WhatsApp si se cuenta con el ID del mensaje
        if message_id:
            try:
                await self._client.mark_message_as_read(message_id)
            except Exception as e:
                logger.debug(f"[WhatsAppService] Error no bloqueante al marcar leído: {e}")

        # 2. Resolver vendedor asignado en Odoo ERP
        user_id = await self._resolver.resolve_user_id(normalized_phone)

        # 3. Construir identificador de sesión unívoco por teléfono
        session_id = f"wa_{normalized_phone}"

        logger.info(
            f"[WhatsAppService] Procesando mensaje de {normalized_phone} "
            f"(user_id={user_id}, session_id={session_id}): '{clean_text}'"
        )

        # 4. Invocación asíncrona del Grafo Principal de Spark Agent
        graph = self._get_graph_app()
        initial_state = {
            "raw_query": clean_text,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [HumanMessage(content=clean_text)],
        }
        config = {"configurable": {"thread_id": session_id}}

        try:
            final_state = await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            logger.error(f"[WhatsAppService] Error durante la ejecución del grafo: {exc}", exc_info=True)
            error_msg = "Lo siento, ha ocurrido un error temporal al procesar tu consulta. Por favor intenta nuevamente en unos momentos."
            if self._settings.auto_reply:
                await self._client.send_text_message(to=normalized_phone, text=error_msg)
            return error_msg

        # 5. Extraer y sanear la respuesta final conversacional
        raw_response = final_state.get("final_response") or ""
        clean_response = extract_clean_text(raw_response)
        if not clean_response:
            clean_response = "Disculpa, no pude generar una respuesta en este momento."

        # 6. Despachar la respuesta al usuario mediante WhatsApp Cloud API
        if self._settings.auto_reply:
            try:
                await self._client.send_text_message(to=normalized_phone, text=clean_response)
                logger.info(f"[WhatsAppService] Respuesta despachada exitosamente a {normalized_phone}")
            except Exception as exc:
                logger.error(f"[WhatsAppService] Error al enviar respuesta a Meta Graph API: {exc}")

        return clean_response

    async def process_webhook_payload(self, payload: Dict[str, Any]) -> List[str]:
        """Procesa todas las entradas válidas de texto de una notificación push de Meta."""
        messages = self.extract_text_messages(payload)
        responses: List[str] = []

        for msg in messages:
            resp = await self.handle_incoming_message(
                sender_phone=msg["from"],
                text=msg["body"],
                message_id=msg.get("id"),
            )
            if resp:
                responses.append(resp)

        return responses


_GLOBAL_WHATSAPP_SERVICE: Optional[WhatsAppService] = None


def get_whatsapp_service() -> WhatsAppService:
    """Proveedor singleton/dependency injection de WhatsAppService."""
    global _GLOBAL_WHATSAPP_SERVICE
    if _GLOBAL_WHATSAPP_SERVICE is None:
        _GLOBAL_WHATSAPP_SERVICE = WhatsAppService()
    return _GLOBAL_WHATSAPP_SERVICE
