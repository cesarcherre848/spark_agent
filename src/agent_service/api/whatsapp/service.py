"""
src/agent_service/api/whatsapp/service.py - Orquestador de Mensajería para WhatsApp Cloud API
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, RemoveMessage

from src.agent_service.api.service import PhoneUserResolver, get_phone_user_resolver, normalize_phone
from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client
from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.api.whatsapp.responder import WhatsAppResponder, get_whatsapp_responder
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
        responder: Optional[WhatsAppResponder] = None,
    ):
        self._client = client or get_whatsapp_client()
        self._resolver = resolver or get_phone_user_resolver()
        self._graph_app = graph_app
        self._settings = settings or get_whatsapp_settings()
        self._responder = responder or WhatsAppResponder(client=self._client, settings=self._settings)

    def _get_graph_app(self) -> Any:
        """Obtiene de forma perezosa la instancia del grafo principal si no fue inyectada."""
        if self._graph_app is None:
            from src.agent_service.api.webhook import get_agent_graph
            self._graph_app = get_agent_graph()
        return self._graph_app

    async def _typing_loop(
        self,
        message_id: str,
        phone_number_id: Optional[str] = None,
    ) -> None:
        """Bucle asíncrono que refresca el indicador de 'escribiendo...' cada N segundos.

        Meta expira el indicador tras ~25 segundos; refrescamos cada typing_refresh_interval
        (20s por defecto) para mantenerlo visible mientras el grafo procesa.
        """
        interval = self._settings.typing_refresh_interval
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self._client.send_typing_indicator(message_id, phone_number_id=phone_number_id)
                except Exception as exc:
                    logger.debug(f"[WhatsAppService] Error no bloqueante en typing loop: {exc}")
        except asyncio.CancelledError:
            pass

    async def _generate_delay_message(self, user_query: str) -> str:
        """Genera un mensaje empático y contextual para informar al usuario sobre la demora.

        Utiliza un LLM con temperatura alta (configurable, default 0.85) para producir un
        mensaje creativo de 2-3 líneas que referencia lo que el usuario consultó, con emojis
        sutiles y sin jerga técnica interna.
        """
        _FALLBACK_MSG = "⏳ Sigo trabajando en tu consulta, dame unos segundos más... ¡Ya casi! ✨"
        try:
            from src.agent_service.core.llms.factory import create_chat_model, extract_clean_text
            from langchain_core.messages import SystemMessage, HumanMessage as HMsg

            llm = create_chat_model(temperature=self._settings.delay_warning_temperature)

            system_prompt = (
                "Eres un asistente comercial amigable y empático de una empresa de distribución. "
                "El usuario hizo una consulta por WhatsApp y la respuesta está tardando más de lo habitual. "
                "Tu trabajo es generar UN SOLO mensaje breve (máximo 2-3 líneas) para informarle que sigues "
                "trabajando en su solicitud. "
                "REGLAS ESTRICTAS:\n"
                "- Sé cálido, profesional y empático.\n"
                "- Usa 1-2 emojis sutiles (⏳, ✨, 🔍, 📋) para dar vida al mensaje.\n"
                "- Haz referencia contextual a lo que el usuario preguntó (productos, precios, stock, pedidos, etc.).\n"
                "- NUNCA menciones términos técnicos internos: Odoo, grafo, nodo, base de datos, ID, LLM, API, servidor.\n"
                "- NO uses saludos (Hola, Estimado). Ve directo al punto.\n"
                "- Transmite que estás activamente trabajando, no esperando.\n"
                "- Solo responde con el mensaje, sin explicación adicional."
            )

            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HMsg(content=f"La consulta del usuario fue: \"{user_query}\""),
            ])

            clean = extract_clean_text(response.content if hasattr(response, "content") else response)
            return clean.strip() if clean and clean.strip() else _FALLBACK_MSG
        except Exception as exc:
            logger.warning(f"[WhatsAppService] Error generando mensaje de demora con LLM: {exc}")
            return _FALLBACK_MSG

    async def _delay_warning_task(
        self,
        normalized_phone: str,
        clean_text: str,
        message_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
    ) -> None:
        """Tarea asíncrona que envía un mensaje explicativo si el procesamiento supera el umbral.

        Duerme delay_warning_seconds; si no es cancelada antes, genera un mensaje creativo
        con LLM y lo envía al usuario. Luego reactiva el typing indicator (Meta lo cancela
        al enviar un mensaje de texto).
        """
        try:
            await asyncio.sleep(self._settings.delay_warning_seconds)
        except asyncio.CancelledError:
            return

        # El grafo aún está procesando — despachar mensaje explicativo
        try:
            delay_msg = await self._generate_delay_message(clean_text)
            await self._responder.send_response(
                to=normalized_phone,
                text=delay_msg,
                phone_number_id=phone_number_id,
            )
            logger.info(f"[WhatsAppService] Mensaje de demora enviado a {normalized_phone}: '{delay_msg[:80]}...'")

            # Reactivar typing indicator (Meta lo cancela al enviar un mensaje)
            if message_id and self._settings.typing_indicator_enabled:
                try:
                    await self._client.send_typing_indicator(message_id, phone_number_id=phone_number_id)
                except Exception:
                    pass
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(f"[WhatsAppService] Error al enviar mensaje de demora: {exc}")

    @staticmethod
    def extract_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae los mensajes entrantes (texto y audio/voz) desde el payload de Meta."""
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

                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id") if isinstance(metadata, dict) else None

                contacts = value.get("contacts", [])
                contact_wa_id = ""
                contact_user_id = ""
                if isinstance(contacts, list) and len(contacts) > 0 and isinstance(contacts[0], dict):
                    contact_wa_id = (contacts[0].get("wa_id") or "").strip()
                    contact_user_id = (contacts[0].get("user_id") or "").strip()

                messages = value.get("messages", [])
                if not isinstance(messages, list):
                    continue

                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    msg_type = msg.get("type", "")
                    sender = (
                        msg.get("from")
                        or contact_wa_id
                        or msg.get("from_user_id")
                        or contact_user_id
                        or ""
                    ).strip()
                    msg_id = msg.get("id", "")
                    timestamp = msg.get("timestamp", "")

                    if not sender:
                        continue

                    if msg_type == "text" and "text" in msg:
                        text_dict = msg.get("text", {})
                        body = text_dict.get("body", "").strip() if isinstance(text_dict, dict) else ""
                        if body:
                            extracted.append({
                                "type": "text",
                                "from": sender,
                                "body": body,
                                "id": msg_id,
                                "timestamp": timestamp,
                                "phone_number_id": phone_number_id,
                            })
                    elif msg_type in ("audio", "voice"):
                        audio_dict = msg.get("audio") or msg.get("voice") or {}
                        media_id = audio_dict.get("id", "") if isinstance(audio_dict, dict) else ""
                        mime_type = audio_dict.get("mime_type", "audio/ogg") if isinstance(audio_dict, dict) else "audio/ogg"
                        if media_id:
                            extracted.append({
                                "type": "audio",
                                "from": sender,
                                "media_id": media_id,
                                "mime_type": mime_type,
                                "id": msg_id,
                                "timestamp": timestamp,
                                "phone_number_id": phone_number_id,
                            })

        return extracted

    @staticmethod
    def extract_text_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae los mensajes entrantes de tipo 'text' desde la estructura del payload de Meta."""
        return [m for m in WhatsAppService.extract_messages(payload) if m.get("type") == "text"]

    async def handle_incoming_message(
        self,
        sender_phone: str,
        text: str,
        message_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
    ) -> Optional[str]:
        """Procesa un mensaje de usuario: resuelve vendedor en Odoo, ejecuta el grafo y envía respuesta.

        Gestiona concurrentemente:
        - Indicador de 'escribiendo...' (typing indicator) con refresco periódico.
        - Timer de demora (>60s) que envía un mensaje explicativo creativo vía LLM.
        """
        normalized_phone = normalize_phone(sender_phone)
        if not normalized_phone:
            logger.warning(f"[WhatsAppService] Número de teléfono inválido o vacío: '{sender_phone}'")
            return None

        clean_text = text.strip()
        if not clean_text:
            return None

        # 1. Marcar como leído y enviar typing indicator inicial
        if message_id:
            try:
                if self._settings.typing_indicator_enabled:
                    await self._client.send_typing_indicator(message_id, phone_number_id=phone_number_id)
                elif phone_number_id:
                    await self._client.mark_message_as_read(message_id, phone_number_id=phone_number_id)
                else:
                    await self._client.mark_message_as_read(message_id)
            except Exception as e:
                logger.debug(f"[WhatsAppService] Error no bloqueante al marcar leído/typing: {e}")

        # 2. Resolver vendedor asignado en Odoo ERP
        user_id = await self._resolver.resolve_user_id(normalized_phone)

        # 3. Construir identificador de sesión unívoco por teléfono
        session_id = f"wa_{normalized_phone}"

        logger.info(
            f"[WhatsAppService] Procesando mensaje de {normalized_phone} "
            f"(user_id={user_id}, session_id={session_id}, phone_number_id={phone_number_id}): '{clean_text}'"
        )

        # 4. Lanzar tareas concurrentes de UX: typing loop y timer de demora
        typing_task: Optional[asyncio.Task] = None
        delay_task: Optional[asyncio.Task] = None

        if message_id and self._settings.typing_indicator_enabled:
            typing_task = asyncio.create_task(
                self._typing_loop(message_id, phone_number_id=phone_number_id)
            )

        if self._settings.delay_warning_seconds > 0:
            delay_task = asyncio.create_task(
                self._delay_warning_task(
                    normalized_phone=normalized_phone,
                    clean_text=clean_text,
                    message_id=message_id,
                    phone_number_id=phone_number_id,
                )
            )

        # 5. Invocación asíncrona del Grafo Principal de Spark Agent
        graph = self._get_graph_app()
        initial_state = {
            "raw_query": clean_text,
            "user_id": user_id,
            "session_id": session_id,
            "channel": "whatsapp",
            "messages": [HumanMessage(content=clean_text)],
        }
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Comprobar si el hilo tiene un interrupt (HITL) pendiente o si expiró la sesión (> 24h)
            from langgraph.types import Command
            has_pending_interrupt = False
            if hasattr(graph, "aget_state"):
                state_call = graph.aget_state(config)
                if hasattr(state_call, "__await__"):
                    thread_state = await state_call
                    if thread_state and getattr(thread_state, "values", None):
                        ttl_seconds = self._settings.session_ttl_seconds
                        created_at_val = getattr(thread_state, "created_at", None)
                        if created_at_val and ttl_seconds > 0:
                            try:
                                if isinstance(created_at_val, str):
                                    created_dt = datetime.fromisoformat(created_at_val)
                                else:
                                    created_dt = created_at_val
                                if datetime.now(timezone.utc) - created_dt > timedelta(seconds=ttl_seconds):
                                    logger.info(
                                        f"[WhatsAppService] Sesión {session_id} inactiva por más de {ttl_seconds}s "
                                        f"({created_at_val}). Purgando memoria de corto plazo de la sesión anterior."
                                    )
                                    old_msgs = thread_state.values.get("messages", [])
                                    remove_ops = [RemoveMessage(id=m.id) for m in old_msgs if getattr(m, "id", None)]
                                    if hasattr(graph, "aupdate_state"):
                                        reset_values = {
                                            "active_customer": None,
                                            "active_partner_id": None,
                                            "active_order": None,
                                            "active_order_id": None,
                                            "customer_name": None,
                                            "target_customer_name": None,
                                            "target_order_name": None,
                                            "user_context": None,
                                            "retrieved_memories": [],
                                        }
                                        if remove_ops:
                                            reset_values["messages"] = remove_ops
                                        await graph.aupdate_state(config, reset_values)
                                    thread_state = None
                            except Exception as e:
                                logger.warning(f"[WhatsAppService] Error validando TTL de sesión: {e}")

                        # Poda de mensajes excedentes (> 15) para mantener la memoria ágil y evitar arrastre histórico
                        if thread_state and getattr(thread_state, "values", None) and thread_state.values.get("messages"):
                            old_msgs = thread_state.values["messages"]
                            if len(old_msgs) > 15:
                                excess_msgs = old_msgs[:-15]
                                remove_excess = [RemoveMessage(id=m.id) for m in excess_msgs if getattr(m, "id", None)]
                                if remove_excess and hasattr(graph, "aupdate_state"):
                                    logger.info(f"[WhatsAppService] Purgando {len(remove_excess)} mensajes excedentes (>15) en {session_id}.")
                                    await graph.aupdate_state(config, {"messages": remove_excess})

                    has_pending_interrupt = bool(
                        thread_state and getattr(thread_state, "tasks", None) and any(t.interrupts for t in thread_state.tasks)
                    )

            if has_pending_interrupt:
                logger.info(f"[WhatsAppService] Reanudando interrupt pendiente en {session_id} con respuesta: '{clean_text}'")
                final_state = await graph.ainvoke(Command(resume=clean_text), config=config)
            else:
                final_state = await graph.ainvoke(initial_state, config=config)
        except Exception as exc:
            logger.error(f"[WhatsAppService] Error durante la ejecución del grafo: {exc}", exc_info=True)
            error_msg = "Lo siento, ha ocurrido un error temporal al procesar tu consulta. Por favor intenta nuevamente en unos momentos."
            if self._settings.auto_reply:
                await self._responder.send_response(
                    to=normalized_phone,
                    text=error_msg,
                    phone_number_id=phone_number_id,
                )
            return error_msg
        finally:
            # Garantizar cancelación limpia de tareas concurrentes al concluir el grafo
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
            if delay_task and not delay_task.done():
                delay_task.cancel()
                try:
                    await delay_task
                except asyncio.CancelledError:
                    pass

        # 6. Extraer respuesta final o pregunta de confirmación HITL (interrupt)
        if final_state.get("__interrupt__"):
            interrupts = final_state["__interrupt__"]
            intr_val = interrupts[0].value if interrupts else {}
            if isinstance(intr_val, dict) and "question" in intr_val:
                raw_response = intr_val["question"]
            elif isinstance(intr_val, dict) and "clarification_question" in intr_val:
                raw_response = intr_val["clarification_question"]
            else:
                raw_response = str(intr_val)
            logger.info(f"[WhatsAppService] Capturado interrupt HITL conversacional para WhatsApp: '{raw_response}'")
        else:
            raw_response = final_state.get("final_response") or ""
            if not raw_response:
                for m in reversed(final_state.get("messages", [])):
                    if isinstance(m, AIMessage) and m.content:
                        raw_response = m.content
                        break

        clean_response = self._responder.format_response(raw_response)
        if not clean_response:
            clean_response = "Disculpa, no pude generar una respuesta en este momento."

        # 7. Despachar la respuesta al usuario mediante WhatsAppResponder
        if self._settings.auto_reply:
            try:
                await self._responder.send_response(
                    to=normalized_phone,
                    text=clean_response,
                    phone_number_id=phone_number_id,
                )
                logger.info(f"[WhatsAppService] Respuesta despachada exitosamente a {normalized_phone}")
            except Exception as exc:
                logger.error(f"[WhatsAppService] Error al enviar respuesta a Meta Graph API: {exc}")

        return clean_response

    async def handle_incoming_audio_message(
        self,
        sender_phone: str,
        media_id: str,
        mime_type: str = "audio/ogg",
        message_id: Optional[str] = None,
        phone_number_id: Optional[str] = None,
    ) -> Optional[str]:
        """Descarga una nota de voz o audio de WhatsApp, la transcribe mediante whisper y procesa el texto resultante."""
        from src.agent_service.api.whatsapp.audio_transcriber import transcribe_audio

        normalized_phone = normalize_phone(sender_phone)
        if not normalized_phone:
            logger.warning(f"[WhatsAppService] Teléfono inválido para audio: '{sender_phone}'")
            return None

        # Enviar indicador de escribiendo de inmediato para dar feedback al usuario
        if message_id and self._settings.typing_indicator_enabled:
            try:
                await self._client.send_typing_indicator(message_id, phone_number_id=phone_number_id)
            except Exception:
                pass

        try:
            logger.info(f"[WhatsAppService] Descargando audio {media_id} ({mime_type}) de {normalized_phone}...")
            audio_bytes, actual_mime = await self._client.download_audio(media_id)
            effective_mime = actual_mime or mime_type

            logger.info(f"[WhatsAppService] Transcribiendo audio {media_id} ({len(audio_bytes)} bytes)...")
            transcribed_text = await transcribe_audio(audio_bytes, mime_type=effective_mime)

            if not transcribed_text or not transcribed_text.strip():
                logger.warning(f"[WhatsAppService] Transcripción vacía para audio {media_id} de {normalized_phone}.")
                msg = "Disculpa, no pude entender claramente el audio. ¿Podrías escribir tu consulta o enviarlo nuevamente?"
                if self._settings.auto_reply:
                    await self._responder.send_response(to=normalized_phone, text=msg, phone_number_id=phone_number_id)
                return msg

            logger.info(f"[WhatsAppService] Audio transcrito con éxito: '{transcribed_text}'")
            return await self.handle_incoming_message(
                sender_phone=sender_phone,
                text=transcribed_text,
                message_id=message_id,
                phone_number_id=phone_number_id,
            )
        except Exception as exc:
            logger.error(f"[WhatsAppService] Error procesando mensaje de audio: {exc}", exc_info=True)
            err_msg = "Lo siento, tuve un problema al procesar tu nota de voz. Por favor envía tu consulta en texto."
            if self._settings.auto_reply:
                await self._responder.send_response(to=normalized_phone, text=err_msg, phone_number_id=phone_number_id)
            return err_msg

    async def process_webhook_payload(self, payload: Dict[str, Any]) -> List[str]:
        """Procesa todas las entradas válidas (texto y audio) de una notificación push de Meta."""
        messages = self.extract_messages(payload)
        responses: List[str] = []

        for msg in messages:
            msg_type = msg.get("type", "text")
            if msg_type == "text":
                resp = await self.handle_incoming_message(
                    sender_phone=msg["from"],
                    text=msg["body"],
                    message_id=msg.get("id"),
                    phone_number_id=msg.get("phone_number_id"),
                )
                if resp:
                    responses.append(resp)
            elif msg_type == "audio":
                resp = await self.handle_incoming_audio_message(
                    sender_phone=msg["from"],
                    media_id=msg["media_id"],
                    mime_type=msg.get("mime_type", "audio/ogg"),
                    message_id=msg.get("id"),
                    phone_number_id=msg.get("phone_number_id"),
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
