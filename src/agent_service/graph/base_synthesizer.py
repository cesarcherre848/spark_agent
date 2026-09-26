"""
src/agent_service/graph/base_synthesizer.py - Clase base heredada para síntesis de respuestas.
"""

import re
import logging
from typing import Dict, Any, Optional, Literal, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agent_service.core.formatters.currency import format_currency, normalize_currencies_in_text
from src.agent_service.core.templates.dialogs import WhitelabelSanitizer

logger = logging.getLogger(__name__)

ChannelType = Literal["whatsapp", "web", "other"]


class BaseSynthesizerNode:
    """Clase base transversal para todos los nodos de síntesis en Spark Agent.

    Proporciona:
    1. Detección y normalización del canal ('whatsapp', 'web', 'other').
    2. Inyección de directrices de estilo y restricciones técnicas según el canal.
    3. Estandarización de divisas mediante diccionario centralizado (S/. para PEN).
    4. Post-procesamiento unificado (sanitización de marcas blancas Odoo/ERP/Postgres,
       corrección de divisas y adaptación de sintaxis para WhatsApp).
    5. Empaquetado consistente de la respuesta final (final_response y AIMessage).
    """

    def __init__(self, llm: Optional[BaseChatModel] = None, *args, **kwargs):
        self._llm = llm
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_channel(state: Optional[Dict[str, Any]]) -> str:
        """Obtiene y normaliza el canal activo desde el estado, por defecto 'whatsapp'."""
        if not state:
            return "whatsapp"
        raw_ch = state.get("channel") or "whatsapp"
        ch = str(raw_ch).lower().strip()
        if ch in ("whatsapp", "wa"):
            return "whatsapp"
        elif ch in ("web", "webchat", "portal", "rest"):
            return "web"
        return "other"

    @classmethod
    def get_channel_prompt_instructions(cls, channel_or_state: Union[str, Dict[str, Any], None]) -> str:
        """Genera directrices de formato y estilo para inyectar en los prompts de los LLMs."""
        ch = (
            cls.get_channel(channel_or_state)
            if isinstance(channel_or_state, dict) or channel_or_state is None
            else cls.get_channel({"channel": channel_or_state})
        )

        if ch == "whatsapp":
            return (
                "DIRECTRICES OBLIGATORIAS DE FORMATO PARA WHATSAPP:\n"
                "- Estilo para móvil: conciso, estructurado con viñetas limpias con guiones o asteriscos.\n"
                "- Usa formato nativo de WhatsApp con *negrita* y _cursiva_.\n"
                "- PROHIBIDO estrictamente el uso de tablas Markdown (no se visualizan bien en pantallas móviles de WhatsApp). Usa listas con viñetas.\n"
                "- PROHIBIDO usar etiquetas HTML.\n"
                "- MONEDA: Expresa siempre los montos en moneda nacional como 'S/.' (ej: 'S/. 35.00'). NUNCA uses 'PEN', '$ (PEN)' ni 'S/' sin punto.\n"
            )
        elif ch == "web":
            return (
                "DIRECTRICES DE FORMATO PARA CANAL WEB / CHATBOT:\n"
                "- Puedes utilizar tablas Markdown completas para comparar productos o detallar órdenes si aportan claridad.\n"
                "- Emplea formato Markdown enriquecido (encabezados, listas numeradas, negritas y enlaces).\n"
                "- MONEDA: Expresa siempre los montos en moneda nacional como 'S/.' (ej: 'S/. 35.00'). NUNCA uses 'PEN'.\n"
            )
        else:
            return (
                "DIRECTRICES GENERALES DE FORMATO:\n"
                "- Presenta la información de forma clara, profesional y ordenada en texto directo.\n"
                "- MONEDA: Expresa siempre los montos en moneda nacional como 'S/.' (ej: 'S/. 35.00'). NUNCA uses 'PEN'.\n"
            )

    @classmethod
    def format_currency(cls, amount: Union[float, int, str, None], currency: Optional[str] = "PEN") -> str:
        """Formatea un monto numérico con la nomenclatura comercial oficial."""
        return format_currency(amount, currency=currency)

    @classmethod
    def post_process_response(cls, text: str, channel: str = "whatsapp") -> str:
        """Post-procesa y sanitiza el texto antes de entregarlo al usuario."""
        if not text:
            return ""

        # 1. Normalización de monedas heterogéneas
        processed = normalize_currencies_in_text(text)

        # 2. Sanitización estricta de marca blanca (Whitelabel compliance)
        processed = WhitelabelSanitizer.clean_text(processed)

        # 3. Adaptaciones específicas para el canal
        if channel == "whatsapp":
            processed = cls._adapt_for_whatsapp(processed)

        return processed.strip()

    @staticmethod
    def _adapt_for_whatsapp(text: str) -> str:
        """Adapta tablas Markdown accidentales y remueve tags HTML para WhatsApp."""
        # Remover etiquetas HTML
        cleaned = re.sub(r"<[^>]+>", "", text)

        # Transformar tablas Markdown a listas de viñetas legibles
        lines = cleaned.split("\n")
        new_lines = []
        table_headers = []
        is_table = False

        for line in lines:
            stripped = line.strip()
            # Detectar fila separadora: |---|---| o |:---|:---|
            if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", stripped):
                is_table = True
                continue

            # Detectar filas de tabla
            if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not table_headers:
                    table_headers = cells
                    continue
                else:
                    # Fila de datos: transformar a viñeta legible
                    item_parts = []
                    for idx, c in enumerate(cells):
                        hdr = table_headers[idx] if idx < len(table_headers) else ""
                        if hdr and c:
                            item_parts.append(f"*{hdr}*: {c}")
                        elif c:
                            item_parts.append(c)
                    new_lines.append("* " + " | ".join(item_parts))
                    continue
            else:
                table_headers = []
                is_table = False

            new_lines.append(line)

        return "\n".join(new_lines)

    @classmethod
    def format_final_response(
        cls,
        raw_text: str,
        state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Empaqueta la respuesta final procesada con 'final_response' y 'messages'."""
        channel = cls.get_channel(state)
        cleaned_text = cls.post_process_response(raw_text, channel=channel)
        out = {
            "final_response": cleaned_text,
            "messages": [AIMessage(content=cleaned_text)],
        }
        if extra:
            out.update(extra)
        return out
