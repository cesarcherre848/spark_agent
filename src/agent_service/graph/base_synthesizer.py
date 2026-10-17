"""
src/agent_service/graph/base_synthesizer.py - Clase base heredada para síntesis de respuestas.
"""

import re
import logging
from typing import Dict, Any, Optional, Literal, Union, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.agent_service.core.formatters.currency import format_currency, normalize_currencies_in_text
from src.agent_service.core.templates.dialogs import WhitelabelSanitizer
from src.agent_service.soul import inject_soul, SoulRole

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
                "- Usa formato nativo de WhatsApp: *negrita* (un solo asterisco) y _cursiva_ (guion bajo). NUNCA uses doble asterisco (**negrita**) ya que no se visualiza bien en WhatsApp móvil.\n"
                "- CADA PRODUCTO O ELEMENTO DE LISTA DEBE IR OBLIGATORIAMENTE EN UN RENGLÓN INDEPENDIENTE CON UN SALTO DE LÍNEA (\\n) AL INICIO. PROHIBIDO ESTRICTAMENTE COLOCAR VIÑETAS CONSECUTIVAS EN EL MISMO PÁRRAFO O RENGLÓN CORRIDO.\n"
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
    def format_products_context(cls, products: List[Any], max_desc_len: int = 250) -> str:
        """Formatea de forma universal una lista de productos (Document, dict o ProductCanonicalFamily)
        en texto conciso y libre de ambigüedades de números de página para prevenir alucinaciones de SKU."""
        if not products:
            return "No se encontraron productos disponibles en el catálogo."

        formatted_lines = []
        for idx, item in enumerate(products, start=1):
            if hasattr(item, "family_name") and hasattr(item, "variants"):
                # Estructura jerárquica de ProductCanonicalFamily
                fam_name = getattr(item, "family_name") or "Producto"
                marca = getattr(item, "marca", "") or ""
                summary = (getattr(item, "summary_description", "") or "")[:max_desc_len].strip()
                marca_str = f" (Marca: {marca})" if marca else ""
                lines = [f"CONCEPTO / FAMILIA: {fam_name}{marca_str}"]
                if summary:
                    lines.append(f"  Descripción general: {summary}")
                lines.append("  FACETAS / VARIANTES DISPONIBLES:")
                for var in getattr(item, "variants", []):
                    v_name = getattr(var, "name", "")
                    v_sku = getattr(var, "sku", "")
                    v_price = getattr(var, "price", None)
                    v_curr = getattr(var, "currency", "PEN")
                    v_attr = getattr(var, "attribute_details", "")
                    p_str = cls.format_currency(v_price, currency=v_curr) if v_price is not None else "Consultar"
                    lines.append(f"  • {v_name} | SKU OFICIAL: {v_sku} | Precio: {p_str} | Detalle: {v_attr}")
                formatted_lines.append("\n".join(lines))
                continue

            if isinstance(item, Document):
                meta = item.metadata or {}
                desc = (item.page_content or "")[:max_desc_len].strip()
                name = meta.get("name") or "Producto"
                sku = meta.get("sku")
                marca = meta.get("marca")
                vendor_name = meta.get("vendor_name")
                edicion = meta.get("edicion")
                pagina = meta.get("pagina")
                price = meta.get("price")
                normal_price = meta.get("normal_price")
                discount_tag = meta.get("discount_tag")
                catalog_variants = meta.get("catalog_variants")
                currency = meta.get("currency") or "PEN"
            elif isinstance(item, dict):
                desc = (item.get("description") or item.get("page_content") or "")[:max_desc_len].strip()
                name = item.get("name") or "Producto"
                sku = item.get("sku")
                marca = item.get("marca")
                vendor_name = item.get("vendor_name")
                edicion = item.get("edicion")
                pagina = item.get("pagina")
                price = item.get("price")
                normal_price = item.get("normal_price")
                discount_tag = item.get("discount_tag")
                catalog_variants = item.get("catalog_variants")
                currency = item.get("currency") or "PEN"
            else:
                continue

            sku_str = f"SKU OFICIAL: {sku}" if sku else "SKU: No disponible"
            meta_details = []
            if marca:
                meta_details.append(f"Marca: {marca}")
            if vendor_name and str(vendor_name).strip().lower() != str(marca or "").strip().lower():
                meta_details.append(f"Proveedor: {vendor_name}")
            if edicion:
                meta_details.append(f"Campaña: {edicion}")
            if pagina is not None:
                # Sin corchetes para no confundir al LLM con SKUs
                meta_details.append(f"Página de catálogo: {pagina}")
            if price is not None:
                price_str = cls.format_currency(price, currency=currency)
                meta_details.append(f"Precio Oferta: {price_str}")
            if normal_price is not None and price is not None and float(normal_price) > float(price):
                norm_str = cls.format_currency(normal_price, currency=currency)
                meta_details.append(f"Precio Normal: {norm_str}")
            if discount_tag:
                meta_details.append(f"Promoción: {discount_tag}")

            meta_str = " | ".join(meta_details)
            variants_str = ""
            if catalog_variants and isinstance(catalog_variants, list) and len(catalog_variants) > 1:
                v_items = []
                for cv in catalog_variants[:12]:
                    if isinstance(cv, dict) and cv.get("sku"):
                        v_lbl = cv.get("label") or "Variante"
                        v_items.append(f"{v_lbl} -> SKU OFICIAL: {cv['sku']}")
                if v_items:
                    variants_str = f"\n  • TONOS / VARIANTES EN ESTA PÁGINA: {'; '.join(v_items)}"

            formatted_lines.append(
                f"PRODUCTO: {name}\n"
                f"  • {sku_str} | {meta_str}\n"
                f"  • Detalle: {desc}{variants_str}"
            )

        return "\n\n".join(formatted_lines)

    @classmethod
    def build_synthesizer_system_prompt(
        cls,
        task_specific_rules: str,
        role: Union[SoulRole, str] = SoulRole.RECOMMENDER,
        state: Optional[Dict[str, Any]] = None,
        include_multi_vendor: bool = True,
        extra_context: Optional[str] = None,
    ) -> str:
        """Construye el System Prompt unificado del sintetizador obedeciendo al contrato de MIA (SOUL).

        Garantiza que todos los subgrafos cumplan con:
        1. Identidad, tono y estilo de MIA (SOUL).
        2. Directrices de canal (WhatsApp vs Web).
        3. Formato estándar de viñetas comerciales para productos:
           * **[SKU] Nombre Comercial** (S/. XX.XX) - [Marca]: Beneficio o motivo de recomendación.
        4. Continuidad conversacional (sin saludos redundantes si ya existe diálogo en curso).
        5. Cumplimiento estricto de marca blanca (Whitelabel) y diferenciación multi-marca.
        """
        channel_instructions = cls.get_channel_prompt_instructions(state)

        ch = cls.get_channel(state)
        bullet_example = (
            "- *[SKU] Nombre Comercial* (S/. XX.XX): Breve beneficio o motivo de recomendación."
            if ch == "whatsapp"
            else "* **[SKU] Nombre Comercial** (S/. XX.XX): Breve beneficio o motivo de recomendación."
        )
        sku_format = (
            "entre corchetes en negrita WhatsApp (ej: *[SKU]*)"
            if ch == "whatsapp"
            else "entre corchetes en negrita (ej: **[SKU]**)"
        )
        newline_rule = (
            "- CADA VIÑETA DEBE IR EN UN RENGLÓN INDEPENDIENTE CON UN SALTO DE LÍNEA (\\n). PROHIBIDO escribir viñetas seguidas en la misma línea."
            if ch == "whatsapp"
            else ""
        )

        standard_product_rules = f"""
ESTRUCTURA Y AGRUPACIÓN DE PRODUCTOS:
- Si las opciones recomendadas provienen de DOS O MÁS marcas comerciales distintas (consulta abierta, comparativa o catálogo variado):
  AGRÚPALAS de forma limpia y ordenada por cada marca comercial utilizando un subtítulo destacado para cada una (ejemplo: '*En [Nombre de Marca]:*' o '**[Nombre de Marca]**:'), seguido de sus viñetas correspondientes.
- Si todos los productos pertenecen a una ÚNICA marca comercial (consulta mono-marca o filtro específico):
  Menciona la marca en la introducción y presenta las viñetas directamente sin subtítulos repetitivos.
- ESTRUCTURA DE CADA VIÑETA:
  {bullet_example}
{newline_rule}
- CÓDIGO SKU (GROUNDING ESTRICTO): Obligatorio si está disponible en los datos, siempre {sku_format}.
  REGLA DE ORO: El SKU es ÚNICAMENTE el código oficial indicado en 'SKU OFICIAL:'. Queda TERMINANTEMENTE PROHIBIDO inventar códigos, usar números de página de catálogo o números de índice como SKUs.
- PRECIOS: Expresa siempre los montos en moneda nacional como 'S/.' (ej: 'S/. 35.00'). NUNCA inventes precios ni códigos SKU ausentes en los productos proporcionados.
- Si el catálogo no cuenta con opciones exactas, sé cortés, transparente y ofrece la alternativa disponible más cercana.
""".strip()

        multi_vendor_rules = ""
        if include_multi_vendor:
            multi_vendor_rules = """
GESTIÓN MULTI-MARCA Y WHITELABEL (PROMPT GENERAL):
- NEUTRALIDAD COMERCIAL: Trata a todas las marcas comerciales del catálogo de manera equitativa, neutral y objetiva sin favorecer ni sesgar hacia ninguna marca en particular.
- RESTRICCIÓN DE MARCA SOLICITADA: Si el usuario solicita explícitamente una marca específica, restringe las recomendaciones estrictamente a esa marca y NO mezcles opciones de otras marcas no solicitadas.
- 100% Whitelabel: NUNCA expongas identificadores internos, bases de datos ni términos de backend ('vendor_id', 'res_partner', 'partner_id', 'Odoo', 'PostgreSQL', 'view_user_authorized_products'). Utiliza siempre el nombre comercial visible de cara al cliente.
""".strip()

        conversational_rules = """
CONTINUIDAD CONVERSACIONAL (MULTI-TURNO):
- Si hay mensajes previos en la conversación, NO repitas saludos de bienvenida ("Hola", "Buen día", "¿En qué puedo ayudarte?"). Continúa fluidamente respondiendo directo a la consulta del usuario.
""".strip()

        clarity_and_unambiguity_rules = """
CLARIDAD Y NO AMBIGÜEDAD (COMUNICACIÓN PRECISA Y CONSULTIVA):
- PRECISIÓN EN CONSULTAS DE SEGUIMIENTO Y REFERENCIAS RELATIVAS ('esto', 'esos', '¿a qué catálogo pertenece?'):
  Si el usuario pregunta sobre productos o recomendaciones previas del diálogo:
  1. Identifica inequívocamente a qué productos específicos se refiere la consulta (mencionando su nombre comercial y SKU).
  2. NUNCA respondas con generalidades ambiguas ni mezcles marcas que no corresponden (PROHIBIDO terminantemente decir 'pertenecen a nuestras campañas de [Marca A] y [Marca B]' si los productos recomendados corresponden únicamente a [Marca A] o a [Marca B]).
  3. Detalla con exactitud quirúrgica la marca y edición/campaña del catálogo a la que pertenece cada producto o el conjunto discutido.
- ATRIBUCIÓN TRANSPARENTE: Toda mención de campañas, páginas de catálogo o precios debe ser veraz, inequívoca y sustentada en los datos disponibles.
""".strip()

        combined_task_instructions = f"""
{task_specific_rules.strip()}

{standard_product_rules}

{multi_vendor_rules}

{clarity_and_unambiguity_rules}

{conversational_rules}

{channel_instructions}
""".strip()

        return inject_soul(
            combined_task_instructions,
            role=role,
            extra_context=extra_context,
        )

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
        """Adapta tablas Markdown accidentales, separa viñetas inline y estandariza formato para WhatsApp."""
        if not text:
            return ""

        # 1. Remover etiquetas HTML
        cleaned = re.sub(r"<[^>]+>", "", text)

        # 2. Transformar tablas Markdown a listas de viñetas legibles
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

        cleaned = "\n".join(new_lines)

        # 3. Separar subtítulos de marca como '*En Ésika:*' o '**En Ésika:**' si están precedidos por texto en la misma línea
        cleaned = re.sub(r"(?<=[^\n])\s+(\*+En\s+[^:]+:\*+)", r"\n\n\1\n", cleaned)

        # 4. Separar viñetas inline pegadas con espacio: ej. '... pág 60 - [SKU]' o '... : * [SKU]' o '... - **[SKU]'
        cleaned = re.sub(r"(?<=[^\n])\s+([*-]\s+(?:\*+|\b\[))", r"\n\1", cleaned)

        # 5. Separar notas complementarias o de transición (ej: 'Para complementar...', 'Como verás...')
        cleaned = re.sub(
            r"(?<=[.!?])\s+((?:Para complementar|También contamos con|Como puedes ver|Como verás)[^.]*?\.)",
            r"\n\n\1",
            cleaned,
        )

        # 6. Separar preguntas de cierre si están pegadas al final de un producto o nota
        cleaned = re.sub(
            r"(?<=[.!?])\s+(¿(?:Te gustaría|Deseas|Prefieres|Confirmamos|Quieres)[^?]+\?)",
            r"\n\n\1",
            cleaned,
        )

        # 7. Convertir **bold** de Markdown a *bold* de WhatsApp
        cleaned = re.sub(r"\*\*([^*]+?)\*\*", r"*\1*", cleaned)

        # 8. Normalizar saltos de línea (máximo 2 saltos consecutivos)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

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
