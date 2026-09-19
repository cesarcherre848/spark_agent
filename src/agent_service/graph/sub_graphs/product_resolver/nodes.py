import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.types import interrupt

from src.agent_service.graph.sub_graphs.product_resolver.state import (
    ProductResolverState,
    SKUItem,
)
from src.agent_service.graph.sub_graphs.product_resolver.schemas import (
    ExtractionResult,
    SynthesizeResponse,
)
from src.agent_service.tools.product_tools import (
    get_product_by_skus,
    search_suppliers,
    validate_product_ownership,
)

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.soul import inject_soul, SoulRole

logger = logging.getLogger(__name__)


class ProductResolverNodes:
    """Nodos del subgrafo product_resolver implementando el flujo del .drawio y soporte HITL."""

    def __init__(
        self,
        llm: BaseChatModel,
        product_tool: Any = get_product_by_skus,
        ownership_tool: Any = validate_product_ownership,
        supplier_tool: Any = search_suppliers,
        partner_resolver: Optional[
            Callable[
                [str, Optional[int], Optional[str]],
                Awaitable[Tuple[bool, Optional[str], List[str], Optional[Dict[str, Any]]]]
            ]
        ] = None,
    ):
        self._llm = llm
        self._product_tool = product_tool
        self._ownership_tool = ownership_tool
        self._supplier_tool = supplier_tool
        self._partner_resolver = partner_resolver

        # Extracción estricta sin alucinación (temp 0.0) y síntesis comercial balanceada (temp 0.35)
        self._extractor = bind_structured_output(bind_temperature(llm, 0.0), ExtractionResult)
        self._synthesizer = bind_structured_output(bind_temperature(llm, 0.35), SynthesizeResponse)


    async def extract_skus_and_attributes(self, state: ProductResolverState) -> dict:
        """Nodo 1: Extrae SKUs, cantidades, atributos y partners desde la entrada del usuario."""
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = m.content
                    break
        raw_query = (raw_query or "").strip()
        user_id = state.get("user_id", "5")

        # Recortar historial para que el extractor entienda referencias contextuales (ej. '¿qué precio tienen?')
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres un especialista en extracción de pedidos y catálogo para un ERP comercial.

            Tareas:
                - Extraer con precisión todos los códigos SKU o referencias de producto solicitados por el cliente.
                - IMPORTANTE: Si el cliente hace referencia a productos recomendados o discutidos previamente en el historial de la conversación (ej: '¿qué precio tienen?', 'cotízamelos', 'el primero', 'los recomendados', 'cuánto cuestan', o menciona sus nombres), revisa el historial y los SKUs sugeridos en el contexto, y extrae los códigos SKU correspondientes con cantidad 1 por defecto.
                - Extraer cantidades y atributos técnicos (ej. cantidad, medidas, color). Si no se especifica cantidad, asume 1 por defecto para cotización.
                - Identificar el nombre del proveedor o partner si el usuario lo menciona (ej. 'de Siderperu', 'de Unique').
                - Evaluar si la información es completa (is_complete = True) si se identificó al menos un SKU.
                  Si no hay ningún SKU identificable en la consulta ni en el historial reciente, marca is_complete = False y explica qué falta.
        """

        context_parts = []
        user_context = state.get("user_context")
        if user_context:
            context_parts.append(f"Antecedentes de interacciones previas:\n{user_context}")

        matched_skus = state.get("matched_skus")
        if matched_skus:
            context_parts.append(f"Códigos SKU recientemente sugeridos o identificados en el catálogo: {', '.join(map(str, matched_skus))}")

        context_block = f"\n{chr(10).join(context_parts)}\n" if context_parts else ""

        user_prompt = f"""
            {context_block}Consulta actual del cliente:
            {raw_query}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_prompt),
        ]

        extraction: Optional[ExtractionResult] = await self._extractor.ainvoke(messages)

        if extraction is None:
            extraction = ExtractionResult(is_complete=False, missing_info_prompt="Por favor especifica el código SKU o detalles de los productos que deseas cotizar.")
        elif isinstance(extraction, dict):
            extraction = ExtractionResult(**extraction)

        items: Dict[str, SKUItem] = {}

        if getattr(extraction, "is_complete", False) and getattr(extraction, "items", None):
            # Opción A: Si se inyectó un partner_resolver personalizado (compatibilidad con mocks directos)
            if self._partner_resolver is not None:
                for item in extraction.items:
                    clean_sku = item.sku.strip()
                    is_owner, resolved_p_id, candidates, metadata = await self._partner_resolver(
                        clean_sku, user_id, item.partner_name
                    )
                    items[clean_sku] = SKUItem(
                        attributes=item.attributes,
                        is_owner=is_owner,
                        partner_id=resolved_p_id,
                        candidate_partner_ids=candidates,
                        partner_metadata=metadata,
                    )
            else:
                # Opción B: Resolución oficial mediante validate_product_ownership y search_suppliers
                clean_skus = [item.sku.strip() for item in extraction.items if item.sku]
                try:
                    user_id_int = int(user_id)
                except (ValueError, TypeError):
                    user_id_int = 5

                ownership_output = await self._ownership_tool.ainvoke({
                    "user_id": user_id_int,
                    "skus": clean_skus,
                })
                validation_results = ownership_output.get("results", {})

                for item in extraction.items:
                    clean_sku = item.sku.strip()
                    val = validation_results.get(clean_sku, {})
                    is_owner = val.get("is_owner", False)
                    candidates_info = val.get("candidate_partners", [])

                    candidate_ids = [str(c["partner_id"]) for c in candidates_info]
                    metadata = {str(c["partner_id"]): c["vendor_name"] for c in candidates_info}

                    resolved_p_id = None
                    if item.partner_name:
                        mentioned_clean = item.partner_name.strip().lower()
                        # 1. Coincidencia directa contra los proveedores autorizados
                        for p_id, p_name in metadata.items():
                            if mentioned_clean in p_name.lower() or p_name.lower() in mentioned_clean:
                                resolved_p_id = p_id
                                break

                        # 2. Si no coincide de forma directa, consultar search_suppliers en Odoo
                        if not resolved_p_id and self._supplier_tool:
                            supp_res = await self._supplier_tool.ainvoke({
                                "query": item.partner_name,
                                "limit": 5,
                            })
                            found_supps = supp_res.get("suppliers", [])
                            for s in found_supps:
                                s_id_str = str(s.get("id"))
                                if s_id_str in candidate_ids:
                                    resolved_p_id = s_id_str
                                    break

                    # 3. Si no especificó partner y solo hay 1 candidato único autorizado
                    if not resolved_p_id and len(candidate_ids) == 1:
                        resolved_p_id = candidate_ids[0]

                    items[clean_sku] = SKUItem(
                        attributes=item.attributes,
                        is_owner=is_owner,
                        partner_id=resolved_p_id,
                        candidate_partner_ids=candidate_ids,
                        partner_metadata=metadata,
                    )

        all_authorized = bool(items) and all(it.get("is_owner", False) for it in items.values())
        return {
            "items": items,
            "is_extraction_complete": extraction.is_complete and all_authorized,
            "raw_query": raw_query,
        }

    async def feedback_ask_missing(
        self,
        state: ProductResolverState,
        config: RunnableConfig,
    ) -> dict:
        """Nodo 2 (HITL 1): Pausa la ejecución para solicitar información faltante o SKUs autorizados al usuario."""
        var_child_runnable_config.set(config)
        current_clarifications = state.get("clarification_count", 0)
        matched_skus = state.get("matched_skus", [])
        items = state.get("items", {})
        unauthorized = [sku for sku, it in items.items() if not it.get("is_owner", False)]

        if unauthorized:
            prompt_message = (
                f"Los siguientes productos no están disponibles o no cuentas con autorización en tu catálogo comercial: "
                f"{', '.join(unauthorized)}. Por favor indica un código SKU válido o confirma si deseas continuar "
                f"únicamente con los productos autorizados."
            )
            interrupt_payload = {
                "type": "unauthorized_or_missing",
                "unauthorized_skus": unauthorized,
                "question": prompt_message,
            }
        else:
            hint = f" (SKUs recientemente sugeridos: {', '.join(map(str, matched_skus))})" if matched_skus else ""
            prompt_message = (
                f"Para procesar tu pedido necesitamos más información: "
                f"por favor indícanos el código SKU exacto o los detalles de los productos que deseas{hint}."
            )
            interrupt_payload = {
                "type": "missing_info",
                "question": prompt_message,
                "suggested_skus": matched_skus,
            }

        # Interrupción Human-in-the-Loop de LangGraph
        user_response = interrupt(interrupt_payload)
        user_clarification = str(user_response).strip()

        return {
            "raw_query": user_clarification,
            "clarification_count": current_clarifications + 1,
            "messages": [
                AIMessage(content=prompt_message),
                HumanMessage(content=user_clarification),
            ],
        }

    async def check_partner_conflicts(self, state: ProductResolverState) -> dict:
        """Nodo 3: Evalúa si algún SKU comparte partners o existe ambigüedad de proveedor."""
        items = state.get("items", {})
        has_conflicts = False

        for sku, item in items.items():
            candidates = item.get("candidate_partner_ids", [])
            partner_id = item.get("partner_id")

            # Conflicto: más de un candidato posible y ningún partner resuelto
            if len(candidates) > 1 and not partner_id:
                has_conflicts = True
                break

        return {
            "has_partner_conflicts": has_conflicts,
        }

    async def feedback_clarify_partners(
        self,
        state: ProductResolverState,
        config: RunnableConfig,
    ) -> dict:
        """Nodo 4 (HITL 2): Pausa la ejecución para solicitar al usuario aclarar qué partner usar."""
        var_child_runnable_config.set(config)
        items = state.get("items", {})
        conflicted_sku = None
        candidate_options: List[str] = []
        partner_meta: Dict[str, str] = {}

        for sku, item in items.items():
            candidates = item.get("candidate_partner_ids", [])
            if len(candidates) > 1 and not item.get("partner_id"):
                conflicted_sku = sku
                candidate_options = candidates
                partner_meta = item.get("partner_metadata", {})
                break

        options_str = ", ".join(f"{partner_meta.get(p, p)} ({p})" for p in candidate_options)
        prompt_message = (
            f"El producto SKU '{conflicted_sku}' está disponible con varios proveedores: {options_str}. "
            f"¿Con cuál proveedor deseas procesar este producto?"
        )

        interrupt_payload = {
            "type": "partner_conflict",
            "sku": conflicted_sku,
            "options": candidate_options,
            "question": prompt_message,
        }

        # Interrupción Human-in-the-Loop de LangGraph
        user_choice = interrupt(interrupt_payload)
        chosen_str = str(user_choice).strip()

        # Asignar el partner elegido al item en conflicto (por ID o por coincidencia de nombre)
        if conflicted_sku and conflicted_sku in items:
            resolved_id = None
            if chosen_str in candidate_options:
                resolved_id = chosen_str
            else:
                for p_id, p_name in partner_meta.items():
                    if chosen_str.lower() in p_name.lower():
                        resolved_id = p_id
                        break
            items[conflicted_sku]["partner_id"] = resolved_id or chosen_str

        return {
            "items": items,
            "has_partner_conflicts": False,
            "messages": [
                AIMessage(content=prompt_message),
                HumanMessage(content=chosen_str),
            ],
        }

    async def call_get_product_by_skus(self, state: ProductResolverState) -> dict:
        """Nodo 5: Invoca la tool get_product_by_skus para obtener datos vivos de Odoo."""
        items = state.get("items", {})
        skus_to_fetch = list(items.keys())

        if not skus_to_fetch:
            return {"tool_raw_output": []}

        tool_result = await self._product_tool.ainvoke({
            "skus": skus_to_fetch,
            "fields": ["price", "description", "uom", "category", "barcode", "taxes"],
        })

        products = tool_result.get("products", []) if isinstance(tool_result, dict) else []

        return {
            "tool_raw_output": products,
        }

    async def group_by_partners(self, state: ProductResolverState) -> dict:
        """Nodo 6: Agrupación determinista por partner_id en Python."""
        items = state.get("items", {})
        tool_products = state.get("tool_raw_output", [])

        # Indexar productos de la tool por SKU
        tool_map = {str(p.get("sku")): p for p in tool_products}

        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for sku, item in items.items():
            partner_id = item.get("partner_id") or "unassigned"
            product_data = tool_map.get(sku, {
                "sku": sku,
                "name": f"Producto {sku}",
                "price": 0.0,
                "currency": "PEN",
            })

            # Extraer cantidad solicitada si existe en los atributos
            attrs = item.get("attributes", {})
            qty = attrs.get("cantidad") or attrs.get("qty") or 1
            try:
                qty_num = float(qty)
            except (ValueError, TypeError):
                qty_num = 1.0

            unit_price = float(product_data.get("price") or 0.0)
            subtotal = unit_price * qty_num

            enriched_entry = {
                **product_data,
                "requested_qty": qty_num,
                "subtotal": subtotal,
                "partner_id": partner_id,
                "partner_name": item.get("partner_metadata", {}).get(partner_id, partner_id),
                "custom_attributes": attrs,
            }

            grouped.setdefault(partner_id, []).append(enriched_entry)

        return {
            "grouped_products": grouped,
        }

    async def synthesize_response(self, state: ProductResolverState) -> dict:
        """Nodo 7: Síntesis final de la respuesta organizada por partners."""
        grouped = state.get("grouped_products", {})
        raw_query = state.get("raw_query", "")

        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        # Formatear resumen para el prompt del LLM
        lines = []
        for partner_id, prods in grouped.items():
            partner_name = prods[0].get("partner_name", partner_id) if prods else partner_id
            lines.append(f"### Proveedor: {partner_name} ({partner_id})")
            total_partner = 0.0
            currency = "PEN"
            for p in prods:
                qty = p.get("requested_qty", 1)
                price = p.get("price", 0.0)
                subtotal = p.get("subtotal", 0.0)
                currency = p.get("currency", "PEN")
                total_partner += subtotal
                lines.append(
                    f"- SKU: {p.get('sku')} | {p.get('name')} | Cant: {qty} {p.get('uom', 'Units')} | "
                    f"P.Unit: {price} {currency} | Subtotal: {subtotal:.2f} {currency}"
                )
            lines.append(f"**Total Proveedor:** {total_partner:.2f} {currency}\n")

        context_text = "\n".join(lines) if lines else "No se cotizaron productos específicos en este turno."

        system_prompt = inject_soul(
            """
            Tareas de cotización multimarca:
            - Si hay productos cotizados por proveedor: Organízalos claramente por proveedor con cantidades, precios unitarios y subtotales en formato limpio y estructurado.
            - Si no hay productos cotizados (ej. el cliente hizo una pregunta, solicitó códigos o la información fue insuficiente para cotizar): Responde con amabilidad aclarando sus dudas a partir del historial, indícale los códigos SKU disponibles si los conoces o explícale con claridad cómo puede solicitarlos para cotizar.
            """,
            role=SoulRole.QUOTATION,
        )

        matched_skus = state.get("matched_skus", [])
        skus_hint = f"\nCódigos SKU sugeridos previamente en la conversación: {', '.join(map(str, matched_skus))}\n" if matched_skus else ""

        user_prompt = f"""
            Solicitud o consulta del cliente: {raw_query}
            {skus_hint}
            Productos cotizados por proveedor:
            {context_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_prompt),
        ]

        answer: Optional[SynthesizeResponse] = await self._synthesizer.ainvoke(messages)
        final_text = (
            answer.response_text
            if answer and hasattr(answer, "response_text") and answer.response_text
            else "Se completó la cotización de los productos solicitados."
        )

        # Generar resumen compacto para memoria a largo plazo si hubo productos cotizados
        recap_items = []
        for p_id, prods in grouped.items():
            p_name = prods[0].get("partner_name", p_id) if prods else p_id
            for p in prods:
                recap_items.append(
                    f"{p.get('requested_qty', 1)} unidades del SKU {p.get('sku')} ({p.get('name')}) con {p_name}"
                )
        memory_to_save = f"Cotización realizada: {', '.join(recap_items)}." if recap_items else None

        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
            "memory_to_save": memory_to_save,
        }
