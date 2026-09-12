import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
            Callable[[str, str, Optional[str]], Awaitable[Tuple[bool, Optional[str], List[str], Dict[str, str]]]]
        ] = None,
    ):
        self._llm = llm
        self._product_tool = product_tool
        self._ownership_tool = ownership_tool
        self._supplier_tool = supplier_tool
        self._partner_resolver = partner_resolver

        self._extractor = llm.with_structured_output(ExtractionResult)
        self._synthesizer = llm.with_structured_output(SynthesizeResponse)

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

        system_prompt = """
            Eres un especialista en extracción de pedidos y catálogo para un ERP comercial.

            Tareas:
                - Extraer con precisión todos los códigos SKU o referencias de producto mencionados por el usuario.
                - Extraer cantidades y atributos técnicos (ej. cantidad, medidas, color).
                - Identificar el nombre del proveedor o partner si el usuario lo menciona (ej. 'de Siderperu', 'de Aceros Arequipa').
                - Evaluar si la información es completa (is_complete = True) o si falta información indispensable para procesar (is_complete = False).
                  Si no hay ningún SKU o la solicitud es totalmente incomprensible, marca is_complete = False y explica qué falta.
        """

        user_prompt = f"""
            Consulta del cliente:
            {raw_query}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        extraction: ExtractionResult = await self._extractor.ainvoke(messages)

        items: Dict[str, SKUItem] = {}

        if extraction.is_complete and extraction.items:
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
            prompt_message = (
                "Para procesar tu pedido necesitamos más información: "
                "por favor indícanos el código SKU exacto o los detalles de los productos que deseas."
            )
            interrupt_payload = {
                "type": "missing_info",
                "question": prompt_message,
            }

        # Interrupción Human-in-the-Loop de LangGraph
        user_response = interrupt(interrupt_payload)
        user_clarification = str(user_response).strip()

        return {
            "raw_query": user_clarification,
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

        context_text = "\n".join(lines) if lines else "No se encontraron productos disponibles."

        system_prompt = """
            Eres un asesor comercial experto en ERP para consolidación de pedidos multimarca y multiproveedor.

            Tareas:
                - Redactar una respuesta clara, profesional y estructurada para el cliente.
                - Organizar los productos agrupados por cada proveedor (partner).
                - Especificar cantidades solicitadas, precios unitarios, subtotales y total consolidado por proveedor.
                - Mantener un tono servicial y cordial.
        """

        user_prompt = f"""
            Solicitud original del cliente: {raw_query}

            Productos cotizados por proveedor:
            {context_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        answer: SynthesizeResponse = await self._synthesizer.ainvoke(messages)

        return {
            "final_response": answer.response_text,
            "messages": [AIMessage(content=answer.response_text)],
        }
