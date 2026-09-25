"""
src/agent_service/graph/sub_graphs/sales_manage/nodes.py - Nodos ejecutores del subgrafo sales_manage

Implementa el flujo 1:1 de flow_diagrams/sales_manage.drawio con herramientas Odoo desacopladas,
resolución inequívoca de clientes, evaluación de duplicados y decisiones Human-in-the-Loop 100% en lenguaje natural.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.types import interrupt

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.core.llms.factory import extract_clean_text
from src.agent_service.soul import inject_soul, SoulRole
from src.agent_service.graph.sub_graphs.sales_manage.state import SalesManageState
from src.agent_service.graph.sub_graphs.sales_manage.schemas import (
    SalesExtractionResult,
    SalesDuplicateCheckResult,
    SalesSynthesizeResponse,
    SalesReflectionResult,
)
from src.agent_service.core.hitl.parser import (
    parse_hitl_binary_decision,
    parse_hitl_entity_selection,
    parse_hitl_order_choice,
)
from src.agent_service.tools.sales_tools import (
    normalize_order_name,
    extract_order_code,
    odoo_list_sales_orders,
    odoo_list_current_sales_orders,
    odoo_create_quotation,
    odoo_update_quotation,
    odoo_view_quotation,
    odoo_confirm_order,
    odoo_unlock_order,
    odoo_update_order,
    odoo_lock_order,
    odoo_remove_sale_order,
)
from src.agent_service.tools.contact_tools import (
    odoo_list_current_customers,
    odoo_upsert_customer,
)

logger = logging.getLogger(__name__)


def _customers_match(name1: Optional[str], name2: Optional[str]) -> bool:
    if not name1 or not name2:
        return False
    n1 = name1.strip().lower()
    n2 = name2.strip().lower()
    if n1 == n2 or n1 in n2 or n2 in n1:
        return True
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    common = [t for t in tokens1 if len(t) >= 3 and t in tokens2]
    return len(common) > 0


class SalesManageNodes:
    """Nodos del subgrafo de gestión de ventas, órdenes y cotizaciones en Odoo ERP."""

    def __init__(
        self,
        llm: BaseChatModel,
        list_orders_tool: Callable[..., Awaitable[Dict[str, List[Dict[str, Any]]]]] = odoo_list_sales_orders,
        list_current_orders_tool: Callable[..., Awaitable[List[Dict[str, Any]]]] = odoo_list_current_sales_orders,
        create_quotation_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_create_quotation,
        update_quotation_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_update_quotation,
        view_quotation_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_view_quotation,
        confirm_order_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_confirm_order,
        unlock_order_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_unlock_order,
        update_order_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_update_order,
        lock_order_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_lock_order,
        remove_order_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_remove_sale_order,
        list_customers_tool: Callable[..., Awaitable[List[Dict[str, Any]]]] = odoo_list_current_customers,
        upsert_customer_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_upsert_customer,
    ):
        self._llm = llm
        self._list_orders_tool = list_orders_tool
        self._list_current_orders_tool = list_current_orders_tool
        self._create_quotation_tool = create_quotation_tool
        self._update_quotation_tool = update_quotation_tool
        self._view_quotation_tool = view_quotation_tool
        self._confirm_order_tool = confirm_order_tool
        self._unlock_order_tool = unlock_order_tool
        self._update_order_tool = update_order_tool
        self._lock_order_tool = lock_order_tool
        self._remove_order_tool = remove_order_tool
        self._list_customers_tool = list_customers_tool
        self._upsert_customer_tool = upsert_customer_tool

        # Modelos estructurados
        self._extractor = bind_structured_output(bind_temperature(llm, 0.0), SalesExtractionResult)
        self._judge = bind_structured_output(bind_temperature(llm, 0.1), SalesDuplicateCheckResult)
        self._synthesizer = bind_structured_output(bind_temperature(llm, 0.35), SalesSynthesizeResponse)
        self._reflector = bind_structured_output(bind_temperature(llm, 0.0), SalesReflectionResult)


    async def extract_sales_info(self, state: SalesManageState) -> dict:
        """Nodo 1 (extraction): Extrae atributos (SKUs x cantidad, customer, action, status, period, order_name)."""
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=6,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres un asistente comercial de ventas para Odoo ERP.
            Tu misión es analizar la solicitud del vendedor y el contexto conversacional reciente para extraer de forma estructurada:
            
            1. 'action':
               - 'view': si solicita ver, mostrar o consultar el detalle de una orden o cotización específica (ej: 'muéstrame la cotización 03 de Adhara', 'ver cotización S00003', 'detalles del pedido 3').
               - 'list': si solicita ver o consultar órdenes, pedidos o cotizaciones en general (ej: 'muéstrame los pedidos de Carlos', 'cuáles cotizaciones tengo abiertas este mes').
               - 'add_items': si solicita agregar o sumar productos/unidades a una orden o cotización activa o existente (ej: 'agrega 2 del SKU 6189', 'suma 2 delineadores a la cotización').
               - 'remove_items': si solicita quitar, restar o eliminar unidades o productos de una orden o cotización activa o existente (ej: 'quita 2 unidades del Delineador', 'elimina 2 items de 6189', 'resta 1 unidad').
               - 'upsert': si desea cotizar productos nuevos desde cero o crear una cotización nueva (ej: 'cotízame 5 del SKU 1 para Transportes Lima').
               - 'confirm': si solicita confirmar, aprobar o pasar a pedido una cotización (ej: 'confirma la cotización SO001', 'pasa a pedido la orden de Carlos').
               - 'edit_order': si solicita modificar cantidades o ítems en un pedido que ya está confirmado/bloqueado.
               - 'remove': si solicita anular, cancelar o borrar una orden completa (ej: 'cancela el pedido SO015', 'anula la cotización').
            
            2. 'customer': Nombre del cliente o empresa mencionada (ej: 'Carlos Pérez', 'Inversiones Alfa', 'Adhara Banda'). Si no se menciona explícitamente en este mensaje pero se venía tratando con él, mantenlo.
            3. 'items': Lista de objetos referenciados en la orden:
               - 'sku': Código SKU (ej: '6189', '5104') o nombre comercial si no hay código numérico explícito (ej: 'Delineador', 'Sexy Glam'). Si en el mensaje previo del asistente se mostró '[5104] Delineador Plumón Tattoo', extrae preferentemente el SKU '5104' o el nombre 'Delineador'.
               - 'qty': Cantidad de unidades referenciadas (siempre número positivo, ej: 2.0).
               - 'action': La operación a realizar sobre este ítem específico:
                 * 'add': Si se desea agregar o sumar unidades al pedido (ej: 'agrega 2 del SKU 6189', 'ponle 3 unidades').
                 * 'set': Si se desea fijar la cantidad total a un número exacto (ej: 'cambia la cantidad a 5', 'deja solo 3 unidades').
                 * 'remove': Si se desea borrar o eliminar completamente el producto de la orden (ej: 'elimina el producto 6189', 'quita por completo el delineador').
                 * 'subtract': Si se desea restar, reducir o quitar una cantidad parcial de unidades (ej: 'elimina 2 items de 6189', 'quita 2 unidades del Delineador', 'resta 1 unidad').
               - 'price_unit': Precio unitario acordado si se especificó explícitamente.
            4. 'status': Estado comercial específico ÚNICAMENTE si el usuario lo restringe de forma explícita:
               - 'draft': si pide expresamente 'cotizaciones', 'presupuestos' o 'en borrador' (ej: 'cuáles cotizaciones tengo abiertas').
               - 'sale': únicamente si pide expresamente 'pedidos confirmados', 'órdenes aprobadas' o 'ventas cerradas'.
               - 'cancel': si pide expresamente 'canceladas' o 'anuladas'.
               - None: si el usuario utiliza términos generales como 'pedidos', 'mis pedidos', 'cómo van los pedidos de X', 'órdenes', 'mis órdenes', 'qué tengo con X' (sin especificar si son borradores o confirmadas), deja 'status' como None para consultar tanto cotizaciones como pedidos confirmados y cancelados.
            5. 'period': Si se especifica periodo temporal ('hoy', 'este mes', 'semana').
            6. 'order_name': Si se especifica un código o número de orden comercial (ej: '03', 'SO001', 'S00003').
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=f"Solicitud comercial:\n{raw_query}"),
        ]

        extraction: Optional[SalesExtractionResult] = await self._extractor.ainvoke(messages)
        if extraction is None:
            extraction = SalesExtractionResult(action="list")
        elif isinstance(extraction, dict):
            try:
                valid_keys = SalesExtractionResult.model_fields.keys()
                clean_kwargs = {k: v for k, v in extraction.items() if k in valid_keys and v is not None}
                extraction = SalesExtractionResult(**clean_kwargs)
            except Exception as exc:
                logger.warning(f"[SalesManage] Error parseando dict de extracción ({exc}). Usando fallback.")
                extraction = SalesExtractionResult(action="list")

        items_dict = [
            it.model_dump() if hasattr(it, "model_dump") else (it.dict() if hasattr(it, "dict") else dict(it))
            for it in getattr(extraction, "items", [])
        ]

        action = getattr(extraction, "action", "list")

        # 1. Extraer o normalizar código de orden comercial
        extracted_order = getattr(extraction, "order_name", None)
        target_order = extract_order_code(extracted_order) if extracted_order else None

        if not target_order and raw_query:
            target_order = extract_order_code(raw_query)

        extracted_customer = getattr(extraction, "customer", None)
        prev_customer = state.get("customer_name")
        target_customer = extracted_customer or prev_customer

        # Detectar si el usuario especificó explícitamente un código de orden en su mensaje actual
        has_explicit_order_in_query = bool(extracted_order or (raw_query and extract_order_code(raw_query)))

        # 2. Si no se especificó orden en el mensaje actual:
        if not target_order:
            if extracted_customer and not has_explicit_order_in_query:
                # El usuario mencionó un cliente explícito; NO heredar órdenes activas previas de otro cliente
                target_order = None
            else:
                prev_order = state.get("target_order_name")
                if prev_order:
                    target_order = normalize_order_name(prev_order)
                elif state.get("messages"):
                    for m in reversed(state["messages"]):
                        found = extract_order_code(str(m.content))
                        if found:
                            target_order = found
                            break

        # 3. Si es una creación nueva desde cero (upsert) y el usuario no indicó orden explícita
        if action == "upsert" and not has_explicit_order_in_query:
            raw_lower = raw_query.lower()
            is_modifying_active = any(k in raw_lower for k in ("agrega", "modifica", "cambia", "quita", "resta", "elimina", "en la orden", "en la cotizac"))
            if not is_modifying_active:
                target_order = None

        customer_switched = bool(
            extracted_customer and prev_customer and extracted_customer.strip().lower() != prev_customer.strip().lower()
        )

        output_state = {
            "sales_action": action,
            "customer_name": target_customer,
            "items": items_dict,
            "status_filter": getattr(extraction, "status", None),
            "period_filter": getattr(extraction, "period", None),
            "target_order_name": target_order,
            "operation_result": None,
            "order_view_data": None,
            "orders_list": None,
            "cancellation_reason": None,
            "final_response": None,
        }

        # Si el usuario especificó un cliente explícito sin código de orden, limpiar IDs para evitar contaminación
        if extracted_customer and (customer_switched or not has_explicit_order_in_query):
            output_state["target_order_id"] = None
            output_state["partner_id"] = None
            output_state["customer_resolved"] = False
            output_state["customer_not_found"] = False

        return output_state

    async def reflect_intent_node(self, state: SalesManageState) -> dict:
        """Nodo de Reflexión: Autoanálisis crítico de la intención del usuario y coherencia comercial.
        
        Evalúa y rectifica la hipótesis del extractor en 4 dimensiones:
        1. Alcance: ¿Consulta global del vendedor ('mis pedidos') o cliente específico?
        2. Semántica de estado: ¿'pedidos' abarca cotizaciones en draft o exclusivamente confirmados?
        3. Coherencia orden vs cliente: ¿La orden pertenece al cliente solicitado?
        4. Claridad de intención: ¿Es inequívoca o requiere aclaración humana (HITL)?
        """
        raw_query = state.get("raw_query", "")
        raw_lower = raw_query.strip().lower()
        action = state.get("sales_action", "list")
        customer_name = state.get("customer_name")
        target_order = state.get("target_order_name")
        target_order_id = state.get("target_order_id")
        status_filter = state.get("status_filter")
        has_explicit_order = bool(extract_order_code(raw_query))

        # ==============================================================================
        # 1. Reflexión de Alcance: ¿Consulta global del vendedor o de cliente específico?
        # ==============================================================================
        global_markers = (
            "mis pedidos", "todos mis pedidos", "mis cotizaciones",
            "todas mis cotizaciones", "todas mis órdenes", "mis órdenes",
            "mis ventas", "todos los pedidos", "todas las órdenes",
            "dame todos", "dame mis", "muéstrame todos", "muéstrame mis",
            "listar todo", "lista todo", "ver mis pedidos", "ver todos",
        )
        is_global_seller_query = any(m in raw_lower for m in global_markers)

        # Si el usuario hace una consulta global de su cartera comercial
        if action == "list" and is_global_seller_query:
            # Reflexión: El usuario consulta sus pedidos en general, NO un cliente en específico.
            # Purgar cualquier cliente u orden residual heredado de turnos anteriores.
            is_strictly_confirmed = any(k in raw_lower for k in ("confirmad", "cerrad", "aprobada", "ganad"))
            cleaned_status = status_filter
            if status_filter in ("sale", "pedido", "pedidos") and not is_strictly_confirmed:
                cleaned_status = None

            return {
                "sales_action": "list",
                "customer_name": None,
                "partner_id": None,
                "target_order_name": None,
                "target_order_id": None,
                "customer_resolved": False,
                "customer_not_found": False,
                "status_filter": cleaned_status,
                "is_intent_clear": True,
                "reflection_reasoning": "Consulta global del vendedor ('mis pedidos'). Purgados cliente y orden previos para listar toda la cartera.",
            }

        # ==============================================================================
        # 2. Reflexión Semántica de Estado: ¿"pedidos" vs "cotizaciones"?
        # ==============================================================================
        if action == "list" and customer_name:
            # Si el usuario pregunta por un cliente específico (ej: "cómo van los pedidos de Janet")
            # y no exigió taxativamente "confirmados" / "aprobados",
            # la reflexión rectifica que el vendedor desea ver el pipeline completo del cliente.
            is_strictly_confirmed = any(k in raw_lower for k in ("confirmad", "cerrad", "aprobada", "ganad"))
            if status_filter in ("sale", "pedido", "pedidos") and not is_strictly_confirmed:
                status_filter = None

        # ==============================================================================
        # 3. Reflexión de Coherencia: Orden explícita vs Cliente solicitado
        # ==============================================================================
        if has_explicit_order and target_order:
            if customer_name and action in ("upsert", "add_items", "remove_items", "confirm", "edit_order"):
                try:
                    view_res = await self._view_quotation_tool(order_name=target_order)
                    ord_owner = view_res.get("customer_name") if view_res else None
                    if ord_owner and not _customers_match(customer_name, ord_owner):
                        # Conflicto explícito: orden S00003 de Adhara vs cliente Janet -> Escalar a HITL
                        return {
                            "is_intent_clear": False,
                            "clarification_question": (
                                f"La cotización {target_order} pertenece a {ord_owner}, pero indicaste a {customer_name}. "
                                f"¿Deseas modificar la cotización {target_order} de {ord_owner}, o crear una nueva para {customer_name}?"
                            ),
                            "clarification_options": [
                                f"Modificar {target_order} de {ord_owner}",
                                f"Crear nueva cotización para {customer_name}",
                                "Cancelar operación",
                            ],
                            "reflection_reasoning": f"Conflicto explícito: orden {target_order} ({ord_owner}) vs cliente {customer_name}.",
                        }
                except Exception as e:
                    logger.warning(f"Error al verificar orden {target_order} en reflect_intent_node: {e}")

            return {
                "sales_action": action,
                "customer_name": customer_name,
                "target_order_name": target_order,
                "target_order_id": target_order_id,
                "status_filter": status_filter,
                "is_intent_clear": True,
                "reflection_reasoning": f"Orden explícita '{target_order}' válida sin conflicto.",
            }

        # 4. Acciones de consulta o gestión sin orden explícita
        if action in ("list", "view", "remove", "edit_order", "confirm"):
            return {
                "sales_action": action,
                "customer_name": customer_name,
                "target_order_name": target_order,
                "target_order_id": target_order_id,
                "status_filter": status_filter,
                "is_intent_clear": True,
                "reflection_reasoning": f"Acción comercial '{action}' clara sin conflictos.",
            }

        # 5. Caso: Cliente especificado sin orden explícita en la consulta actual (Upsert / Add)
        if customer_name and not has_explicit_order:
            # Si había una orden previa en memoria, verificar si pertenecía a este cliente
            if target_order:
                try:
                    view_res = await self._view_quotation_tool(order_name=target_order)
                    ord_owner = view_res.get("customer_name") if view_res else None
                    if ord_owner and not _customers_match(customer_name, ord_owner):
                        # La orden en memoria era de otro cliente (ej: Adhara Banda); purgarla
                        target_order = None
                        target_order_id = None
                except Exception as e:
                    logger.warning(f"Error al verificar orden previa {target_order}: {e}")
                    target_order = None
                    target_order_id = None

            return {
                "sales_action": action,
                "customer_name": customer_name,
                "target_order_name": target_order,
                "target_order_id": target_order_id,
                "partner_id": None if not target_order else state.get("partner_id"),
                "customer_resolved": False if not target_order else state.get("customer_resolved", False),
                "status_filter": status_filter,
                "is_intent_clear": True,
                "reflection_reasoning": f"Intención clara para el cliente '{customer_name}'. Orden previa purgada si correspondía a otro cliente.",
            }

        return {
            "sales_action": action,
            "customer_name": customer_name,
            "target_order_name": target_order,
            "target_order_id": target_order_id,
            "status_filter": status_filter,
            "is_intent_clear": True,
            "reflection_reasoning": "Intención clara por defecto.",
        }



    async def feedback_intent_clarification_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """HITL Interrupt: Aclara la intención del usuario cuando la reflexión detecta ambigüedad."""
        var_child_runnable_config.set(config)
        question = state.get("clarification_question") or "Por favor aclara cómo deseas procesar esta cotización o pedido."
        options = state.get("clarification_options") or []

        user_resume = interrupt({
            "type": "intent_clarification",
            "question": question,
            "options": options,
        })

        user_reply = str(user_resume).strip()
        if not user_reply or user_reply.lower() in ("/cancel", "/salir", "cancelar", "salir"):
            return {
                "is_intent_clear": False,
                "cancellation_reason": "Operación cancelada por el usuario al solicitar aclaración de intención.",
            }

        if options:
            sel = await parse_hitl_entity_selection(user_input=user_reply, options=options, llm=self._llm)
            if sel.action == "cancel":
                return {
                    "is_intent_clear": False,
                    "cancellation_reason": "Operación cancelada por el usuario.",
                }
            elif sel.action == "new":
                return {
                    "is_intent_clear": True,
                    "sales_action": "upsert",
                    "target_order_name": None,
                    "target_order_id": None,
                    "partner_id": None,
                    "customer_resolved": False,
                }
            elif sel.action == "select" and sel.selected_entity_name:
                ord_code = extract_order_code(sel.selected_entity_name)
                return {
                    "is_intent_clear": True,
                    "target_order_name": ord_code or sel.selected_entity_name,
                }

        is_ok = await parse_hitl_binary_decision(user_input=user_reply, context_question=question, llm=self._llm)
        if is_ok:
            return {
                "is_intent_clear": True,
                "sales_action": "upsert",
                "target_order_name": None,
                "target_order_id": None,
                "partner_id": None,
                "customer_resolved": False,
            }
        else:
            return {
                "is_intent_clear": False,
                "cancellation_reason": "Operación cancelada por el usuario tras la aclaración.",
            }

    async def resolve_customer_node(self, state: SalesManageState) -> dict:
        """Valida que el cliente exista en la cartera del vendedor sin ambigüedades."""
        user_id = int(state.get("user_id") or 5)
        cust_name = state.get("customer_name")
        partner_id = state.get("partner_id")
        customer_resolved = state.get("customer_resolved", False)

        if partner_id and customer_resolved:
            return {"customer_resolved": True, "customer_not_found": False}


        # Si no hay nombre de cliente especificado, pero hay código de orden, intentar resolverlo de la orden
        if not cust_name:
            order_name = state.get("target_order_name")
            if order_name:
                view_res = await self._view_quotation_tool(order_name=order_name)
                if view_res and view_res.get("partner_id"):
                    return {
                        "partner_id": view_res["partner_id"],
                        "customer_name": view_res.get("customer_name"),
                        "customer_resolved": True,
                        "customer_not_found": False,
                    }
            # Sin cliente y sin orden: no se puede resolver
            return {"customer_resolved": False, "customer_not_found": False}

        candidates = await self._list_customers_tool(user_id=user_id, name=cust_name, limit=5)

        if len(candidates) == 1:
            return {
                "partner_id": candidates[0]["id"],
                "customer_name": candidates[0]["name"],
                "customer_resolved": True,
                "customer_not_found": False,
            }
        elif len(candidates) > 1:
            return {
                "candidate_partners": candidates,
                "customer_resolved": False,
                "customer_not_found": False,
            }
        else:
            return {
                "candidate_partners": [],
                "customer_resolved": False,
                "customer_not_found": True,
            }

    async def feedback_ambiguous_customer_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """HITL Interrupt: Resuelve ambigüedad de clientes presentando opciones legibles sin exponer IDs técnicos."""
        var_child_runnable_config.set(config)
        candidates = state.get("candidate_partners", [])
        
        # Presentar solo nombre comercial y teléfono
        options = [
            f"{c['name']} (Tel: {c.get('phone') or 'Sin teléfono'})"
            for c in candidates
        ]
        options_text = "; ".join(f"'{opt}'" for opt in options)
        question = (
            f"Encontré varios clientes coincidentes en tu cartera: {options_text}. "
            f"¿A cuál de ellos te refieres, o prefieres cancelar?"
        )

        user_resume = interrupt({
            "type": "ambiguous_customer_conflict",
            "question": question,
            "options": options,
        })

        selection = await parse_hitl_entity_selection(
            user_input=str(user_resume),
            options=options,
            llm=self._llm,
        )

        if selection.action == "select" and selection.selected_index is not None and selection.selected_index < len(candidates):
            chosen = candidates[selection.selected_index]
            return {
                "partner_id": chosen["id"],
                "customer_name": chosen["name"],
                "customer_resolved": True,
                "customer_not_found": False,
            }
        elif selection.action == "new":
            return {
                "customer_resolved": False,
                "customer_not_found": True,
            }
        else:
            return {
                "customer_resolved": False,
                "cancellation_reason": "Operación cancelada por el usuario al no definir el cliente.",
            }

    async def feedback_unknown_customer_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """HITL Interrupt: Ofrece registrar de inmediato un cliente no encontrado en la cartera."""
        var_child_runnable_config.set(config)
        user_id = int(state.get("user_id") or 5)
        cust_name = state.get("customer_name") or "el cliente"
        question = (
            f"El cliente '{cust_name}' no se encuentra en tu cartera comercial de Odoo. "
            f"¿Deseas darlo de alta en este momento para continuar con la venta?"
        )

        user_resume = interrupt({
            "type": "unknown_customer_conflict",
            "question": question,
            "suggested_actions": ["yes", "no"],
        })

        is_confirmed = await parse_hitl_binary_decision(
            user_input=str(user_resume),
            context_question=question,
            llm=self._llm,
        )

        if is_confirmed and state.get("customer_name"):
            res = await self._upsert_customer_tool(
                user_id=user_id,
                name=state["customer_name"],
            )
            if res.get("success") and res.get("id"):
                return {
                    "partner_id": res["id"],
                    "customer_resolved": True,
                    "customer_not_found": False,
                }

        return {
            "customer_resolved": False,
            "cancellation_reason": f"Operación cancelada. No se pudo asociar la venta al cliente '{cust_name}'.",
        }

    async def list_sales_orders_node(self, state: SalesManageState) -> dict:
        """Rama List: Invoca tool (list_sales_orders) y agrupa por estado comercial."""
        user_id = int(state.get("user_id") or 5)
        partner_id = state.get("partner_id")
        customer_name = state.get("customer_name")
        status_filter = state.get("status_filter")
        period_filter = state.get("period_filter")

        grouped_orders = await self._list_orders_tool(
            user_id=user_id,
            partner_id=partner_id,
            customer_name=customer_name,
            status=status_filter,
            period=period_filter,
        )

        # Si se filtró por un estado restrictivo y devolvió 0 registros para un cliente específico,
        # verificar si el cliente tiene cotizaciones u órdenes en otros estados para no dar falso negativo.
        total_found = sum(len(v) for v in grouped_orders.values()) if isinstance(grouped_orders, dict) else 0
        if total_found == 0 and status_filter and (customer_name or partner_id):
            try:
                fallback_orders = await self._list_orders_tool(
                    user_id=user_id,
                    partner_id=partner_id,
                    customer_name=customer_name,
                    status=None,
                    period=period_filter,
                )
                fallback_total = sum(len(v) for v in fallback_orders.values()) if isinstance(fallback_orders, dict) else 0
                if fallback_total > 0:
                    grouped_orders = fallback_orders
            except Exception as e:
                logger.warning(f"Error en búsqueda fallback de órdenes: {e}")

        target_name = state.get("target_order_name")
        view_data = None
        if target_name:
            try:
                view_data = await self._view_quotation_tool(order_name=target_name)
            except Exception as e:
                logger.warning(f"No se pudo consultar vista de orden {target_name}: {e}")

        return {
            "orders_list": grouped_orders,
            "order_view_data": view_data,
        }

    async def list_current_sales_orders_node(self, state: SalesManageState) -> dict:
        """Subflujo Duplicados: Recupera cotizaciones abiertas del cliente para el LLM Judge."""
        user_id = int(state.get("user_id") or 5)
        partner_id = state.get("partner_id")
        customer_name = state.get("customer_name")

        candidates = await self._list_current_orders_tool(
            user_id=user_id,
            partner_id=partner_id,
            customer_name=customer_name,
        )
        return {"current_candidates": candidates}

    async def duplicate_candidates_judge_node(self, state: SalesManageState) -> dict:
        """Subflujo Duplicados: LLM evalúa si las líneas coinciden con cotizaciones ya abiertas."""
        candidates = state.get("current_candidates", [])
        target_name = state.get("target_order_name")

        # Si el usuario ya especificó un código de orden comercial (ej: SO001), buscar coincidencia directa
        if target_name:
            target_norm = normalize_order_name(target_name)
            for c in candidates:
                c_norm = normalize_order_name(c.get("name", ""))
                if c_norm and target_norm and c_norm.upper() == target_norm.upper():
                    return {
                        "has_possible_duplicates": False,
                        "target_order_id": c["id"],
                        "target_order_name": c["name"],
                        "duplicate_choice": "selected",
                    }
            # Si no estaba en candidates (ej. orden confirmada 'sale'), resolver ID vía view_quotation_tool
            view_res = await self._view_quotation_tool(order_name=target_norm or target_name)
            if view_res and view_res.get("id"):
                return {
                    "has_possible_duplicates": False,
                    "target_order_id": view_res["id"],
                    "target_order_name": view_res.get("name") or target_norm or target_name,
                    "duplicate_choice": "selected",
                }

        if not candidates:
            return {
                "has_possible_duplicates": False,
                "duplicate_choice": "new",
            }

        items = state.get("items", [])
        items_desc = ", ".join(f"{it.get('qty', 1)} de SKU {it.get('sku', '')}" for it in items)

        cand_lines_str = []
        for c in candidates:
            lines = c.get("lines", [])
            if lines:
                lines_desc = ", ".join(f"{l.get('quantity', 1):.0f}x {l.get('product_name', '')}" for l in lines)
            else:
                lines_desc = "Sin detalle de líneas"
            cand_lines_str.append(
                f"- Cotización: {c['name']}, Fecha: {c.get('date_order', '')}, Total: S/ {c.get('amount_total', 0):.2f}, Productos: [{lines_desc}]"
            )
        cand_text = "\n".join(cand_lines_str)

        system_prompt = """
            Eres un auditor de órdenes de venta para Odoo ERP.
            Evalúa si los productos que el vendedor desea agregar o cotizar coinciden o podrían corresponder
            a una de las cotizaciones ya abiertas para este cliente, evitando órdenes huérfanas o duplicadas.

            Criterios de evaluación:
            1. Si una cotización abierta del cliente contiene los mismos productos o SKUs solicitados (total o parcialmente), marca has_duplicates=True e indica matched_order_name con el código de la cotización (ej: 'S00002').
            2. Si las cotizaciones abiertas del cliente tienen fecha muy reciente (ej. hoy) y montos o productos similares, sugiere que se evalúe como duplicado potencial (has_duplicates=True).
            3. Si las cotizaciones abiertas tienen productos completamente distintos y ajenos a la solicitud, marca has_duplicates=False.
        """

        user_prompt = f"""
            Ítems solicitados en la consulta actual:
            {items_desc or 'Productos varios'}

            Cotizaciones abiertas en Odoo para este cliente:
            {cand_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        judge: Optional[SalesDuplicateCheckResult] = await self._judge.ainvoke(messages)
        if judge is None:
            judge = SalesDuplicateCheckResult(
                has_duplicates=len(candidates) > 0,
                matched_order_name=candidates[0]["name"] if candidates else None,
                reasoning="Coincidencia sugerida por fecha reciente.",
            )
        elif isinstance(judge, dict):
            try:
                valid_keys = SalesDuplicateCheckResult.model_fields.keys()
                clean_kwargs = {k: v for k, v in judge.items() if k in valid_keys and v is not None}
                judge = SalesDuplicateCheckResult(**clean_kwargs)
            except Exception as exc:
                logger.warning(f"[SalesManage] Error parseando dict de duplicados ({exc}). Usando fallback.")
                judge = SalesDuplicateCheckResult(has_duplicates=False)

        has_dups = getattr(judge, "has_duplicates", False)
        matched_name = getattr(judge, "matched_order_name", None)

        # Encontrar ID interno si hubo match
        matched_id = None
        if matched_name:
            for c in candidates:
                if str(c.get("name")).upper() == str(matched_name).upper():
                    matched_id = c["id"]
                    break

        return {
            "has_possible_duplicates": has_dups,
            "duplicate_rationale": getattr(judge, "reasoning", ""),
            "target_order_name": matched_name or state.get("target_order_name"),
            "target_order_id": matched_id or state.get("target_order_id"),
        }

    async def feedback_duplicate_sales_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """Subflujo Duplicados: HITL interrupt evaluando en lenguaje natural si actualiza orden existente o abre nueva."""
        var_child_runnable_config.set(config)
        candidates = state.get("current_candidates", [])
        matched_name = state.get("target_order_name")

        cand_order_strings = []
        for c in candidates:
            lines = c.get("lines", [])
            lines_summary = f" - {', '.join(l.get('product_name', '') for l in lines[:2])}" if lines else ""
            cand_order_strings.append(f"{c['name']} (S/ {c.get('amount_total', 0):.2f}{lines_summary})")

        cand_desc = "; ".join(cand_order_strings)

        if matched_name:
            question = (
                f"Detecté que ya existe una cotización abierta para este cliente ({matched_name}) que parece un duplicado o contiene productos coincidentes. "
                f"Cotizaciones abiertas encontradas: {cand_desc}. "
                f"¿Deseas actualizar la cotización existente ({matched_name}) o prefieres abrir una nueva orden?"
            )
        else:
            question = (
                f"Encontré cotizaciones abiertas recientes para este cliente: {cand_desc}. "
                f"¿Deseas actualizar una cotización existente o prefieres abrir una nueva orden?"
            )

        user_resume = interrupt({
            "type": "duplicate_order_conflict",
            "question": question,
            "candidate_orders": cand_order_strings,
        })

        choice_result = await parse_hitl_order_choice(
            user_input=str(user_resume),
            candidate_orders=cand_order_strings,
            llm=self._llm,
        )

        if choice_result.choice == "selected":
            target_name = choice_result.selected_order_name or (candidates[0]["name"] if candidates else None)
            target_id = None
            if target_name:
                for c in candidates:
                    if str(c.get("name", "")).upper() == str(target_name).upper():
                        target_id = c["id"]
                        break
            if not target_id and candidates:
                target_id = candidates[0]["id"]
                target_name = candidates[0]["name"]

            return {
                "duplicate_choice": "selected",
                "target_order_name": target_name,
                "target_order_id": target_id,
            }
        elif choice_result.choice == "new":
            return {
                "duplicate_choice": "new",
                "target_order_name": None,
                "target_order_id": None,
            }
        else:
            return {
                "duplicate_choice": "cancel",
                "cancellation_reason": "Operación cancelada por el usuario ante la selección de orden.",
            }

    async def create_quotation_node(self, state: SalesManageState) -> dict:
        """Rama Upsert (new): Crea cotización en borrador en Odoo."""
        user_id = int(state.get("user_id") or 5)
        partner_id = state.get("partner_id")
        items = state.get("items", [])

        if not partner_id:
            return {
                "operation_result": {
                    "success": False,
                    "error": "No se pudo crear la cotización porque falta el cliente.",
                }
            }

        result = await self._create_quotation_tool(
            user_id=user_id,
            partner_id=partner_id,
            items=items,
        )
        return {
            "operation_result": result,
            "target_order_id": result.get("order_id"),
            "target_order_name": result.get("name"),
        }

    async def update_quotation_node(self, state: SalesManageState) -> dict:
        """Rama Upsert (selected): Actualiza cotización existente en Odoo."""
        order_id = state.get("target_order_id")
        order_name = state.get("target_order_name")
        items = state.get("items", [])

        view_data = {}
        if not order_id and order_name:
            view_data = await self._view_quotation_tool(order_name=order_name)
            order_id = view_data.get("id")
        elif order_id:
            view_data = await self._view_quotation_tool(order_id=order_id)

        if not order_id and not order_name:
            return {
                "operation_result": {
                    "success": False,
                    "error": "No se identificó el número de cotización a actualizar.",
                }
            }

        # Guardrail de cliente: verificar que la orden a actualizar pertenezca al cliente pedido
        req_cust = state.get("customer_name")
        ord_cust = view_data.get("customer_name")
        if req_cust and ord_cust and not _customers_match(req_cust, ord_cust):
            logger.warning(
                f"Guardrail activado en update_quotation_node: orden '{order_name or order_id}' "
                f"pertenece a '{ord_cust}', pero se solicitó para '{req_cust}'"
            )
            return {
                "operation_result": {
                    "success": False,
                    "error": f"La cotización {order_name or order_id} pertenece a {ord_cust}, no a {req_cust}. Operación abortada para evitar alterar el pedido incorrecto.",
                }
            }

        result = await self._update_quotation_tool(
            order_id=int(order_id) if order_id else None,
            order_name=order_name,
            items=items,
        )

        return {
            "operation_result": result,
            "target_order_id": result.get("order_id") or order_id,
            "target_order_name": result.get("name") or order_name,
        }

    async def view_quotation_node(self, state: SalesManageState) -> dict:
        """Inspecciona el resumen financiero y detalle de líneas de la orden en Odoo."""
        order_id = state.get("target_order_id")
        order_name = state.get("target_order_name")
        if order_name:
            order_name = normalize_order_name(order_name)

        view_data = await self._view_quotation_tool(order_id=order_id, order_name=order_name)
        return {
            "order_view_data": view_data,
            "target_order_id": view_data.get("id") or order_id,
            "target_order_name": view_data.get("name") or order_name,
            "partner_id": state.get("partner_id") or view_data.get("partner_id"),
            "customer_name": state.get("customer_name") or view_data.get("customer_name"),
        }

    async def confirm_order_node(self, state: SalesManageState) -> dict:
        """Rama Confirm: Confirma cotización pasando a pedido oficial de venta en Odoo."""
        order_id = state.get("target_order_id")
        order_name = state.get("target_order_name")

        if not order_id and order_name:
            view_data = await self._view_quotation_tool(order_name=order_name)
            order_id = view_data.get("id")

        if not order_id:
            return {
                "operation_result": {
                    "success": False,
                    "error": "No se encontró el ID de la orden para confirmar.",
                }
            }

        result = await self._confirm_order_tool(order_id=int(order_id))
        return {
            "operation_result": result,
            "target_order_id": order_id,
            "target_order_name": result.get("name") or order_name,
        }

    async def guardrail_unlock_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """Rama Edit Order: Guardrail HITL en lenguaje natural antes de desbloquear pedido confirmado."""
        var_child_runnable_config.set(config)
        order_name = state.get("target_order_name") or "la orden"
        question = (
            f"El pedido {order_name} ya está confirmado y bloqueado en Odoo. "
            f"Para modificar sus productos es necesario desbloquearlo primero. "
            f"¿Deseas desbloquearlo y aplicar las modificaciones?"
        )

        user_resume = interrupt({
            "type": "unlock_order_guardrail",
            "question": question,
            "suggested_actions": ["yes", "no"],
        })

        is_confirmed = await parse_hitl_binary_decision(
            user_input=str(user_resume),
            context_question=question,
            llm=self._llm,
        )
        return {"unlock_confirmed": is_confirmed}

    async def unlock_order_node(self, state: SalesManageState) -> dict:
        """Rama Edit Order: Desbloquea la orden en Odoo (action_unlock)."""
        order_id = state.get("target_order_id")
        if not order_id and state.get("target_order_name"):
            view_data = await self._view_quotation_tool(order_name=state["target_order_name"])
            order_id = view_data.get("id")

        if order_id:
            await self._unlock_order_tool(order_id=int(order_id))
            return {"target_order_id": order_id}
        return {}

    async def update_order_node(self, state: SalesManageState) -> dict:
        """Rama Edit Order: Actualiza las líneas del pedido desbloqueado."""
        order_id = state.get("target_order_id")
        items = state.get("items", [])
        if order_id:
            res = await self._update_order_tool(order_id=int(order_id), items=items)
            return {"operation_result": res}
        return {"operation_result": {"success": False, "error": "Falta ID de orden para actualizar."}}

    async def lock_order_node(self, state: SalesManageState) -> dict:
        """Rama Edit Order: Re-bloquea automáticamente la orden (action_lock)."""
        order_id = state.get("target_order_id")
        if order_id:
            await self._lock_order_tool(order_id=int(order_id))
        return {}

    async def feedback_remove_order_node(
        self,
        state: SalesManageState,
        config: RunnableConfig,
    ) -> dict:
        """Rama Remove: Confirmación HITL en lenguaje natural antes de cancelar."""
        var_child_runnable_config.set(config)
        order_name = state.get("target_order_name") or "la orden seleccionada"
        question = f"¿Estás seguro de que deseas cancelar la orden {order_name} en Odoo?"

        user_resume = interrupt({
            "type": "remove_order_confirmation",
            "question": question,
            "suggested_actions": ["yes", "no"],
        })

        is_confirmed = await parse_hitl_binary_decision(
            user_input=str(user_resume),
            context_question=question,
            llm=self._llm,
        )
        return {"remove_confirmed": is_confirmed}

    async def remove_order_node(self, state: SalesManageState) -> dict:
        """Rama Remove: Cancela la orden en Odoo (action_cancel)."""
        order_id = state.get("target_order_id")
        if not order_id and state.get("target_order_name"):
            view_data = await self._view_quotation_tool(order_name=state["target_order_name"])
            order_id = view_data.get("id")

        if not order_id:
            return {
                "operation_result": {
                    "success": False,
                    "error": "No se identificó la orden a cancelar en Odoo.",
                }
            }

        result = await self._remove_order_tool(order_id=int(order_id))
        return {"operation_result": result}

    async def synthesize_sales_response(self, state: SalesManageState) -> dict:
        """Nodo Final: Sintetiza una respuesta profesional sin exponer IDs técnicos internos."""
        cancellation = state.get("cancellation_reason")
        if cancellation:
            return {
                "final_response": cancellation,
                "messages": [AIMessage(content=cancellation)],
            }

        raw_query = state.get("raw_query") or ""
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=8,
            strategy="last",
            token_counter=len,
        )

        action = state.get("sales_action", "list")
        op_res = state.get("operation_result") or {}
        view_data = state.get("order_view_data") or {}
        orders_list = state.get("orders_list") or {}

        system_prompt = inject_soul(
            """
            Directrices específicas de gestión de órdenes y pedidos:
            - NUNCA menciones códigos internos como 'user_id', 'partner_id' ni IDs numéricos de base de datos.
            - NUNCA menciones términos de backend como 'ERP' u 'Odoo'.
            - Utiliza el nombre comercial del cliente y el código comercial de la orden (ej: 'SO001', 'S00003').
            - Si el usuario pide ver, explicar o consultar los productos de una orden o cotización (ej: 'explícame la cotización S00003', 'dame el detalle de los productos', 'lista los productos', 'qué productos tiene'):
              * NUNCA preguntes al usuario si desea que consultes el detalle ni ofrezcas hacerlo más adelante.
              * LISTA INMEDIATAMENTE de forma clara cada producto de la orden con su nombre comercial, cantidad y precio o subtotal a partir de los datos en 'Detalle de la orden' o 'Listado de órdenes'.
            - Si el usuario solicitó agregar, restar o eliminar ítems (ej: 'elimina 2 items de 6189'):
              * Explica claramente los cambios realizados y presenta el detalle resultante actualizado de la orden con las nuevas cantidades y montos totales actualizados.
            - Si el usuario hace una pregunta de seguimiento (ej: '¿cuántas hay?', '¿cuál fue la última?', '¿cuál es el total?'), responde directamente a esa pregunta con precisión basándote en la información comercial y el historial conversacional.
            - Si es un listado, presenta las órdenes agrupadas por estado (Cotizaciones, Pedidos Confirmados, Cancelados) indicando cliente, fecha y monto total.
            - Si es una cotización creada o actualizada, resume los productos incluidos, subtotal, impuestos y total general.
            - Si es una confirmación de pedido, celebra la venta con entusiasmo profesional y sobrio.
            - Si es una cancelación, confirma cordialmente que la orden ha sido cancelada.
            """,
            role=SoulRole.SALES_ORDERS,
        )

        user_content = f"""
            Consulta actual del usuario: {raw_query}

            Contexto de la operación:
            - Acción ejecutada: {action}
            - Resultado de la operación: {op_res}
            - Detalle de la orden (vista): {view_data}
            - Listado de órdenes agrupadas en el sistema: {orders_list}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_content),
        ]

        synth_result: Optional[SalesSynthesizeResponse] = None
        try:
            synth_result = await self._synthesizer.ainvoke(messages)
        except Exception as e:
            logger.warning(f"Error en _synthesizer.ainvoke: {e}")
            try:
                # Intento de recuperación con invocación directa al LLM
                direct_msg = await self._llm.ainvoke(messages)
                content_text = extract_clean_text(getattr(direct_msg, "content", ""))
                if content_text:
                    synth_result = SalesSynthesizeResponse(response_text=content_text)

            except Exception as e2:
                logger.warning(f"Error en recuperación directa de síntesis: {e2}")

        resp_text = getattr(synth_result, "response_text", "") if synth_result else ""
        if not resp_text and isinstance(synth_result, dict):
            resp_text = synth_result.get("response_text", "")

        if not resp_text:
            target_name = state.get("target_order_name") or op_res.get("name") or view_data.get("name")
            cust = state.get("customer_name") or op_res.get("customer_name") or view_data.get("customer_name")
            
            if op_res.get("success") is False:
                err = op_res.get("error") or "No se pudo procesar la operación en Odoo."
                resp_text = f"No se pudo completar la operación{' para ' + cust if cust else ''}: {err}"
            elif action in ("upsert", "create") or op_res.get("action") == "created":
                tot = op_res.get("amount_total") or view_data.get("amount_total") or 0.0
                ord_str = f" **{target_name}**" if target_name else ""
                cust_str = f" para **{cust}**" if cust else ""
                resp_text = f"Se ha creado exitosamente la cotización{ord_str}{cust_str} por un total de **S/ {tot:.2f}**."
            elif action in ("add_items", "remove_items", "update") or op_res.get("action") == "updated":
                tot = op_res.get("amount_total") or view_data.get("amount_total") or 0.0
                ord_str = f" **{target_name}**" if target_name else ""
                cust_str = f" para **{cust}**" if cust else ""
                resp_text = f"Se ha actualizado exitosamente la cotización{ord_str}{cust_str}. El nuevo total es **S/ {tot:.2f}**."
            elif action == "list" or orders_list:
                # Manejador determinista y profesional para listados
                draft_orders = orders_list.get("draft", []) if isinstance(orders_list, dict) else []
                sale_orders = orders_list.get("sale", []) if isinstance(orders_list, dict) else []
                cancel_orders = orders_list.get("cancel", []) if isinstance(orders_list, dict) else []

                parts = []
                if cust:
                    header = f"Aquí tienes el estado de las cotizaciones y pedidos de **{cust}** en Odoo:"
                else:
                    header = "Aquí tienes el listado general de tus cotizaciones y pedidos en Odoo:"
                parts.append(header)

                if draft_orders:
                    parts.append("\n### 📋 Cotizaciones en Borrador")
                    for o in draft_orders:
                        lines_desc = ""
                        if o.get("lines"):
                            lines_desc = " (" + ", ".join(f"{l.get('quantity', 1):.0f}x {l.get('product_name', '')}" for l in o["lines"]) + ")"
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **S/ {o.get('amount_total', 0.0):.2f}**{lines_desc}")

                if sale_orders:
                    parts.append("\n### ✅ Pedidos Confirmados")
                    for o in sale_orders:
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **S/ {o.get('amount_total', 0.0):.2f}**")

                if cancel_orders:
                    parts.append("\n### 🚫 Órdenes Canceladas")
                    for o in cancel_orders:
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **S/ {o.get('amount_total', 0.0):.2f}**")

                if not draft_orders and not sale_orders and not cancel_orders:
                    target_str = f" para **{cust}**" if cust else ""
                    parts = [f"Actualmente no se encontraron pedidos ni cotizaciones registradas en Odoo{target_str}."]

                resp_text = "\n".join(parts)
            elif action == "view" and view_data:
                v_name = view_data.get("name") or target_name or "la orden"
                v_cust = view_data.get("customer_name") or cust or ""
                v_tot = view_data.get("amount_total", 0.0)
                v_lines = view_data.get("lines", [])
                parts = [f"Detalle de cotización **{v_name}** de **{v_cust}** (Total: **S/ {v_tot:.2f}**):"]
                for l in v_lines:
                    parts.append(f"* {l.get('quantity', 1):.0f}x {l.get('product_name', '')} a S/ {l.get('price_unit', 0.0):.2f}")
                resp_text = "\n".join(parts)
            else:
                cust_str = f" para **{cust}**" if cust else ""
                ord_str = f" ({target_name})" if target_name else ""
                resp_text = f"Detalle de consulta comercial en Odoo{cust_str}{ord_str}."

        return {
            "final_response": resp_text,
            "messages": [AIMessage(content=resp_text)],
        }
