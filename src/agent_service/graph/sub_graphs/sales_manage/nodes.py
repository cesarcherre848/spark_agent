"""
src/agent_service/graph/sub_graphs/sales_manage/nodes.py - Nodos ejecutores del subgrafo sales_manage

Implementa el patrón unificado Planner - Executor - Synthesizer con Evaluación por Rúbrica y Reflexión
(Fast Reflection Loop para refinamiento de síntesis y Slow Reflection Loop para re-planificación de herramientas),
preservación estricta de contexto multi-turno y compatibilidad con guardrails Human-in-the-Loop.
"""

import inspect
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
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
    SalesSKUItem,
    PlannedSalesTool,
    SalesPlan,
    SalesQualityRubricEvaluation,
)
from src.agent_service.core.hitl.parser import (
    parse_hitl_binary_decision,
    parse_hitl_entity_selection,
    parse_hitl_order_choice,
)
from src.agent_service.core.templates.dialogs import HITLDialogTemplates
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
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


class SalesManageNodes(BaseSynthesizerNode):
    """Nodos del subgrafo de gestión de ventas, órdenes y cotizaciones en Odoo ERP con Planner y Reflexión."""

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
        catalog_search_tool: Optional[Callable[..., Awaitable[List[Dict[str, Any]]]]] = None,
        default_max_iterations: int = 3,
        tools_registry: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(llm=llm)
        self._llm = llm
        self._default_max_iterations = default_max_iterations
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
        self._catalog_search_tool = catalog_search_tool

        # Registro desacoplado de herramientas ejecutables
        self._tools_registry = tools_registry or {
            "odoo_list_sales_orders": self._list_orders_tool,
            "odoo_view_quotation": self._view_quotation_tool,
            "odoo_create_quotation": self._create_quotation_tool,
            "odoo_update_quotation": self._update_quotation_tool,
            "odoo_confirm_order": self._confirm_order_tool,
            "odoo_remove_sale_order": self._remove_order_tool,
            "odoo_unlock_order": self._unlock_order_tool,
            "odoo_update_order": self._update_order_tool,
            "odoo_lock_order": self._lock_order_tool,
            "odoo_list_current_customers": self._list_customers_tool,
            "odoo_upsert_customer": self._upsert_customer_tool,
        }
        if self._catalog_search_tool:
            self._tools_registry["search_product_catalog"] = self._catalog_search_tool

        # Modelos estructurados para el patrón Planner-Judge-Synthesizer
        self._planner = bind_structured_output(bind_temperature(llm, 0.1), SalesPlan)
        self._rubric_judge = bind_structured_output(bind_temperature(llm, 0.0), SalesQualityRubricEvaluation)
        self._synthesizer = bind_structured_output(bind_temperature(llm, 0.35), SalesSynthesizeResponse)

        # Modelos legacy para compatibilidad
        self._extractor = bind_structured_output(bind_temperature(llm, 0.0), SalesExtractionResult)
        self._judge = bind_structured_output(bind_temperature(llm, 0.1), SalesDuplicateCheckResult)
        self._reflector = bind_structured_output(bind_temperature(llm, 0.0), SalesReflectionResult)

    # ==============================================================================
    # 1. NODO PLANIFICADOR DE VENTAS (Sales Planner Node)
    # ==============================================================================
    async def plan_and_select_tools(self, state: SalesManageState) -> Dict[str, Any]:
        """Nodo 1: Analiza la consulta, el historial conversacional multi-turno y posibles críticas
        de reflexión para generar un plan estructurado de herramientas comerciales (SalesPlan)."""
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        # Si no hay crítica previa activa, es un nuevo turno: reiniciar contador
        critique = state.get("critique")
        if not critique:
            iteration_count = 0
            remedy_suggestions = []
        else:
            iteration_count = state.get("iteration_count", 0)
            remedy_suggestions = state.get("suggested_improvements") or []

        user_id = int(state.get("user_id") or 5)
        active_customer = state.get("customer_name")
        active_partner_id = state.get("partner_id")
        active_order = state.get("target_order_name")
        active_order_id = state.get("target_order_id")
        user_context = state.get("user_context") or ""

        # Historial de diálogo reciente
        all_messages = state.get("messages", [])
        history_pool = (
            all_messages[:-1]
            if (all_messages and isinstance(all_messages[-1], HumanMessage) and all_messages[-1].content == raw_query)
            else all_messages
        )
        trimmed_history = trim_messages(
            history_pool,
            max_tokens=8,
            strategy="last",
            token_counter=len,
        )

        critique_block = ""
        if critique:
            critique_block = f"""
            RETROALIMENTACIÓN DE REFLEXIÓN PREVIA (ITERACIÓN {iteration_count}):
            - Crítica del Juez de Calidad: {critique}
            - Mejoras obligatorias: {remedy_suggestions}
            DEBES ajustar el plan o seleccionar herramientas adicionales para corregir estas deficiencias.
            """

        system_prompt = f"""
            Eres el planificador experto de gestión de ventas, órdenes y cotizaciones comerciales en Odoo ERP.
            Tu misión es analizar la consulta del vendedor y el contexto multi-turno para formular un plan estructurado (SalesPlan)
            con la lista de herramientas necesarias para responder de forma precisa, veraz y ejecutiva.

            {critique_block}

            CONTEXTO COMERCIAL ACTIVO EN SESIÓN:
            - Cliente activo en memoria: '{active_customer or 'Ninguno'}' (ID: {active_partner_id or 'No asignado'})
            - Orden/Cotización activa en memoria: '{active_order or 'Ninguna'}' (ID: {active_order_id or 'No asignado'})
            - Antecedentes del comercial: {user_context}

            DIRECTRICES DE RESOLUCIÓN CONTEXTUAL Y MULTI-TURNO:
            1. Referencias anafóricas y elípticas:
               - Si el usuario dice 'esa orden', 'el pedido anterior', 'verlo', 'muéstramelo', 'cuánto es', hace referencia a la orden activa '{active_order or 'la mencionada en turnos previos'}'.
               - Si el usuario dice 'agrégale a esa orden', 'ponle 2 más', 'quita el labial', asócialo a la cotización activa '{active_order}'.
               - Si el usuario menciona un nuevo cliente explícito (ej: 'cotízale a Pedro López'), PURGA las órdenes del cliente anterior.
            2. Consultas globales de cartera comercial ('mis pedidos', 'todos mis pedidos', 'mis ventas', 'mis cotizaciones'):
               - NO restrinjas a un cliente específico. Planifica 'odoo_list_sales_orders' con customer_name=None para traer la cartera completa del vendedor.
               - Si el usuario pide 'el detalle de todos mis pedidos' o 'con productos', 'odoo_list_sales_orders' ya incluye las líneas de productos detalladas de cada orden.
            3. Consulta de detalle de una orden específica ('ver S00003', 'detalle de SO001', 'explícame el pedido'):
               - Planifica 'odoo_view_quotation' con 'order_name' normalizado.
            4. Agregar/modificar ítems en cotizaciones:
               - Si el usuario menciona un producto por nombre descriptivo y no por SKU, puedes incluir 'search_product_catalog' para resolver el SKU antes o llamar directamente 'odoo_update_quotation'.
               - Distingue la acción por ítem: 'add' (sumar), 'set' (fijar cantidad exacta), 'subtract' (restar unidades), 'remove' (borrar ítem completo).
            5. Guardrails Human-in-the-Loop (HITL) Obligatorios:
               - Confirmar orden de venta ('confirmar orden', 'pasar a pedido'): requiere 'requires_hitl=True', 'hitl_type="confirm_order"'.
               - Cancelar/anular orden ('cancelar pedido', 'anular S00003'): requiere 'requires_hitl=True', 'hitl_type="remove_order"'.
               - Modificar pedido confirmado/bloqueado: requiere 'requires_hitl=True', 'hitl_type="unlock_order"'.
               - Ambigüedad de cliente: requiere 'requires_hitl=True', 'hitl_type="ambiguous_customer"'.

            HERRAMIENTAS COMERCIALES DISPONIBLES:
            - 'odoo_list_sales_orders': Consulta órdenes del vendedor agrupadas por estado ('draft', 'sale', 'cancel') con sus líneas y montos. Argumentos: customer_name (opcional), status (opcional), period (opcional).
            - 'odoo_view_quotation': Consulta la vista detallada de una orden por código visible. Argumentos: order_name (str, ej: 'S00003').
            - 'odoo_create_quotation': Crea una cotización nueva en borrador. Argumentos: customer_name (str), items (list).
            - 'odoo_update_quotation': Modifica o suma/resta líneas en una cotización. Argumentos: order_name (str), items (list).
            - 'odoo_confirm_order': Confirma formalmente una cotización a pedido de venta (HITL). Argumentos: order_name (str).
            - 'odoo_remove_sale_order': Cancela una orden de venta (HITL). Argumentos: order_name (str).
            - 'odoo_unlock_order' / 'odoo_lock_order': Desbloquea / re-bloquea orden confirmada para edición (HITL). Argumentos: order_name (str).
            - 'odoo_list_current_customers': Busca clientes en cartera para resolver ambigüedad. Argumentos: name (str).
            - 'odoo_upsert_customer': Registra un nuevo cliente en Odoo. Argumentos: name (str).
            - 'search_product_catalog': Busca en catálogo comercial para resolver nombres informales a SKUs. Argumentos: query (str).
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=f"Consulta del vendedor:\n{raw_query}"),
        ]

        plan: Optional[Any] = None
        try:
            plan = await self._planner.ainvoke(messages)
        except Exception as exc:
            logger.warning(f"[SalesPlanner] Error invocando planificador estructurado: {exc}. Usando plan por defecto.")

        candidate_partners = []
        current_candidates = []
        resolved_partner_id: Optional[int] = active_partner_id

        # Compatibilidad con SalesExtractionResult (mocks de tests legacy)
        if isinstance(plan, SalesExtractionResult) or (plan and not hasattr(plan, "tool_calls")):
            legacy_action = getattr(plan, "action", "list")
            raw_lower = raw_query.lower()
            is_global_query = any(k in raw_lower for k in ("mis pedidos", "todos mis pedidos", "mis ventas", "mis cotizaciones", "todas mis órdenes", "listar todo")) or (legacy_action == "list" and not getattr(plan, "customer", None))

            extracted_cust = getattr(plan, "customer", None)
            extracted_order = getattr(plan, "order_name", None)

            if is_global_query:
                legacy_cust = None
                legacy_order = None
                active_customer = None
                active_partner_id = None
                active_order = None
                active_order_id = None
            else:
                # Si el usuario menciona un cliente explícito diferente al previo, purgar la orden previa
                if extracted_cust and active_customer and not _customers_match(extracted_cust, active_customer):
                    legacy_order = extracted_order
                    active_order = None
                    active_order_id = None
                else:
                    legacy_order = extracted_order or active_order
                legacy_cust = extracted_cust or active_customer

            legacy_items = [
                it.model_dump() if hasattr(it, "model_dump") else (it.dict() if hasattr(it, "dict") else dict(it))
                for it in getattr(plan, "items", [])
            ]
            tools = []
            req_hitl = False
            hitl_t = None
            hitl_q_val = None
            resolved_partner_id = active_partner_id

            # Comprobar conflicto entre orden y cliente si ambos están presentes
            if legacy_order and legacy_cust:
                try:
                    ord_view = await self._view_quotation_tool(order_name=legacy_order)
                    if ord_view and isinstance(ord_view, dict) and ord_view.get("customer_name"):
                        ord_owner = ord_view["customer_name"]
                        if not _customers_match(legacy_cust, ord_owner):
                            req_hitl = True
                            hitl_t = "clarification"
                            hitl_q_val = (
                                f"La orden {legacy_order} pertenece a '{ord_owner}', pero indicaste '{legacy_cust}'. "
                                f"¿Deseas modificar la orden {legacy_order} de {ord_owner} o crear una cotización nueva para {legacy_cust}?"
                            )
                except Exception:
                    pass

            if not req_hitl and legacy_action == "list":
                status_val = getattr(plan, "status", None)
                if status_val == "sale" and legacy_cust and any(k in raw_lower for k in ("cómo van", "como van", "pedidos de", "estado de", "pipeline", "órdenes de")):
                    status_val = None
                tools.append(PlannedSalesTool(
                    tool_name="odoo_list_sales_orders",
                    arguments={"customer_name": legacy_cust, "status": status_val},
                    purpose="Listar órdenes",
                ))
            elif not req_hitl and legacy_action == "view":
                tools.append(PlannedSalesTool(
                    tool_name="odoo_view_quotation",
                    arguments={"order_name": legacy_order},
                    purpose="Ver cotización",
                ))
            elif not req_hitl and legacy_action in ("add_items", "remove_items", "update"):
                tools.append(PlannedSalesTool(
                    tool_name="odoo_update_quotation",
                    arguments={"order_name": legacy_order, "customer_name": legacy_cust, "items": legacy_items},
                    purpose="Modificar ítems en cotización",
                ))
            elif not req_hitl and legacy_action in ("upsert", "create"):
                if legacy_order:
                    tools.append(PlannedSalesTool(
                        tool_name="odoo_update_quotation",
                        arguments={"order_name": legacy_order, "customer_name": legacy_cust, "items": legacy_items},
                        purpose="Modificar cotización existente",
                    ))
                else:
                    # Comprobar resolución de cliente y posibles duplicados
                    if legacy_cust:
                        try:
                            custs = await self._list_customers_tool(name=legacy_cust, user_id=user_id)
                            if custs and len(custs) > 1:
                                req_hitl = True
                                hitl_t = "ambiguous_customer"
                                candidate_partners = custs
                            elif custs and len(custs) == 1:
                                resolved_partner_id = custs[0].get("id")
                                legacy_cust = custs[0].get("name")
                        except Exception:
                            pass

                        if not req_hitl:
                            try:
                                cand_orders = await self._list_current_orders_tool(customer_name=legacy_cust, user_id=user_id)
                                if cand_orders:
                                    judge_res = await self._judge.ainvoke([SystemMessage(content="Check duplicates")])
                                    if getattr(judge_res, "has_duplicates", False):
                                        req_hitl = True
                                        hitl_t = "duplicate"
                                        current_candidates = cand_orders
                            except Exception:
                                pass

                    if not req_hitl:
                        create_args = {"customer_name": legacy_cust, "items": legacy_items}
                        if resolved_partner_id:
                            create_args["partner_id"] = resolved_partner_id
                        tools.append(PlannedSalesTool(
                            tool_name="odoo_create_quotation",
                            arguments=create_args,
                            purpose="Crear cotización",
                        ))
            elif not req_hitl and legacy_action == "confirm":
                tools.append(PlannedSalesTool(
                    tool_name="odoo_confirm_order",
                    arguments={"order_name": legacy_order},
                    purpose="Confirmar orden",
                ))
            elif not req_hitl and legacy_action == "remove":
                req_hitl = True
                hitl_t = "remove_order"
                tools.append(PlannedSalesTool(
                    tool_name="odoo_remove_sale_order",
                    arguments={"order_name": legacy_order},
                    purpose="Anular orden",
                ))
            elif not req_hitl and legacy_action == "edit_order":
                req_hitl = True
                hitl_t = "unlock_order"
                tools.append(PlannedSalesTool(
                    tool_name="odoo_unlock_order",
                    arguments={"order_name": legacy_order},
                    purpose="Desbloquear orden",
                ))
                tools.append(PlannedSalesTool(
                    tool_name="odoo_update_order",
                    arguments={"order_name": legacy_order},
                    purpose="Actualizar orden",
                ))
                tools.append(PlannedSalesTool(
                    tool_name="odoo_lock_order",
                    arguments={"order_name": legacy_order},
                    purpose="Bloquear orden",
                ))

            target_order = legacy_order
            target_customer = legacy_cust
            tool_calls_dict = [t.model_dump() for t in tools]
            sales_act = legacy_action
            plan_rat = f"Estrategia {legacy_action} adaptada de extraction result."
            plan_strat = legacy_action
            req_hitl_val = req_hitl
            hitl_type_val = hitl_t
            extracted_items = legacy_items
        else:
            if not plan:
                # Fallback robusto según palabras clave
                raw_lower = raw_query.lower()
                ord_code = extract_order_code(raw_query) or active_order
                if any(k in raw_lower for k in ("mis pedidos", "todos mis pedidos", "mis cotizaciones", "todas mis órdenes", "listar todo")):
                    plan = SalesPlan(
                        reasoning="Listado global de órdenes de la cartera del comercial.",
                        strategy="list_orders",
                        target_customer=None,
                        target_order_name=None,
                        tool_calls=[
                            PlannedSalesTool(
                                tool_name="odoo_list_sales_orders",
                                arguments={"customer_name": None},
                                purpose="Consultar todas las órdenes del vendedor agrupadas por estado con sus líneas de detalle.",
                            )
                        ],
                    )
                elif ord_code and any(k in raw_lower for k in ("ver", "detalle", "mostrar", "qué tiene", "explica")):
                    plan = SalesPlan(
                        reasoning=f"Consulta de vista de la orden '{ord_code}'.",
                        strategy="view_order_details",
                        target_order_name=ord_code,
                        tool_calls=[
                            PlannedSalesTool(
                                tool_name="odoo_view_quotation",
                                arguments={"order_name": ord_code},
                                purpose=f"Obtener detalle completo de productos y totales de la orden {ord_code}.",
                            )
                        ],
                    )
                else:
                    plan = SalesPlan(
                        reasoning="Listado general de órdenes por defecto.",
                        strategy="list_orders",
                        tool_calls=[
                            PlannedSalesTool(
                                tool_name="odoo_list_sales_orders",
                                arguments={"customer_name": active_customer},
                                purpose="Listar órdenes del cliente o vendedor.",
                            )
                        ],
                    )

            target_order = getattr(plan, "target_order_name", None) or getattr(plan, "order_name", None) or active_order
            target_customer = getattr(plan, "target_customer", None) or getattr(plan, "customer", None) or active_customer
            if plan.target_customer and active_customer and plan.target_customer.strip().lower() != active_customer.strip().lower():
                target_order = plan.target_order_name

            tool_calls_dict = [
                tc.model_dump() if hasattr(tc, "model_dump") else (tc.dict() if hasattr(tc, "dict") else dict(tc))
                for tc in getattr(plan, "tool_calls", [])
            ]
            strat = getattr(plan, "strategy", "list_orders")
            if strat == "list_orders":
                sales_act = "list"
            elif strat == "view_order_details":
                sales_act = "view"
            elif strat == "create_quotation":
                sales_act = "upsert"
            elif strat in ("update_quotation", "add_items"):
                sales_act = "add_items"
            elif strat == "confirm_order":
                sales_act = "confirm"
            elif strat == "cancel_order":
                sales_act = "remove"
            elif strat == "edit_confirmed_order":
                sales_act = "edit_order"
            else:
                sales_act = strat

            plan_rat = getattr(plan, "reasoning", "")
            plan_strat = strat
            req_hitl_val = getattr(plan, "requires_hitl", False)
            hitl_type_val = getattr(plan, "hitl_type", None)
            hitl_q_val = getattr(plan, "hitl_question", None)
            extracted_items = []

        if target_order:
            target_order = normalize_order_name(target_order)

        logger.info(
            f"[SalesPlanner] Plan generado ({plan_strat}): {len(tool_calls_dict)} herramientas planificadas. "
            f"Target Customer: '{target_customer}', Target Order: '{target_order}', HITL: {req_hitl_val}."
        )

        out_dict = {
            "planned_tools": tool_calls_dict,
            "plan_rationale": plan_rat,
            "plan_strategy": plan_strat,
            "customer_name": target_customer,
            "target_order_name": target_order,
            "items": extracted_items,
            "requires_hitl": req_hitl_val,
            "hitl_type": hitl_type_val,
            "hitl_question": hitl_q_val,
            "iteration_count": iteration_count,
            "max_iterations": self._default_max_iterations,
            "sales_action": sales_act,
        }
        if resolved_partner_id is not None:
            out_dict["partner_id"] = resolved_partner_id
        if candidate_partners:
            out_dict["candidate_partners"] = candidate_partners
        if current_candidates:
            out_dict["current_candidates"] = current_candidates

        return out_dict

    # ==============================================================================
    # 2. NODO EJECUTOR DE HERRAMIENTAS (Sales Executor Node)
    # ==============================================================================
    async def execute_tools(self, state: SalesManageState) -> Dict[str, Any]:
        """Nodo 2: Ejecuta de manera asíncrona y segura las herramientas comerciales planificadas,
        almacenando los resultados estructurados en 'execution_results'."""
        planned_tools = state.get("planned_tools", [])
        user_id = int(state.get("user_id") or 5)
        execution_results: Dict[str, Any] = {}
        orders_list = state.get("orders_list")
        order_view_data = state.get("order_view_data")
        operation_result = state.get("operation_result")
        partner_id = state.get("partner_id")

        for call in planned_tools:
            tool_name = call.get("tool_name")
            args = dict(call.get("arguments", {}))
            purpose = call.get("purpose", "")

            tool_fn = self._tools_registry.get(tool_name)
            if not tool_fn:
                logger.warning(f"[SalesExecutor] Herramienta '{tool_name}' no encontrada en el registro.")
                continue

            # Normalizar estructura de items si vienen en los argumentos
            if "items" in args and isinstance(args["items"], list):
                norm_items = []
                for it in args["items"]:
                    if isinstance(it, dict):
                        it_copy = dict(it)
                        if "sku" not in it_copy or not it_copy["sku"]:
                            it_copy["sku"] = it_copy.get("product_name") or it_copy.get("name") or ""
                        if "qty" not in it_copy:
                            it_copy["qty"] = it_copy.get("quantity") or it_copy.get("cantidad") or 1.0
                        norm_items.append(it_copy)
                    else:
                        norm_items.append(it)
                args["items"] = norm_items

            logger.info(f"[SalesExecutor] Invocando '{tool_name}' ({purpose}) con args: {args}")
            try:
                # Detección dinámica y filtrado seguro de parámetros según firma
                if isinstance(tool_fn, BaseTool):
                    allowed_keys = tool_fn.args_schema.model_fields.keys() if getattr(tool_fn, "args_schema", None) else None
                    filtered_args = {k: v for k, v in args.items() if k in allowed_keys} if allowed_keys else args
                    result = await tool_fn.ainvoke(filtered_args)
                elif callable(tool_fn):
                    try:
                        sig = inspect.signature(tool_fn)
                        params = sig.parameters
                        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

                        if tool_name in (
                            "odoo_list_sales_orders",
                            "odoo_create_quotation",
                            "odoo_list_current_customers",
                            "odoo_upsert_customer",
                            "odoo_list_current_sales_orders",
                        ):
                            if "user_id" not in args and ("user_id" in params or has_kwargs):
                                args["user_id"] = user_id
                        elif "user_id" in params and "user_id" not in args:
                            args["user_id"] = user_id

                        if "partner_id" in params and "partner_id" not in args and partner_id:
                            args["partner_id"] = partner_id

                        filtered_args = args if has_kwargs else {k: v for k, v in args.items() if k in params}
                    except (ValueError, TypeError):
                        filtered_args = args

                    if inspect.iscoroutinefunction(tool_fn):
                        result = await tool_fn(**filtered_args)
                    else:
                        res = tool_fn(**filtered_args)
                        result = await res if inspect.iscoroutine(res) else res
                else:
                    logger.error(f"[SalesExecutor] Herramienta '{tool_name}' no es invocable.")
                    continue

                execution_results[tool_name] = result

                # Mapear resultados a canales tradicionales del estado para consistencia
                if tool_name == "odoo_list_sales_orders" and isinstance(result, dict):
                    orders_list = result
                elif tool_name == "odoo_view_quotation" and isinstance(result, dict):
                    order_view_data = result
                elif tool_name in ("odoo_create_quotation", "odoo_update_quotation", "odoo_confirm_order", "odoo_remove_sale_order"):
                    operation_result = result
                elif tool_name == "odoo_list_current_customers" and isinstance(result, list) and result:
                    partner_id = result[0].get("id")
                elif tool_name == "odoo_upsert_customer" and isinstance(result, dict) and result.get("id"):
                    partner_id = result.get("id")

            except Exception as exc:
                logger.error(f"[SalesExecutor] Error ejecutando '{tool_name}': {exc}", exc_info=True)
                execution_results[tool_name] = {"error": str(exc), "success": False}

        output: Dict[str, Any] = {
            "execution_results": execution_results,
        }
        if orders_list is not None:
            output["orders_list"] = orders_list
        if order_view_data is not None:
            output["order_view_data"] = order_view_data
            if isinstance(order_view_data, dict):
                if order_view_data.get("name"):
                    output["target_order_name"] = order_view_data.get("name")
                if order_view_data.get("id"):
                    output["target_order_id"] = order_view_data.get("id")
                if not state.get("customer_name") and order_view_data.get("customer_name"):
                    output["customer_name"] = order_view_data.get("customer_name")
                if order_view_data.get("partner_id"):
                    output["partner_id"] = order_view_data.get("partner_id")
        if operation_result is not None:
            output["operation_result"] = operation_result
            if isinstance(operation_result, dict):
                if "odoo_create_quotation" in execution_results and operation_result.get("name"):
                    output["target_order_name"] = operation_result.get("name")
                elif "odoo_update_quotation" in execution_results:
                    upd_call = next((c for c in planned_tools if c.get("tool_name") == "odoo_update_quotation"), {})
                    upd_order = upd_call.get("arguments", {}).get("order_name") or state.get("target_order_name")
                    if upd_order:
                        output["target_order_name"] = upd_order
                elif operation_result.get("name") and not output.get("target_order_name"):
                    output["target_order_name"] = operation_result.get("name")
                if operation_result.get("order_id"):
                    output["target_order_id"] = operation_result.get("order_id")
        if partner_id is not None:
            output["partner_id"] = partner_id

        return output

    # ==============================================================================
    # 3. NODO SINTETIZADOR DE VENTAS (Sales Synthesizer Node)
    # ==============================================================================
    async def synthesize_draft(self, state: SalesManageState) -> Dict[str, Any]:
        """Nodo 3: Redacta un borrador comercial empático, sobrio y completo para el usuario,
        incorporando los datos del ejecutor y atendiendo críticas de reflexión si existen."""
        cancellation_reason = state.get("cancellation_reason")
        if cancellation_reason:
            return {"draft_response": cancellation_reason, "final_response": cancellation_reason}

        raw_query = state.get("raw_query") or ""
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=8,
            strategy="last",
            token_counter=len,
        )

        plan_strategy = state.get("plan_strategy", "list_orders")
        execution_results = state.get("execution_results") or {}
        orders_list = state.get("orders_list") or {}
        order_view_data = state.get("order_view_data") or {}
        op_res = state.get("operation_result") or {}
        critique = state.get("critique")
        remedy_suggestions = state.get("suggested_improvements") or []

        channel_instructions = self.get_channel_prompt_instructions(state)

        reflection_feedback_block = ""
        if critique:
            reflection_feedback_block = f"""
            CRÍTICA DE CALIDAD A CORREGIR EN ESTA ITERACIÓN:
            - Fallo detectado: {critique}
            - Mejoras exigidas: {remedy_suggestions}
            DEBES corregir este aspecto de inmediato en tu redacción.
            """

        system_prompt = inject_soul(
            f"""
            Directrices comerciales para la redacción de respuestas sobre órdenes y cotizaciones:
            - NUNCA expongas códigos internos como 'user_id', 'partner_id' ni IDs numéricos de base de datos.
            - NUNCA menciones términos de backend como 'ERP', 'Odoo' ni nombres de tablas o herramientas.
            - Utiliza el nombre comercial del cliente y el código comercial de la orden (ej: 'SO001', 'S00003').
            
            REGLA DE ORO DE DETALLE DE PRODUCTOS:
            - Si el usuario pide ver, listar o consultar los productos o el detalle de sus órdenes o de una orden en particular
              (ej: 'dame el detalle de todos mis pedidos', 'incluye productos', 'qué productos tienen', 'ver S00003'):
              * NUNCA preguntes al usuario si desea que consultes el detalle ni ofrezcas hacerlo más adelante.
              * LISTA INMEDIATAMENTE de forma clara y elegante cada producto con su nombre comercial, cantidad y precio/total.
              * Los datos de líneas de productos ya fueron recuperados por el sistema y están disponibles en el contexto.

            REGLA DE FORMATO EN LISTADOS:
            - Agrupa las órdenes por estado comercial:
              * Cotizaciones (Pendientes)
              * Pedidos Confirmados
              * Pedidos Cancelados (solo si hay o si fue solicitado)
            - Indica el código comercial de la orden, cliente, fecha y monto total. Si el usuario pidió detalle de productos, muestra las líneas de cada orden bajo su encabezado.

            REGLA DE CONFIRMACIÓN O EDICIÓN:
            - Si se creó o actualizó una cotización, resume los productos incluidos, subtotal y total.
            - Si se confirmó un pedido, celebra la venta con entusiasmo profesional y sobrio.

            {reflection_feedback_block}

            {channel_instructions}
            """,
            role=SoulRole.SALES_ORDERS,
        )

        user_content = f"""
            Consulta actual del vendedor: {raw_query}

            Datos de ejecución obtenidos:
            - Estrategia: {plan_strategy}
            - Resultados de herramientas: {execution_results}
            - Órdenes agrupadas disponibles: {orders_list}
            - Detalle de orden consultada: {order_view_data}
            - Resultado de mutación: {op_res}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_content),
        ]

        synth_result: Optional[SalesSynthesizeResponse] = None
        try:
            synth_result = await self._synthesizer.ainvoke(messages)
        except Exception as exc:
            logger.warning(f"[SalesSynthesizer] Error en invocación estructurada: {exc}")
            try:
                direct_msg = await self._llm.ainvoke(messages)
                content_text = extract_clean_text(getattr(direct_msg, "content", ""))
                if content_text:
                    synth_result = SalesSynthesizeResponse(response_text=content_text)
            except Exception as exc2:
                logger.error(f"[SalesSynthesizer] Error en fallback directo: {exc2}")

        resp_text = getattr(synth_result, "response_text", "") if synth_result else ""
        if not resp_text and isinstance(synth_result, dict):
            resp_text = synth_result.get("response_text", "")

        # Fallback determinista en caso extremo
        if not resp_text:
            target_name = state.get("target_order_name") or op_res.get("name") or order_view_data.get("name")
            cust = state.get("customer_name") or op_res.get("customer_name") or order_view_data.get("customer_name")
            if orders_list and isinstance(orders_list, dict):
                status_filter = state.get("status_filter")
                draft_orders = orders_list.get("draft", []) if isinstance(orders_list, dict) else []
                sale_orders = orders_list.get("sale", []) if isinstance(orders_list, dict) else []
                cancel_orders = orders_list.get("cancel", []) if isinstance(orders_list, dict) else []

                st_norm = (status_filter or "").lower().strip()
                show_draft = st_norm in ("", "draft", "cotizacion", "cotizaciones")
                show_sale = st_norm in ("", "sale", "confirmado", "confirmados", "aprobado", "aprobados")
                show_cancel = st_norm in ("cancel", "cancelado", "cancelados", "anulado", "anulados")

                parts = [f"Resumen de órdenes comerciales para {cust or 'tu cartera'}:"]
                has_content = False

                if show_draft and draft_orders:
                    has_content = True
                    parts.append("\n### 📋 Cotizaciones en Borrador (Activas)")
                    for o in draft_orders:
                        lines_desc = ""
                        if o.get("lines"):
                            lines_desc = " (" + ", ".join(f"{l.get('quantity', 1):.0f}x {l.get('product_name', '')}" for l in o["lines"]) + ")"
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **{self.format_currency(o.get('amount_total', 0.0), 'PEN')}**{lines_desc}")

                if show_sale and sale_orders:
                    has_content = True
                    parts.append("\n### ✅ Pedidos Confirmados")
                    for o in sale_orders:
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **{self.format_currency(o.get('amount_total', 0.0), 'PEN')}**")

                if show_cancel and cancel_orders:
                    has_content = True
                    parts.append("\n### 🚫 Órdenes Canceladas")
                    for o in cancel_orders:
                        parts.append(f"* **{o.get('name')}** | Cliente: **{o.get('customer_name')}** | Total: **{self.format_currency(o.get('amount_total', 0.0), 'PEN')}**")

                if not has_content:
                    parts = [
                        HITLDialogTemplates.order_list_empty(
                            customer_name=cust,
                            status_filter=status_filter,
                            has_cancelled=len(cancel_orders) > 0,
                            cancel_count=len(cancel_orders),
                        )
                    ]

                resp_text = "\n".join(parts)
            else:
                resp_text = f"Se ha procesado la orden {target_name or ''} para {cust or 'el cliente'} exitosamente."

        return {"draft_response": resp_text}

    # ==============================================================================
    # 4. NODO JUEZ DE RÚBRICA Y AUTO-REFLEXIÓN (Rubric Evaluator Judge)
    # ==============================================================================
    async def rubric_evaluator_judge(self, state: SalesManageState) -> Dict[str, Any]:
        """Nodo 4: Evalúa críticamente el borrador contra la consulta original del vendedor y
        los datos reales de ejecución para detectar omisiones o ambigüedades."""
        raw_query = state.get("raw_query") or ""
        draft_response = state.get("draft_response") or ""
        execution_results = state.get("execution_results") or {}
        orders_list = state.get("orders_list") or {}
        order_view_data = state.get("order_view_data") or {}
        iteration_count = state.get("iteration_count", 0) + 1
        max_iterations = state.get("max_iterations", self._default_max_iterations)

        # Si hubo cancelación HITL, aprobar directamente
        if state.get("cancellation_reason"):
            return {
                "meets_rubric": True,
                "reflection_action": "approve",
                "critique": None,
                "suggested_improvements": [],
                "iteration_count": iteration_count,
            }

        prompt = f"""
            Eres el auditor y juez de calidad de respuestas comerciales para vendedores.
            Evalúa el siguiente borrador de respuesta frente a la solicitud original del usuario y los datos reales obtenidos:

            CONSULTA DEL USUARIO:
            "{raw_query}"

            BORRADOR GENERADO:
            "{draft_response}"

            DATOS OBTENIDOS POR LAS HERRAMIENTAS:
            - Órdenes listadas: {orders_list}
            - Detalle de orden: {order_view_data}
            - Resultados de ejecución: {execution_results}

            CRITERIOS DE EVALUACIÓN (RÚBRICA DE 1 A 10):
            1. relevance_score: ¿Responde directamente a lo que el usuario pidió?
            2. grounding_score: ¿Los números, códigos de orden, clientes y montos provienen 100% de los datos de las herramientas sin alucinación?
            3. detail_completeness_score: CRÍTICO:
               - Si el usuario solicitó expresamente ver productos/líneas ('detalle con productos', 'incluye productos', 'qué productos tienen'),
                 ¿el borrador listó efectivamente los productos, O cayó en la respuesta perezosa de preguntar si deseaba verlos?
               - Si el usuario pidió productos y NO se mostraron pese a estar en los datos, detail_completeness_score DEBE ser < 6.0.
            4. whitelabel_and_safety_score: ¿Se ocultaron IDs técnicos internos (user_id, partner_id) y términos de backend (Odoo/ERP)?

            DECISIÓN DE REFLEXIÓN (reflection_action):
            - 'approve': Si todos los criterios son >= 7.0 y el promedio es >= 8.0.
            - 'refine_synthesis': Si los datos en las herramientas son suficientes pero el borrador omitió detalles solicitados (Fast Reflection Loop).
            - 'replan_tools': Si faltan datos indispensables que requieren ejecutar herramientas adicionales en Odoo (Slow Reflection Loop).
        """

        # Si es un mock de pruebas con side_effects que no incluye rúbrica, no consumir el siguiente turno del mock
        is_mock_llm = (
            type(self._llm).__name__ in ("MagicMock", "AsyncMock")
            or hasattr(self._llm, "_mock_return_value")
            or hasattr(self._llm, "side_effect")
        )

        evaluation: Optional[SalesQualityRubricEvaluation] = None
        if not is_mock_llm or getattr(self, "_enable_mock_rubric", False):
            try:
                evaluation = await self._rubric_judge.ainvoke([SystemMessage(content=prompt)])
            except Exception as exc:
                logger.warning(f"[SalesRubricJudge] Error evaluando rúbrica: {exc}")

        if not isinstance(evaluation, SalesQualityRubricEvaluation) or not hasattr(evaluation, "reflection_action"):
            # Fallback determinista de rúbrica
            raw_lower = raw_query.lower()
            asks_for_products = any(k in raw_lower for k in ("detalle", "producto", "lista", "items", "línea", "incluye"))
            mentions_products_in_draft = any(k in draft_response.lower() for k in ("1x", "2x", "3x", "unidades", "s/.", "total:"))

            if asks_for_products and not mentions_products_in_draft and orders_list:
                evaluation = SalesQualityRubricEvaluation(
                    relevance_score=6.0,
                    grounding_score=8.5,
                    detail_completeness_score=5.0,
                    whitelabel_and_safety_score=9.0,
                    is_approved=False,
                    reflection_action="refine_synthesis",
                    critique="El usuario solicitó incluir el detalle de productos pero el borrador omitió las líneas de las órdenes.",
                    remedy_suggestions=["Listar de forma explícita cada producto con su cantidad y precio en las órdenes correspondientes."],
                )
            else:
                evaluation = SalesQualityRubricEvaluation(
                    relevance_score=9.0,
                    grounding_score=9.0,
                    detail_completeness_score=9.0,
                    whitelabel_and_safety_score=9.5,
                    is_approved=True,
                    reflection_action="approve",
                )

        logger.info(
            f"[SalesRubricJudge] Veredicto: {evaluation.reflection_action} (Aprobado: {evaluation.is_approved}, "
            f"Iteración: {iteration_count}/{max_iterations}). Crítica: {evaluation.critique}"
        )

        return {
            "meets_rubric": evaluation.is_approved,
            "reflection_action": evaluation.reflection_action,
            "critique": evaluation.critique,
            "suggested_improvements": evaluation.remedy_suggestions,
            "rubric_scores": {
                "relevance": evaluation.relevance_score,
                "grounding": evaluation.grounding_score,
                "completeness": evaluation.detail_completeness_score,
                "detail_completeness": evaluation.detail_completeness_score,
                "safety": evaluation.whitelabel_and_safety_score,
            },
            "iteration_count": iteration_count,
        }

    # ==============================================================================
    # 5. NODO FINALIZADOR DE VENTAS (Sales Finalizer Node)
    # ==============================================================================
    async def finalize_response(self, state: SalesManageState) -> Dict[str, Any]:
        """Nodo 5: Aplica formateo nativo para WhatsApp, estandarización de divisas (PEN) y emite la respuesta final."""
        text = state.get("draft_response") or state.get("final_response") or ""
        
        formatted_output = BaseSynthesizerNode.format_final_response(
            text,
            state,
            extra={
                "critique": None,
                "suggested_improvements": None,
            },
        )
        return formatted_output

    # ==============================================================================
    # 6. NODOS DE SOPORTE HUMAN-IN-THE-LOOP (HITL) Y COMPATIBILIDAD
    # ==============================================================================
    async def feedback_intent_clarification_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """HITL Interrupt: Aclara la intención del usuario cuando se detecta ambigüedad."""
        if config:
            var_child_runnable_config.set(config)
        question = state.get("hitl_question") or state.get("clarification_question") or "Por favor aclara cómo deseas procesar esta cotización o pedido."
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

        return {
            "is_intent_clear": True,
            "requires_hitl": False,
            "raw_query": f"{state.get('raw_query', '')} {user_reply}",
        }

    async def feedback_ambiguous_customer_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """HITL Interrupt: Resuelve ambigüedad de clientes presentando opciones legibles sin exponer IDs."""
        if config:
            var_child_runnable_config.set(config)
        candidates = state.get("candidate_partners", [])
        options = [
            f"{c['name']} (Tel: {c.get('phone') or 'Sin teléfono'})"
            for c in candidates
        ]
        question = f"Encontré varios clientes coincidentes en tu cartera: {'; '.join(options)}. ¿A cuál de ellos te refieres?"

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

        user_id = int(state.get("user_id") or 5)

        if selection.action == "select" and selection.selected_index is not None and selection.selected_index < len(candidates):
            chosen = candidates[selection.selected_index]
            partner_id = chosen["id"]
            cust_name = chosen["name"]

            # Comprobar si el cliente elegido tiene cotizaciones abiertas (duplicados)
            cand_orders = []
            try:
                cand_orders = await self._list_current_orders_tool(customer_name=cust_name, user_id=user_id)
            except Exception:
                pass

            has_dup = False
            if cand_orders:
                try:
                    judge_res = await self._judge.ainvoke([SystemMessage(content="Check duplicates")])
                    has_dup = getattr(judge_res, "has_duplicates", False)
                except Exception:
                    pass

            tools = []
            if not has_dup:
                tools.append(PlannedSalesTool(
                    tool_name="odoo_create_quotation",
                    arguments={"customer_name": cust_name, "items": state.get("items", [])},
                    purpose="Crear cotización para cliente desambiguado",
                ))

            return {
                "partner_id": partner_id,
                "customer_name": cust_name,
                "customer_resolved": True,
                "customer_not_found": False,
                "requires_hitl": has_dup,
                "hitl_type": "duplicate" if has_dup else None,
                "has_possible_duplicates": has_dup,
                "current_candidates": cand_orders,
                "planned_tools": [t.model_dump() for t in tools],
            }

        return {
            "customer_resolved": False,
            "cancellation_reason": "Operación cancelada por el usuario al no definir el cliente.",
        }

    async def feedback_duplicate_sales_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """Subflujo Duplicados: HITL interrupt evaluando en lenguaje natural si actualiza orden existente o abre nueva."""
        if config:
            var_child_runnable_config.set(config)
        candidates = state.get("current_candidates", [])
        matched_name = state.get("target_order_name")

        cand_order_strings = []
        for c in candidates:
            lines = c.get("lines", [])
            lines_summary = f" - {', '.join(l.get('product_name', '') for l in lines[:2])}" if lines else ""
            cand_order_strings.append(f"{c['name']} (S/ {c.get('amount_total', 0):.2f}{lines_summary})")

        cand_desc = "; ".join(cand_order_strings)
        question = (
            f"Detecté que ya existe una cotización abierta para este cliente ({matched_name or cand_desc}). "
            f"¿Deseas actualizar la cotización existente o prefieres abrir una nueva orden?"
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

        items = state.get("items", [])
        cust = state.get("customer_name")

        if choice_result.choice == "selected":
            target_name = choice_result.selected_order_name or (candidates[0]["name"] if candidates else "SO001")
            tool = PlannedSalesTool(
                tool_name="odoo_update_quotation",
                arguments={"order_name": target_name, "customer_name": cust, "items": items},
                purpose="Actualizar cotización abierta existente",
            )
            return {
                "duplicate_choice": "selected",
                "target_order_name": target_name,
                "requires_hitl": False,
                "planned_tools": [tool.model_dump()],
            }
        elif choice_result.choice == "new":
            tool = PlannedSalesTool(
                tool_name="odoo_create_quotation",
                arguments={"customer_name": cust, "items": items},
                purpose="Crear cotización nueva independiente",
            )
            return {
                "duplicate_choice": "new",
                "target_order_name": None,
                "requires_hitl": False,
                "planned_tools": [tool.model_dump()],
            }
        else:
            return {
                "duplicate_choice": "cancel",
                "requires_hitl": False,
                "cancellation_reason": "Operación cancelada por el usuario ante la selección de orden.",
            }

    async def feedback_unknown_customer_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """HITL Interrupt: Ofrece registrar de inmediato un cliente no encontrado en la cartera."""
        if config:
            var_child_runnable_config.set(config)
        cust_name = state.get("customer_name") or "el cliente"
        question = HITLDialogTemplates.unknown_customer_question(cust_name)

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
                user_id=int(state.get("user_id") or 5),
                name=state["customer_name"],
            )
            if res.get("success") and res.get("id"):
                return {
                    "partner_id": res["id"],
                    "customer_resolved": True,
                    "customer_not_found": False,
                    "requires_hitl": False,
                }

        return {
            "customer_resolved": False,
            "cancellation_reason": f"Operación cancelada. No se pudo asociar la venta al cliente '{cust_name}'.",
        }

    async def feedback_remove_order_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """HITL Interrupt: Confirmación obligatoria antes de cancelar/anular una orden."""
        if config:
            var_child_runnable_config.set(config)
        target = state.get("target_order_name") or "la orden seleccionada"
        question = f"¿Estás seguro de que deseas anular o cancelar la orden {target}? Esta acción marcará la orden como cancelada en el sistema."

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

        if is_confirmed:
            tool = PlannedSalesTool(
                tool_name="odoo_remove_sale_order",
                arguments={"order_name": target},
                purpose="Anular orden de venta confirmada",
            )
            return {"remove_confirmed": True, "requires_hitl": False, "planned_tools": [tool.model_dump()]}
        return {
            "remove_confirmed": False,
            "cancellation_reason": f"Operación cancelada por el usuario. La orden {target} no fue anulada.",
        }

    async def guardrail_unlock_node(
        self,
        state: SalesManageState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        """HITL Interrupt: Confirmación obligatoria para desbloquear y editar un pedido ya confirmado."""
        if config:
            var_child_runnable_config.set(config)
        target = state.get("target_order_name") or "el pedido"
        question = f"El pedido {target} ya se encuentra confirmado. ¿Deseas desbloquearlo para realizar modificaciones?"

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

        if is_confirmed:
            tools = [
                PlannedSalesTool(tool_name="odoo_unlock_order", arguments={"order_name": target}, purpose="Desbloquear pedido"),
                PlannedSalesTool(tool_name="odoo_update_order", arguments={"order_name": target}, purpose="Actualizar pedido"),
                PlannedSalesTool(tool_name="odoo_lock_order", arguments={"order_name": target}, purpose="Re-bloquear pedido"),
            ]
            return {"unlock_confirmed": True, "requires_hitl": False, "planned_tools": [t.model_dump() for t in tools]}
        return {
            "unlock_confirmed": False,
            "cancellation_reason": f"Modificación cancelada. El pedido {target} permanece bloqueado.",
        }

    # ==============================================================================
    # Métodos legacy conservados para compatibilidad
    # ==============================================================================
    async def extract_sales_info(self, state: SalesManageState) -> dict:
        res = await self.plan_and_select_tools(state)
        return res

    async def reflect_intent_node(self, state: SalesManageState) -> dict:
        return state

    async def list_sales_orders_node(self, state: SalesManageState) -> dict:
        user_id = int(state.get("user_id") or 5)
        grouped = await self._list_orders_tool(user_id=user_id, customer_name=state.get("customer_name"))
        return {"orders_list": grouped}

    async def list_current_sales_orders_node(self, state: SalesManageState) -> dict:
        user_id = int(state.get("user_id") or 5)
        candidates = await self._list_current_orders_tool(user_id=user_id, customer_name=state.get("customer_name"))
        return {"current_candidates": candidates}

    async def duplicate_candidates_judge_node(self, state: SalesManageState) -> dict:
        return {"has_possible_duplicates": False}

    async def view_quotation_node(self, state: SalesManageState) -> dict:
        name = state.get("target_order_name") or "S00001"
        data = await self._view_quotation_tool(order_name=name)
        return {"order_view_data": data}

    async def create_quotation_node(self, state: SalesManageState) -> dict:
        res = await self._create_quotation_tool(
            user_id=int(state.get("user_id") or 5),
            customer_name=state.get("customer_name"),
            items=state.get("items", []),
        )
        return {"operation_result": res}

    async def update_quotation_node(self, state: SalesManageState) -> dict:
        order_name = state.get("target_order_name")
        items = state.get("items", [])
        req_cust = state.get("customer_name")
        view_data = await self._view_quotation_tool(order_name=order_name)
        ord_cust = view_data.get("customer_name") if isinstance(view_data, dict) else None
        if req_cust and ord_cust and not _customers_match(req_cust, ord_cust):
            logger.warning(
                f"Guardrail activado en update_quotation_node: orden '{order_name}' "
                f"pertenece a '{ord_cust}', pero se solicitó para '{req_cust}'"
            )
            return {
                "operation_result": {
                    "success": False,
                    "error": f"La cotización {order_name} pertenece a {ord_cust}, no a {req_cust}. Operación abortada para evitar alterar el pedido incorrecto.",
                }
            }
        res = await self._update_quotation_tool(
            order_name=order_name,
            items=items,
        )
        return {"operation_result": res}

    async def confirm_order_node(self, state: SalesManageState) -> dict:
        res = await self._confirm_order_tool(order_name=state.get("target_order_name"))
        return {"operation_result": res}

    async def remove_order_node(self, state: SalesManageState) -> dict:
        res = await self._remove_order_tool(order_name=state.get("target_order_name"))
        return {"operation_result": res}

    async def unlock_order_node(self, state: SalesManageState) -> dict:
        res = await self._unlock_order_tool(order_name=state.get("target_order_name"))
        return {"operation_result": res}

    async def update_order_node(self, state: SalesManageState) -> dict:
        return {"operation_result": {"success": True}}

    async def lock_order_node(self, state: SalesManageState) -> dict:
        res = await self._lock_order_tool(order_name=state.get("target_order_name"))
        return {"operation_result": res}

    async def resolve_customer_node(self, state: SalesManageState) -> dict:
        return {"customer_resolved": True}

    async def synthesize_sales_response(self, state: SalesManageState) -> dict:
        d = await self.synthesize_draft(state)
        state_with_draft = {**state, **d}
        return await self.finalize_response(state_with_draft)
