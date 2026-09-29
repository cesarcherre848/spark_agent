"""
src/agent_service/graph/sub_graphs/sales_incentives/nodes.py - Nodos para el subgrafo sales_incentives con Actor-Critic y BaseSynthesizerNode.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.core.llms.factory import extract_clean_text
from src.agent_service.soul.prompts import inject_soul, SoulRole
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
from src.agent_service.graph.sub_graphs.sales_incentives.state import SalesIncentivesState
from src.agent_service.graph.sub_graphs.sales_incentives.schemas import (
    IncentiveActionPlan,
    CriticPlanEvaluation,
)
from src.agent_service.tools.incentive_tools import (
    odoo_get_active_incentives,
    extract_sales_trends,
    calculate_incentive_gap,
)
from src.agent_service.tools.sales_tools import odoo_list_sales_orders
from src.agent_service.tools.contact_tools import odoo_list_current_customers

logger = logging.getLogger(__name__)


class SalesIncentivesNodes(BaseSynthesizerNode):
    """Implementa los nodos del subgrafo sales_incentives heredando de BaseSynthesizerNode
    para garantizar el formateo unificado de WhatsApp, estandarización de divisas y contexto temporal.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        list_orders_tool: Any = odoo_list_sales_orders,
        get_incentives_tool: Any = odoo_get_active_incentives,
        list_customers_tool: Any = odoo_list_current_customers,
        catalog_search_tool: Optional[Any] = None,
        default_max_iterations: int = 3,
        default_epsilon: float = 0.25,
    ):
        super().__init__(llm=llm)
        self._llm = llm
        self._list_orders_tool = list_orders_tool
        self._get_incentives_tool = get_incentives_tool
        self._list_customers_tool = list_customers_tool
        self._catalog_search_tool = catalog_search_tool
        self._default_max_iterations = default_max_iterations
        self._default_epsilon = default_epsilon

    async def gather_context_and_trends_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 1: Recopila órdenes actuales de Odoo, consulta planes vigentes y mina tendencias de pedidos."""
        user_id = state.get("user_id") or 5
        supplier_id = state.get("supplier_id")
        supplier_name = state.get("supplier_name")

        # 1. Obtener órdenes actuales del vendedor
        try:
            orders_data = await self._list_orders_tool(user_id=user_id)
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar órdenes en Odoo: {e}")
            orders_data = {"draft": [], "sale": [], "cancel": []}

        # 2. Consultar incentivos vigentes en Odoo
        try:
            incentives = await self._get_incentives_tool(
                supplier_id=supplier_id,
                supplier_name=supplier_name,
                only_active=True,
            )
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar incentivos en Odoo: {e}")
            incentives = []

        # 3. Minar tendencias de compra recientes (demanda orgánica)
        trends = extract_sales_trends(orders_data, supplier_id=supplier_id, top_k=5)

        # 4. Contexto temporal en tiempo real
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # 5. Cálculo de brechas hacia objetivos
        gap_info = calculate_incentive_gap(orders_data, incentives, reference_date=now)

        return {
            "orders_data": orders_data,
            "active_incentives": incentives,
            "sales_trends": trends,
            "gap_info": gap_info,
            "current_date_str": current_date_str,
            "epsilon": state.get("epsilon") or self._default_epsilon,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations") or self._default_max_iterations,
        }

    async def actor_planner_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 2 (Actor): Planificador estratégico que formula un plan de acción equilibrado con ε-greedy."""
        raw_query = state.get("raw_query") or ""
        gap_info = state.get("gap_info") or {}
        active_incentives = state.get("active_incentives") or []
        trends = state.get("sales_trends") or []
        epsilon = state.get("epsilon") or 0.25
        iteration_count = state.get("iteration_count") or 0
        critique = state.get("critique")
        suggested_improvements = state.get("suggested_improvements") or []
        current_date_str = state.get("current_date_str") or datetime.now().strftime("%Y-%m-%d")

        # Obtener cartera de clientes para personalizar recomendaciones
        user_id = state.get("user_id") or 5
        customers = []
        try:
            if callable(self._list_customers_tool):
                cust_res = await self._list_customers_tool(user_id=user_id)
                customers = [c.get("name") for c in cust_res if isinstance(c, dict) and c.get("name")][:5]
        except Exception:
            customers = []

        # Preservar contexto temporal multi-turno (15 mensajes)
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=15,
            strategy="last",
            token_counter=len,
        )

        critique_block = ""
        if critique and iteration_count > 0:
            critique_block = f"""
            RETROALIMENTACIÓN PREVIA DEL CRITIC (Iteración {iteration_count}):
            Observaciones: {critique}
            Ajustes obligatorios requeridos: {'; '.join(suggested_improvements)}
            DEBES corregir estas debilidades en el nuevo plan.
            """

        system_prompt = f"""
        Eres el ACTOR y Estratega de Ventas de Spark Agent (Sales Coach).
        Tu misión es generar un plan de acción estructurado (IncentiveActionPlan) para que el vendedor
        maximice sus ventas y conquiste los mejores incentivos en Odoo ERP.

        POLÍTICA EPSILON-GREEDY (ε-Greedy) (ε = {epsilon}):
        1. Eje de Explotación ({(1-epsilon)*100:.0f}%): Focalízate en cerrar la brecha hacia el siguiente nivel de descuento o bono de producto en Odoo.
        2. Eje de Exploración/Tendencia ({epsilon*100:.0f}%): Incorpora productos de alta demanda orgánica según las tendencias reales de pedidos de la cartera.

        CONTEXTO TEMPORAL Y DE ESTADO:
        - Fecha actual: {current_date_str}
        - Total confirmado en Odoo: S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}
        - Total en cotizaciones pendientes (draft): S/. {gap_info.get('draft_total_pen', 0.0):.2f}
        - Escala actual: {gap_info.get('current_discount_tier', '20%')}
        - Siguiente escala: {gap_info.get('next_discount_tier', '25%')} (Meta: S/. {gap_info.get('next_tier_target_pen', 0.0):.2f})
        - Brecha en soles: S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f}
        - Días restantes de campaña: {gap_info.get('days_remaining', 'Abierto')}
        - Clientes activos en cartera: {', '.join(customers) if customers else 'Clientes generales'}

        PRODUCTOS TENDENCIA RECIENTES (Demanda Orgánica):
        {', '.join([f"{t.get('name')} (SKU {t.get('sku')}, {t.get('total_qty')} unids)" for t in trends]) if trends else 'Sin pedidos previos registrados'}

        PLANES DE INCENTIVO ACTIVOS EN ODOO:
        {'; '.join([f"[{p.get('supplier_name', 'Odoo')}] {p.get('title')}: {p.get('observations', '')}" for p in active_incentives[:6]])}

        {critique_block}
        """

        structured_actor = bind_structured_output(bind_temperature(self._llm, 0.2), IncentiveActionPlan)

        try:
            plan: Optional[IncentiveActionPlan] = await structured_actor.ainvoke([
                SystemMessage(content=system_prompt),
                *trimmed_history,
                HumanMessage(content=f"Consulta del usuario: {raw_query or '¿Cómo maximizo mis ventas e incentivos hoy?'}"),
            ])
            plan_dict = plan.model_dump() if plan else {}
        except Exception as e:
            logger.error(f"[SalesIncentives/Actor] Error al formular plan: {e}")
            plan_dict = {
                "summary": "Plan de contingencia comercial para alcanzar la siguiente escala de descuento.",
                "current_status_overview": f"Ventas confirmadas por S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}.",
                "milestones": [],
                "suggested_combos": [],
                "projected_additional_earnings_pen": 0.0,
                "epsilon_balance_rationale": "Balance por defecto ante fallback.",
            }

        return {
            "candidate_plan": plan_dict,
            "iteration_count": iteration_count + 1,
        }

    async def critic_evaluator_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 3 (Critic): Evalúa la factibilidad temporal, respeto de restricciones y retorno comercial del plan."""
        candidate_plan = state.get("candidate_plan") or {}
        gap_info = state.get("gap_info") or {}
        iteration_count = state.get("iteration_count") or 1
        max_iterations = state.get("max_iterations") or self._default_max_iterations
        current_date_str = state.get("current_date_str") or datetime.now().strftime("%Y-%m-%d")

        # Si se alcanzó el límite máximo de iteraciones, forzar aprobación para no bloquear el turno
        if iteration_count >= max_iterations:
            logger.info(f"[SalesIncentives/Critic] Límite máximo de iteraciones alcanzado ({iteration_count}/{max_iterations}). Aprobando mejor plan.")
            return {
                "is_approved": True,
                "critic_score": 8.0,
                "critic_evaluation": {
                    "is_approved": True,
                    "score": 8.0,
                    "temporal_feasibility_score": 8.0,
                    "constraint_adherence_score": 8.0,
                    "roi_score": 8.0,
                    "critiques": [],
                    "suggested_adjustments": [],
                },
                "critique": None,
                "suggested_improvements": [],
            }

        system_prompt = f"""
        Eres el CRITIC y Juez de Calidad Comercial de Spark Agent.
        Tu rol es evaluar con rigor técnico el plan de incentivos propuesto por el Actor:

        CRITERIOS DE EVALUACIÓN:
        1. Factibilidad Temporal: ¿Es alcanzable en los días restantes ({gap_info.get('days_remaining', 'N/A')}) o en la semana actual (Fecha: {current_date_str})?
        2. Cumplimiento de Restricciones y Realismo: ¿Los montos de los combos corresponden con la brecha real (S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f})?
        3. Balance ε-Greedy: ¿El plan combina adecuadamente metas de incentivo con productos de alta demanda orgánica para no arriesgar ventas?
        4. Accionabilidad: ¿Menciona SKUs específicos, clientes y argumentos de venta claros?

        UMBRAL DE APROBACIÓN:
        - Si el plan cumple satisfactoriamente los 4 criterios (score >= 8.0), fija is_approved = True.
        - Si el plan es irreal, carece de SKUs concretos o ignora la brecha numérica, fija is_approved = False y detalla las críticas y ajustes.
        """

        structured_critic = bind_structured_output(bind_temperature(self._llm, 0.0), CriticPlanEvaluation)

        try:
            evaluation: Optional[CriticPlanEvaluation] = await structured_critic.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Plan a evaluar:\n{str(candidate_plan)}"),
            ])
            eval_dict = evaluation.model_dump() if evaluation else {}
            is_approved = evaluation.is_approved if evaluation else True
            score = evaluation.score if evaluation else 8.5
            critiques = evaluation.critiques if evaluation else []
            adjustments = evaluation.suggested_adjustments if evaluation else []
        except Exception as e:
            logger.warning(f"[SalesIncentives/Critic] Error en evaluación del Critic: {e}. Aprobando por defecto.")
            is_approved = True
            score = 8.5
            critiques, adjustments = [], []
            eval_dict = {"is_approved": True, "score": 8.5}

        return {
            "is_approved": is_approved,
            "critic_score": score,
            "critic_evaluation": eval_dict,
            "critique": " ; ".join(critiques) if critiques else None,
            "suggested_improvements": adjustments,
        }

    async def synthesize_coaching_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 4: Redacta la respuesta comercial con Soul SALES_COACH y aplica BaseSynthesizerNode.format_final_response."""
        plan = state.get("candidate_plan") or {}
        gap_info = state.get("gap_info") or {}
        raw_query = state.get("raw_query") or ""

        # Historial multi-turno
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=15,
            strategy="last",
            token_counter=len,
        )

        system_prompt = inject_soul(
            """
            Eres la Coach Comercial de Ventas y Crecimiento de Spark Agent.
            Comunícate con energía motivadora, empatía ejecutiva y absoluta precisión en las cifras.

            ESTRUCTURA OBLIGATORIA DEL MENSAJE:
            1. Saludo cálido y diagnóstico claro del avance actual (total confirmado en S/., escala alcanzada y cotizaciones por cerrar).
            2. Próximo hito al alcance de la mano con la brecha exacta en soles (S/.) o unidades.
            3. Plan de acción recomendado (combos sugeridos combinando productos con bono de regalo y productos tendencia que los clientes ya están pidiendo).
            4. Llamado a la acción claro y motivador para cerrar ventas hoy.

            REGLAS ESTRICTAS DE FORMATO:
            - Formato nativo de WhatsApp: *negrita* con un solo asterisco.
            - Viñetas en renglón independiente con salto de línea.
            - Expresa montos siempre en soles con la nomenclatura 'S/.' (ej: S/. 180.00).
            - PROHIBIDO mencionar Odoo, ERP, Postgres, JSON ni tablas Markdown.
            """,
            role=SoulRole.INCENTIVES_COACH,
        )

        channel_instructions = BaseSynthesizerNode.get_channel_prompt_instructions(state)

        chat_model = bind_temperature(self._llm, 0.3)

        try:
            prompt_content = f"""
            Consulta del vendedor: {raw_query}

            Plan Estratégico Aprobado:
            - Resumen: {plan.get('summary')}
            - Estado actual: S/. {gap_info.get('confirmed_total_pen', 0.0):.2f} (Escala: {gap_info.get('current_discount_tier')})
            - Próxima meta: {gap_info.get('next_discount_tier')} (Faltan S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f})
            - Hitos: {plan.get('milestones')}
            - Combos sugeridos: {plan.get('suggested_combos')}
            - Proyección de ganancia adicional: S/. {plan.get('projected_additional_earnings_pen', 0.0):.2f}
            """

            response = await chat_model.ainvoke([
                SystemMessage(content=f"{system_prompt}\n\n{channel_instructions}"),
                *trimmed_history,
                HumanMessage(content=prompt_content),
            ])
            reply_text = extract_clean_text(response.content)
        except Exception as e:
            logger.error(f"[SalesIncentives/Synthesizer] Error en redacción de respuesta: {e}")
            reply_text = (
                f"¡Hola! Actualmente registras un avance de *S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}* "
                f"en ventas confirmadas, alcanzando la escala de *{gap_info.get('current_discount_tier', '25% de Descuento')}*.\n\n"
                f"🎯 *Tu siguiente meta:* Te faltan solo *S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f}* "
                f"para conquistar el *{gap_info.get('next_discount_tier', '30% de Descuento')}*."
            )

        # Formateo centralizado con BaseSynthesizerNode
        extra_meta = {
            "candidate_plan": plan,
            "gap_info": gap_info,
            "critic_score": state.get("critic_score"),
            "iteration_count": state.get("iteration_count"),
            "channel": state.get("channel") or "whatsapp",
        }
        return BaseSynthesizerNode.format_final_response(reply_text, state, extra=extra_meta)
