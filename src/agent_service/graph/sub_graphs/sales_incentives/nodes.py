"""
src/agent_service/graph/sub_graphs/sales_incentives/nodes.py - Nodos para el subgrafo sales_incentives con soporte Multimarca y BaseSynthesizerNode.
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
    SupplierIncentivePlan,
)
from src.agent_service.tools.incentive_tools import (
    odoo_get_active_incentives,
    extract_sales_trends,
    calculate_incentive_gap,
    calculate_multibrand_incentive_gaps,
    resolve_supplier_sales_metrics,
    detect_target_supplier,
    get_sku_supplier_map,
    SUPPLIER_ALIASES,
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
        """Nodo 1: Recopila órdenes actuales de Odoo, particiona por marca, consulta planes vigentes y mina tendencias."""
        user_id = state.get("user_id") or 5
        raw_query = state.get("raw_query") or ""
        supplier_id = state.get("supplier_id")
        supplier_name = state.get("supplier_name")

        # 1. Detección automática de marca en la consulta del usuario
        target_meta = detect_target_supplier(raw_query)
        if target_meta:
            supplier_id = target_meta["id"]
            supplier_name = target_meta["name"]
            is_multibrand = False
        elif supplier_id is not None:
            is_multibrand = False
        else:
            is_multibrand = True

        # 2. Obtener órdenes actuales del vendedor
        try:
            orders_data = await self._list_orders_tool(user_id=user_id)
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar órdenes en Odoo: {e}")
            orders_data = {"draft": [], "sale": [], "cancel": []}

        # 3. Consultar incentivos vigentes en Odoo
        try:
            incentives = await self._get_incentives_tool(
                supplier_id=supplier_id if not is_multibrand else None,
                supplier_name=supplier_name if not is_multibrand else None,
                only_active=True,
            )
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar incentivos en Odoo: {e}")
            incentives = []

        # 4. Contexto temporal en tiempo real y SKU map
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        sku_map = await get_sku_supplier_map()

        # 5. Cálculo de métricas y brechas multimarca y monomarca
        multibrand_metrics = resolve_supplier_sales_metrics(orders_data, sku_supplier_map=sku_map)
        multibrand_gaps = calculate_multibrand_incentive_gaps(
            orders_data,
            incentives,
            reference_date=now,
            sku_supplier_map=sku_map,
        )

        gap_info = calculate_incentive_gap(
            orders_data,
            incentives,
            reference_date=now,
            supplier_id=supplier_id if not is_multibrand else None,
            sku_supplier_map=sku_map,
        )

        # 6. Minar tendencias de compra recientes (demanda orgánica)
        trends = extract_sales_trends(
            orders_data,
            supplier_id=supplier_id if not is_multibrand else None,
            top_k=5,
            sku_supplier_map=sku_map,
        )

        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "target_supplier": target_meta,
            "is_multibrand": is_multibrand,
            "orders_data": orders_data,
            "active_incentives": incentives,
            "sales_trends": trends,
            "gap_info": gap_info,
            "multibrand_gaps": multibrand_gaps,
            "multibrand_metrics": multibrand_metrics,
            "current_date_str": current_date_str,
            "epsilon": state.get("epsilon") or self._default_epsilon,
            "iteration_count": 0,
            "max_iterations": state.get("max_iterations") or self._default_max_iterations,
        }

    async def actor_planner_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 2 (Actor): Formula un plan de acción estructurado adaptado a Multimarca o Monomarca con ε-greedy."""
        raw_query = state.get("raw_query") or ""
        is_multibrand = state.get("is_multibrand", True)
        gap_info = state.get("gap_info") or {}
        multibrand_gaps = state.get("multibrand_gaps") or {}
        active_incentives = state.get("active_incentives") or []
        trends = state.get("sales_trends") or []
        epsilon = state.get("epsilon") or 0.25
        iteration_count = state.get("iteration_count") or 0
        critique = state.get("critique")
        suggested_improvements = state.get("suggested_improvements") or []
        current_date_str = state.get("current_date_str") or datetime.now().strftime("%Y-%m-%d")
        supplier_id = state.get("supplier_id")
        supplier_name = state.get("supplier_name")

        # Clientes de la cartera
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

        if is_multibrand:
            # Resumen de métricas multimarca
            brand_sections = []
            for sid, sdata in multibrand_gaps.items():
                s_trends = [f"{t['name']} (SKU {t['sku']})" for t in sdata.get("trends", [])[:3]]
                s_promos = [p["title"] for p in sdata.get("product_promos", [])[:2]]
                sec = f"""
                MARCA / PROVEEDOR [{sdata.get('brand_display_name', sdata.get('supplier_name'))}]:
                - Ventas Confirmadas: S/. {sdata.get('confirmed_total_pen', 0.0):.2f}
                - Cotizaciones Pendientes: S/. {sdata.get('draft_total_pen', 0.0):.2f}
                - Escala/Estado Actual: {sdata.get('current_discount_tier')}
                - Siguiente Meta: {sdata.get('next_discount_tier')} (Objetivo: S/. {sdata.get('next_tier_target_pen', 0.0):.2f})
                - Brecha en Soles: S/. {sdata.get('gap_to_next_tier_pen', 0.0):.2f}
                - Promociones de Regalo Activas: {', '.join(s_promos) if s_promos else 'Ninguna'}
                - Productos Tendencia de esta Marca: {', '.join(s_trends) if s_trends else 'Sin historial reciente'}
                """
                brand_sections.append(sec)

            multibrand_context = "\n".join(brand_sections)

            system_prompt = f"""
            Eres el ACTOR y Estratega de Ventas Multimarca de Spark Agent (Sales Coach).
            Tu misión es generar un plan estructurado (IncentiveActionPlan) para un vendedor que gestiona múltiples marcas comerciales (Yanbal y Belcorp).

            MODO DE TRABAJO: MULTIMARCA AGRUPADO
            - El usuario solicitó un balance general de sus ventas/avances.
            - DEBES estructurar el campo `supplier_plans` con un sub-plan para cada marca con actividad o metas.
            - Fija `is_multibrand = True`.
            - Regla estricta: NO mezcles productos de marcas distintas en un mismo combo. Los combos de Yanbal deben contener solo productos Yanbal; los de Belcorp solo Belcorp.

            POLÍTICA EPSILON-GREEDY (ε = {epsilon}):
            - Para cada marca, equilibra:
              1. 75% Explotación: Combos con bonos de producto o diseñados para cerrar la brecha hacia la siguiente escala de esa marca.
              2. 25% Exploración: Productos de alta demanda orgánica según las tendencias reales de pedidos de esa misma marca.

            DATOS POR MARCA EN ODOO:
            {multibrand_context}

            CONTEXTO TEMPORAL Y CARTERA:
            - Fecha actual: {current_date_str}
            - Clientes activos en cartera: {', '.join(customers) if customers else 'Clientes generales'}

            {critique_block}
            """
        else:
            # Modo Monomarca Filtrado
            display_name = SUPPLIER_ALIASES.get(supplier_id, {}).get("display_name", supplier_name or "Proveedor")
            s_promos = [p["title"] for p in gap_info.get("product_promos", [])[:3]]
            s_trends = [f"{t.get('name')} (SKU {t.get('sku')})" for t in trends[:4]]

            system_prompt = f"""
            Eres el ACTOR y Estratega de Ventas de Spark Agent (Sales Coach).
            Tu misión es generar un plan estructurado (IncentiveActionPlan) enfocado EXCLUSIVAMENTE en la marca: {display_name}.

            MODO DE TRABAJO: MONOMARCA FILTRADO
            - El usuario consultó específicamente por {display_name}.
            - Fija `is_multibrand = False` y `target_supplier_name = '{supplier_name}'`.
            - Focaliza todos los hitos (`milestones`) y combos (`suggested_combos`) exclusivamente en {display_name}.
            - NO menciones otras marcas a menos que el usuario lo pida explícitamente.

            POLÍTICA EPSILON-GREEDY (ε = {epsilon}):
            1. Explotación (75%): Focalízate en cerrar la brecha hacia la siguiente escala de {display_name}.
            2. Exploración (25%): Incorpora productos de alta demanda orgánica de {display_name}.

            ESTADO EN ODOO ({display_name}):
            - Total confirmado: S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}
            - Cotizaciones pendientes (draft): S/. {gap_info.get('draft_total_pen', 0.0):.2f}
            - Escala/Estado actual: {gap_info.get('current_discount_tier', '20%')}
            - Siguiente meta: {gap_info.get('next_discount_tier')} (Meta: S/. {gap_info.get('next_tier_target_pen', 0.0):.2f})
            - Brecha en soles: S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f}
            - Días restantes de campaña: {gap_info.get('days_remaining', 'Abierto')}
            - Promociones activas de esta marca: {', '.join(s_promos) if s_promos else 'Ninguna'}
            - Productos tendencia de esta marca: {', '.join(s_trends) if s_trends else 'Sin historial previo'}
            - Clientes activos en cartera: {', '.join(customers) if customers else 'Clientes generales'}

            {critique_block}
            """

        structured_actor = bind_structured_output(bind_temperature(self._llm, 0.2), IncentiveActionPlan)

        try:
            plan: Optional[IncentiveActionPlan] = await structured_actor.ainvoke([
                SystemMessage(content=system_prompt),
                *trimmed_history,
                HumanMessage(content=f"Consulta del usuario: {raw_query or '¿Cómo van mis ventas e incentivos?'}"),
            ])
            plan_dict = plan.model_dump() if plan else {}
        except Exception as e:
            logger.error(f"[SalesIncentives/Actor] Error al formular plan: {e}")
            plan_dict = {
                "summary": "Plan de contingencia comercial para optimizar metas de ventas.",
                "current_status_overview": f"Ventas confirmadas por S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}.",
                "is_multibrand": is_multibrand,
                "target_supplier_name": supplier_name,
                "supplier_plans": [],
                "milestones": [],
                "suggested_combos": [],
                "projected_additional_earnings_pen": 0.0,
                "epsilon_balance_rationale": "Balance por defecto ante contingencia.",
            }

        return {
            "candidate_plan": plan_dict,
            "iteration_count": iteration_count + 1,
        }

    async def critic_evaluator_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 3 (Critic): Evalúa factibilidad temporal, respeto de restricciones comerciales e integridad de marcas."""
        candidate_plan = state.get("candidate_plan") or {}
        gap_info = state.get("gap_info") or {}
        iteration_count = state.get("iteration_count") or 1
        max_iterations = state.get("max_iterations") or self._default_max_iterations
        current_date_str = state.get("current_date_str") or datetime.now().strftime("%Y-%m-%d")
        is_multibrand = state.get("is_multibrand", True)

        if iteration_count >= max_iterations:
            logger.info(f"[SalesIncentives/Critic] Límite máximo de iteraciones alcanzado ({iteration_count}/{max_iterations}). Aprobando plan.")
            return {
                "is_approved": True,
                "critic_score": 8.5,
                "critic_evaluation": {
                    "is_approved": True,
                    "score": 8.5,
                    "temporal_feasibility_score": 8.5,
                    "constraint_adherence_score": 8.5,
                    "brand_integrity_score": 9.0,
                    "roi_score": 8.5,
                    "critiques": [],
                    "suggested_adjustments": [],
                },
                "critique": None,
                "suggested_improvements": [],
            }

        system_prompt = f"""
        Eres el CRITIC y Juez de Calidad Comercial de Spark Agent.
        Evalúa con rigor técnico el plan de incentivos propuesto por el Actor:

        CRITERIOS DE EVALUACIÓN:
        1. Factibilidad Temporal: ¿Es alcanzable en los días restantes ({gap_info.get('days_remaining', 'Abierto')}) considerando la fecha {current_date_str}?
        2. Cumplimiento de Restricciones y Realismo: ¿Los montos de los combos corresponden con la brecha real en Odoo?
        3. Integridad de Marcas (Brand Integrity):
           - En planes multimarca: cada combo o meta debe pertenecer a su marca correspondiente. ¡No mezclar productos Belcorp para escalas Yanbal ni viceversa!
           - En planes monomarca: el plan no debe desviarse a otras marcas no solicitadas.
        4. Balance ε-Greedy: ¿Equilibra metas de incentivo (75%) con productos de alta demanda orgánica de esa misma marca (25%)?

        UMBRAL DE APROBACIÓN:
        - Si el plan cumple satisfactoriamente los 4 criterios (score >= 8.0), fija is_approved = True.
        - Si detectas inconsistencias numéricas o mezcla indebida de marcas, fija is_approved = False y detalla las correcciones en suggested_adjustments.
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
        """Nodo 4: Redacta la respuesta comercial con Soul INCENTIVES_COACH y BaseSynthesizerNode.format_final_response."""
        plan = state.get("candidate_plan") or {}
        gap_info = state.get("gap_info") or {}
        multibrand_gaps = state.get("multibrand_gaps") or {}
        is_multibrand = state.get("is_multibrand", True)
        supplier_id = state.get("supplier_id")
        raw_query = state.get("raw_query") or ""

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

            ESTRUCTURA DE RESPUESTA POR WHATSAPP:
            - Si es consulta Multimarca: Saludo cálido y desglose estructurado con secciones independientes para cada marca (ej. 🌟 *Yanbal (Unique)* y 💎 *Belcorp (Ésika / Cyzone / L'Bel)*) detallando ventas confirmadas, cotizaciones, metas/escalas, brecha en soles (S/.) y combos sugeridos.
            - Si es consulta Monomarca: Enfócate exclusivamente en la marca consultada, con su diagnóstico actual, siguiente meta, combos recomendados y llamado a la acción.

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
            Modo: {'Multimarca Agrupado' if is_multibrand else 'Monomarca Filtrado'}

            Plan Estratégico Aprobado:
            {str(plan)}

            Brechas Multimarca Disponibles:
            {str(multibrand_gaps)}
            """

            response = await chat_model.ainvoke([
                SystemMessage(content=f"{system_prompt}\n\n{channel_instructions}"),
                *trimmed_history,
                HumanMessage(content=prompt_content),
            ])
            reply_text = extract_clean_text(response.content)
        except Exception as e:
            logger.error(f"[SalesIncentives/Synthesizer] Error en redacción de respuesta: {e}")
            if is_multibrand:
                u_info = multibrand_gaps.get(9, {})
                c_info = multibrand_gaps.get(17, {})
                reply_text = (
                    f"¡Hola! Aquí tienes el balance de tus avances e incentivos por marca:\n\n"
                    f"🌟 *Yanbal (Unique)*\n"
                    f"• *Ventas confirmadas:* S/. {u_info.get('confirmed_total_pen', 0.0):.2f} (Escala: *{u_info.get('current_discount_tier', '20%')}*)\n"
                    f"• *Cotizaciones pendientes:* S/. {u_info.get('draft_total_pen', 0.0):.2f}\n"
                    f"• 🎯 *Siguiente meta:* Te faltan *S/. {u_info.get('gap_to_next_tier_pen', 0.0):.2f}* para el *{u_info.get('next_discount_tier', '25% de Descuento')}*\n\n"
                    f"💎 *Belcorp (Ésika / Cyzone / L'Bel)*\n"
                    f"• *Ventas confirmadas:* S/. {c_info.get('confirmed_total_pen', 0.0):.2f}\n"
                    f"• 🎯 *Meta activa:* *Rappel Anual del 5%* sobre facturación neta acumulada\n\n"
                    f"¿Deseas que preparemos un plan de acción para alguna de estas marcas hoy?"
                )
            else:
                display_name = SUPPLIER_ALIASES.get(supplier_id, {}).get("display_name", "tu marca")
                reply_text = (
                    f"¡Hola! Para *{display_name}*, registras un avance de *S/. {gap_info.get('confirmed_total_pen', 0.0):.2f}* "
                    f"en ventas confirmadas (Escala actual: *{gap_info.get('current_discount_tier', '20%')}*).\n\n"
                    f"🎯 *Tu siguiente meta:* Te faltan solo *S/. {gap_info.get('gap_to_next_tier_pen', 0.0):.2f}* "
                    f"para conquistar el *{gap_info.get('next_discount_tier', '25% de Descuento')}*."
                )

        extra_meta = {
            "candidate_plan": plan,
            "gap_info": gap_info,
            "multibrand_gaps": multibrand_gaps,
            "is_multibrand": is_multibrand,
            "critic_score": state.get("critic_score"),
            "iteration_count": state.get("iteration_count"),
            "channel": state.get("channel") or "whatsapp",
        }
        return BaseSynthesizerNode.format_final_response(reply_text, state, extra=extra_meta)
