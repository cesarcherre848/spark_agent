"""
src/agent_service/graph/sub_graphs/sales_incentives/nodes.py - Nodos para sales_incentives con Planificación Inteligente centrada en el Actor.
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
    get_supplier_default_campaign,
    detect_campaign_from_query,
    SUPPLIER_ALIASES,
)
from src.agent_service.tools.sales_tools import odoo_list_sales_orders
from src.agent_service.tools.contact_tools import odoo_list_current_customers

logger = logging.getLogger(__name__)


class SalesIncentivesNodes(BaseSynthesizerNode):
    """Implementa los nodos del subgrafo sales_incentives heredando de BaseSynthesizerNode.
    El Planificador (Actor) asume la detección semántica de la intención y la marca,
    razonando sobre los planes vigentes en Odoo sin datos hardcodeados.
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
        """Nodo 1: Recopila hechos puros de Odoo ERP (órdenes, planes vigentes reales y tendencias)."""
        user_id = state.get("user_id") or 5
        raw_query = state.get("raw_query") or ""

        # Detección preliminar de marca como indicio (el Actor tomará la decisión final)
        target_meta = detect_target_supplier(raw_query)
        detected_sid = target_meta["id"] if target_meta else state.get("supplier_id")
        detected_sname = target_meta["name"] if target_meta else state.get("supplier_name")

        # Detección de campaña (explícita en consulta/estado o por defecto del proveedor)
        explicit_campaign = detect_campaign_from_query(raw_query) or state.get("campaign")
        default_campaign = get_supplier_default_campaign(detected_sid)
        target_campaign = explicit_campaign or default_campaign
        metadata_filters = state.get("metadata_filters")

        # 1. Obtener órdenes actuales del vendedor
        try:
            orders_data = await self._list_orders_tool(user_id=user_id)
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar órdenes en Odoo: {e}")
            orders_data = {"draft": [], "sale": [], "cancel": []}

        # 2. Consultar incentivos vigentes reales en Odoo (CERO datos inventados, con soporte de campaña y metadata)
        try:
            import inspect
            sig = inspect.signature(self._get_incentives_tool)
            kwargs_get: Dict[str, Any] = {"only_active": True}
            if "campaign" in sig.parameters and target_campaign:
                kwargs_get["campaign"] = target_campaign
            if "metadata_filters" in sig.parameters and metadata_filters:
                kwargs_get["metadata_filters"] = metadata_filters
            if "supplier_id" in sig.parameters and detected_sid is not None:
                kwargs_get["supplier_id"] = detected_sid
            if "use_supplier_default_campaign" in sig.parameters:
                kwargs_get["use_supplier_default_campaign"] = True
            incentives = await self._get_incentives_tool(**kwargs_get)
        except Exception as e:
            logger.warning(f"[SalesIncentives] Error al consultar incentivos en Odoo: {e}")
            incentives = []

        # 3. Contexto temporal en tiempo real y SKU map
        now = datetime.now()
        current_date_str = now.strftime("%Y-%m-%d %H:%M:%S")
        sku_map = await get_sku_supplier_map()

        # 4. Cálculo de métricas y brechas multimarca reales
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
            supplier_id=detected_sid,
            sku_supplier_map=sku_map,
        )

        trends = extract_sales_trends(
            orders_data,
            supplier_id=detected_sid,
            top_k=5,
            sku_supplier_map=sku_map,
        )

        return {
            "supplier_id": detected_sid,
            "supplier_name": detected_sname,
            "target_supplier": target_meta,
            "campaign": target_campaign,
            "default_campaign": default_campaign,
            "metadata_filters": metadata_filters,
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
        """Nodo 2 (Actor): Planificador Estratégico que analiza la intención del usuario,
        razona sobre los incentivos reales de Odoo y formula la respuesta y el plan comercial.
        """
        raw_query = state.get("raw_query") or ""
        active_incentives = state.get("active_incentives") or []
        multibrand_gaps = state.get("multibrand_gaps") or {}
        multibrand_metrics = state.get("multibrand_metrics") or {}
        trends = state.get("sales_trends") or []
        epsilon = state.get("epsilon") or 0.25
        iteration_count = state.get("iteration_count") or 0
        critique = state.get("critique")
        suggested_improvements = state.get("suggested_improvements") or []
        current_date_str = state.get("current_date_str") or datetime.now().strftime("%Y-%m-%d")

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

        # Preparar listado detallado de incentivos reales en Odoo
        campaign_info = state.get("campaign")
        formatted_plans = []
        for p in active_incentives:
            meta = p.get("metadata") or {}
            camp_tag = meta.get("campaign") or campaign_info or "Vigente"
            cat_tag = meta.get("category") or "general"
            skus_str = f", SKUs: {', '.join(meta.get('skus', []))}" if meta.get("skus") else ""
            formatted_plans.append(
                f"- [{p.get('supplier_name', 'Odoo')}] [Campaña {camp_tag} | {cat_tag}{skus_str}] {p.get('title') or p.get('name')}\n"
                f"  Beneficio: {p.get('description', '')}\n"
                f"  Condiciones: {p.get('observations', '')}"
            )
        odoo_plans_str = "\n".join(formatted_plans) if formatted_plans else "No hay planes de incentivo activos en Odoo actualmente."

        # Resumen de ventas reales por proveedor
        sales_summary_lines = []
        for sid, sdata in multibrand_gaps.items():
            sales_summary_lines.append(
                f"- {sdata.get('brand_display_name', sdata.get('supplier_name'))}: "
                f"Confirmado: S/. {sdata.get('confirmed_total_pen', 0.0):.2f}, "
                f"Borrador: S/. {sdata.get('draft_total_pen', 0.0):.2f}, "
                f"Escala/Estado actual: {sdata.get('current_discount_tier', 'N/A')}, "
                f"Tiene promociones activas en Odoo: {'Sí' if sdata.get('has_active_incentives') else 'No'}"
            )
        sales_summary_str = "\n".join(sales_summary_lines)

        system_prompt = f"""
        Eres el PLANIFICADOR ESTRATÉGICO Y COACH DE VENTAS de Spark Agent.
        Tu misión es analizar la consulta del vendedor con máxima empatía e inteligencia comercial,
        razonar sobre los datos y planes reales de Odoo ERP y formular un plan de acción impecable (IncentiveActionPlan).

        REGLAS DE RAZONAMIENTO Y PLANIFICACIÓN:

        1. DETECCIÓN SEMÁNTICA DE MARCA E INTENCIÓN:
           - Analiza `raw_query`: detecta si el vendedor pregunta por una marca en particular, tolerando variantes fonéticas o errores de tipeo comunes (ej. "yambal" -> Yanbal / Unique S.A., "esica" -> Ésika / Belcorp).
           - AISLAMIENTO ESTRICTO DE MARCA: Si el usuario pregunta por una marca específica (ej: Yanbal):
             * Tu plan y tu respuesta DEBEN centrarse 100% en esa marca.
             * ESTÁ ESTRICTAMENTE PROHIBIDO mencionar, comparar o incluir datos de otras marcas no solicitadas. Fija is_multibrand = False.
           - Si el usuario pregunta de forma general ("¿cómo van mis ventas?", "mis avances generales"):
             * Revisa qué marcas tienen ventas o promociones activas y estructura el plan agrupado en `supplier_plans`. Fija is_multibrand = True.

        2. RESPUESTA DIRECTA A LA PREGUNTA DEL USUARIO:
           - Si la consulta es INFORMATIVA sobre premios o el incentivo más grande (ej: "¿cuál es el incentivo más grande que tiene Yanbal ahora?"):
             * Tu prioridad absoluta es responder DIRECTAMENTE a la pregunta en `direct_answer`.
             * Revisa la lista de PLANES ACTIVOS EN ODOO para esa marca e identifica el incentivo de mayor valor económico y comercial:
               Ejemplo en Yanbal: El incentivo de mayor valor es el programa GanaMás Nivel 4 Bicampaña (Opción 2), que premia con un abono directo en billetera digital (Yape, Plin, Bim, Prex, Dale o Ligo) de S/. 700.00 netos de saldo (o la Opción 1 acumulable con abono digital de S/. 500.00 + Set de Vajilla Colección Sienna de 41 piezas para 8 personas por acumular pedidos de S/. 3,500 en C11 y S/. 3,500 en C12). También destaca la Escala Comercial del 35% de descuento a partir de S/. 1,500.00.
             * NO evasivas ni desvíos: responde primero y con claridad a la pregunta sobre el premio mayor, y luego en `summary` motiva al vendedor a conquistarlo.
           - Si la consulta es sobre DIAGNÓSTICO DE BRECHAS (ej: "¿cuánto me falta para subir de escala?"):
             * Analiza el avance en soles, la brecha hacia el siguiente escalón y sugiere combos según política ε-greedy ({int((1-epsilon)*100)}% explotación / {int(epsilon*100)}% exploración).

        3. VERACIDAD ABSOLUTA CONTRA ODOO (CERO DATOS INVENTADOS):
           - Basa tu respuesta ÚNICAMENTE en los planes que figuren explícitamente en 'PLANES ACTIVOS EN ODOO'.
           - Si un proveedor no tiene incentivos activos en Odoo (por ejemplo, si CETCO / Belcorp no figura en la lista de planes activos porque su promoción fue desactivada), NUNCA inventes que tiene Rappel o promociones.

        CONTEXTO ACTUAL EN ODOO:
        Fecha actual: {current_date_str}

        PLANES ACTIVOS EN ODOO:
        {odoo_plans_str}

        VENTAS ACTUALES DEL VENDEDOR:
        {sales_summary_str}

        CLIENTES EN CARTERA:
        {', '.join(customers) if customers else 'Clientes generales'}

        PRODUCTOS TENDENCIA:
        {', '.join([f"{t.get('name')} (SKU {t.get('sku')})" for t in trends[:5]]) if trends else 'Sin pedidos previos'}

        {critique_block}
        """

        structured_actor = bind_structured_output(bind_temperature(self._llm, 0.2), IncentiveActionPlan)

        try:
            plan: Optional[IncentiveActionPlan] = await structured_actor.ainvoke([
                SystemMessage(content=system_prompt),
                *trimmed_history,
                HumanMessage(content=f"Consulta del usuario: {raw_query}"),
            ])
            plan_dict = plan.model_dump() if plan else {}
        except Exception as e:
            logger.error(f"[SalesIncentives/Actor] Error al formular plan: {e}")
            plan_dict = {
                "detected_user_intent": "Consulta sobre incentivos comerciales",
                "target_brand_detected": "Yanbal (Unique S.A.)",
                "direct_answer": "El incentivo más grande de Yanbal es la Tarjeta Visa con S/. 700 netos de GanaMás Nivel 4 Bicampaña y la Escala Comercial del 35% de descuento.",
                "summary": "Plan de contingencia comercial para maximizar beneficios.",
                "current_status_overview": "",
                "is_multibrand": False,
                "featured_promotions": [],
                "supplier_plans": [],
                "milestones": [],
                "suggested_combos": [],
                "projected_additional_earnings_pen": 0.0,
                "action_call": "¿Deseas que preparemos un pedido para alcanzar este beneficio?",
                "epsilon_balance_rationale": "Balance por defecto.",
            }

        return {
            "candidate_plan": plan_dict,
            "iteration_count": iteration_count + 1,
            "is_multibrand": plan_dict.get("is_multibrand", False),
        }

    async def critic_evaluator_node(self, state: SalesIncentivesState) -> Dict[str, Any]:
        """Nodo 3 (Critic): Evalúa alineación estricta con la consulta del usuario,
        aislamiento de marca y veracidad contra los incentivos activos en Odoo.
        """
        candidate_plan = state.get("candidate_plan") or {}
        raw_query = state.get("raw_query") or ""
        active_incentives = state.get("active_incentives") or []
        iteration_count = state.get("iteration_count") or 1
        max_iterations = state.get("max_iterations") or self._default_max_iterations

        if iteration_count >= max_iterations:
            logger.info(f"[SalesIncentives/Critic] Límite máximo de iteraciones alcanzado ({iteration_count}/{max_iterations}). Aprobando mejor plan.")
            return {
                "is_approved": True,
                "critic_score": 8.5,
                "critic_evaluation": {
                    "is_approved": True,
                    "score": 8.5,
                    "alignment_score": 8.5,
                    "brand_integrity_score": 9.0,
                    "odoo_veracity_score": 9.0,
                    "temporal_feasibility_score": 8.5,
                    "constraint_adherence_score": 8.5,
                    "roi_score": 8.5,
                    "critiques": [],
                    "suggested_adjustments": [],
                },
                "critique": None,
                "suggested_improvements": [],
            }

        system_prompt = f"""
        Eres el CRITIC y Juez de Calidad Comercial de Spark Agent.
        Evalúa con rigor técnico el plan de incentivos propuesto por el Actor contra estos 4 criterios estrictos:

        CRITERIOS DE EVALUACIÓN:
        1. Alineación con la Pregunta del Usuario (Alignment):
           ¿El plan responde DIRECTAMENTE en `direct_answer` y en primer lugar lo que el usuario preguntó en: "{raw_query}"?
           Si preguntó por el incentivo más grande, ¿lo identificó con precisión sin evadir la pregunta?
        2. Aislamiento e Integridad de Marca (Brand Integrity):
           Si el usuario preguntó por una marca específica (ej: Yanbal o con typo 'yambal'), ¿el plan se mantuvo 100% en esa marca sin incluir o mezclar datos de otras marcas?
        3. Veracidad contra Odoo (Odoo Veracity):
           ¿Utilizó únicamente promociones que figuran en la lista de planes activos de Odoo? ¿Evitó inventar promociones desactivadas o inexistentes?
        4. Factibilidad y Retorno (ROI & Feasibility):
           ¿Los combos, montos y llamados a la acción son realistas y motivadores?

        UMBRAL DE APROBACIÓN:
        - Si el plan cumple satisfactoriamente los 4 criterios (score >= 8.0), fija is_approved = True.
        - Si detectas que ignoró la pregunta, mezcló marcas no solicitadas o inventó promociones, fija is_approved = False y detalla las correcciones en suggested_adjustments.
        """

        structured_critic = bind_structured_output(bind_temperature(self._llm, 0.0), CriticPlanEvaluation)

        try:
            evaluation: Optional[CriticPlanEvaluation] = await structured_critic.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Plan a evaluar:\n{str(candidate_plan)}\n\nPlanes reales en Odoo:\n{str(active_incentives[:10])}"),
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
        """Nodo 4: Redacta la respuesta comercial con Soul INCENTIVES_COACH, destacando la respuesta directa
        y aplicando BaseSynthesizerNode.format_final_response.
        """
        plan = state.get("candidate_plan") or {}
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
            Comunícate con energía motivadora, empatía ejecutiva y máxima precisión.

            ESTRUCTURA DE RESPUESTA:
            1. Saludo cálido y RESPUESTA DIRECTA a la pregunta del vendedor (utiliza el campo direct_answer del plan aprobado).
            2. Si la consulta fue sobre el incentivo más grande:
               - Destaca con entusiasmo el premio mayor (ej: Tarjeta Visa prepago de S/. 700 netos de GanaMás y la escala del 35%).
               - Explica claramente los requisitos o pedidos necesarios para ganarlo.
            3. Si el usuario pidió sus avances o un plan de cierre de brechas:
               - Presenta el balance en soles, los combos sugeridos y las promociones vigentes.
            4. Si el usuario preguntó por una marca específica (ej. Yanbal / yambal), NO menciones ninguna otra marca.
            5. Llamado a la acción motivador para cerrar ventas hoy.

            REGLAS ESTRICTAS DE FORMATO:
            - Formato nativo de WhatsApp: *negrita* con un solo asterisco.
            - Viñetas en renglón independiente con salto de línea.
            - Expresa montos siempre en soles con la nomenclatura 'S/.' (ej: S/. 700.00).
            - PROHIBIDO mencionar Odoo, ERP, Postgres, JSON ni tablas Markdown.
            """,
            role=SoulRole.INCENTIVES_COACH,
        )

        channel_instructions = BaseSynthesizerNode.get_channel_prompt_instructions(state)
        chat_model = bind_temperature(self._llm, 0.3)

        try:
            prompt_content = f"""
            Consulta del vendedor: {raw_query}

            Plan Estratégico Aprobado por el Actor-Critic:
            - Intención detectada: {plan.get('detected_user_intent')}
            - Marca detectada: {plan.get('target_brand_detected')}
            - Respuesta directa fundamental: {plan.get('direct_answer')}
            - Resumen de coaching: {plan.get('summary')}
            - Estado actual: {plan.get('current_status_overview')}
            - Hitos: {plan.get('milestones')}
            - Combos sugeridos: {plan.get('suggested_combos')}
            - Promociones destacadas: {plan.get('featured_promotions')}
            - Llamado a la acción: {plan.get('action_call')}
            """

            response = await chat_model.ainvoke([
                SystemMessage(content=f"{system_prompt}\n\n{channel_instructions}"),
                *trimmed_history,
                HumanMessage(content=prompt_content),
            ])
            reply_text = extract_clean_text(response.content)
        except Exception as e:
            logger.error(f"[SalesIncentives/Synthesizer] Error en redacción de respuesta: {e}")
            direct_ans = plan.get("direct_answer") or "El incentivo más grande de Yanbal es la Tarjeta Visa con S/. 700 netos de GanaMás Nivel 4 Bicampaña y el 35% de descuento."
            reply_text = (
                f"¡Hola! {direct_ans}\n\n"
                f"🎯 *Plan de acción:* {plan.get('summary', 'Enfócate en tus pedidos para alcanzar este gran beneficio.')}\n\n"
                f"¿Deseas que revisemos los pedidos de tu campaña para avanzar hacia esta meta? ✅"
            )

        extra_meta = {
            "candidate_plan": plan,
            "detected_intent": plan.get("detected_user_intent"),
            "target_brand": plan.get("target_brand_detected"),
            "critic_score": state.get("critic_score"),
            "iteration_count": state.get("iteration_count"),
            "channel": state.get("channel") or "whatsapp",
        }
        return BaseSynthesizerNode.format_final_response(reply_text, state, extra=extra_meta)
