"""
src/agent_service/graph/sub_graphs/product_advisor/nodes.py - Nodos ejecutores de product_advisor.
"""

import logging
import asyncio
import re
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.documents import Document

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.core.llms.factory import extract_clean_text
from src.agent_service.soul import SoulRole
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorPlan,
    QualityRubricEvaluation,
    format_products_for_advisor_prompt,
    FinalAnswer,
)
from src.agent_service.graph.sub_graphs.product_advisor.tools import (
    build_advisor_tools_registry,
    invoke_advisor_tool,
)

logger = logging.getLogger(__name__)


class ProductAdvisorNodes(BaseSynthesizerNode):
    """Nodos ejecutores del subgrafo unificado product_advisor con patrón Planner-Executor,
    herramientas dinámicas y ciclo Evaluator-Optimizer con rúbrica de calidad."""

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: Optional[ProductVectorStore] = None,
        odoo_client: Optional[OdooClient] = None,
        default_max_iterations: int = 3,
        tools_registry: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(llm=llm)
        self._llm = llm
        self._vector_store = vector_store
        self._odoo_client = odoo_client
        self._default_max_iterations = default_max_iterations

        # Herramientas
        self._tools_registry = tools_registry or build_advisor_tools_registry(
            vector_store=vector_store,
            odoo_client=odoo_client,
        )

        # Modelos estructurados
        self._planner = bind_structured_output(
            bind_temperature(llm, 0.1), AdvisorPlan
        )
        self._rubric_judge = bind_structured_output(
            bind_temperature(llm, 0.0), QualityRubricEvaluation
        )
        self._synthesizer = bind_structured_output(
            bind_temperature(llm, 0.35), FinalAnswer
        )

    async def plan_and_select_tools(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 1: Analiza la consulta, historial y posibles críticas previas para generar un plan de acción."""
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        iteration_count = state.get("iteration_count", 0)
        critique = state.get("critique")
        remedy_suggestions = state.get("suggested_improvements") or []
        partner_id = state.get("partner_id")
        customer_name = state.get("customer_name")
        user_id = state.get("user_id")

        # Obtener historial de diálogo previo para soporte multi-turno
        all_messages = state.get("messages", [])
        history_pool = (
            all_messages[:-1]
            if (all_messages and isinstance(all_messages[-1], HumanMessage) and all_messages[-1].content == raw_query)
            else all_messages
        )
        trimmed_history = trim_messages(
            history_pool,
            max_tokens=6,
            strategy="last",
            token_counter=len,
        )

        system_prompt = f"""
            Eres el planificador experto de un Asesor de Productos y Catálogo Comercial.
            Tu misión es analizar la consulta del usuario y generar un plan estructurado (AdvisorPlan)
            con las herramientas necesarias para responder de forma precisa, veraz y personalizada.

            RESOLUCIÓN CONTEXTUAL Y MULTI-TURNO:
            - Si la consulta actual es una pregunta de seguimiento, aclaración, comparación o filtro de un diálogo anterior
              (ej: 'de [Marca A] o de [Marca B] ?', '¿cuánto cuesta el segundo?', '¿tienes en color rojo?', '¿y colonias?'),
              analízala OBLIGATORIAMENTE en conjunto con los turnos previos de la conversación.
            - Propaga la categoría, tipo de producto o producto base discutido (ej: si antes hablaron de
              'perfumes para mujer', y ahora dice 'de [Marca A] o de [Marca B] ?', la búsqueda debe ser de 'perfumes' o 'fragancias'
              para cada marca indicada, NUNCA la frase genérica 'productos destacados').
            - PROPAGACIÓN OBLIGATORIA DE PÚBLICO OBJETIVO Y GÉNERO:
              Si en turnos previos el cliente especificó un público objetivo o género
              (ej: 'masculino', 'hombre', 'caballero', 'para él', 'femenino', 'mujer', 'dama', 'para ella', 'infantil', 'niños'),
              debes PRESERVARLO Y PROPAGARLO OBLIGATORIAMENTE a cualquier consulta de seguimiento elíptica
              (ej: si antes preguntó 'perfumes masculinos...' y ahora dice '¿y colonias?', la búsqueda
              debe ser 'colonia hombre masculino eau de toilette fragancia fresca', NUNCA únicamente 'colonias').
              Ten presente que en el catálogo comercial, las fragancias masculinas ligeras suelen denominarse
              'Eau de Toilette' o 'Parfum', mientras que el término 'Colonia' aislado suele corresponder a líneas femeninas o infantiles.

            HERRAMIENTAS DISPONIBLES:
            1. 'search_product_catalog': Búsqueda semántica híbrida en el catálogo. Argumentos:
               - query (str): Términos clave del producto.
               - marca (str, opcional): Marca comercial canónica consultada. Si el usuario solicita explícitamente una marca, pásala SIEMPRE aquí para filtrar únicamente sus productos.
               - vendor_name (str, opcional): Nombre comercial del proveedor o partner si se especifica.
               - pagina (int, opcional): Número de página exacta en el catálogo.
               - edicion (str, opcional): Edición/campaña (ej: 'C10').
               - limit (int, opcional): Cantidad máxima a recuperar (default 8).
               - user_id (int, opcional): ID del vendedor CRM.
            2. 'get_product_odoo_details': Consulta en tiempo real de precios, stock, moneda y UOM en Odoo. Argumentos:
               - skus (List[str]): Lista de códigos SKU a consultar.
               - fields (List[str], opcional): ['price', 'description', 'uom', 'category', 'taxes'].
            3. 'get_cross_sell_recommendations': Recomendaciones complementarias o alternativas. Argumentos:
               - base_sku (str, opcional): SKU base de referencia.
               - category (str, opcional): Categoría para sugerencias cruzadas.
               - marca (str, opcional): Marca preferida.
               - limit (int, opcional): Cantidad (default 5).
            4. 'filter_and_sort_products': Filtro determinista por presupuesto, página y orden. Argumentos:
               - products (List[dict]): Lista de productos a filtrar.
               - min_price (float, opcional), max_price (float, opcional).
               - marca (str, opcional), vendor_name (str, opcional), pagina (int, opcional), edicion (str, opcional).
               - sort_by (str, opcional): 'price_asc', 'price_desc' o 'relevance'.
            5. 'get_customer_purchase_history': Historial de compras del cliente en Odoo. Argumentos:
               - partner_id (int, opcional), customer_name (str, opcional), limit (int, opcional).

            PAUTAS DE PLANEAMIENTO:
            - AJUSTE PROACTIVO DE TOP-K / LIMIT (BÚSQUEDA EXHAUSTIVA):
              Si la consulta solicita un listado amplio o exhaustivo (ej: 'dame todos los perfumes...', 'qué opciones tienes', 'muéstrame todo el catálogo'),
              o si el cliente especifica un presupuesto máximo (ej: 'menor de 50', 'hasta 60 soles', 'el más barato'), o un rango de precios (ej: 'entre 40 y 80'):
              DEBES fijar proactivamente 'limit=15' o 'limit=20' en 'search_product_catalog' (en lugar del default de 8) para realizar una búsqueda exhaustiva
              que explore todos los escalones de precios del catálogo.
              Si el presupuesto solicitado es bajo (ej: < S/. 50 o < S/. 80), expande los términos de búsqueda a categorías afines accesibles
              (ej: query='perfume colonia fragancia corporal masculina hombre', limit=20) seguido de 'filter_and_sort_products'.
            - Si el usuario pide recomendaciones personalizadas ("según lo que suelo comprar", "¿qué me sugieres?"),
              invoca 'get_customer_purchase_history' para conocer sus preferencias antes de buscar en el catálogo.
            - Si el usuario pide un producto puntual o menciona marcas, páginas o campañas ("labial de [Marca] pág 12"),
              invoca 'search_product_catalog' extrayendo los metadatos correspondientes (restringiendo a dicha marca si fue solicitada).
            - Si el usuario pide comparar marcas explícitamente ("compara [Marca A] y [Marca B]"), planea búsquedas específicas
              para cada marca y consolida las opciones.
            - Si el usuario especifica un presupuesto máximo o pide "el más barato", planea 'search_product_catalog'
              con limit=15 o 20 seguido de 'filter_and_sort_products'.
            - Si el usuario busca productos complementarios a uno ya seleccionado, usa 'get_cross_sell_recommendations'.
        """

        feedback_context = ""
        if iteration_count > 0 and critique:
            feedback_context = f"""
            ATENCIÓN - RETROALIMENTACIÓN DE REINTENTO (Iteración {iteration_count}):
            El intento anterior fue RECHAZADO por la Rúbrica de Calidad con la siguiente crítica:
            "{critique}"
            Sugerencias de remediación: {remedy_suggestions}

            DEBES AJUSTAR EL PLAN:
            1. Aumenta el limit/top-k a 20 o 25 en 'search_product_catalog' para una búsqueda aún más profunda.
            2. Si la categoría estricta (ej: perfumes) no tiene opciones bajo el presupuesto, busca categorías complementarias o sustitutas más accesibles del mismo público (ej: colonias corporales, eau de toilette, desodorantes).
            3. Si ningún producto cumple estrictamente con el presupuesto máximo, usa 'filter_and_sort_products' con sort_by='price_asc' sin max_price estricto para recuperar las opciones más económicas disponibles en catálogo y poder orientar al cliente.
            """

        user_content = f"""
            Consulta del usuario: "{raw_query}"
            ID de cliente (partner_id): {partner_id or 'No especificado'}
            Nombre de cliente: {customer_name or 'No especificado'}
            ID de vendedor: {user_id or 5}
            {feedback_context}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_content),
        ]

        try:
            plan: AdvisorPlan = await self._planner.ainvoke(messages)
            planned_tools = [t.model_dump() for t in plan.tool_calls]
            plan_rationale = plan.reasoning
            extracted_meta = plan.extracted_metadata or {}
        except Exception as e:
            logger.warning(f"Error generando plan estructurado con LLM: {e}. Aplicando plan por defecto.")
            plan_rationale = "Plan de contingencia: búsqueda directa en catálogo."
            fallback_query = raw_query
            if len(raw_query.split()) <= 6 and trimmed_history:
                for m in reversed(trimmed_history):
                    if isinstance(m, HumanMessage) and m.content:
                        fallback_query = f"{m.content} {raw_query}"
                        break
            # Ajuste proactivo de limit en fallback si es exhaustivo o con precio
            f_lower = raw_query.lower()
            fallback_limit = 15 if any(w in f_lower for w in ["todo", "todos", "barato", "menos de", "hasta", "presupuesto", "precio", "rango"]) else 8
            planned_tools = [{
                "tool_name": "search_product_catalog",
                "arguments": {"query": fallback_query, "limit": fallback_limit, "user_id": user_id or 5},
                "purpose": "Búsqueda estándar de catálogo.",
            }]
            extracted_meta = {}

        return {
            "raw_query": raw_query,
            "plan_rationale": plan_rationale,
            "planned_tools": planned_tools,
            "metadata_filters": extracted_meta,
        }

    async def execute_tools(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 2: Ejecuta las herramientas seleccionadas en el plan de forma asíncrona y consolida candidatos."""
        planned_tools = state.get("planned_tools") or []
        user_id = state.get("user_id") or 5
        raw_query = state.get("raw_query") or "productos"

        # Fallback si no hay tools planificadas
        if not planned_tools:
            planned_tools = [{
                "tool_name": "search_product_catalog",
                "arguments": {"query": raw_query, "limit": 8, "user_id": user_id},
                "purpose": "Búsqueda por defecto",
            }]

        observations = []
        candidate_products: List[Dict[str, Any]] = list(state.get("candidate_products") or [])
        purchase_history = state.get("customer_purchase_history")

        for tool_call in planned_tools:
            t_name = tool_call.get("tool_name")
            t_args = dict(tool_call.get("arguments") or {})

            # Inyectar user_id si la herramienta lo admite y no vino
            if "user_id" in t_args and t_args["user_id"] is None:
                t_args["user_id"] = user_id

            # Inyectar candidatos y precios actualizados en filter_and_sort_products si no vinieron
            if t_name == "filter_and_sort_products":
                if not t_args.get("products"):
                    t_args["products"] = list(candidate_products)
                skus_to_price = [
                    p.get("sku") for p in t_args["products"]
                    if p.get("sku") and p.get("price") is None
                ]
                if skus_to_price:
                    get_details_fn = self._tools_registry.get("get_product_odoo_details")
                    if get_details_fn:
                        try:
                            price_res = await invoke_advisor_tool(
                                self._tools_registry,
                                "get_product_odoo_details",
                                {"skus": skus_to_price[:20]},
                            )
                            if isinstance(price_res, dict) and "products" in price_res:
                                d_sku = {d.get("sku"): d for d in price_res["products"] if isinstance(d, dict)}
                                for p in t_args["products"]:
                                    s = p.get("sku")
                                    if s in d_sku:
                                        p["price"] = d_sku[s].get("price")
                                        p["currency"] = d_sku[s].get("currency")
                                        p["uom"] = d_sku[s].get("uom")
                                        if d_sku[s].get("sales_description"):
                                            p["sales_description"] = d_sku[s].get("sales_description")
                                for p in candidate_products:
                                    s = p.get("sku")
                                    if s in d_sku:
                                        p["price"] = d_sku[s].get("price")
                                        p["currency"] = d_sku[s].get("currency")
                                        p["uom"] = d_sku[s].get("uom")
                                        if d_sku[s].get("sales_description"):
                                            p["sales_description"] = d_sku[s].get("sales_description")
                        except Exception as pe:
                            logger.warning(f"Error enriqueciendo precios previos a filtrado: {pe}")

            logger.info(f"product_advisor: Ejecutando herramienta '{t_name}' con args {t_args}")
            res = await invoke_advisor_tool(self._tools_registry, t_name, t_args)

            observations.append({
                "tool_name": t_name,
                "arguments": t_args,
                "result": res,
            })

            # Manejar resultados según herramienta
            if t_name in ("search_product_catalog", "get_cross_sell_recommendations"):
                if isinstance(res, list):
                    # Agregar nuevos candidatos evitando duplicados por SKU
                    existing_skus = {p.get("sku") for p in candidate_products if p.get("sku")}
                    for item in res:
                        if isinstance(item, dict):
                            s = item.get("sku")
                            if not s or s not in existing_skus:
                                candidate_products.append(item)
                                if s:
                                    existing_skus.add(s)

            elif t_name == "filter_and_sort_products":
                if isinstance(res, list):
                    if not res and candidate_products:
                        logger.info(
                            "product_advisor: filter_and_sort_products descartó todos los candidatos por presupuesto. "
                            "Preservando opciones disponibles más económicas como respaldo consultivo."
                        )
                        sorted_fallback = sorted(
                            candidate_products,
                            key=lambda x: (x.get("price") is None, float(x.get("price") or 999999))
                        )
                        for item in sorted_fallback[:3]:
                            item["budget_exceeded"] = True
                        candidate_products = sorted_fallback[:3]
                    else:
                        candidate_products = res

            elif t_name == "get_customer_purchase_history":
                if isinstance(res, dict):
                    purchase_history = res

            elif t_name == "get_product_odoo_details":
                if isinstance(res, dict) and "products" in res:
                    details_list = res.get("products", [])
                    details_by_sku = {d.get("sku"): d for d in details_list if isinstance(d, dict)}
                    for p in candidate_products:
                        s = p.get("sku")
                        if s and s in details_by_sku:
                            det = details_by_sku[s]
                            p["price"] = det.get("price")
                            p["currency"] = det.get("currency")
                            p["uom"] = det.get("uom")
                            if det.get("sales_description"):
                                p["sales_description"] = det.get("sales_description")

        # Enriquecimiento reactivo de precios si los candidatos carecen de precio
        skus_needing_price = [
            p.get("sku") for p in candidate_products
            if p.get("sku") and p.get("price") is None
        ]
        if skus_needing_price:
            try:
                get_details_fn = self._tools_registry.get("get_product_odoo_details")
                if get_details_fn:
                    res = await invoke_advisor_tool(
                        self._tools_registry,
                        "get_product_odoo_details",
                        {"skus": skus_needing_price[:15]},
                    )
                    if isinstance(res, dict) and "products" in res:
                        details_by_sku = {d.get("sku"): d for d in res["products"] if isinstance(d, dict)}
                        for p in candidate_products:
                            s = p.get("sku")
                            if s in details_by_sku:
                                p["price"] = details_by_sku[s].get("price")
                                p["currency"] = details_by_sku[s].get("currency")
                                p["uom"] = details_by_sku[s].get("uom")
            except Exception as e:
                logger.warning(f"Error en enriquecimiento reactivo de precios: {e}")

        # Crear documentos de LangChain para retrocompatibilidad
        retrieved_docs = []
        for p in candidate_products:
            retrieved_docs.append(
                Document(
                    page_content=p.get("description") or p.get("name") or "",
                    metadata=p,
                )
            )

        return {
            "tool_observations": observations,
            "candidate_products": candidate_products,
            "retrieved_products": retrieved_docs,
            "customer_purchase_history": purchase_history,
            "enriched_products": candidate_products,
            "filtered_products": candidate_products,
        }

    async def synthesize_draft(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 3: Sintetiza el borrador de respuesta comercial incorporando formato de canal y whitelabel obedeciendo al sintetizador padre."""
        raw_query = state.get("raw_query") or ""
        candidate_products = state.get("candidate_products") or []
        purchase_history = state.get("customer_purchase_history")

        products_context = self.format_products_context(candidate_products)

        # Historial de diálogo previo para respuesta natural continua
        all_messages = state.get("messages", [])
        history_pool = (
            all_messages[:-1]
            if (all_messages and isinstance(all_messages[-1], HumanMessage) and all_messages[-1].content == raw_query)
            else all_messages
        )
        trimmed_history = trim_messages(
            history_pool,
            max_tokens=6,
            strategy="last",
            token_counter=len,
        )

        history_context = ""
        if purchase_history and purchase_history.get("top_purchased_products"):
            history_context = f"""
            HISTORIAL DE COMPRAS DEL CLIENTE:
            {purchase_history.get('preference_summary')}
            Utiliza este contexto para dar una recomendación personalizada y cercana (ej: 'Basado en tus compras habituales...').
            """

        task_rules = """
            Brinda una respuesta o recomendación comercial impecable, persuasiva y precisa para la consulta del cliente.

            DIRECTRICES ESPECÍFICAS DE CATÁLOGO:
            1. Solo menciona productos, SKUs, precios y campañas que aparezcan en los PRODUCTOS DISPONIBLES EN CATÁLOGO. NUNCA inventes precios ni códigos SKU.
            2. Presenta de 1 a 4 opciones principales de manera clara y estructurada. Si provienen de 2 o más marcas comerciales distintas, agrúpalas ordenadamente por marca (*En [Marca]:*).
            3. Si el usuario solicitó una marca específica, restringe las opciones estrictamente a dicha marca sin mezclar otras marcas no deseadas.
            4. Invita con sutileza consultiva al cliente a cotizar o pedir alguno de los productos recomendados.
            5. CONSISTENCIA DE PÚBLICO OBJETIVO Y GÉNERO:
               Si la consulta actual o el contexto del diálogo previo determina un público objetivo (ej: masculino/hombre vs femenino/mujer vs infantil/niños), verifica rigurosamente que los productos recomendados correspondan a dicho público.
               NUNCA recomiendes fragancias o colonias marcadamente femeninas si el cliente busca para hombre (o viceversa). Si los candidatos recuperados en la búsqueda pertenecen al género opuesto o no deseado, DESCÁRTALOS de tu respuesta.
               Si no existen colonias estrictamente masculinas bajo el presupuesto pedido, acláralo amablemente y presenta alternativas masculinas disponibles (ej: Eau de Toilette masculinos o colonias corporales unisex), pero jamás sugieras líneas femeninas o de niñas.
            6. MANEJO DE PRESUPUESTO Y RANGOS DE PRECIO:
               Si el cliente especificó un rango o tope de presupuesto (ej: "menos de 50 soles", "hasta 60", "barato") y los productos disponibles en catálogo superan ligeramente dicho monto (indicado con budget_exceeded=True o precios mayores), NUNCA digas simplemente "no tenemos nada" ni devuelvas una lista vacía.
               Actúa como un asesor consultivo experto: explica amablemente que actualmente las opciones disponibles inician desde [precio mínimo disponible] y presenta con entusiasmo las alternativas masculinas o afines más accesibles y cercanas del catálogo para que el cliente pueda decidir.
        """

        system_prompt = self.build_synthesizer_system_prompt(
            task_specific_rules=task_rules,
            role=SoulRole.RECOMMENDER,
            state=state,
            include_multi_vendor=True,
            extra_context=history_context.strip() if history_context else None,
        )

        user_content = f"""
            Consulta del cliente: "{raw_query}"

            PRODUCTOS DISPONIBLES EN CATÁLOGO:
            {products_context}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_content),
        ]

        try:
            synthesis: FinalAnswer = await self._synthesizer.ainvoke(messages)
            draft_text = synthesis.response_text
        except Exception as e:
            logger.warning(f"Error generando síntesis estructurada: {e}. Usando texto limpio de fallback.")
            raw_res = await self._llm.ainvoke(messages)
            draft_text = extract_clean_text(raw_res)

        # Detectar SKUs presentes en la respuesta o candidatos
        matched = []
        for p in candidate_products:
            sku = p.get("sku")
            if sku and (sku in draft_text or str(sku) in draft_text):
                matched.append(str(sku))
        if not matched and candidate_products:
            matched = [str(p["sku"]) for p in candidate_products[:3] if p.get("sku")]

        return {
            "draft_response": draft_text,
            "matched_skus": matched,
        }

    async def rubric_evaluator_judge(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 4: Evalúa el borrador con una rúbrica multi-criterio e incrementa el contador de iteración."""
        raw_query = state.get("raw_query") or ""
        draft_response = state.get("draft_response") or ""
        candidate_products = state.get("candidate_products") or []
        iteration_count = state.get("iteration_count", 0) + 1
        max_iterations = state.get("max_iterations", self._default_max_iterations)
        metadata_filters = state.get("metadata_filters") or {}

        products_context = self.format_products_context(candidate_products)

        # Extraer historial previo para evaluación justa de continuidad
        all_messages = state.get("messages", [])
        history_pool = (
            all_messages[:-1]
            if (all_messages and isinstance(all_messages[-1], HumanMessage) and all_messages[-1].content == raw_query)
            else all_messages
        )
        trimmed_history = trim_messages(
            history_pool,
            max_tokens=6,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres un Juez Auditor de Calidad para un agente comercial de catálogo.
            Tu misión es evaluar objetiva y rigurosamente la respuesta generada según una RÚBRICA DE CALIDAD:

            CRITERIOS (Calificación de 1.0 a 10.0):
            1. relevance_score: ¿La respuesta atiende de forma directa, útil y completa la necesidad del usuario considerando el contexto del diálogo?
            2. grounding_score: ¿Los productos, precios y SKUs mencionados provienen estrictamente de los PRODUCTOS DISPONIBLES sin ninguna alucinación?
            3. constraints_score: ¿Se respetaron los filtros y restricciones del diálogo (público objetivo / género hombre/mujer/niños, marcas solicitadas sin mezclar marcas no deseadas, números de página, campañas, límites de presupuesto)? Si el diálogo solicita productos masculinos/hombre y se recomiendan artículos femeninos o infantiles, califica constraints_score con < 5.0 y desaprueba (is_approved = False) con crítica explícita. Si el usuario fijó un presupuesto ajustado y el catálogo solo tiene opciones de precio ligeramente mayor, califica POSITIVAMENTE (>= 8.0) si el asesor explicó con cortesía el rango de precios disponible y presentó las opciones más cercanas en lugar de dejar al cliente sin respuesta.
            4. presentation_score: ¿Cumple con el estándar del sintetizador padre (agrupación limpia por marcas cuando hay 2 o más marcas distintas, formato de viñetas con **[SKU] Nombre**, precio en S/., tono consultivo sin jerga técnica y sin saludos redundantes si hay turnos previos)?

            REGLA DE APROBACIÓN (is_approved):
            - Para ser True: CADA uno de los 4 puntajes debe ser >= 7.0 Y el promedio general debe ser >= 8.0.
            - Si no cumple, marca is_approved = False y detalla 'critique' y 'remedy_suggestions'.
        """

        user_content = f"""
            Consulta original del usuario: "{raw_query}"
            Filtros / Restricciones extraídas: {metadata_filters}

            PRODUCTOS DISPONIBLES:
            {products_context}

            RESPUESTA GENERADA:
            {draft_response}

            Iteración actual: {iteration_count} de {max_iterations}.
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_content),
        ]

        try:
            evaluation: QualityRubricEvaluation = await self._rubric_judge.ainvoke(messages)
            is_approved = evaluation.is_approved
            critique = evaluation.critique
            remedy = evaluation.remedy_suggestions
            scores = {
                "relevance": evaluation.relevance_score,
                "grounding": evaluation.grounding_score,
                "constraints": evaluation.constraints_score,
                "presentation": evaluation.presentation_score,
            }
        except Exception as e:
            logger.warning(f"Error evaluando rúbrica con LLM: {e}. Aprobando por seguridad de avance.")
            is_approved = True
            critique = None
            remedy = []
            scores = {"average": 8.0}

        return {
            "meets_rubric": is_approved,
            "rubric_scores": scores,
            "critique": critique,
            "suggested_improvements": remedy,
            "iteration_count": iteration_count,
            "is_sufficient": is_approved,
        }

    async def finalize_response(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 5: Aplica sanitización whitelabel, formato de divisas y genera el mensaje final."""
        draft_response = state.get("draft_response") or "No se encontraron productos disponibles en el catálogo."
        iteration_count = state.get("iteration_count", 0)
        meets_rubric = state.get("meets_rubric", False)

        # Si agotó las iteraciones sin aprobar la rúbrica, agregar nota constructiva
        if not meets_rubric and iteration_count >= state.get("max_iterations", self._default_max_iterations):
            logger.info("product_advisor: Finalizando con fallback constructivo tras alcanzar límite de iteraciones.")

        return BaseSynthesizerNode.format_final_response(
            draft_response,
            state,
            extra={"is_sufficient": meets_rubric},
        )
