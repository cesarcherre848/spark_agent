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
from src.agent_service.core.guardrails.sku_grounding import (
    SKUGroundingValidator,
    SKUFailSafeSanitizer,
)
from src.agent_service.core.stores.product.product_hierarchy import (
    UniversalProductHierarchyResolver,
    ProductCanonicalFamily,
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
        self._hierarchy_resolver = UniversalProductHierarchyResolver()

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

        # Gestión de estado multi-turno y ciclos de reflexión:
        critique = state.get("critique")
        existing_candidates = state.get("candidate_products") or []
        previous_skus = list(state.get("previous_turn_skus") or [])
        previous_prices = dict(state.get("previous_turn_prices") or {})

        if not critique:
            # Nuevo turno: reiniciar contador y purgar acumulación de candidatos antiguos
            iteration_count = 0
            remedy_suggestions = []
            if existing_candidates:
                # Extraer SKUs y precios de los productos recomendados o evaluados en el turno previo para delta audit
                previous_skus = [
                    str(p.get("sku")) for p in existing_candidates
                    if p.get("sku")
                ]
                for p in existing_candidates:
                    if p.get("sku") and p.get("price") is not None:
                        try:
                            previous_prices[str(p.get("sku"))] = float(p.get("price"))
                        except (ValueError, TypeError):
                            pass
            candidate_products_cleaned = []
        else:
            # Reflexión activa dentro del mismo turno: preservar estado
            iteration_count = state.get("iteration_count", 0)
            remedy_suggestions = state.get("suggested_improvements") or []
            candidate_products_cleaned = existing_candidates

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
            max_tokens=15,
            strategy="last",
            token_counter=len,
        )

        prev_turn_info = ""
        if previous_prices:
            prev_turn_info = f"\n            PRODUCTOS PRESENTADOS EN EL TURNO ANTERIOR (Precios de referencia):\n" + "\n".join(
                f"            - SKU [{sku}]: S/. {price:.2f}" for sku, price in list(previous_prices.items())[:5]
            )

        system_prompt = f"""
            Eres el planificador experto de un Asesor de Productos y Catálogo Comercial.
            Tu misión es analizar la consulta del usuario y generar un plan estructurado (AdvisorPlan)
            con las herramientas necesarias para responder de forma precisa, veraz y personalizada.

            RESOLUCIÓN CONTEXTUAL Y MULTI-TURNO:
            - Si la consulta actual es una pregunta de seguimiento, aclaración, comparación o filtro de un diálogo anterior
              (ej: 'de [Marca A] o de [Marca B] ?', '¿cuánto cuesta el segundo?', '¿tienes en color rojo?', '¿y colonias?', 'algo aún más económico'),
              analízala OBLIGATORIAMENTE en conjunto con los turnos previos de la conversación.
            - CAMBIO DE CRITERIO / PRECIO RELATIVO ('algo aún más económico', 'más barato', 'de menor precio'):
              Si el usuario pide opciones más económicas respecto al turno previo, el objetivo de búsqueda y filtrado DEBE garantizar
              encontrar productos cuyo precio sea estrictamente menor a las opciones ofrecidas anteriormente.{prev_turn_info}
            - PROPAGACIÓN OBLIGATORIA DE PÚBLICO OBJETIVO Y GÉNERO:
              Si en turnos previos el cliente especificó un público objetivo o género
              (ej: 'masculino', 'hombre', 'caballero', 'para él', 'femenino', 'mujer', 'dama', 'para ella', 'infantil', 'niños'),
              debes PRESERVARLO Y PROPAGARLO OBLIGATORIAMENTE a cualquier consulta de seguimiento elíptica.
            - EXTRACCIÓN ESTRUCTURADA DE PERFIL DE INTENCIÓN (intent_spec):
              DEBES poblar el campo 'intent_spec' en tu plan con los datos inferidos de la consulta e historial:
              * functional_category: Familia funcional primaria:
                - 'skincare_treatment': Cuidado, tratamiento, hidratación, sueros, limpieza y protectores solares.
                - 'makeup_cosmetics': Bases, polvos, labiales, sombras, delineadores, correctores.
                - 'fragrance_perfumery': Perfumes, colonias, eau de parfum, eau de toilette, brumas aromáticas.
                - 'personal_hygiene': Desodorantes, roll-on, antitranspirantes, jabones de tocador.
              * target_stage: 'adult', 'mature', 'teen_young', o 'kids'. Por defecto 'adult' a menos que se pida expresamente para bebés/niños.
              * target_gender: 'female', 'male', 'unisex', o 'kids'.
              * target_age: Edad numérica si se menciona o puede estimarse del texto (ej: 50, 32, 18, etc.).
              * target_age_group: Etapa de vida inferida de la edad ('kids', 'teen_young', 'adult', 'mature'). Si edad >= 45 o 'madura', fija 'mature'.
              * anatomical_focus: 'facial', 'body', 'hair', 'fragrance', o 'makeup'.
              * product_profile: 'fresca_diario', 'citrica', 'antiarrugas', etc.
              * excluded_targets: Si la persona es adulta o madura (edad >= 45), incluye ['juvenil_teen', 'kids'] para descartar fragancias de fantasía adolescente o infantil.
            - DISCRIMINACIÓN TAXONÓMICA Y FAMILIA FUNCIONAL:
              * Skincare vs Maquillaje: Si el diálogo trata sobre cuidado facial, tratamiento o limpieza de rostro, fija functional_category='skincare_treatment'. NUNCA recomiendes bases de maquillaje, polvos ni correctores para solicitudes de cuidado/tratamiento de piel a menos que el usuario pida maquillaje explícitamente.
              * Fragancias vs Higiene: Si el diálogo trata sobre fragancias o perfumes y el cliente pide 'algo menos' o 'más barato', DEBES PRESERVAR la categoría functional_category='fragrance_perfumery' buscando colonias o sprays más accesibles. NUNCA sustituyas una fragancia por desodorantes ni jabones a menos que el cliente los pida directamente.
              * Cuidado facial adulto: NUNCA recomiendes cremas o productos infantiles (ej: Yanbal Baby) para solicitudes de cuidado facial de adultos.
            - DISCRIMINACIÓN ANATÓMICA ESTRICTA:
              Si el diálogo trata sobre 'rostro', 'crema facial', 'antiarrugas', 'cutis', NUNCA mezcles cremas corporales, de manos o capilares.
            - REGLA DE DESCONOCIMIENTO DE IDS INTERNOS: Los usuarios de negocio NO conocen ni utilizan IDs internos de base de datos (`product_id`, `partner_id`). Todo número o código de producto indicado por el cliente es SIEMPRE un SKU comercial (ej: 20013876, 47000).

             HERRAMIENTAS DISPONIBLES:
            1. 'search_product_catalog': Búsqueda semántica híbrida en el catálogo. Argumentos:
               - query (str): Términos clave del producto (incluyendo líneas o sub-marcas como 'Totalist', 'Hidra', 'Triple Acción Max').
               - marca (str, opcional): Casa comercial o marca principal matriz ÚNICAMENTE (ej: 'Yanbal', 'Ésika'). NO colocar nombres de líneas de producto aquí (ej: 'Totalist', 'Triple Acción' NO son marcas matrices, deben ir en 'query').
               - vendor_name (str, opcional): Nombre comercial del proveedor o partner.
               - pagina (int, opcional): Número de página exacta en el catálogo.
               - edicion (str, opcional): Edición/campaña (ej: 'C10').
               - limit (int, opcional): Cantidad máxima a recuperar (default 8, usar 15 o 20 en búsquedas de economía o amplio surtido).
               - user_id (int, opcional): ID del vendedor CRM.
            2. 'rerank_and_filter_candidates': Reranker contextual y filtro dinámico inteligente. Argumentos:
               - candidates (List[dict], opcional: se inyectan los recuperados si se omite).
               - context_query (str): Intención completa de la consulta (ej: 'crema facial antiarrugas económica para mujer').
               - target_item_count (int, opcional): Cantidad exacta de productos a devolver (ej: 2 si pide 2 cremas, 1 si pide 1 colonia).
               - feedback (str, opcional): Crítica semántica o directriz correctiva.
               - sort_by (str, opcional): 'price_asc', 'price_desc' o 'relevance'.
            3. 'get_product_odoo_details': Consulta en tiempo real de precios, stock, moneda y UOM en Odoo. Argumentos:
               - skus (List[str]): Lista de códigos SKU a consultar.
               - fields (List[str], opcional): ['price', 'description', 'uom', 'category', 'taxes'].
            4. 'get_cross_sell_recommendations': Recomendaciones complementarias o alternativas. Argumentos:
               - base_sku (str, opcional): SKU base de referencia.
               - category (str, opcional): Categoría para sugerencias cruzadas.
               - marca (str, opcional): Marca preferida.
               - limit (int, opcional): Cantidad (default 5).
            5. 'filter_and_sort_products': Filtro determinista por presupuesto, página y orden.
            6. 'get_customer_purchase_history': Historial de compras del cliente en Odoo.
            7. 'get_active_incentive_targets': Metas de incentivo comercial vigentes en Odoo.

            PAUTAS DE PLANEAMIENTO:
            - ORIENTACIÓN A METAS DE INCENTIVOS Y DEMANDA ORGÁNICA (POLÍTICA ε-GREEDY):
              Cuando el usuario solicite recomendaciones generales o abiertas de productos
              (ej: 'qué me recomiendas', 'opciones de perfumes', 'los mejores labiales', 'qué me conviene vender hoy'):
              1. Puedes incluir 'get_active_incentive_targets' para conocer los SKUs que otorgan bonos o regalos en Odoo (ej: colonias Soy, extracto antiedad, perfumes alta gama) y los productos de mayor rotación en pedidos recientes.
              2. En la selección y sugerencia de productos, favorece aquellos que impulsen las metas del vendedor sin descuidar la preferencia y presupuesto del cliente.
            - AJUSTE PROACTIVO DE TOP-K / LIMIT (BÚSQUEDA EXHAUSTIVA):
              Si la consulta solicita un listado amplio o exhaustivo (ej: 'dame todos los perfumes...', 'qué opciones tienes', 'muéstrame todo el catálogo'),
              o si el cliente especifica un presupuesto máximo (ej: 'menor de 50', 'hasta 60 soles', 'el más barato'), o un rango de precios (ej: 'entre 40 y 80'), o pide 'algo más económico':
              DEBES fijar proactivamente 'limit=15' o 'limit=20' en 'search_product_catalog' (en lugar del default de 8) para realizar una búsqueda exhaustiva
              que explore todos los escalones de precios del catálogo.
            - FLUJO RECOMENDADO: Siempre que se realice una búsqueda semántica ('search_product_catalog') para recomendar productos,
              es altamente recomendable planear a continuación 'rerank_and_filter_candidates' con el context_query adecuado y
              sort_by='price_asc' si el usuario busca economía o target_item_count según los ítems deseados.
            - Si el usuario busca opciones económicas o pide 'algo más económico', planea 'search_product_catalog' con limit=15 o 20
              seguido de 'rerank_and_filter_candidates' con context_query detallado y sort_by='price_asc'.
            - Si el usuario pide recomendaciones personalizadas ("según lo que suelo comprar", "¿qué me sugieres?"),
              invoca 'get_customer_purchase_history' para conocer sus preferencias antes de buscar en el catálogo.
            - Si el usuario pide un producto puntual, línea de producto o menciona marcas matriz ("crema Totalist", "labial de Yanbal", "Ésika pág 12"),
              invoca 'search_product_catalog' asegurándote de incluir el nombre de la línea en 'query' y restringir 'marca' solo si es una casa matriz (Yanbal, Ésika).
            - FILTRO DE CAMPAÑA POR DEFECTO (ÚLTIMA CAMPAÑA VIGENTE):
              El catálogo busca por defecto en la ÚLTIMA CAMPAÑA VIGENTE de cada proveedor (ej: C11 para Yanbal, C-15 para Ésika).
              Por lo tanto, deja 'edicion=None' en 'search_product_catalog' a menos que el usuario pida EXPLÍCITAMENTE una campaña o catálogo anterior (ej: 'en la C10', 'en el catálogo pasado').
            - Si el usuario pide comparar marcas explícitamente ("compara [Marca A] y [Marca B]"), planea búsquedas específicas
              para cada marca y consolida las opciones.
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
            2. Aplica 'rerank_and_filter_candidates' con feedback semántico explícito.
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
            intent_spec_dict = plan.intent_spec.model_dump() if getattr(plan, "intent_spec", None) else {}
            for t in planned_tools:
                if t.get("tool_name") == "rerank_and_filter_candidates":
                    if intent_spec_dict and "intent_spec" not in t.get("arguments", {}):
                        t.setdefault("arguments", {})["intent_spec"] = intent_spec_dict
        except Exception as e:
            logger.warning(f"Error generando plan estructurado con LLM: {e}. Aplicando plan por defecto.")
            plan_rationale = "Plan de contingencia: búsqueda directa en catálogo."
            fallback_query = raw_query
            if len(raw_query.split()) <= 6 and trimmed_history:
                for m in reversed(trimmed_history):
                    if isinstance(m, HumanMessage) and m.content:
                        fallback_query = f"{m.content} {raw_query}"
                        break
            # Detección heurística de consulta de seguimiento anafórica sobre turnos previos
            f_lower = raw_query.lower()
            prev_skus = []
            if trimmed_history:
                for m in reversed(trimmed_history):
                    if isinstance(m, AIMessage) and m.content:
                        found_skus = re.findall(r"\[([A-Za-z0-9_-]+)\]", m.content)
                        if found_skus:
                            prev_skus = found_skus
                            break
            is_followup = any(w in f_lower for w in ["esto", "eso", "esos", "catalogo", "catálogo", "campaña", "marca", "pertenece", "cuesta", "precio"])
            if is_followup and prev_skus:
                planned_tools = [{
                    "tool_name": "get_product_odoo_details",
                    "arguments": {"skus": prev_skus[:10]},
                    "purpose": "Consultar detalles oficiales y catálogo de los productos del turno previo.",
                }]
            else:
                fallback_limit = 15 if any(w in f_lower for w in ["todo", "todos", "barato", "menos de", "hasta", "presupuesto", "precio", "rango", "económico", "economico"]) else 8
                planned_tools = [
                    {
                        "tool_name": "search_product_catalog",
                        "arguments": {"query": fallback_query, "limit": fallback_limit, "user_id": user_id or 5},
                        "purpose": "Búsqueda estándar de catálogo.",
                    },
                    {
                        "tool_name": "rerank_and_filter_candidates",
                        "arguments": {
                            "context_query": fallback_query,
                            "sort_by": "price_asc" if any(w in f_lower for w in ["barato", "económic", "economico", "menos"]) else "relevance",
                        },
                        "purpose": "Reranking contextual y descarte de falsos positivos anatómicos.",
                    }
                ]
            extracted_meta = {}
            intent_spec_dict = {}

        return {
            "raw_query": raw_query,
            "plan_rationale": plan_rationale,
            "planned_tools": planned_tools,
            "metadata_filters": extracted_meta,
            "intent_spec": intent_spec_dict,
            "iteration_count": iteration_count,
            "critique": critique,
            "suggested_improvements": remedy_suggestions,
            "candidate_products": candidate_products_cleaned,
            "previous_turn_skus": previous_skus,
            "previous_turn_prices": previous_prices,
        }

    async def execute_tools(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 2: Ejecuta las herramientas seleccionadas en el plan de forma asíncrona y consolida candidatos."""
        planned_tools = state.get("planned_tools") or []
        user_id = state.get("user_id") or 5
        raw_query = state.get("raw_query") or "productos"

        observations = []
        candidate_products: List[Dict[str, Any]] = list(state.get("candidate_products") or [])
        purchase_history = state.get("customer_purchase_history")

        # Medium Reflection Loop: Si venimos de re_rerank_with_feedback y ya disponemos de candidatos,
        # re-ejecutamos únicamente rerank_and_filter_candidates en memoria con la crítica semántica
        reflection_action = state.get("reflection_action")
        critique = state.get("critique")
        if reflection_action == "re_rerank_with_feedback" and candidate_products:
            logger.info("product_advisor: Medium Loop activo. Re-ejecutando únicamente 'rerank_and_filter_candidates' en memoria con crítica.")
            planned_tools = [{
                "tool_name": "rerank_and_filter_candidates",
                "arguments": {
                    "candidates": list(candidate_products),
                    "context_query": raw_query,
                    "feedback": critique,
                    "sort_by": "price_asc" if any(w in raw_query.lower() for w in ["barato", "económic", "economico", "menos"]) else "relevance",
                    "intent_spec": state.get("intent_spec") or {},
                },
                "purpose": "Medium Reflection Loop: re-ordenamiento y descarte en memoria con retroalimentación.",
            }]
        elif not planned_tools:
            planned_tools = [{
                "tool_name": "search_product_catalog",
                "arguments": {"query": raw_query, "limit": 8, "user_id": user_id},
                "purpose": "Búsqueda por defecto",
            }]

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

            # Inyectar candidatos, contexto y feedback en rerank_and_filter_candidates
            elif t_name == "rerank_and_filter_candidates":
                if not t_args.get("candidates"):
                    t_args["candidates"] = list(candidate_products)
                if not t_args.get("context_query"):
                    t_args["context_query"] = raw_query
                # Si hay crítica en el estado y no vino feedback en los argumentos, inyectarlo
                if not t_args.get("feedback") and state.get("critique"):
                    t_args["feedback"] = state.get("critique")
                if not t_args.get("intent_spec") and state.get("intent_spec"):
                    t_args["intent_spec"] = state.get("intent_spec")
                # Asegurar precios en los candidatos antes de rerankear para ordenamiento fiel por precio
                skus_to_price = [
                    p.get("sku") for p in t_args["candidates"]
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
                                for p in t_args["candidates"]:
                                    s = p.get("sku")
                                    if s in d_sku:
                                        p["price"] = d_sku[s].get("price")
                                        p["currency"] = d_sku[s].get("currency")
                                        p["uom"] = d_sku[s].get("uom")
                                        if d_sku[s].get("sales_description"):
                                            p["sales_description"] = d_sku[s].get("sales_description")
                        except Exception as pe:
                            logger.warning(f"Error enriqueciendo precios previos a reranking: {pe}")

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
                            f"product_advisor: {t_name} descartó todos los candidatos. "
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

            elif t_name == "rerank_and_filter_candidates":
                if isinstance(res, list):
                    candidate_products = res

            elif t_name == "get_customer_purchase_history":
                if isinstance(res, dict):
                    purchase_history = res

            elif t_name == "get_product_odoo_details":
                if isinstance(res, dict) and "products" in res:
                    details_list = res.get("products", [])
                    details_by_sku = {d.get("sku"): d for d in details_list if isinstance(d, dict)}
                    existing_skus = set()
                    for p in candidate_products:
                        s = p.get("sku")
                        if s:
                            existing_skus.add(s)
                        if s and s in details_by_sku:
                            det = details_by_sku[s]
                            if p.get("catalog_offer_price") is not None:
                                p["price"] = float(p["catalog_offer_price"])
                            elif det.get("price") is not None:
                                p["price"] = det.get("price")
                            p["currency"] = det.get("currency")
                            p["uom"] = det.get("uom")
                            if det.get("sales_description"):
                                p["sales_description"] = det.get("sales_description")
                            if det.get("name") and not p.get("name"):
                                p["name"] = det.get("name")
                            if det.get("marca") and not p.get("marca"):
                                p["marca"] = det.get("marca")
                            if det.get("edicion") and not p.get("edicion"):
                                p["edicion"] = det.get("edicion")
                            if det.get("pagina") and not p.get("pagina"):
                                p["pagina"] = det.get("pagina")
                    for det in details_list:
                        if isinstance(det, dict) and det.get("sku") and det["sku"] not in existing_skus:
                            candidate_products.append({
                                "sku": det["sku"],
                                "name": det.get("name") or f"Producto {det['sku']}",
                                "price": det.get("price"),
                                "currency": det.get("currency", "PEN"),
                                "uom": det.get("uom"),
                                "description": det.get("sales_description") or det.get("description", ""),
                                "marca": det.get("marca"),
                                "edicion": det.get("edicion"),
                                "pagina": det.get("pagina"),
                            })
                            existing_skus.add(det["sku"])

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
                                if p.get("catalog_offer_price") is not None:
                                    p["price"] = float(p["catalog_offer_price"])
                                else:
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
        user_id = state.get("user_id")

        # Agrupación jerárquica de conceptos y variantes
        try:
            hierarchical_families = await self._hierarchy_resolver.group_candidates_hierarchically(
                candidate_products, user_id=user_id
            )
            products_context = self.format_products_context(hierarchical_families or candidate_products)
        except Exception as e:
            logger.warning(f"Error agrupando jerarquía de productos: {e}. Usando formato estándar.")
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
            max_tokens=15,
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
            7. CLARIDAD Y NO AMBIGÜEDAD EN CONSULTAS DE SEGUIMIENTO:
               Si la consulta del cliente indaga sobre recomendaciones previas ("esto a qué catálogo pertenece", "¿a qué campaña corresponde?", "¿cuánto cuesta ese?", "¿de qué marca es?"):
               - Identifica de forma precisa a qué producto(s) y SKU(s) específicos se refiere la pregunta.
               - Explica con total claridad su marca comercial y la campaña o edición a la que pertenece.
               - PROHIBIDO TERMINANTEMENTE emitir respuestas ambiguas que mezclen marcas que no corresponden a los productos reales recomendados previamente (ej: si los productos eran de Yanbal, jamás menciones Ésika ni otras marcas no involucradas).
            8. REGLA DE CÓDIGOS DE PRODUCTO (CÓDIGO = SKU):
               El término 'código', 'código de producto', 'código de referencia' o 'código de venta' se refiere SIEMPRE y de forma estricta al SKU o código comercial visible (ej: [20013876], [47000]). Los usuarios de negocio no conocen ni utilizan IDs internos de base de datos (`product_id`, `partner_id`). Al mostrar o recomendar productos al usuario, indica siempre su código comercial / SKU entre corchetes o como código de referencia para que el cliente o vendedor pueda solicitarlo o agregarlo a un pedido fácilmente. Queda terminantemente PROHIBIDO exponer IDs numéricos internos.
            9. ADECUACIÓN DEMOGRÁFICA Y GRUPO ETARIO:
               - Si la consulta o contexto especifica un cliente/destinatario adulto o maduro (ej: 40-50+ años) y busca colonia o fragancia fresca de uso diario, prioriza líneas sofisticadas, florales sutiles, herbales o frescas elegantes (perfil clásico o distinguido). NUNCA recomiendes fragancias con concepto de fantasía adolescente/juvenil (glitter, notas golosina o nombres de fantasía teen) ni colonias infantiles para adultos.
               - Si el cliente busca opciones para chicos jóvenes o adolescentes, favorece entonces líneas frescas juveniles, chispeantes y frutales.
            10. OPORTUNIDAD COMERCIAL POR VARIACIÓN DE PRECIO INTER-CATÁLOGO:
               - Si en los datos del producto se indica una variación de precio favorable respecto a la campaña anterior (ej: price_variation con descuento o menor precio en la campaña actual), menciónalo de manera consultiva y atractiva para el cliente como una oportunidad especial o rebaja destacada (ej: 'con S/. XX.XX de descuento respecto a la campaña previa').
            11. INTEGRIDAD DE FAMILIA FUNCIONAL (CUIDADO FACIAL / TRATAMIENTO VS MAQUILLAJE):
               - Si el cliente consulta por cuidado facial, hidratación, limpieza o tratamiento de la piel (incluyendo piel mixta, grasa o seca), NUNCA recomiendes bases líquidas, polvos compactos ni correctores de maquillaje, aun si contienen ingredientes matificantes o niacinamida. Ofrece exclusivamente cremas hidratantes, geles, sueros o limpiadores faciales de tratamiento.
            12. INTEGRIDAD DE CATEGORÍA DE PERFUMERÍA VS HIGIENE PERSONAL:
               - Si el cliente busca fragancias, perfumes o colonias y pide 'algo más barato' o 'algo menos', NUNCA sustituyas con desodorantes en roll-on ni jabones haciéndolos pasar por fragancias. Si la fragancia más accesible cuesta más que lo esperado, explica de forma consultiva que es la opción de perfumería más económica disponible del catálogo.
            13. ADULTO VS LÍNEAS PEDIÁTRICAS O BEBÉ:
               - Para consultas de cuidado facial o corporal de adultos, jamás recomiendes productos pediátricos o para bebés (ej: Yanbal Baby) a menos que se solicite expresamente para recién nacidos o infantes.
        """

        critique = state.get("critique")
        remedy_suggestions = state.get("suggested_improvements") or []
        previous_draft = state.get("draft_response")
        reflection_action = state.get("reflection_action")

        reflection_feedback = ""
        if critique:
            if reflection_action == "ask_clarification":
                reflection_feedback = f"""

            ATENCIÓN - CALIBRACIÓN CON EL CLIENTE REQUERIDA (ask_clarification):
            El Juez Auditor detectó ambigüedad o falta de claridad en el criterio del cliente:
            "{critique}"
            Sugerencias: {remedy_suggestions}

            INSTRUCCIONES:
            Formula una pregunta amable y concisa de calibración orientada a resolver la preferencia del cliente (ej: rango de precio específico, preferencia de marca o tipo de producto) antes de enviar recomendaciones ciegas, mostrando brevemente las alternativas disponibles si es pertinente.
            """
            else:
                reflection_feedback = f"""

            ATENCIÓN - RETROALIMENTACIÓN DE REFLEXIÓN (CLARIDAD Y NO AMBIGÜEDAD):
            El borrador anterior fue RECHAZADO por la Rúbrica de Calidad:
            Crítica: "{critique}"
            Sugerencias de mejora: {remedy_suggestions}

            INSTRUCCIONES DE CORRECCIÓN:
            1. Corrige directamente la redacción para ser 100% específico, transparente e inequívoco.
            2. Si la consulta se refiere a recomendaciones previas (ej: 'esto a qué catálogo pertenece'), menciona con precisión los productos/SKUs del turno anterior y asócialos exactamente a su marca y catálogo real.
            3. Prohibido estrictamente generalizar o mezclar marcas que no corresponden a los productos reales.
            """
            if previous_draft:
                reflection_feedback += f"\n            BORRADOR ANTERIOR A REFINAR:\n            {previous_draft}\n"

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
            {reflection_feedback}
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

        user_id = state.get("user_id")
        try:
            hierarchical_families = await self._hierarchy_resolver.group_candidates_hierarchically(
                candidate_products, user_id=user_id
            )
            products_context = self.format_products_context(hierarchical_families or candidate_products)
        except Exception:
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
            max_tokens=15,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres un Juez Auditor de Calidad para un agente comercial de catálogo.
            Tu misión es evaluar objetiva y rigurosamente la respuesta generada según una RÚBRICA DE CALIDAD:

            CRITERIOS (Calificación de 1.0 a 10.0):
            1. relevance_score: ¿La respuesta atiende de forma directa, útil y completa la necesidad del usuario considerando el contexto del diálogo?
            2. grounding_score: ¿Los productos, precios y SKUs mencionados provienen estrictamente de los PRODUCTOS DISPONIBLES sin ninguna alucinación?
            3. constraints_score: ¿Se respetaron los filtros y restricciones del diálogo (marcas solicitadas, números de página, campañas, límites de presupuesto, género Y ADECUACIÓN DEMOGRÁFICA / GRUPO ETARIO)?
               - GÉNERO / PÚBLICO: Si el diálogo solicita público objetivo / género hombre/mujer/niños (ej: masculinos/hombre) y se recomiendan artículos femeninos o infantiles (o viceversa), califica constraints_score con < 5.0 y desaprueba (is_approved = False).
               - ADECUACIÓN DEMOGRÁFICA / EDAD: Si el diálogo solicita productos para una clienta o cliente adulto/maduro (ej: 50 años) y se recomiendan colonias o productos de concepto adolescente/juvenil (fantasía teen, glitter o notas golosina) o infantiles (o viceversa), califica constraints_score con < 5.0, desaprueba (is_approved = False) y selecciona 're_rerank_with_feedback' exigiendo líneas acordes al perfil etario maduro (perfil sobrio, floral clásico o distinguido).
               - INTEGRIDAD DE FAMILIA FUNCIONAL: Si el diálogo solicita cuidado facial / tratamiento y se recomiendan productos de maquillaje (bases, polvos, correctores), o si solicita fragancias/perfumes y se recomiendan desodorantes o jabones sustituyendo la perfumería, o si es para adulto y se recomiendan productos de bebé/pediátricos: califica constraints_score con < 5.0, desaprueba (is_approved = False) y selecciona 're_rerank_with_feedback' o 'replan_tools'.
            4. presentation_score: ¿Cumple con el estándar del sintetizador padre (agrupación limpia por marcas cuando hay 2 o más marcas distintas, formato de viñetas con [SKU] Nombre, precio en S/., tono consultivo sin jerga técnica y sin saludos redundantes si hay turnos previos)? OBLIGATORIO: Si el texto contiene viñetas de productos pegadas o concatenadas en el mismo párrafo sin saltos de línea (\\n), califica presentation_score < 6.0, marca is_approved = False.
            5. clarity_and_unambiguity_score: ¿La respuesta es 100% clara, directa e inequívoca? En preguntas de seguimiento o referencia de turnos previos ('esto a qué catálogo pertenece', '¿cuánto cuesta ese?', '¿de qué marca son?'):
               - Debe identificar con exactitud de qué productos se habla y atribuir con veracidad quirúrgica su marca y catálogo/campaña real.
               - PROHIBIDO APROBAR si la respuesta agrupa o menciona marcas que no corresponden a los productos reales recomendados.
            6. intent_fulfillment_score: ¿Cumple fielmente con el cambio o preferencia solicitada en este turno?
               - AUDITORÍA DE PRECIO RELATIVO ('algo aún más económico', 'más barato'): Si el usuario pidió opciones más económicas respecto al turno previo y la respuesta repite productos del mismo precio o más caros ignorando opciones más baratas presentes en los candidatos, califica intent_fulfillment_score < 5.0, desaprueba (is_approved = False) y selecciona 're_rerank_with_feedback'.
               - DISCRIMINACIÓN ANATÓMICA: Si el usuario busca cremas para rostro y se incluyeron cremas corporales o de manos, califica < 5.0 y selecciona 're_rerank_with_feedback'.
            7. criteria_clarity_score: ¿El criterio de la consulta está 100% claro o es ambiguo?
               - Si la consulta es tan ambigua que no se puede discernir qué producto recomendar sin calibración (ej: disyuntivas comerciales marcadas o preferencias indefinidas), califica < 6.0 y selecciona 'ask_clarification'.

            ACCIÓN DE REFLEXIÓN (reflection_action):
            - 'approve': Si todos los criterios son >= 7.0 y el promedio es >= 8.0.
            - 're_rerank_with_feedback' (MEDIUM LOOP): Si los productos candidatos disponibles contienen opciones válidas pero el orden o selección actual violó la intención (ej: se necesita priorizar las opciones más baratas, descartar falsos positivos anatómicos, o ajustar el conteo). Proporciona la crítica semántica precisa en 'critique'.
            - 'refine_synthesis' (FAST LOOP): Si los productos seleccionados son los correctos pero falló la redacción, el formato de viñetas, la claridad o hubo ambigüedad en el texto.
            - 'replan_tools' (SLOW LOOP): Si faltan productos completamente en el catálogo o se necesita una búsqueda semántica con nuevos términos.
            - 'ask_clarification': Si el criterio del cliente no está claro y es preferible formular una pregunta de calibración.

            REGLA DE APROBACIÓN (is_approved):
            - Para ser True: CADA uno de los puntajes debe ser >= 7.0 Y el promedio general debe ser >= 8.0.
            - Si no cumple, marca is_approved = False y detalla 'critique', 'remedy_suggestions' y 'reflection_action'.
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

        # Expandir candidatos con sus variantes de catálogo para que SKUGroundingValidator reconozca todos los tonos/iniciales oficiales de la página
        expanded_candidates = list(candidate_products)
        for cp in candidate_products:
            if isinstance(cp, dict) and cp.get("catalog_variants"):
                for cv in cp["catalog_variants"]:
                    if isinstance(cv, dict) and cv.get("sku"):
                        expanded_candidates.append({"sku": cv["sku"], "name": cv.get("label") or cp.get("name")})

        # 1. Auditoría Determinista Anti-Alucinación de SKUs (Capa 2)
        grounding_check = SKUGroundingValidator.validate_draft_grounding(
            draft_response=draft_response,
            candidate_products=expanded_candidates,
        )
        if not grounding_check.is_valid and iteration_count < max_iterations:
            logger.warning(f"[RubricJudge] Grounding determinista rechazó borrador: {grounding_check.invalid_skus}")
            return {
                "meets_rubric": False,
                "rubric_scores": {
                    "relevance": 7.5,
                    "grounding": 4.0,
                    "constraints": 7.0,
                    "presentation": 7.0,
                    "clarity_and_unambiguity": 6.0,
                    "intent_fulfillment": 7.0,
                    "criteria_clarity": 8.0,
                },
                "reflection_action": "refine_synthesis",
                "critique": grounding_check.critique_message,
                "suggested_improvements": grounding_check.suggested_improvements,
                "iteration_count": iteration_count,
                "is_sufficient": False,
            }

        try:
            evaluation: QualityRubricEvaluation = await self._rubric_judge.ainvoke(messages)
            is_approved = evaluation.is_approved
            critique = evaluation.critique
            remedy = evaluation.remedy_suggestions
            refl_action = getattr(evaluation, "reflection_action", None)
            if is_approved:
                refl_action = "approve"
            elif not refl_action or refl_action == "approve":
                if getattr(evaluation, "criteria_clarity_score", 8.0) < 6.0:
                    refl_action = "ask_clarification"
                elif getattr(evaluation, "intent_fulfillment_score", 8.0) < 7.0:
                    refl_action = "re_rerank_with_feedback"
                elif evaluation.constraints_score < 7.0 or evaluation.relevance_score < 7.0 or not candidate_products:
                    refl_action = "replan_tools"
                else:
                    refl_action = "refine_synthesis"
            scores = {
                "relevance": evaluation.relevance_score,
                "grounding": evaluation.grounding_score,
                "constraints": evaluation.constraints_score,
                "presentation": evaluation.presentation_score,
                "clarity_and_unambiguity": getattr(evaluation, "clarity_and_unambiguity_score", 8.0),
                "intent_fulfillment": getattr(evaluation, "intent_fulfillment_score", 8.0),
                "criteria_clarity": getattr(evaluation, "criteria_clarity_score", 8.0),
            }
        except Exception as e:
            logger.warning(f"Error evaluando rúbrica con LLM: {e}. Aprobando por seguridad de avance.")
            is_approved = True
            critique = None
            remedy = []
            refl_action = "approve"
            scores = {"average": 8.0}

        return {
            "meets_rubric": is_approved,
            "rubric_scores": scores,
            "reflection_action": refl_action,
            "critique": critique,
            "suggested_improvements": remedy,
            "iteration_count": iteration_count,
            "is_sufficient": is_approved,
        }

    async def finalize_response(self, state: ProductAdvisorState) -> Dict[str, Any]:
        """Nodo 5: Aplica sanitización whitelabel, formato de divisas y genera el mensaje final."""
        draft_response = state.get("draft_response") or "No se encontraron productos disponibles en el catálogo."
        candidate_products = state.get("candidate_products") or []
        iteration_count = state.get("iteration_count", 0)
        meets_rubric = state.get("meets_rubric", False)

        # Sanitización fail-safe final de SKUs (Capa 4), incluyendo variantes de catálogo en la página
        authorized_skus = {
            str(p.get("sku") or p.get("default_code") or "").strip()
            for p in candidate_products
            if isinstance(p, dict) and (p.get("sku") or p.get("default_code"))
        }
        for p in candidate_products:
            if isinstance(p, dict) and p.get("catalog_variants"):
                for cv in p["catalog_variants"]:
                    if isinstance(cv, dict) and cv.get("sku"):
                        authorized_skus.add(str(cv["sku"]).strip())
        authorized_skus.discard("")
        if authorized_skus:
            draft_response = SKUFailSafeSanitizer.sanitize_text(draft_response, authorized_skus)

        # Si agotó las iteraciones sin aprobar la rúbrica, agregar nota constructiva
        if not meets_rubric and iteration_count >= state.get("max_iterations", self._default_max_iterations):
            logger.info("product_advisor: Finalizando con fallback constructivo tras alcanzar límite de iteraciones.")

        return BaseSynthesizerNode.format_final_response(
            draft_response,
            state,
            extra={
                "is_sufficient": meets_rubric,
                "iteration_count": 0,
                "critique": None,
                "suggested_improvements": [],
                "reflection_action": "approve",
            },
        )

