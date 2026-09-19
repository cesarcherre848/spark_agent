"""
src/agent_service/graph/sub_graphs/product_recomender/nodes.py - Nodos ejecutores de product_recomender

Implementa el ciclo de recomendación relacional con:
1. Extracción de intención, filtros y top-k dinámico.
2. Búsqueda híbrida con regla de sobre-recuperación 2x (k >= max(top_k * 2, 6)).
3. Enriquecimiento oficial en Odoo ERP (precios de lista, stock, ficha técnica).
4. Filtrado determinista de usuario (presupuesto, categoría, exclusión de auto-recomendación).
5. Evaluación objetiva frente a Rúbrica de 4 dimensiones (Evaluator-Optimizer).
6. Auto-reflexión y reformulación en caso de no cumplir la rúbrica (hasta max_iterations).
7. Síntesis ejecutiva comercial para el cliente/vendedor.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.documents import Document

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.schemas import ProductCatalogFilter
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.sales_tools import get_shared_odoo_client
from src.agent_service.graph.sub_graphs.product_recomender.state import ProductRecomenderState
from src.agent_service.graph.sub_graphs.product_recomender.schemas import (
    RecommendationIntentExtraction,
    EnrichedRecommendedProduct,
    RubricEvaluationResult,
    RecommendationSynthesisResponse,
)

from src.agent_service.soul import inject_soul, SoulRole

logger = logging.getLogger(__name__)


class ProductRecomenderNodes:
    """Nodos del subgrafo de recomendación de productos con patrón de reflexión y rúbrica."""

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: Optional[ProductVectorStore] = None,
        odoo_client: Optional[OdooClient] = None,
        default_top_k: int = 15,
        default_max_iterations: int = 2,
    ):
        self._llm = llm
        self._vector_store = vector_store
        self._odoo_client = odoo_client
        self._default_top_k = default_top_k
        self._default_max_iterations = default_max_iterations

        # Modelos estructurados para cada tarea del subgrafo
        self._intent_extractor = bind_structured_output(
            bind_temperature(llm, 0.0), RecommendationIntentExtraction
        )
        self._rubric_judge = bind_structured_output(
            bind_temperature(llm, 0.0), RubricEvaluationResult
        )
        self._refinement_llm = bind_temperature(llm, 0.2)
        self._synthesizer = bind_structured_output(
            bind_temperature(llm, 0.35), RecommendationSynthesisResponse
        )

    async def extract_recommendation_intent(self, state: ProductRecomenderState) -> dict:
        """Nodo 1: Extrae la intención, producto base, filtros y el top-k dinámico solicitado."""
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
            Eres un asesor comercial experto en catálogo y recomendaciones de productos.
            Tu misión es analizar la consulta del usuario y extraer de forma estructurada:
            
            1. 'base_product': Código SKU (ej: '6189', '5104') o nombre comercial del producto de partida si el usuario
               busca algo que combine, complemente o reemplace a uno específico. Si no se indica producto base, deja None.
            2. 'relation_type':
               - 'cross_sell': si busca productos complementarios que combinen o se usen juntos (ej: 'qué combina con Sexy Glam', 'recomiéndame un fijador para este maquillaje').
               - 'up_sell': si busca una alternativa superior, más duradera, premium o de mayor gama.
               - 'substitute': si busca un reemplazo directo o equivalente.
               - 'general_recommendation': si busca productos basados en una necesidad o categoría sin producto base (ej: 'qué me recomiendas para cejas').
            3. 'top_k': Número exacto de recomendaciones solicitadas por el usuario (ej: 'dame 2 opciones' -> 2, 'muéstrame 5 productos' -> 5).
               Si el usuario NO especificó un número, asigna 15 por defecto.
            4. 'min_price' y 'max_price': Rango de presupuesto si se mencionó (ej: 'menos de 100 soles' -> max_price=100.0).
            5. 'category': Categoría o tipo de producto solicitado (ej: 'labiales', 'delineadores', 'sombras').
            6. 'required_attributes': Lista de beneficios o atributos explícitos (ej: ['a prueba de agua', 'mate', 'larga duración']).
            7. 'search_query': Consulta de búsqueda semántica limpia para catálogo (solo términos descriptivos y técnicos, sin saludos ni rodeos).
            8. 'sort_by': Criterio de ordenamiento si el usuario lo pide implícita o explícitamente:
               - 'price_asc': si pide 'más baratos', 'económicos', 'menor precio', 'desde el más bajo', 'gangas'.
               - 'price_desc': si pide 'más caros', 'premium', 'alta gama', 'mayor precio'.
               - 'relevance': orden por defecto si no se piden superlativos de precio.
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=f"Consulta de recomendación:\n{raw_query}"),
        ]

        extraction: Optional[RecommendationIntentExtraction] = await self._intent_extractor.ainvoke(messages)
        if extraction is None or not isinstance(extraction, RecommendationIntentExtraction):
            if isinstance(extraction, dict):
                extraction = RecommendationIntentExtraction(**extraction)
            else:
                extraction = RecommendationIntentExtraction(
                    search_query=raw_query or "productos recomendados",
                    top_k=self._default_top_k,
                )

        target_top_k = int(extraction.top_k or self._default_top_k)
        if target_top_k < 1:
            target_top_k = self._default_top_k

        sort_by_val = getattr(extraction, "sort_by", "relevance") or "relevance"

        filters_dict = {
            "min_price": extraction.min_price,
            "max_price": extraction.max_price,
            "category": extraction.category,
            "required_attributes": extraction.required_attributes,
        }

        return {
            "base_product": extraction.base_product,
            "relation_type": extraction.relation_type,
            "top_k": target_top_k,
            "sort_by": sort_by_val,
            "filters": filters_dict,
            "search_query": extraction.search_query,
            "iteration_count": 0,
            "max_iterations": self._default_max_iterations,
        }

    async def retrieve_candidate_products(self, state: ProductRecomenderState) -> dict:
        """Nodo 2: Recupera candidatos con la Regla 3x de sobre-recuperación (k >= max(top_k * 3, 6))."""
        search_query = state.get("search_query") or state.get("raw_query") or ""
        top_k = int(state.get("top_k") or self._default_top_k)

        # Regla 3x para garantizar suficientes candidatos antes de filtrar por precio, categoría y rúbrica
        retrieval_k = max(top_k * 3, 6)

        user_id = state.get("user_id")
        user_id_int = int(user_id) if user_id and str(user_id).isdigit() else None
        catalog_filter = ProductCatalogFilter(user_id=user_id_int) if user_id_int is not None else None

        docs: List[Document] = []
        if self._vector_store:
            try:
                docs = await self._vector_store.ahybrid_search(
                    query=search_query,
                    k=retrieval_k,
                    filters=catalog_filter,
                )
            except Exception as e:
                logger.error(f"Error al ejecutar ahybrid_search en ProductVectorStore: {e}")
                docs = []

        return {"candidate_documents": docs}

    async def enrich_product_details(self, state: ProductRecomenderState) -> dict:
        """Nodo 3: Cruza los candidatos con Odoo ERP para asociar precios oficiales, moneda y ficha técnica."""
        docs: List[Document] = state.get("candidate_documents", [])
        if not docs:
            return {"enriched_products": []}

        # Extraer SKUs únicos de los documentos
        skus_list: List[str] = []
        doc_map_by_sku: Dict[str, Document] = {}
        for d in docs:
            sku = d.metadata.get("sku")
            if sku:
                sku_str = str(sku).strip()
                if sku_str not in doc_map_by_sku:
                    doc_map_by_sku[sku_str] = d
                    skus_list.append(sku_str)

        odoo_client = self._odoo_client or get_shared_odoo_client()
        odoo_products_map: Dict[str, Any] = {}
        if skus_list and hasattr(odoo_client, "get_products_by_skus"):
            try:
                prod_res = await odoo_client.get_products_by_skus(
                    skus=skus_list,
                    fields=["price", "description", "category", "uom", "barcode"],
                )
                for p in getattr(prod_res, "products", []):
                    odoo_products_map[str(p.sku).strip()] = p
            except Exception as e:
                logger.warning(f"Error al consultar detalles en Odoo para SKUs {skus_list}: {e}")

        relation_type = state.get("relation_type", "general_recommendation")
        enriched: List[Dict[str, Any]] = []

        for sku, doc in doc_map_by_sku.items():
            odoo_prod = odoo_products_map.get(sku)
            name = (
                getattr(odoo_prod, "name", None)
                or doc.metadata.get("name")
                or f"Producto SKU {sku}"
            )
            price = getattr(odoo_prod, "price", None)
            currency = getattr(odoo_prod, "currency", None) or "PEN"
            category = getattr(odoo_prod, "category", None) or doc.metadata.get("category")
            uom = getattr(odoo_prod, "uom", None) or "Units"
            desc = (
                getattr(odoo_prod, "sales_description", None)
                or doc.page_content
                or ""
            )
            pid = getattr(odoo_prod, "product_id", None) or doc.metadata.get("product_id")

            enriched.append({
                "product_id": pid,
                "sku": sku,
                "name": name,
                "price": price,
                "currency": currency,
                "category": category,
                "uom": uom,
                "description": desc.strip(),
                "relation_type": relation_type,
            })

        return {"enriched_products": enriched}

    async def apply_user_filters(self, state: ProductRecomenderState) -> dict:
        """Nodo 4: Aplica filtros deterministas (presupuesto, categoría, exclusión de producto base y ordenamiento)."""
        enriched: List[Dict[str, Any]] = state.get("enriched_products", [])
        filters = state.get("filters", {}) or {}
        base_product = state.get("base_product")
        sort_by = state.get("sort_by") or "relevance"

        min_price = filters.get("min_price")
        max_price = filters.get("max_price")
        cat_filter = str(filters.get("category") or "").strip().lower()

        filtered: List[Dict[str, Any]] = []
        for p in enriched:
            p_sku = str(p.get("sku", "")).strip().lower()
            p_name = str(p.get("name", "")).strip().lower()
            p_price = p.get("price")
            p_cat = str(p.get("category") or "").strip().lower()
            p_desc = str(p.get("description") or "").strip().lower()

            # 1. Descartar auto-recomendación (el producto base no debe recomendarse a sí mismo)
            if base_product:
                base_clean = str(base_product).strip().lower()
                if p_sku == base_clean or base_clean in p_name:
                    continue

            # 2. Filtro de presupuesto máximo
            if max_price is not None and p_price is not None:
                if float(p_price) > float(max_price):
                    continue

            # 3. Filtro de presupuesto mínimo
            if min_price is not None and p_price is not None:
                if float(p_price) < float(min_price):
                    continue

            # 4. Filtro de categoría (si fue especificado explícitamente)
            if cat_filter:
                terms = [t for t in cat_filter.replace("-", " ").replace("_", " ").split() if len(t) > 2]
                stem_terms = set(terms)
                for t in terms:
                    if t.endswith("es") and len(t) > 3:
                        stem_terms.add(t[:-2])
                    elif t.endswith("s") and len(t) > 2:
                        stem_terms.add(t[:-1])

                if any(k in cat_filter for k in ["labial", "labio", "lip"]):
                    stem_terms.update(["labial", "labio", "lip", "lips", "gloss", "rouge", "bálsamo", "balm"])
                elif any(k in cat_filter for k in ["perfum", "colonia", "fraganc"]):
                    stem_terms.update(["perfum", "colonia", "fraganc", "eau de", "cologne"])
                elif any(k in cat_filter for k in ["ceja"]):
                    stem_terms.update(["ceja", "brow"])
                elif any(k in cat_filter for k in ["pestaña", "rimel", "rímel", "mascara"]):
                    stem_terms.update(["pestaña", "rimel", "rímel", "mascara", "lash"])

                matches_category = p_cat not in ["general", "all", "todos", "venta", ""] and any(term in p_cat for term in stem_terms)
                matches_name = any(term in p_name for term in stem_terms)
                matches_desc = any(term in p_desc for term in stem_terms)

                if not (matches_category or matches_name or matches_desc):
                    continue

            filtered.append(p)

        # 5. Ordenamiento determinista matemático
        if sort_by == "price_asc":
            with_price = [p for p in filtered if p.get("price") is not None]
            without_price = [p for p in filtered if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]))
            filtered = with_price + without_price
        elif sort_by == "price_desc":
            with_price = [p for p in filtered if p.get("price") is not None]
            without_price = [p for p in filtered if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]), reverse=True)
            filtered = with_price + without_price

        return {"filtered_products": filtered}

    async def rubric_evaluator_judge(self, state: ProductRecomenderState) -> dict:
        """Nodo 5: Juez Evaluador con Rúbrica de Calidad en 4 Criterios (Evaluator-Optimizer)."""
        filtered = state.get("filtered_products", [])
        top_k = int(state.get("top_k") or self._default_top_k)
        raw_query = state.get("raw_query") or ""
        base_product = state.get("base_product") or "No especificado"
        relation_type = state.get("relation_type", "general_recommendation")
        filters = state.get("filters", {}) or {}

        # Si no hay candidatos tras el filtrado, falla automáticamente para disparar reflexión
        if not filtered:
            return {
                "meets_rubric": False,
                "rubric_scores": {"relevance": 1.0, "filter_compliance": 1.0, "diversity": 1.0, "data_completeness": 1.0},
                "critique": "No se encontraron productos disponibles que cumplan los filtros de búsqueda iniciales.",
                "suggested_refinement": "Flexibilizar términos de búsqueda o ampliar el rango de presupuesto.",
                "final_recommended_products": [],
            }

        candidates_text = []
        for idx, p in enumerate(filtered, start=1):
            price_str = f"{p.get('currency', 'PEN')} {p.get('price'):.2f}" if p.get("price") is not None else "Precio no disponible"
            candidates_text.append(
                f"[{idx}] SKU: {p.get('sku')} | Nombre: {p.get('name')} | Precio: {price_str} | Categoría: {p.get('category')}\n"
                f"    Descripción: {p.get('description', '')[:200]}"
            )
        cand_block = "\n\n".join(candidates_text)

        system_prompt = """
            Eres un auditor de calidad comercial y evaluación de recomendaciones de catálogo.
            Tu misión es evaluar objetivamente la lista de productos candidatos frente a la solicitud del cliente
            utilizando la siguiente RÚBRICA DE 4 CRITERIOS (escala 1.0 a 5.0):

            1. RELEVANCIA TEMÁTICO-COMERCIAL:
               - ¿Los productos complementan o sustituyen adecuadamente al producto base o satisfacen la necesidad del usuario?
            2. CUMPLIMIENTO DE FILTROS Y ORDEN:
               - ¿Se respeta el presupuesto, categoría y criterio de ordenamiento solicitado?
               - Si el cliente solicitó productos más baratos/económicos ('sort_by' = price_asc), los productos de menor precio deben seleccionarse con máxima prioridad y orden ascendente.
            3. DIVERSIDAD:
               - ¿Las opciones ofrecen variedad sin duplicidad ni redundancia excesiva?
            4. COMPLETITUD DE DATOS:
               - ¿Los productos cuentan con precio de lista válido y descripción informativa?

            REGLA DE VEREDICTO (meets_rubric):
            - Marca meets_rubric = True si el promedio de criterios es >= 3.8 y NO hay violaciones flagrantes de presupuesto o categoría.
            - Marca meets_rubric = False si los productos son irrelevantes, violan el presupuesto o no cubren la necesidad.
            - En 'selected_skus', incluye exactamente los mejores SKUs (hasta el top_k solicitado) respetando el orden estricto de conveniencia y precio.
            - En 'critique', detalla tu análisis objetivo.
        """

        sort_by = state.get("sort_by") or "relevance"
        user_prompt = f"""
            Solicitud original del cliente: {raw_query}
            Producto base de referencia: {base_product}
            Tipo de relación esperada: {relation_type}
            Top-K solicitado: {top_k}
            Criterio de ordenamiento: {sort_by}
            Filtros aplicados: {filters}

            Candidatos disponibles:
            {cand_block}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        judge_res: Optional[RubricEvaluationResult] = await self._rubric_judge.ainvoke(messages)
        if judge_res is None or not isinstance(judge_res, RubricEvaluationResult):
            if isinstance(judge_res, dict):
                judge_res = RubricEvaluationResult(**judge_res)
            else:
                # Fallback seguro
                meets = len(filtered) > 0
                return {
                    "meets_rubric": meets,
                    "rubric_scores": {"relevance": 4.0, "filter_compliance": 4.0, "diversity": 4.0, "data_completeness": 4.0},
                    "critique": "Aprobación por fallback.",
                    "final_recommended_products": filtered[:top_k],
                }

        scores_dict = {
            "relevance": judge_res.scores.relevance,
            "filter_compliance": judge_res.scores.filter_compliance,
            "diversity": judge_res.scores.diversity,
            "data_completeness": judge_res.scores.data_completeness,
        }

        # Seleccionar los productos finales según los SKUs elegidos por el juez
        selected_skus = judge_res.selected_skus[:top_k]
        sku_to_prod = {str(p["sku"]).strip(): p for p in filtered}
        final_prods = [sku_to_prod[s] for s in selected_skus if s in sku_to_prod]

        # Si el juez no especificó SKUs pero aprobó, tomar los primeros top_k
        if not final_prods and judge_res.meets_rubric:
            final_prods = filtered[:top_k]

        # Garantizar ordenamiento según sort_by en final_prods
        if sort_by == "price_asc":
            with_price = [p for p in final_prods if p.get("price") is not None]
            without_price = [p for p in final_prods if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]))
            final_prods = with_price + without_price
        elif sort_by == "price_desc":
            with_price = [p for p in final_prods if p.get("price") is not None]
            without_price = [p for p in final_prods if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]), reverse=True)
            final_prods = with_price + without_price

        return {
            "meets_rubric": judge_res.meets_rubric,
            "rubric_scores": scores_dict,
            "critique": judge_res.critique,
            "suggested_refinement": judge_res.suggested_refinement,
            "final_recommended_products": final_prods,
        }

    async def reflection_optimizer(self, state: ProductRecomenderState) -> dict:
        """Nodo 6: Analiza la crítica de la rúbrica y optimiza la búsqueda para la siguiente iteración."""
        current_iter = int(state.get("iteration_count", 0)) + 1
        critique = state.get("critique") or "Candidatos insuficientes o no relevantes."
        suggested = state.get("suggested_refinement") or ""
        old_query = state.get("search_query") or state.get("raw_query") or ""
        base_prod = state.get("base_product") or ""
        filters = state.get("filters", {}) or {}

        system_prompt = """
            Eres un optimizador de búsquedas en catálogo y recomendaciones comerciales.
            La iteración previa de recomendación no superó la rúbrica de calidad requerida.
            Tu misión es generar una nueva consulta de búsqueda semántica mejorada incorporando
            sinónimos técnicos, términos comerciales de catálogo o relajando restricciones demasiado estrictas.
            Devuelve únicamente la nueva frase de búsqueda concisa y directa.
        """

        user_content = f"""
            Consulta previa: {old_query}
            Producto base: {base_prod}
            Filtros: {filters}
            Diagnóstico del Juez (Crítica): {critique}
            Sugerencia de ajuste: {suggested}

            Genera una nueva consulta de búsqueda semántica optimizada:
        """

        resp = await self._refinement_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ])
        new_query = str(resp.content).strip().strip('"').strip("'")
        if not new_query:
            new_query = old_query

        return {
            "iteration_count": current_iter,
            "search_query": new_query,
        }

    async def synthesize_recommendations(self, state: ProductRecomenderState) -> dict:
        """Nodo 7: Redacta la respuesta comercial ejecutiva para el vendedor o cliente."""
        recommended = state.get("final_recommended_products")
        if not recommended:
            recommended = state.get("filtered_products", [])[: int(state.get("top_k") or self._default_top_k)]

        sort_by = state.get("sort_by") or "relevance"
        if sort_by == "price_asc":
            with_price = [p for p in recommended if p.get("price") is not None]
            without_price = [p for p in recommended if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]))
            recommended = with_price + without_price
        elif sort_by == "price_desc":
            with_price = [p for p in recommended if p.get("price") is not None]
            without_price = [p for p in recommended if p.get("price") is None]
            with_price.sort(key=lambda p: float(p["price"]), reverse=True)
            recommended = with_price + without_price

        raw_query = state.get("raw_query") or ""
        base_prod = state.get("base_product")
        relation_type = state.get("relation_type", "general_recommendation")

        if not recommended:
            msg = (
                "Lamentablemente en este momento no encontré productos disponibles en el catálogo "
                "que cumplan exactamente con los filtros y especificaciones solicitadas. "
                "¿Deseas que busquemos en otras categorías o ajustemos el rango de presupuesto?"
            )
            return {
                "final_response": msg,
                "messages": [AIMessage(content=msg)],
            }

        rec_lines = []
        for idx, p in enumerate(recommended, start=1):
            price_str = f"${p.get('price'):.2f}" if p.get("price") is not None else "Consultar precio"
            rec_lines.append(
                f"- [{p.get('sku')}] {p.get('name')}: {price_str} ({p.get('currency', 'PEN')})\n"
                f"  * Categoría: {p.get('category') or 'General'}\n"
                f"  * Detalle: {p.get('description', '')[:220]}"
            )
        recs_text = "\n".join(rec_lines)

        system_prompt = inject_soul(
            """
            Reglas específicas de presentación de recomendaciones:
            - Respeta estrictamente el orden de los productos recomendados que se te entregan (si están ordenados por precio más bajo o ascendente, preséntalos en ese mismo orden exacto sin alterarlo).
            - Presenta cada producto con viñetas claras incluyendo su SKU entre corchetes (ej: **[6189]**), nombre comercial y precio oficial con su divisa.
            - Explica brevemente por qué es una excelente recomendación (por qué combina, complementa o representa una alternativa de gran valor).
            - Invita con sutileza consultiva al usuario a agregar alguno de los productos a su cotización si lo desea.
            """,
            role=SoulRole.RECOMMENDER,
        )

        user_content = f"""
            Consulta original: {raw_query}
            Producto de partida: {base_prod or 'Ninguno en particular'}
            Tipo de relación: {relation_type}

            Productos recomendados seleccionados:
            {recs_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        synth_res: Optional[RecommendationSynthesisResponse] = await self._synthesizer.ainvoke(messages)
        resp_text = getattr(synth_res, "response_text", "") if synth_res else ""
        if not resp_text and isinstance(synth_res, dict):
            resp_text = synth_res.get("response_text", "")

        if not resp_text:
            resp_text = f"Aquí tienes las mejores opciones recomendadas para tu consulta:\n\n{recs_text}"

        return {
            "final_response": resp_text,
            "messages": [AIMessage(content=resp_text)],
        }
