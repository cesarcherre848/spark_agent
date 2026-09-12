from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.schemas import ProductCatalogFilter
from src.agent_service.graph.sub_graphs.product_rag.schemas import (
    NormalizedQuery,
    EvaluationResult,
    QueryRefinementResult,
    FinalAnswer,
    format_candidates_for_prompt,
)
from src.agent_service.graph.sub_graphs.product_rag.state import ProductRagState
from src.agent_service.core.llms import bind_temperature


class ProductRagNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: ProductVectorStore,
        top_k: int = 5,
        default_max_iterations: int = 2,
    ):
        self._llm = llm
        self._vector_store = vector_store
        self._top_k = top_k
        self._default_max_iterations = default_max_iterations

        # Temperaturas por llamada: normalización y evaluación estrictas (0.0),
        # reformulación reflexiva creativa (0.5) y síntesis comercial balanceada (0.35).
        self._normalizer = bind_temperature(llm, 0.15).with_structured_output(NormalizedQuery)
        self._judge = bind_temperature(llm, 0.1).with_structured_output(EvaluationResult)
        self._refiner = bind_temperature(llm, 0.5).with_structured_output(QueryRefinementResult)
        self._synthesizer = bind_temperature(llm, 0.35).with_structured_output(FinalAnswer)


    async def normalize_query(self, state: ProductRagState) -> dict:
        raw_query = state.get("raw_query", "").strip()

        system_prompt = """
            Eres un especialista en preparación y depuración de consultas para motores de búsqueda de catálogo ERP (PostgreSQL Full-Text Search + pgvector).

            Tareas:
                - Extraer la intención comercial pura eliminando saludos, cortesías, rodeos conversacionales y ambigüedades.
                - Convertir descripciones informales en términos técnicos de catálogo (ej. 'alguna crema para las arrugas' -> 'crema antiedad acido hialuronico').
                - Mantener marcas, calibres, medidas o atributos específicos si el usuario los menciona.

            Campos de salida:
                1. 'search_query': Consulta normalizada, concisa y rica en palabras clave para búsqueda.
        """

        user_prompt = f"""
            Consulta original del cliente:
            {raw_query}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result: NormalizedQuery = await self._normalizer.ainvoke(messages)

        return {
            "refined_query": result.search_query,
        }

    async def retrieve_products(self, state: ProductRagState) -> dict:
        query_text = (state.get("refined_query") or state.get("raw_query") or "").strip()
        user_id = state.get("user_id")

        catalog_filter = ProductCatalogFilter(user_id=user_id)
        docs = await self._vector_store.ahybrid_search(
            query=query_text,
            k=self._top_k,
            alpha=0.5,
            filters=catalog_filter,
        )

        current_iteration = state.get("iteration_count", 0)

        return {
            "retrieved_products": docs,
            "iteration_count": current_iteration + 1,
        }

    async def llm_as_judge(self, state: ProductRagState) -> dict:
        docs = state.get("retrieved_products", [])

        if not docs:
            return {
                "is_sufficient": False,
                "selected_indices": [],
                "matched_skus": [],
                "critique": "No se encontraron productos candidatos en el catálogo para esta consulta.",
            }

        query = state.get("refined_query") or state.get("raw_query") or ""
        candidates_text = format_candidates_for_prompt(docs)

        system_prompt = """
            Eres un auditor de búsqueda para ventas por catálogo ERP.

            Tareas:
                - Evaluar si los productos candidatos resuelven la consulta del cliente.
                - Identificar exactamente cuáles candidatos son relevantes según su índice numérico [1], [2], etc.

            Criterios de salida:
                - 'is_sufficient': True si al menos un producto candidato coincide en especificación o intención; False si ninguno es relevante.
                - 'selected_indices': Lista con los índices numéricos de los productos que sí son relevantes (ej. [1, 2]). Vacío si ninguno es relevante.
                - 'critique': Si is_sufficient es False, explica qué faltó o por qué no coinciden. Si es True, resume por qué fueron seleccionados.
        """

        user_prompt = f"""
            Consulta del usuario: {query}

            Candidatos del catálogo:
            {candidates_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        evaluation: EvaluationResult = await self._judge.ainvoke(messages)

        # Extracción determinista de SKUs en Python a partir de los índices seleccionados
        matched_skus = [
            docs[idx - 1].metadata["sku"]
            for idx in evaluation.selected_indices
            if 1 <= idx <= len(docs) and docs[idx - 1].metadata.get("sku")
        ]

        return {
            "is_sufficient": evaluation.is_sufficient,
            "selected_indices": evaluation.selected_indices,
            "matched_skus": matched_skus,
            "critique": evaluation.critique,
        }

    # Alias por retrocompatibilidad con referencias anteriores
    llm_as_jugde = llm_as_judge

    async def reflection_and_refined(self, state: ProductRagState) -> dict:
        raw_query = state.get("raw_query", "")
        current_query = state.get("refined_query") or raw_query
        critique = state.get("critique", "Los productos recuperados no coincidieron con la intención del usuario.")
        failed_products = state.get("retrieved_products", [])
        failed_candidates_summary = format_candidates_for_prompt(failed_products[:4])

        system_prompt = """
            Eres un asistente especialista en optimización de consultas para motores de búsqueda de productos ERP.

            Tareas:
                - Analizar por qué la búsqueda previa falló según la crítica del evaluador y los productos recuperados erróneamente.
                - Proponer una nueva consulta reescrita y optimizada, incorporando sinónimos comerciales o términos técnicos más precisos.

            Campos de salida:
                - 'refined_query': La nueva consulta reescrita y concisa.
        """

        user_prompt = f"""
            Consulta original: {raw_query}
            Consulta usada en el intento anterior: {current_query}
            Crítica del evaluador: {critique}

            Productos que arrojó erróneamente el catálogo:
            {failed_candidates_summary}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        refinement: QueryRefinementResult = await self._refiner.ainvoke(messages)

        return {
            "refined_query": refinement.refined_query,
        }

    async def synthesize_response(self, state: ProductRagState) -> dict:
        docs = state.get("retrieved_products", [])
        raw_query = state.get("raw_query", "")
        is_sufficient = state.get("is_sufficient", False)
        critique = state.get("critique", "")
        selected_indices = state.get("selected_indices", [])

        if docs and is_sufficient:
            selected_docs = [
                docs[idx - 1]
                for idx in selected_indices
                if 1 <= idx <= len(docs)
            ] or docs[:3]
            candidates_text = format_candidates_for_prompt(selected_docs)
        else:
            candidates_text = "No se encontraron productos coincidentes o autorizados en el catálogo."

        system_prompt = """
            Eres un asesor comercial para un catálogo de ventas ERP.

            Tareas:
                - Redactar una respuesta cordial, técnica y orientada a la venta.
                - Si hay productos seleccionados: Recomienda de forma clara las opciones pertinentes, mencionando sus nombres y beneficios clave.
                - Si no hay productos disponibles o is_sufficient es False: Explica amablemente que no disponemos de ese artículo exacto en este momento.
                - No inventes características, precios ni especificaciones ausentes en los candidatos.
        """

        user_prompt = f"""
            Consulta del cliente: {raw_query}
            Veredicto de búsqueda (is_sufficient): {is_sufficient}
            Diagnóstico de la búsqueda: {critique}

            Candidatos seleccionados:
            {candidates_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        answer: FinalAnswer = await self._synthesizer.ainvoke(messages)

        return {
            "final_response": answer.response_text,
        }