# src/agent_service/graph/subgraphs/rag_agent/nodes.py
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from src.agent_service.graph.sub_graphs.rag_product.schemas import (EvaluationResult, FinalAnswer)
from src.agent_service.graph.sub_graphs.rag_product.state import (RagProductState)
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.schemas import ProductCatalogFilter

class ProductRagNodes:
    def __init__(self, llm: BaseChatModel, vector_store: ProductVectorStore, top_k: int = 5):
        self._llm = llm
        self._vector_store = vector_store
        self._top_k = top_k

        self._judge = llm.with_structured_output(EvaluationResult)
        self._synthesizer = llm.with_structured_output(FinalAnswer)

    async def retrieve_products(self, state: RagProductState) -> dict:
        query_text = state.get("raw_query")
        user_id = 5

        catalog_filter = ProductCatalogFilter(user_id=user_id)

        docs = await self._vector_store.ahybrid_search(
            query=query_text,
            k=self._top_k,
            alpha=0.7,
            filters=catalog_filter,
        )

        return {
            "documents": docs,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    
        pass

    async def llm_as_jugde(self, state: RagProductState) -> dict:
        pass

    async def reflection_and_refined(self, state: RagProductState) -> dict:
        pass

    async def synthesize_response(self, state: RagProductState) -> dict:
        pass