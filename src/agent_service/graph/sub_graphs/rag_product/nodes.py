# src/agent_service/graph/subgraphs/rag_agent/nodes.py
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from src.agent_service.graph.sub_graphs.rag_product.schemas import (EvaluationResult, FinalAnswer)
from src.agent_service.graph.sub_graphs.rag_product.state import (RagProductState)

class ProductRagNodes:
    def __init__(self, llm: BaseChatModel, vector_store: VectorStore, top_k: int = 5):
        self._llm = llm
        self._vector_store = vector_store
        self._top_k = top_k

        self._judge = llm.with_structured_output(EvaluationResult)
        self._synthesizer = llm.with_structured_output(FinalAnswer)

    async def retrieve_products(self, state: RagProductState) -> dict:
        pass

    async def llm_as_jugde(self, state: RagProductState) -> dict:
        pass

    async def reflection_and_refined(self, state: RagProductState) -> dict:
        pass

    async def synthesize_response(self, state: RagProductState) -> dict:
        pass