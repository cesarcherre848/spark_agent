from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.graph.sub_graphs.rag_product.state import ProductRagState
from src.agent_service.graph.sub_graphs.rag_product.nodes import ProductRagNodes


def _route_after_judge(state: ProductRagState) -> Literal["synthesize", "reflection_refined"]:
    # Si la búsqueda satisfizo la necesidad, pasamos a generar la respuesta final
    if state.get("is_sufficient", False):
        return "synthesize"

    # Límite de seguridad para evitar ciclos infinitos
    max_iterations = state.get("max_iterations", 2)
    iteration_count = state.get("iteration_count", 0)

    if iteration_count >= max_iterations:
        return "synthesize"

    return "reflection_refined"


def build_rag_product_graph(
    llm: BaseChatModel,
    vector_store: ProductVectorStore,
    top_k: int = 5,
    default_max_iterations: int = 2,
):
    nodes = ProductRagNodes(
        llm=llm,
        vector_store=vector_store,
        top_k=top_k,
        default_max_iterations=default_max_iterations,
    )

    workflow = StateGraph(state_schema=ProductRagState)

    workflow.add_node("normalize", nodes.normalize_query)
    workflow.add_node("retrieve", nodes.retrieve_products)
    workflow.add_node("llm_as_judge", nodes.llm_as_judge)
    workflow.add_node("reflection_refined", nodes.reflection_and_refined)
    workflow.add_node("synthesize", nodes.synthesize_response)

    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", "retrieve")
    workflow.add_edge("retrieve", "llm_as_judge")

    workflow.add_conditional_edges(
        "llm_as_judge",
        _route_after_judge,
        {
            "synthesize": "synthesize",
            "reflection_refined": "reflection_refined",
        },
    )

    # Ciclo de auto-reflexión y reformulación (Loop 1)
    workflow.add_edge("reflection_refined", "retrieve")
    workflow.add_edge("synthesize", END)

    return workflow.compile()