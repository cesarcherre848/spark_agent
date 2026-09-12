from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.graph.sub_graphs.product_resolver.state import ProductResolverState
from src.agent_service.graph.sub_graphs.product_resolver.nodes import ProductResolverNodes
from src.agent_service.tools.product_tools import (
    get_product_by_skus,
    search_suppliers,
    validate_product_ownership,
)


def _route_after_extraction(
    state: ProductResolverState,
) -> Literal["check_partner_conflicts", "feedback_ask_missing"]:
    """Enrutador después de la extracción: si faltan datos esenciales va a feedback; si está completo avanza."""
    if state.get("is_extraction_complete", False):
        return "check_partner_conflicts"
    return "feedback_ask_missing"


def _route_after_partner_check(
    state: ProductResolverState,
) -> Literal["call_get_product_by_skus", "feedback_clarify_partners"]:
    """Enrutador después de verificar partners: si hay conflictos va a feedback; si no hay avanza a la tool."""
    if state.get("has_partner_conflicts", False):
        return "feedback_clarify_partners"
    return "call_get_product_by_skus"


def build_product_resolver_graph(
    llm: BaseChatModel,
    product_tool: Any = get_product_by_skus,
    ownership_tool: Any = validate_product_ownership,
    supplier_tool: Any = search_suppliers,
    partner_resolver: Optional[Any] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el subgrafo product_resolver con soporte nativo de checkpointer e HITL."""
    nodes = ProductResolverNodes(
        llm=llm,
        product_tool=product_tool,
        ownership_tool=ownership_tool,
        supplier_tool=supplier_tool,
        partner_resolver=partner_resolver,
    )

    workflow = StateGraph(state_schema=ProductResolverState)

    # Registro de los 7 nodos
    workflow.add_node("extract_skus_and_attributes", nodes.extract_skus_and_attributes)
    workflow.add_node("feedback_ask_missing", nodes.feedback_ask_missing)
    workflow.add_node("check_partner_conflicts", nodes.check_partner_conflicts)
    workflow.add_node("feedback_clarify_partners", nodes.feedback_clarify_partners)
    workflow.add_node("call_get_product_by_skus", nodes.call_get_product_by_skus)
    workflow.add_node("group_by_partners", nodes.group_by_partners)
    workflow.add_node("synthesize_response", nodes.synthesize_response)

    # Conexiones
    workflow.add_edge(START, "extract_skus_and_attributes")

    # Bucle 1: Extracción completa vs faltan datos
    workflow.add_conditional_edges(
        "extract_skus_and_attributes",
        _route_after_extraction,
        {
            "check_partner_conflicts": "check_partner_conflicts",
            "feedback_ask_missing": "feedback_ask_missing",
        },
    )
    workflow.add_edge("feedback_ask_missing", "extract_skus_and_attributes")

    # Bucle 2: Conflicto de partners vs sin conflictos
    workflow.add_conditional_edges(
        "check_partner_conflicts",
        _route_after_partner_check,
        {
            "call_get_product_by_skus": "call_get_product_by_skus",
            "feedback_clarify_partners": "feedback_clarify_partners",
        },
    )
    workflow.add_edge("feedback_clarify_partners", "check_partner_conflicts")

    # Flujo lineal hacia la síntesis
    workflow.add_edge("call_get_product_by_skus", "group_by_partners")
    workflow.add_edge("group_by_partners", "synthesize_response")
    workflow.add_edge("synthesize_response", END)

    # Por defecto suministramos MemorySaver si no se provee un checkpointer
    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    return workflow.compile(checkpointer=resolved_checkpointer)
