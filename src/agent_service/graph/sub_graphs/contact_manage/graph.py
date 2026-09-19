"""
src/agent_service/graph/sub_graphs/contact_manage/graph.py - Ensamble del subgrafo contact_manage
"""

from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.graph.sub_graphs.contact_manage.state import ContactManageState
from src.agent_service.graph.sub_graphs.contact_manage.nodes import ContactManageNodes
from src.agent_service.tools.contact_tools import (
    odoo_get_customers,
    odoo_list_current_customers,
    odoo_upsert_customer,
    odoo_remove_customer,
)


def _route_after_extraction(
    state: ContactManageState,
) -> Literal["get_contacts_node", "list_current_contacts_node", "feedback_user_remove_node"]:
    """Enrutador posterior a la extracción: bifurca según la acción detectada."""
    action = state.get("contact_action", "list")
    if action == "upsert":
        return "list_current_contacts_node"
    if action == "remove":
        return "feedback_user_remove_node"
    return "get_contacts_node"


def _route_after_judge(
    state: ContactManageState,
) -> Literal["feedback_user_duplicate_node", "upsert_customer_node"]:
    """Enrutador tras evaluación de duplicados: si hay duplicados solicita feedback, sino upsert directo."""
    if state.get("has_possible_duplicates", False):
        return "feedback_user_duplicate_node"
    return "upsert_customer_node"


def _route_after_duplicate_confirm(
    state: ContactManageState,
) -> Literal["upsert_customer_node", "synthesize_response"]:
    """Enrutador tras feedback de duplicados: procede con upsert si confirmó, sino pasa a síntesis."""
    if state.get("upsert_confirmed", True):
        return "upsert_customer_node"
    return "synthesize_response"


def _route_after_remove_confirm(
    state: ContactManageState,
) -> Literal["remove_customer_node", "synthesize_response"]:
    """Enrutador tras feedback de confirmación de eliminación."""
    if state.get("remove_confirmed", True):
        return "remove_customer_node"
    return "synthesize_response"


def build_contact_manage_graph(
    llm: BaseChatModel,
    get_customers_tool: Any = odoo_get_customers,
    list_current_customers_tool: Any = odoo_list_current_customers,
    upsert_customer_tool: Any = odoo_upsert_customer,
    remove_customer_tool: Any = odoo_remove_customer,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el subgrafo contact_manage con soporte de Human-in-the-Loop y herramientas desacopladas."""
    nodes = ContactManageNodes(
        llm=llm,
        get_customers_tool=get_customers_tool,
        list_current_customers_tool=list_current_customers_tool,
        upsert_customer_tool=upsert_customer_tool,
        remove_customer_tool=remove_customer_tool,
    )

    workflow = StateGraph(state_schema=ContactManageState)

    # Registro de nodos
    workflow.add_node("extract_customer_info", nodes.extract_customer_info)
    workflow.add_node("get_contacts_node", nodes.get_contacts_node)
    workflow.add_node("list_current_contacts_node", nodes.list_current_contacts_node)
    workflow.add_node("duplicate_candidates_judge_node", nodes.duplicate_candidates_judge_node)
    workflow.add_node("feedback_user_duplicate_node", nodes.feedback_user_duplicate_node)
    workflow.add_node("feedback_user_remove_node", nodes.feedback_user_remove_node)
    workflow.add_node("upsert_customer_node", nodes.upsert_customer_node)
    workflow.add_node("remove_customer_node", nodes.remove_customer_node)
    workflow.add_node("synthesize_response", nodes.synthesize_response)

    # Flujo de inicio
    workflow.add_edge(START, "extract_customer_info")

    # Bifurcación principal (list vs upsert vs remove)
    workflow.add_conditional_edges(
        "extract_customer_info",
        _route_after_extraction,
        {
            "get_contacts_node": "get_contacts_node",
            "list_current_contacts_node": "list_current_contacts_node",
            "feedback_user_remove_node": "feedback_user_remove_node",
        },
    )

    # Rama List
    workflow.add_edge("get_contacts_node", "synthesize_response")

    # Rama Upsert
    workflow.add_edge("list_current_contacts_node", "duplicate_candidates_judge_node")
    workflow.add_conditional_edges(
        "duplicate_candidates_judge_node",
        _route_after_judge,
        {
            "feedback_user_duplicate_node": "feedback_user_duplicate_node",
            "upsert_customer_node": "upsert_customer_node",
        },
    )
    workflow.add_conditional_edges(
        "feedback_user_duplicate_node",
        _route_after_duplicate_confirm,
        {
            "upsert_customer_node": "upsert_customer_node",
            "synthesize_response": "synthesize_response",
        },
    )
    workflow.add_edge("upsert_customer_node", "synthesize_response")

    # Rama Remove
    workflow.add_conditional_edges(
        "feedback_user_remove_node",
        _route_after_remove_confirm,
        {
            "remove_customer_node": "remove_customer_node",
            "synthesize_response": "synthesize_response",
        },
    )
    workflow.add_edge("remove_customer_node", "synthesize_response")

    # Síntesis a END
    workflow.add_edge("synthesize_response", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=resolved_checkpointer)


def get_contact_manage_graph():
    """Fábrica sin argumentos para inspección en LangGraph Studio o CLI."""
    from src.agent_service.core.llms.factory import get_default_llm
    llm = get_default_llm()
    return build_contact_manage_graph(llm=llm, checkpointer=MemorySaver())
