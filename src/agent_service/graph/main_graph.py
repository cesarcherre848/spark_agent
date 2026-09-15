"""
src/agent_service/graph/main_graph.py - Grafo Principal Unificado de Spark Agent

Integra los tres módulos fundamentales del sistema:
1. Memoria a largo plazo (user_memory) con PostgreSQL y pgvector.
2. Enrutador inteligente (LLM temp=0.0) hacia:
   - general_chat (charla general y asistencia)
   - product_rag (búsqueda semántica en catálogo)
   - product_resolver (cotizaciones de SKUs, Odoo ERP y HITL)
3. Guardado automático de cotizaciones en memoria semántica.
"""

import logging
from typing import Optional, List, Dict, Any, Literal, Annotated, Union
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore
from src.agent_service.graph.sub_graphs.user_memory.nodes import UserMemoryNodes
from src.agent_service.graph.sub_graphs.user_memory.checkpointer import get_postgres_checkpointer
from src.agent_service.graph.sub_graphs.product_resolver.graph import (
    build_product_resolver_graph,
    get_product_resolver_graph,
)
from src.agent_service.graph.sub_graphs.product_rag.graph import (
    build_rag_product_graph,
    get_product_rag_graph,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. ESTADO CONSOLIDADO DEL GRAFO PRINCIPAL
# ==============================================================================
class MainGraphState(TypedDict, total=False):
    """Estado global que fluye a través del Router y los subgrafos especializados."""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: Optional[Union[int, str]]
    raw_query: Optional[str]
    session_id: Optional[str]

    # Clasificación del Router
    intent: Optional[Literal["rag", "resolver", "general"]]
    intent_reasoning: Optional[str]

    # Memoria recuperada
    user_context: Optional[str]
    retrieved_memories: List[Dict[str, Any]]

    # Salida final devuelta al usuario
    final_response: Optional[str]

    # Pasamanos de product_resolver:
    items: Dict[str, Any]
    is_extraction_complete: bool
    has_partner_conflicts: bool
    tool_raw_output: List[Dict[str, Any]]
    grouped_products: Dict[str, List[Dict[str, Any]]]
    memory_to_save: Optional[str]
    saved_memory_id: Optional[int]
    clarification_count: int

    # Pasamanos de product_rag:
    refined_query: Optional[str]
    retrieved_products: List[Any]
    is_sufficient: bool
    matched_skus: List[str]


# ==============================================================================
# 2. ESQUEMA DEL ENRUTADOR
# ==============================================================================
class RouterDecision(BaseModel):
    """Decisión estructurada de enrutamiento basada en la intención del usuario."""
    intent: Literal["rag", "resolver", "general"] = Field(
        description=(
            "'rag': si el usuario busca, explora o pide recomendaciones de productos en lenguaje natural SIN especificar SKUs. "
            "'resolver': si el usuario desea cotizar, comprar, consultar precios/stock o menciona códigos SKU específicos. "
            "'general': saludos, agradecimientos, preguntas sobre qué puede hacer el asistente o charla general."
        )
    )
    reasoning: str = Field(description="Breve justificación de la decisión")


# ==============================================================================
# 3. NODOS DEL GRAFO PRINCIPAL
# ==============================================================================
def create_router_node(llm: BaseChatModel):
    """Crea el nodo enrutador determinista con temperatura 0.0."""
    structured_router = bind_structured_output(bind_temperature(llm, 0.0), RouterDecision)

    async def router_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        # Recortar historial a los últimos mensajes para contexto relevante
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres el supervisor y enrutador del sistema comercial Spark ERP.
            Tu misión es clasificar con precisión la intención de la consulta del usuario en una de 3 categorías:

            1. 'general': Saludos, despedidas, agradecimientos o preguntas sobre qué servicios o ayuda puedes brindar.
            2. 'rag': Consultas exploratorias en lenguaje natural donde el cliente busca recomendaciones, características o disponibilidad de productos SIN mencionar SKUs específicos (ej: '¿qué cremas faciales tienen?', 'recomiéndame esmaltes rojos').
            3. 'resolver': Solicitudes explícitas de compra, cotización, precios o consultas donde el cliente especifica códigos SKU (ej: 'cotízame 3 unidades del SKU 1', 'precio del SKU 14', 'quiero comprar').
        """

        user_context = state.get("user_context")
        context_block = f"\nAntecedentes de memoria:\n{user_context}\n" if user_context else ""

        user_prompt = f"""
            {context_block}Consulta actual: {raw_query}
        """

        decision: Optional[RouterDecision] = await structured_router.ainvoke([
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_prompt),
        ])

        intent = decision.intent if decision and hasattr(decision, "intent") else "general"
        reasoning = decision.reasoning if decision and hasattr(decision, "reasoning") else "Enrutamiento por defecto"

        logger.info(f"Router clasificó consulta como: '{intent}' (Razón: {reasoning})")
        return {
            "intent": intent,
            "intent_reasoning": reasoning,
            "raw_query": raw_query,
        }

    return router_node


def create_general_chat_node(llm: BaseChatModel):
    """Crea el nodo de conversación general para saludos y orientación."""
    chat_model = bind_temperature(llm, 0.35)

    async def general_chat_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query", "")
        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres el asistente inteligente de Spark ERP, especializado en gestión comercial y catálogo multimarca.
            Responde de manera amable, profesional, concisa y servicial.
            Explica con claridad que puedes ayudar al usuario a:
            1. Explorar y recomendar productos del catálogo comercial (búsqueda semántica).
            2. Cotizar y gestionar pedidos con validación de proveedores autorizados y Odoo ERP.
            3. Recordar acuerdos o cotizaciones de sesiones anteriores.
        """

        user_context = state.get("user_context")
        context_block = f"\nAntecedentes del usuario:\n{user_context}\n" if user_context else ""

        response = await chat_model.ainvoke([
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=f"{context_block}{raw_query}"),
        ])

        reply_text = str(response.content)
        return {
            "final_response": reply_text,
            "messages": [AIMessage(content=reply_text)],
        }

    return general_chat_node


# ==============================================================================
# 4. CONSTRUCCIÓN DEL GRAFO PRINCIPAL (SPARK_AGENT)
# ==============================================================================
def _route_after_router(state: MainGraphState) -> Literal["general_chat", "product_rag", "product_resolver"]:
    """Enrutamiento condicional según la intención detectada."""
    intent = state.get("intent", "general")
    if intent == "rag":
        return "product_rag"
    elif intent == "resolver":
        return "product_resolver"
    return "general_chat"


def build_main_graph(
    llm: BaseChatModel,
    resolver_graph: Optional[Any] = None,
    rag_graph: Optional[Any] = None,
    memory_store: Optional[UserMemoryStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el Grafo Principal unificado con memoria, router y subgrafos."""
    workflow = StateGraph(state_schema=MainGraphState)

    # 1. Nodos de memoria
    mem_nodes = UserMemoryNodes(store=memory_store) if memory_store else None

    async def retrieve_memory_wrapper(state: MainGraphState) -> dict:
        if mem_nodes:
            return await mem_nodes.retrieve_memory_node(state)
        return {}

    async def save_memory_wrapper(state: MainGraphState) -> dict:
        if mem_nodes:
            return await mem_nodes.save_memory_node(state)
        return {}

    # 2. Router y Chat General
    router_node = create_router_node(llm)
    general_node = create_general_chat_node(llm)

    # 3. Subgrafos compilados
    compiled_resolver = resolver_graph if resolver_graph is not None else get_product_resolver_graph()
    compiled_rag = rag_graph if rag_graph is not None else get_product_rag_graph()

    # Registro de nodos en el grafo
    workflow.add_node("retrieve_memory", retrieve_memory_wrapper)
    workflow.add_node("router", router_node)
    workflow.add_node("general_chat", general_node)
    workflow.add_node("product_rag", compiled_rag)
    workflow.add_node("product_resolver", compiled_resolver)
    workflow.add_node("save_memory", save_memory_wrapper)

    # Flujo de ejecución
    workflow.add_edge(START, "retrieve_memory")
    workflow.add_edge("retrieve_memory", "router")

    # Bifurcación condicional del router
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "general_chat": "general_chat",
            "product_rag": "product_rag",
            "product_resolver": "product_resolver",
        },
    )

    # Rutas hacia el fin
    workflow.add_edge("general_chat", END)
    workflow.add_edge("product_rag", END)
    workflow.add_edge("product_resolver", "save_memory")
    workflow.add_edge("save_memory", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=resolved_checkpointer)


def get_main_graph():
    """Fábrica sin argumentos para LangGraph Studio y punto de entrada oficial."""
    llm = get_default_llm()
    pool = get_db_pool()
    embeddings = get_embedding_service()
    store = UserMemoryStore(pool=pool, embedding_service=embeddings)
    return build_main_graph(llm=llm, memory_store=store, checkpointer=MemorySaver())
