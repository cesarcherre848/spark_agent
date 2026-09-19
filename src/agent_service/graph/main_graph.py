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
from src.agent_service.graph.sub_graphs.contact_manage.graph import (
    build_contact_manage_graph,
    get_contact_manage_graph,
)
from src.agent_service.graph.sub_graphs.sales_manage.graph import (
    build_sales_manage_graph,
    get_sales_manage_graph,
)
from src.agent_service.graph.sub_graphs.product_recomender.graph import (
    build_product_recomender_graph,
    get_product_recomender_graph,
)

from src.agent_service.soul import inject_soul, SoulRole
from src.agent_service.core.guardrails import (
    evaluate_input_guardrail,
    format_guardrail_refusal,
    ViolationCategory,
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

    # Guardrail de entrada y seguridad
    is_blocked: Optional[bool]
    guardrail_category: Optional[str]
    guardrail_reason: Optional[str]

    # Clasificación del Router
    intent: Optional[Literal["rag", "resolver", "general", "contact", "sales", "recommender", "out_of_scope"]]
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

    # Pasamanos de contact_manage:
    contact_action: Optional[Literal["list", "upsert", "remove"]]
    extracted_name: Optional[str]
    extracted_phones: Optional[List[str]]
    target_contact_id: Optional[int]
    customers_list: Optional[List[Dict[str, Any]]]
    current_candidates: Optional[List[Dict[str, Any]]]
    has_possible_duplicates: Optional[bool]
    duplicate_rationale: Optional[str]
    upsert_confirmed: Optional[bool]
    remove_confirmed: Optional[bool]
    operation_result: Optional[Dict[str, Any]]

    # Pasamanos de sales_manage:
    sales_action: Optional[Literal["list", "view", "upsert", "add_items", "remove_items", "confirm", "edit_order", "remove"]]
    customer_name: Optional[str]
    partner_id: Optional[int]
    candidate_partners: Optional[List[Dict[str, Any]]]
    customer_resolved: Optional[bool]
    customer_not_found: Optional[bool]
    orders_list: Optional[Dict[str, List[Dict[str, Any]]]]
    duplicate_choice: Optional[Literal["new", "selected", "cancel"]]
    target_order_name: Optional[str]
    target_order_id: Optional[int]
    unlock_confirmed: Optional[bool]
    order_view_data: Optional[Dict[str, Any]]
    cancellation_reason: Optional[str]
    is_intent_clear: Optional[bool]
    reflection_reasoning: Optional[str]
    clarification_question: Optional[str]
    clarification_options: Optional[List[str]]


    # Pasamanos de product_recomender:
    recommendation_type: Optional[str]
    target_top_k: Optional[int]
    base_product_name: Optional[str]
    base_product_sku: Optional[str]
    max_price_budget: Optional[float]
    min_price_budget: Optional[float]
    candidate_products: Optional[List[Any]]
    enriched_products: Optional[List[Any]]
    filtered_products: Optional[List[Any]]
    meets_rubric: Optional[bool]
    rubric_scores: Optional[Dict[str, Any]]
    rubric_reasoning: Optional[str]
    recommended_products: Optional[List[Dict[str, Any]]]


# ==============================================================================
# 2. ESQUEMA DEL ENRUTADOR
# ==============================================================================
class RouterDecision(BaseModel):
    """Clasificación estructurada de la consulta comercial."""
    intent: Literal["rag", "resolver", "general", "contact", "sales", "recommender", "out_of_scope"] = Field(
        description="Intención comercial detectada: 'rag', 'recommender', 'resolver', 'contact', 'sales', 'general' u 'out_of_scope'."
    )
    reasoning: str = Field(description="Breve justificación de la decisión.")


# ==============================================================================
# 3. NODOS DEL GRAFO PRINCIPAL
# ==============================================================================
def create_input_guardrail_node():
    """Crea el nodo de evaluación de Guardrail Capa 1 (Fast-Path determinista)."""
    async def input_guardrail_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        eval_result = evaluate_input_guardrail(raw_query)

        if eval_result.is_blocked:
            refusal_msg = eval_result.refusal_message or format_guardrail_refusal(eval_result.category)
            logger.warning(
                f"[Capa 1 Guardrail] Petición bloqueada. Categoría: {eval_result.category.value}. "
                f"Razón: {eval_result.reason}"
            )
            return {
                "is_blocked": True,
                "guardrail_category": eval_result.category.value,
                "guardrail_reason": eval_result.reason,
                "raw_query": raw_query,
                "final_response": refusal_msg,
                "messages": [AIMessage(content=refusal_msg)],
            }

        return {
            "is_blocked": False,
            "guardrail_category": ViolationCategory.NONE.value,
            "guardrail_reason": eval_result.reason,
            "raw_query": raw_query,
        }

    return input_guardrail_node


def create_guardrail_blocked_node():
    """Crea el nodo que emite la respuesta de rechazo educado y redirección comercial."""
    async def guardrail_blocked_node(state: MainGraphState) -> dict:
        final_response = state.get("final_response")
        category_str = state.get("guardrail_category")
        category = (
            ViolationCategory(category_str)
            if category_str in [c.value for c in ViolationCategory]
            else ViolationCategory.OUT_OF_SCOPE
        )

        if not final_response:
            final_response = format_guardrail_refusal(category)

        return {
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }

    return guardrail_blocked_node


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
            Clasifica la consulta del usuario en exactamente una de estas 7 intenciones comerciales:
            1. 'recommender': Recomendación, sugerencia, ranking, presupuesto o selección de productos:
               - Peticiones con superlativos de precio o rankings (ej: 'los más baratos', 'más económicos', 'más caros', 'mejores opciones', 'los 3 mejores').
               - Peticiones de recomendación o sugerencia general o por categoría (ej: 'recomiéndame cremas', 'qué me recomiendas para la piel', 'me podrías decir 3 cremas hidratantes').
               - Recomendaciones relacionales (cross-selling, up-selling o sustitutos: ej: 'qué combina con X', 'alternativas a Y').
            2. 'rag': Consultas puramente informativas, técnicas o de existencia de catálogo SIN pedir ranking, recomendación ni comparación de precios:
               - Preguntas sobre ingredientes, ficha técnica, modo de uso o existencia puntual (ej: '¿tienen protector solar?', '¿qué componentes tiene la crema Bio Milk?', '¿tienen labiales mate disponibles?').
            3. 'resolver': Cotización directa o precios/stock con códigos SKU numéricos específicos (ej: 'precio del SKU 1', 'cotiza 3 del SKU 5').
            4. 'contact': Gestión de cartera de clientes comerciales (ej: 'mis clientes', 'agregar cliente Carlos').
            5. 'sales': Gestión de pedidos y cotizaciones (ej: 'mis pedidos', 'confirmar orden SO001', 'cotizar a cliente').
            6. 'general': Saludos de cortesía, despedidas o preguntas sobre qué servicios puedes brindar.
            7. 'out_of_scope': Solicitud ajena al negocio (poemas, chistes, código/programación, tareas escolares, ciencias, consejos personales).
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
        output: dict = {
            "intent": intent,
            "intent_reasoning": reasoning,
            "raw_query": raw_query,
            "final_response": None,
            "cancellation_reason": None,
        }

        # Si el Router clasifica como out_of_scope, preparar bloqueo semántico y mensaje educado
        if intent == "out_of_scope":
            refusal_msg = format_guardrail_refusal(ViolationCategory.OUT_OF_SCOPE)
            output["is_blocked"] = True
            output["guardrail_category"] = ViolationCategory.OUT_OF_SCOPE.value
            output["guardrail_reason"] = reasoning
            output["final_response"] = refusal_msg
            output["messages"] = [AIMessage(content=refusal_msg)]

        # Asegurar que el mensaje del usuario quede registrado en el historial si no venía en messages
        existing_msgs = state.get("messages", [])
        if raw_query and (not existing_msgs or not (isinstance(existing_msgs[-1], HumanMessage) and existing_msgs[-1].content == raw_query)):
            user_msg = HumanMessage(content=raw_query)
            if "messages" in output:
                output["messages"] = [user_msg, *output["messages"]]
            else:
                output["messages"] = [user_msg]

        return output

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

        system_prompt = inject_soul(
            """
            Orienta al usuario con calidez ejecutiva sobre tus capacidades comerciales:
            1. Explorar y recomendar productos del catálogo comercial multimarca.
            2. Cotizar precios y gestionar pedidos o cotizaciones.
            3. Consultar y gestionar la cartera de clientes.

            HARD CONSTRAINT: No atiendas solicitudes ajenas a la gestión comercial (código, poemas, tareas escolares).
            """,
            role=SoulRole.GENERAL,
        )

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
def _route_after_router(state: MainGraphState) -> Literal[
    "general_chat",
    "product_rag",
    "product_resolver",
    "contact_manage",
    "sales_manage",
    "product_recomender",
    "guardrail_blocked",
]:
    """Enrutamiento condicional según la intención detectada."""
    intent = state.get("intent", "general")
    if intent == "out_of_scope":
        return "guardrail_blocked"
    elif intent == "rag":
        return "product_rag"
    elif intent == "recommender":
        return "product_recomender"
    elif intent == "resolver":
        return "product_resolver"
    elif intent == "contact":
        return "contact_manage"
    elif intent == "sales":
        return "sales_manage"
    return "general_chat"


def build_main_graph(
    llm: BaseChatModel,
    resolver_graph: Optional[Any] = None,
    rag_graph: Optional[Any] = None,
    contact_graph: Optional[Any] = None,
    sales_graph: Optional[Any] = None,
    recommender_graph: Optional[Any] = None,
    memory_store: Optional[UserMemoryStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Construye y compila el Grafo Principal unificado con guardrails, memoria, router y subgrafos."""
    workflow = StateGraph(state_schema=MainGraphState)

    # 0. Nodos de Guardrail
    input_guardrail = create_input_guardrail_node()
    guardrail_blocked = create_guardrail_blocked_node()

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
    compiled_contact = contact_graph if contact_graph is not None else get_contact_manage_graph()
    compiled_sales = sales_graph if sales_graph is not None else get_sales_manage_graph()
    compiled_recommender = recommender_graph if recommender_graph is not None else get_product_recomender_graph()

    # Registro de nodos en el grafo
    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("guardrail_blocked", guardrail_blocked)
    workflow.add_node("retrieve_memory", retrieve_memory_wrapper)
    workflow.add_node("router", router_node)
    workflow.add_node("general_chat", general_node)
    workflow.add_node("product_rag", compiled_rag)
    workflow.add_node("product_recomender", compiled_recommender)
    workflow.add_node("product_resolver", compiled_resolver)
    workflow.add_node("contact_manage", compiled_contact)
    workflow.add_node("sales_manage", compiled_sales)
    workflow.add_node("save_memory", save_memory_wrapper)

    # Flujo de ejecución:
    # 1. Capa 1 Guardrail (Fast-Path Determinista al entrar)
    workflow.add_edge(START, "input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        lambda s: "guardrail_blocked" if s.get("is_blocked") else "retrieve_memory",
        {
            "guardrail_blocked": "guardrail_blocked",
            "retrieve_memory": "retrieve_memory",
        },
    )

    # 2. Memoria hacia Router
    workflow.add_edge("retrieve_memory", "router")

    # 3. Bifurcación condicional del router (incluye out_of_scope -> guardrail_blocked)
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "general_chat": "general_chat",
            "product_rag": "product_rag",
            "product_recomender": "product_recomender",
            "product_resolver": "product_resolver",
            "contact_manage": "contact_manage",
            "sales_manage": "sales_manage",
            "guardrail_blocked": "guardrail_blocked",
        },
    )

    # Rutas hacia el fin
    workflow.add_edge("guardrail_blocked", END)
    workflow.add_edge("general_chat", END)
    workflow.add_edge("product_rag", END)
    workflow.add_edge("product_recomender", END)
    workflow.add_edge("product_resolver", "save_memory")
    workflow.add_edge("contact_manage", END)
    workflow.add_edge("sales_manage", END)
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
