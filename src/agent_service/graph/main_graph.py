"""
src/agent_service/graph/main_graph.py - Grafo Principal Unificado de Spark Agent

Integra los tres módulos fundamentales del sistema:
1. Memoria a largo plazo (user_memory) con PostgreSQL y pgvector.
2. Enrutador inteligente (LLM temp=0.0) hacia:
   - general_chat (charla general y asistencia)
   - product_advisor (asesor comercial de productos y recomendaciones)
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

from src.agent_service.core.llms.factory import get_default_llm, extract_clean_text
from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.graph.sub_graphs.user_memory.store import UserMemoryStore
from src.agent_service.graph.sub_graphs.user_memory.nodes import UserMemoryNodes
from src.agent_service.graph.sub_graphs.user_memory.checkpointer import get_postgres_checkpointer
from src.agent_service.graph.sub_graphs.product_resolver.graph import (
    build_product_resolver_graph,
    get_product_resolver_graph,
)
from src.agent_service.graph.sub_graphs.contact_manage.graph import (
    build_contact_manage_graph,
    get_contact_manage_graph,
)
from src.agent_service.graph.sub_graphs.sales_manage.graph import (
    build_sales_manage_graph,
    get_sales_manage_graph,
)
from src.agent_service.graph.sub_graphs.product_advisor.graph import (
    build_product_advisor_graph,
    get_product_advisor_graph,
)
from src.agent_service.graph.sub_graphs.sales_incentives.graph import (
    build_sales_incentives_graph,
    get_sales_incentives_graph,
)

from src.agent_service.soul import inject_soul, SoulRole
from src.agent_service.core.guardrails import (
    evaluate_input_guardrail,
    aevaluate_input_guardrail,
    format_guardrail_refusal,
    format_guardrail_warning,
    ViolationCategory,
    GuardrailAction,
    GuardrailResult,
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
    channel: Optional[Literal["whatsapp", "web", "other"]]

    # Guardrail de entrada y seguridad
    is_blocked: Optional[bool]
    is_warning: Optional[bool]
    guardrail_action: Optional[str]
    guardrail_category: Optional[str]
    guardrail_reason: Optional[str]
    guardrail_warning: Optional[str]
    guardrail_scores: Optional[Dict[str, float]]

    # Clasificación del Router
    intent: Optional[Literal["rag", "resolver", "general", "contact", "sales", "recommender", "out_of_scope"]]
    intent_reasoning: Optional[str]
    router_model: Optional[str]
    router_confidence: Optional[float]
    router_scores: Optional[Dict[str, Any]]

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

    # Pasamanos unificado de product_advisor:
    refined_query: Optional[str]
    retrieved_products: List[Any]
    is_sufficient: bool
    matched_skus: List[str]
    customer_purchase_history: Optional[Dict[str, Any]]
    plan_rationale: Optional[str]
    planned_tools: Optional[List[Dict[str, Any]]]
    candidate_products: Optional[List[Any]]
    enriched_products: Optional[List[Any]]
    filtered_products: Optional[List[Any]]
    meets_rubric: Optional[bool]
    rubric_scores: Optional[Dict[str, Any]]
    critique: Optional[str]
    suggested_improvements: Optional[List[str]]
    iteration_count: Optional[int]
    draft_response: Optional[str]

    # Contexto comercial activo multi-turno:
    active_customer: Optional[str]
    active_partner_id: Optional[int]
    active_order: Optional[str]
    active_order_id: Optional[int]
    customer_name: Optional[str]
    partner_id: Optional[int]
    target_customer_name: Optional[str]
    target_order_name: Optional[str]
    session_orders: Optional[Dict[str, Dict[str, Any]]]

    # Pasamanos de sales_incentives:
    active_incentives: Optional[List[Dict[str, Any]]]
    sales_trends: Optional[List[Dict[str, Any]]]
    gap_info: Optional[Dict[str, Any]]

    # Control de contexto del Juez (Context Judge)
    is_standalone: Optional[bool]
    context_usefulness: Optional[Literal["fully_useful", "partially_useful", "irrelevant_or_obsolete"]]
    relevant_context_summary: Optional[str]


# ==============================================================================
# 2. ESQUEMAS DEL ENRUTADOR Y JUEZ DE CONTEXTO
# ==============================================================================
class ContextJudgement(BaseModel):
    """Veredicto del Juez de Contexto sobre la consulta actual frente al historial."""
    is_standalone: bool = Field(
        description="True si la consulta actual puede entenderse y responderse por sí sola sin depender del historial (inicia un tema nuevo, cambia de cliente o saluda tras inactividad)."
    )
    context_usefulness: Literal["fully_useful", "partially_useful", "irrelevant_or_obsolete"] = Field(
        description="Grado de utilidad del historial previo para responder a la consulta actual."
    )
    relevant_context_summary: Optional[str] = Field(
        default=None,
        description="Resumen conciso del hecho, producto o cliente previo estrictamente necesario si context_usefulness no es irrelevant."
    )
    suggested_intent: Optional[Literal["advisor", "resolver", "contact", "sales", "incentives", "general"]] = Field(
        default=None,
        description="Intención sugerida tras el análisis contextual para orientar al router."
    )


class RouterDecision(BaseModel):
    """Clasificación estructurada de la consulta comercial."""
    intent: Literal["advisor", "resolver", "general", "contact", "sales", "out_of_scope", "rag", "recommender", "incentives"] = Field(
        description="Intención comercial detectada: 'advisor', 'resolver', 'contact', 'sales', 'incentives', 'general' u 'out_of_scope'."
    )
    reasoning: str = Field(description="Breve justificación de la decisión.")


# ==============================================================================
# 3. NODOS DEL GRAFO PRINCIPAL
# ==============================================================================
def create_input_guardrail_node():
    """Crea el nodo de evaluación de Guardrail híbrido (Capa 1 Regex + Capa 2 LLM)."""
    async def input_guardrail_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        eval_result = await aevaluate_input_guardrail(raw_query)

        if eval_result.is_blocked:
            # Salvaguarda defensiva: si la consulta contiene términos comerciales evidentes
            # de catálogo, clientes, pedidos o incentivos, desactivar falsos positivos de injection/out_of_scope.
            words = [w.strip("?,.:;!¡¿\"'()").lower() for w in raw_query.split() if w.strip("?,.:;!¡¿\"'()")]
            commercial_keywords = {
                "esika", "essika", "ésika", "yanbal", "yambal", "unique", "cyzone", "lbel", "l'bel",
                "precio", "precios", "cuanto", "cuánto", "marca", "marcas", "catalogo", "catálogo",
                "opcion", "opciones", "perfume", "perfumes", "colonia", "colonias", "fragancia", "fragancias",
                "labial", "labiales", "crema", "cremas", "maquillaje", "rimel", "mascara", "máscara",
                "hombre", "hombres", "mujer", "mujeres", "comprar", "cotizar", "pedido", "producto", "productos",
                "cliente", "clientes", "agregar", "anadir", "añadir", "nuevo", "nueva", "orden", "ordenes", "órdenes",
                "ganamas", "ganamás", "meta", "metas", "incentivo", "incentivos", "premio", "premios", "comision", "comisiones", "bono", "bonos"
            }
            has_commercial_kw = any(w in commercial_keywords for w in words)
            if has_commercial_kw and eval_result.category in (ViolationCategory.PROMPT_INJECTION, ViolationCategory.OUT_OF_SCOPE):
                logger.info(
                    f"[Guardrail Graph] Desbloqueando falso positivo para consulta comercial: '{raw_query}'"
                )
                eval_result = GuardrailResult(
                    action=GuardrailAction.ALLOW,
                    is_blocked=False,
                    is_warning=False,
                    category=ViolationCategory.NONE,
                    reason="Consulta comercial permitida por salvaguarda del grafo.",
                    scores=eval_result.scores,
                )
            elif eval_result.category == ViolationCategory.OUT_OF_SCOPE:
                # out_of_scope jamás debe ser un bloqueo duro en el guardrail de entrada
                logger.info(
                    f"[Guardrail Graph] Desclasificando bloqueo out_of_scope a advertencia cordial: '{raw_query}'"
                )
                eval_result = GuardrailResult(
                    action=GuardrailAction.WARN,
                    is_blocked=False,
                    is_warning=True,
                    category=ViolationCategory.OUT_OF_SCOPE,
                    reason=eval_result.reason,
                    warning_message=format_guardrail_warning(ViolationCategory.OUT_OF_SCOPE),
                )

        if eval_result.is_blocked:
            refusal_msg = eval_result.refusal_message or format_guardrail_refusal(eval_result.category)
            logger.warning(
                f"[Capa 1/2 Guardrail] Petición bloqueada. Categoría: {eval_result.category.value}. "
                f"Acción: {eval_result.action.value}. Razón: {eval_result.reason}"
            )
            return {
                "is_blocked": True,
                "is_warning": False,
                "guardrail_action": eval_result.action.value,
                "guardrail_category": eval_result.category.value,
                "guardrail_reason": eval_result.reason,
                "guardrail_warning": None,
                "guardrail_scores": eval_result.scores,
                "raw_query": raw_query,
                "final_response": refusal_msg,
                "messages": [AIMessage(content=refusal_msg)],
            }

        if eval_result.is_warning:
            # Context-aware smoothing: si es una pregunta corta de seguimiento y hay mensajes previos en la conversación,
            # o menciona marcas comerciales o atributos de catálogo, descartar la falsa alarma de out_of_scope.
            has_history = len(state.get("messages", [])) > 1
            words = [w.strip("?,.:;!¡¿") for w in raw_query.lower().split() if w.strip("?,.:;!¡¿")]
            commercial_keywords = {
                "esika", "essika", "ésika", "yanbal", "yambal", "unique", "cyzone", "lbel", "l'bel",
                "precio", "cuanto", "cuánto", "marca", "catalogo", "catálogo", "opcion", "opción",
                "segundo", "primero", "tercero", "perfume", "labial", "crema", "rimel", "mascara",
                "cliente", "clientes", "agregar", "nuevo", "nueva", "pedido", "orden", "ganamas", "meta", "bono"
            }
            has_commercial_kw = any(w in commercial_keywords for w in words)
            if eval_result.category == ViolationCategory.OUT_OF_SCOPE and (has_history or has_commercial_kw) and len(words) <= 8:
                logger.info(
                    f"[Guardrail] Suavizando advertencia de out_of_scope para consulta de seguimiento: '{raw_query}'"
                )
                return {
                    "is_blocked": False,
                    "is_warning": False,
                    "guardrail_action": GuardrailAction.ALLOW.value,
                    "guardrail_category": ViolationCategory.NONE.value,
                    "guardrail_reason": "Consulta de seguimiento contextual válida.",
                    "guardrail_warning": None,
                    "guardrail_scores": eval_result.scores,
                    "raw_query": raw_query,
                }

            logger.info(
                f"[Capa 2 Guardrail] Advertencia comercial emitida. Categoría: {eval_result.category.value}. "
                f"Acción: {eval_result.action.value}. Razón: {eval_result.reason}"
            )
            return {
                "is_blocked": False,
                "is_warning": True,
                "guardrail_action": eval_result.action.value,
                "guardrail_category": eval_result.category.value,
                "guardrail_reason": eval_result.reason,
                "guardrail_warning": eval_result.warning_message,
                "guardrail_scores": eval_result.scores,
                "raw_query": raw_query,
            }

        return {
            "is_blocked": False,
            "is_warning": False,
            "guardrail_action": eval_result.action.value,
            "guardrail_category": ViolationCategory.NONE.value,
            "guardrail_reason": eval_result.reason,
            "guardrail_warning": None,
            "guardrail_scores": eval_result.scores,
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


def create_context_judge_node(llm: BaseChatModel):
    """Crea el nodo Juez de Contexto que evalúa si la consulta actual es independiente o depende del historial."""
    structured_judge = bind_structured_output(bind_temperature(llm, 0.0), ContextJudgement)

    async def context_judge_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        messages = state.get("messages", [])
        # Excluimos el mensaje actual para inspeccionar el historial previo
        history_msgs = [m for m in messages if not (isinstance(m, HumanMessage) and str(m.content).strip() == raw_query)]

        if not history_msgs:
            logger.info(f"[ContextJudge] Sin historial previo. Consulta '{raw_query}' marcada como standalone.")
            return {
                "is_standalone": True,
                "context_usefulness": "irrelevant_or_obsolete",
                "relevant_context_summary": None,
            }

        trimmed_history = trim_messages(
            history_msgs,
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        judge_prompt = """
        Eres un Juez Crítico de Contexto Conversacional para un Asistente Comercial de Belleza y Distribución (Ésika, L'Bel, Cyzone, Yanbal).
        Tu única misión es evaluar la 'Consulta actual' respecto al 'Historial de conversación reciente' para decidir si la consulta actual es INDEPENDIENTE o si DEPENDE del historial previo.

        Criterios de Evaluación:
        1. Consulta INDEPENDIENTE (is_standalone = True, context_usefulness = 'irrelevant_or_obsolete'):
           - La consulta inicia un tema completamente nuevo, sin importar qué se habló antes.
           - Menciona una entidad o acción explícita nueva (ej: 'Agregar Cliente Valeria Osores', 'mis clientes', '¿cómo van mis metas de ganamás?', 'cotizar a Juan Pérez', 'muéstrame labiales mate').
           - Aunque antes se haya hablado de otro cliente o de pedidos pasados, si la consulta actual trae su propia acción o entidad, el historial previo es OBSOLETO e IRRELEVANTE.
           - relevant_context_summary DEBE ser None.

        2. Consulta DEPENDIENTE DE CONTEXTO (is_standalone = False, context_usefulness = 'fully_useful' o 'partially_useful'):
           - Preguntas de seguimiento cortas, anáforas, elipsis o preguntas conectivas:
             (ej: '¿y el segundo?', '¿cuánto cuesta ese?', 'cotízamelo', '¿y a Francia?', '¿y a Pedro?', '¿qué hay de Carlos?', '¿tienes en tono rojo?').
           - ELIPSIS DE ENTIDAD/CLIENTE: Si la consulta anterior involucraba creación, consulta o cotización para una persona o producto, y la consulta actual pregunta por otra persona usando conectores elípticos (ej: '¿Y a Francia?', 'Y Francia', '¿y para Carmen?'), la consulta es DEPENDIENTE DE CONTEXTO para que el asistente continúe la acción correspondiente con la nueva persona.
           - relevant_context_summary: resumen de 1 línea con la acción y entidad previa (ej. 'Acción previa de registro/gestión de clientes').

        REGLA DE ORO ANTI-SOLAPAMIENTO:
        Ante consultas completas que inicien explícitamente un tema nuevo, marca is_standalone = True y context_usefulness = 'irrelevant_or_obsolete'. Pero ante elipsis cortas de seguimiento ('¿Y a X?'), clasifícalas como DEPENDIENTES para continuar el flujo.
        """

        try:
            verdict: Optional[ContextJudgement] = await structured_judge.ainvoke([
                SystemMessage(content=judge_prompt),
                *trimmed_history,
                HumanMessage(content=f"Consulta actual: {raw_query}"),
            ])
            is_standalone = verdict.is_standalone if verdict else True
            context_usefulness = verdict.context_usefulness if verdict else "irrelevant_or_obsolete"
            summary = verdict.relevant_context_summary if verdict else None
        except Exception as exc:
            logger.warning(f"[ContextJudge] Fallback ante error en Juez LLM: {exc}")
            is_standalone = True
            context_usefulness = "irrelevant_or_obsolete"
            summary = None

        logger.info(
            f"[ContextJudge] Consulta: '{raw_query}' -> is_standalone={is_standalone}, "
            f"usefulness={context_usefulness}, summary={summary}"
        )

        return {
            "is_standalone": is_standalone,
            "context_usefulness": context_usefulness,
            "relevant_context_summary": summary,
        }

    return context_judge_node


def create_router_node(llm: BaseChatModel):
    """Crea el nodo enrutador inteligente basado en LLM estructurado."""
    structured_router = bind_structured_output(bind_temperature(llm, 0.0), RouterDecision)

    async def router_node(state: MainGraphState) -> dict:
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()
        user_context = state.get("user_context")

        is_standalone = state.get("is_standalone", True)
        context_usefulness = state.get("context_usefulness", "irrelevant_or_obsolete")
        relevant_context_summary = state.get("relevant_context_summary")

        # Si el Juez determinó que la consulta es autónoma o el contexto previo es obsoleto/irrelevante,
        # NO alimentamos el historial confuso al router para evitar solapamientos o sesgos (off-by-one).
        if is_standalone or context_usefulness == "irrelevant_or_obsolete":
            trimmed_history = []
            context_note = (
                "\n[AVISO DE CONTEXTO]: La consulta actual es completamente NUEVA e INDEPENDIENTE. "
                "Cualquier conversación previa sobre otros clientes, productos o metas es OBSOLETA. "
                "Clasifica EXCLUSIVAMENTE basándote en la consulta actual.\n"
            )
        else:
            trimmed_history = trim_messages(
                state.get("messages", []),
                max_tokens=15,
                strategy="last",
                token_counter=len,
            )
            context_note = (
                f"\n[AVISO DE CONTEXTO]: Consulta de seguimiento. Contexto previo relevante: {relevant_context_summary}\n"
                if relevant_context_summary else ""
            )

        system_prompt = f"""
            Clasifica la consulta del usuario en exactamente una de estas 7 intenciones comerciales:
            1. 'advisor': Asesoría y catálogo comercial de productos (búsquedas, recomendaciones, sugerencias, rankings, existencias, ingredientes o presupuestos):
               - Búsqueda o existencia de productos (ej: 'dame todos los perfumes para hombre de Yanbal', '¿tienen labiales mate?', 'muéstrame opciones de cremas').
               - Recomendaciones o sugerencias generales o por categoría (ej: 'qué me recomiendas para la piel', 'los 3 mejores perfumes', 'opciones más baratas', 'me podrías decir 3 cremas').
               - Preguntas informativas de catálogo (ej: 'ingredientes de crema Bio Milk', 'componentes de fragancia Ohm').
               - Recomendaciones relacionales (cross-selling, up-selling o sustitutos).
            2. 'resolver': Cotización directa o precios/stock con códigos SKU numéricos específicos (ej: 'precio del SKU 1', 'cotiza 3 del SKU 5').
            3. 'contact': Gestión de cartera de clientes comerciales (ej: 'mis clientes', 'agregar cliente Carlos', 'agregar cliente Valeria Osores').
            4. 'sales': Gestión de pedidos, órdenes y cotizaciones comerciales (ej: 'mis pedidos', 'confirmar orden SO001', 'detalle del pedido de Janet', 'cómo van las órdenes de Carlos', 'cotizar a cliente').
            5. 'incentives': Consulta de metas, avances de venta, comisiones, bonos, premios, escalas comerciales o planes de incentivos (ej: '¿cómo voy con mis metas?', '¿qué me falta para el bono?', '¿cuánto me falta para ganar la batidora o la tarjeta Visa?', '¿qué me conviene vender hoy para llegar al siguiente nivel de descuento?', 'planes de incentivos de Unique o Belcorp', 'mis comisiones', 'ganamas nivel 2').
            6. 'general': Saludos de cortesía, despedidas o preguntas sobre qué servicios puedes brindar.
            7. 'out_of_scope': Solicitud ajena al negocio (poemas, chistes, código/programación, tareas escolares, ciencias, consejos personales).

            REGLA DE ELIPSIS CONTEXTUAL:
            Si la consulta es una pregunta corta de seguimiento (ej: '¿Y a Francia?', '¿Y a Pedro?', '¿Y de María?', '¿Y qué hay de Carlos?') y el contexto previo inmediato trataba de creación o consulta de clientes ('contact') o de pedidos ('sales'), clasifícala en ESA MISMA intención ('contact' o 'sales'). NUNCA la clasifiques como 'out_of_scope' ni como país si puede ser el nombre de una persona en la gestión comercial.
            {context_note}
        """

        context_block = f"\nAntecedentes de memoria a largo plazo:\n{user_context}\n" if (user_context and not is_standalone) else ""
        user_prompt = f"""
            {context_block}Consulta actual: {raw_query}
        """

        try:
            decision: Optional[RouterDecision] = await structured_router.ainvoke([
                SystemMessage(content=system_prompt),
                *trimmed_history,
                HumanMessage(content=user_prompt),
            ])
            intent = decision.intent if decision and hasattr(decision, "intent") else "general"
            reasoning = decision.reasoning if decision and hasattr(decision, "reasoning") else "Enrutamiento por defecto"
        except Exception as exc:
            logger.error(f"[Router/LLM] Error en enrutamiento con LLM: {exc}")
            intent = "general"
            reasoning = f"Fallback emergente ante error de LLM: {exc}"

        logger.info(f"Router LLM clasificó consulta como: '{intent}' (Razón: {reasoning})")

        output: dict = {
            "intent": intent,
            "intent_reasoning": reasoning,
            "router_model": "llm",
            "raw_query": raw_query,
            "final_response": None,
            "cancellation_reason": None,
            "iteration_count": 0,
        }

        # Si el Router clasifica como out_of_scope, preparar bloqueo o permitir advertencia si ya venía marcada
        if intent == "out_of_scope":
            if state.get("is_warning"):
                logger.info(
                    "[Router] Consulta clasificada como out_of_scope con advertencia previa (WARN). "
                    "Permitiendo orientación cordial en general_chat."
                )
            else:
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
            max_tokens=15,
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
            FORMAT CONSTRAINT: Responde exclusivamente con texto conversacional directo para el usuario. Nunca generes diccionarios, JSON ni metadatos técnicos.
            """,
            role=SoulRole.GENERAL,
        )

        user_context = state.get("user_context")
        context_block = f"\nAntecedentes del usuario:\n{user_context}\n" if user_context else ""
        channel_instructions = BaseSynthesizerNode.get_channel_prompt_instructions(state)

        response = await chat_model.ainvoke([
            SystemMessage(content=f"{system_prompt}\n\n{channel_instructions}"),
            *trimmed_history,
            HumanMessage(content=f"{context_block}{raw_query}"),
        ])

        reply_text = extract_clean_text(response.content)

        return BaseSynthesizerNode.format_final_response(reply_text, state)


    return general_chat_node


# ==============================================================================
# 4. CONSTRUCCIÓN DEL GRAFO PRINCIPAL (SPARK_AGENT)
# ==============================================================================
def _route_after_router(state: MainGraphState) -> Literal[
    "general_chat",
    "product_advisor",
    "product_resolver",
    "contact_manage",
    "sales_manage",
    "sales_incentives",
    "guardrail_blocked",
]:
    """Enrutamiento condicional según la intención detectada."""
    intent = state.get("intent", "general")
    if intent == "out_of_scope":
        if state.get("is_warning"):
            return "general_chat"
        return "guardrail_blocked"
    elif intent in ("advisor", "product_advisor", "rag", "recommender"):
        return "product_advisor"
    elif intent == "resolver":
        return "product_resolver"
    elif intent == "contact":
        return "contact_manage"
    elif intent == "sales":
        return "sales_manage"
    elif intent in ("incentives", "sales_incentives"):
        return "sales_incentives"
    return "general_chat"


def build_main_graph(
    llm: BaseChatModel,
    resolver_graph: Optional[Any] = None,
    contact_graph: Optional[Any] = None,
    sales_graph: Optional[Any] = None,
    advisor_graph: Optional[Any] = None,
    incentives_graph: Optional[Any] = None,
    memory_store: Optional[UserMemoryStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    rag_graph: Optional[Any] = None,
    recommender_graph: Optional[Any] = None,
):
    """Construye y compila el Grafo Principal unificado con guardrails, memoria, router y subgrafos."""
    workflow = StateGraph(state_schema=MainGraphState)

    # 0. Nodos de Guardrail y Juez de Contexto
    input_guardrail = create_input_guardrail_node()
    guardrail_blocked = create_guardrail_blocked_node()
    context_judge = create_context_judge_node(llm)

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
    compiled_advisor = (
        advisor_graph if advisor_graph is not None
        else (rag_graph if rag_graph is not None else (recommender_graph if recommender_graph is not None else get_product_advisor_graph()))
    )
    compiled_contact = contact_graph if contact_graph is not None else get_contact_manage_graph()
    compiled_sales = sales_graph if sales_graph is not None else get_sales_manage_graph()
    compiled_incentives = incentives_graph if incentives_graph is not None else get_sales_incentives_graph()

    # Registro de nodos en el grafo
    workflow.add_node("input_guardrail", input_guardrail)
    workflow.add_node("guardrail_blocked", guardrail_blocked)
    workflow.add_node("context_judge", context_judge)
    workflow.add_node("retrieve_memory", retrieve_memory_wrapper)
    workflow.add_node("router", router_node)
    workflow.add_node("general_chat", general_node)
    workflow.add_node("product_advisor", compiled_advisor)
    workflow.add_node("product_resolver", compiled_resolver)
    workflow.add_node("contact_manage", compiled_contact)
    workflow.add_node("sales_manage", compiled_sales)
    workflow.add_node("sales_incentives", compiled_incentives)
    workflow.add_node("save_memory", save_memory_wrapper)

    # Flujo de ejecución:
    # 1. Capa 1 y 2 Guardrail (Fast-Path Determinista + LLM Evaluator)
    workflow.add_edge(START, "input_guardrail")
    workflow.add_conditional_edges(
        "input_guardrail",
        lambda s: "guardrail_blocked" if s.get("is_blocked") else "context_judge",
        {
            "guardrail_blocked": "guardrail_blocked",
            "context_judge": "context_judge",
        },
    )

    # 2. Context Judge hacia Recuperación de Memoria y Router
    workflow.add_edge("context_judge", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "router")

    # 3. Bifurcación condicional del router (incluye out_of_scope -> guardrail_blocked)
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "general_chat": "general_chat",
            "product_advisor": "product_advisor",
            "product_resolver": "product_resolver",
            "contact_manage": "contact_manage",
            "sales_manage": "sales_manage",
            "sales_incentives": "sales_incentives",
            "guardrail_blocked": "guardrail_blocked",
        },
    )

    # Rutas hacia el fin
    workflow.add_edge("guardrail_blocked", END)
    workflow.add_edge("general_chat", END)
    workflow.add_edge("product_advisor", END)
    workflow.add_edge("product_resolver", "save_memory")
    workflow.add_edge("contact_manage", END)
    workflow.add_edge("sales_manage", END)
    workflow.add_edge("sales_incentives", END)
    workflow.add_edge("save_memory", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=resolved_checkpointer)


def get_main_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Fábrica para LangGraph Studio, Webhook y punto de entrada oficial."""
    llm = get_default_llm()
    pool = get_db_pool()
    embeddings = get_embedding_service()
    store = UserMemoryStore(pool=pool, embedding_service=embeddings)
    cp = checkpointer
    if cp is None:
        try:
            if pool and not getattr(pool, "closed", True):
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                cp = AsyncPostgresSaver(pool)
            else:
                cp = MemorySaver()
        except Exception as e:
            logger.warning(f"No fue posible inicializar AsyncPostgresSaver ({e}), usando MemorySaver.")
            cp = MemorySaver()
    return build_main_graph(llm=llm, memory_store=store, checkpointer=cp)
