import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.core.hitl.schemas import (
    HITLBinaryDecision,
    HITLEntitySelection,
    HITLOrderChoice,
)

logger = logging.getLogger(__name__)


async def parse_hitl_binary_decision(
    user_input: str,
    context_question: str,
    llm: BaseChatModel,
) -> bool:
    """Evalúa la respuesta del usuario en lenguaje natural y determina si es una aprobación ('accept' -> True) o rechazo ('reject' -> False)."""
    text = (user_input or "").strip()
    if not text:
        return False

    system_prompt = """
        Eres un evaluador especializado de decisiones humanas para un sistema comercial ERP.
        Tu labor es interpretar la intención del usuario ante una pregunta de confirmación y clasificarla de forma binaria:

        1. 'accept': El usuario desea proceder, consiente, autoriza, asiente o confirma (ejemplos: 'sí', 'dale', 'procede', 'haz el cambio', 'adelante', 'por favor actualízalo', 'elimínalo', 'de acuerdo', 'sí hazlo', 'claro').
        2. 'reject': El usuario declina, rechaza, cancela, prefiere no hacerlo, expresa negativa o pide detener la operación (ejemplos: 'no', 'cancelar', 'déjalo ahí', 'mejor no hagas nada', 'para nada', 'no quiero que lo borres', 'aborta').
    """

    user_prompt = f"""
        Pregunta de confirmación realizada al usuario:
        {context_question}

        Respuesta del usuario en lenguaje natural:
        {text}
    """

    try:
        classifier = bind_structured_output(bind_temperature(llm, 0.0), HITLBinaryDecision)
        result: Optional[HITLBinaryDecision] = await classifier.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if result and hasattr(result, "decision"):
            logger.info(f"HITL Binary Decision: '{result.decision}' (Razón: {getattr(result, 'reasoning', '')})")
            return result.decision == "accept"
        if isinstance(result, dict) and "decision" in result:
            logger.info(f"HITL Binary Decision dict: '{result['decision']}'")
            return result["decision"] == "accept"
    except Exception as e:
        logger.warning(f"Error al clasificar decisión HITL binaria con LLM: {e}. Aplicando fallback seguro (reject).")

    return False


async def parse_hitl_entity_selection(
    user_input: str,
    options: list,
    llm: BaseChatModel,
) -> HITLEntitySelection:
    """Interpreta la respuesta en lenguaje natural ante opciones de entidades (ej. clientes) y resuelve la opción elegida."""
    text = (user_input or "").strip()
    if not text:
        return HITLEntitySelection(action="cancel", reasoning="Respuesta vacía del usuario.")

    options_text = "\n".join(f"- Opción {i}: {opt}" for i, opt in enumerate(options))
    system_prompt = """
        Eres un asistente especializado en resolución de entidades en un sistema comercial ERP.
        Tu misión es analizar la respuesta de un usuario que debe elegir una opción entre varias presentadas (por ejemplo, clientes con nombres similares).

        Reglas:
        - Si el usuario elige claramente una opción (por nombre, número ordinal como 'la primera', parte del teléfono o descripción), clasifica action='select', selected_index=<índice 0-based> y selected_entity_name=<nombre>.
        - Si el usuario dice que no es ninguno, que prefiere crearlo nuevo, o dar de alta un nuevo cliente, clasifica action='new'.
        - Si el usuario dice cancelar, salir, o rechaza continuar, clasifica action='cancel'.
    """

    user_prompt = f"""
        Opciones presentadas al usuario:
        {options_text}

        Respuesta del usuario:
        {text}
    """

    try:
        classifier = bind_structured_output(bind_temperature(llm, 0.0), HITLEntitySelection)
        result: Optional[HITLEntitySelection] = await classifier.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if result and hasattr(result, "action"):
            logger.info(f"HITL Entity Selection: action={result.action}, idx={result.selected_index}, name={result.selected_entity_name}")
            return result
        if isinstance(result, dict):
            return HITLEntitySelection(**result)
    except Exception as e:
        logger.warning(f"Error al clasificar selección de entidad con LLM: {e}. Fallback a cancel.")

    return HITLEntitySelection(action="cancel", reasoning="Fallback por error de clasificación.")


async def parse_hitl_order_choice(
    user_input: str,
    candidate_orders: list,
    llm: BaseChatModel,
) -> HITLOrderChoice:
    """Interpreta si el usuario prefiere reutilizar/actualizar una orden existente o abrir una nueva desde cero."""
    text = (user_input or "").strip()
    if not text:
        return HITLOrderChoice(choice="cancel", reasoning="Respuesta vacía.")

    orders_text = "\n".join(f"- {ord_desc}" for ord_desc in candidate_orders)
    system_prompt = """
        Eres un clasificador de decisiones comerciales para un ERP de ventas.
        Se le presentaron al usuario órdenes/cotizaciones existentes similares para un cliente.
        Tu labor es interpretar si el usuario desea:
        1. 'selected': Reutilizar, actualizar o confirmar una de las órdenes existentes (ej: 'usa la primera', 'actualiza la que teníamos', 'usa la orden SO001', 'agrega a la existente').
        2. 'new': Desea ignorar las anteriores y abrir una nueva orden/cotización desde cero (ej: 'haz una nueva', 'crea una orden nueva', 'otra distinta', 'ninguna de esas, una nueva').
        3. 'cancel': Cancela o prefiere no continuar (ej: 'cancela', 'no hagas nada').
    """

    user_prompt = f"""
        Órdenes existentes candidatas:
        {orders_text}

        Respuesta del usuario:
        {text}
    """

    try:
        classifier = bind_structured_output(bind_temperature(llm, 0.0), HITLOrderChoice)
        result: Optional[HITLOrderChoice] = await classifier.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        if result and hasattr(result, "choice"):
            logger.info(f"HITL Order Choice: choice={result.choice}, order={result.selected_order_name}")
            return result
        if isinstance(result, dict):
            return HITLOrderChoice(**result)
    except Exception as e:
        logger.warning(f"Error al clasificar elección de orden con LLM: {e}. Fallback a cancel.")

    return HITLOrderChoice(choice="cancel", reasoning="Fallback por error de clasificación.")
