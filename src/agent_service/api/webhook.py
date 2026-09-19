"""
src/agent_service/api/webhook.py - Aplicación FastAPI y endpoints de Webhook para Spark Agent
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Any
from fastapi import FastAPI, Depends, status, HTTPException

from dotenv import load_dotenv
from src.agent_service.api.schemas import WebhookRequest, WebhookResponse
from src.agent_service.api.service import (
    PhoneUserResolver,
    get_phone_user_resolver,
    normalize_phone,
)
from src.agent_service.graph.main_graph import get_main_graph
from src.agent_service.config.database import close_db_pool
from src.agent_service.core.llms.factory import clean_text_from_tool_call_artifacts

load_dotenv(".env.dev")


logger = logging.getLogger(__name__)

_GLOBAL_GRAPH_APP: Optional[Any] = None


def get_agent_graph() -> Any:
    """Retorna la aplicación del Grafo Principal compilado de Spark Agent."""
    global _GLOBAL_GRAPH_APP
    if _GLOBAL_GRAPH_APP is None:
        _GLOBAL_GRAPH_APP = get_main_graph()
    return _GLOBAL_GRAPH_APP


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida asíncrono de la aplicación FastAPI."""
    logger.info("Iniciando Spark Agent Webhook API...")
    # Precargar el grafo principal al inicio
    try:
        get_agent_graph()
    except Exception as e:
        logger.warning(f"No fue posible pre-cargar el grafo al iniciar: {e}")
    yield
    logger.info("Cerrando recursos de Spark Agent...")
    try:
        await close_db_pool()
    except Exception as e:
        logger.warning(f"Error al cerrar db pool: {e}")


app = FastAPI(
    title="Spark Agent Webhook API",
    version="1.0.0",
    description="Endpoint HTTP Webhook para integración de Spark Agent con canales de mensajería (OpenClaw, WhatsApp, etc.).",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Verifica de forma no bloqueante que el servicio Webhook esté operativo."""
    return {"status": "ok", "service": "spark_agent_webhook"}


@app.post(
    "/api/v1/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    tags=["Webhook"],
)
@app.post(
    "/webhook",
    response_model=WebhookResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def process_webhook(
    payload: WebhookRequest,
    graph: Any = Depends(get_agent_graph),
    resolver: PhoneUserResolver = Depends(get_phone_user_resolver),
):
    """Procesa de forma 100% asíncrona y no bloqueante una consulta entrante:

    1. Normaliza el número de teléfono extraído.
    2. Resuelve el user_id asignado en Odoo mediante caché en memoria TTLCache.
    3. Construye el thread_id a partir de session_id o wa_{phone}.
    4. Invoca asíncronamente get_main_graph().ainvoke(...) sin bloquear el event loop.
    5. Retorna la respuesta estructurada del agente.
    """
    normalized_phone = normalize_phone(payload.phone_number)
    if not normalized_phone:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="El número de teléfono no contiene dígitos válidos.",
        )

    # 1. Determinar el identificador del hilo (thread_id) para la sesión
    thread_id = payload.session_id or f"wa_{normalized_phone}"

    # 2. Resolver user_id en Odoo mediante caché asíncrona en memoria
    user_id = await resolver.resolve_user_id(normalized_phone)
    if user_id is None:
        logger.warning(
            f"Acceso denegado: el número telefónico '{normalized_phone}' no está registrado en Odoo o no cuenta con comercial asignado."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El número de teléfono no está registrado o no cuenta con autorización en el sistema.",
        )

    # 3. Invocar asíncronamente el Grafo Principal de Spark Agent
    try:
        graph_input = {
            "raw_query": payload.raw_query,
            "user_id": user_id,
            "session_id": thread_id,
        }
        result = await graph.ainvoke(
            graph_input,
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        logger.error(f"Error en la ejecución del grafo principal: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno procesando la consulta con el agente: {str(e)}",
        )

    # 4. Extraer respuesta final y metadatos sanitizados
    raw_final_response = result.get("final_response") or ""
    if not raw_final_response:
        messages = result.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            raw_final_response = messages[-1].content

    final_response = clean_text_from_tool_call_artifacts(str(raw_final_response))

    intent = result.get("intent")
    is_topic_finished = bool(result.get("is_topic_finished", False))

    return WebhookResponse(
        status="success",
        user_id=user_id,
        phone_number=normalized_phone,
        session_id=thread_id,
        intent=intent,
        response=final_response,
        is_topic_finished=is_topic_finished,
    )
