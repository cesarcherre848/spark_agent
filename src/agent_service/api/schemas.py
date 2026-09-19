"""
src/agent_service/api/schemas.py - Modelos Pydantic v2 para el Webhook de Spark Agent
"""

from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, field_validator


class WebhookRequest(BaseModel):
    """Esquema de carga útil entrante para el Webhook de Spark Agent."""

    raw_query: str = Field(
        ...,
        description="Texto o consulta del usuario.",
        min_length=1,
    )
    phone_number: str = Field(
        ...,
        validation_alias=AliasChoices("phone_number", "phone", "from_number", "from", "numero"),
        description="Número de teléfono del remitente (WhatsApp / SMS / Móvil).",
        min_length=1,
    )
    session_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("session_id", "thread_id", "conversation_id"),
        description="ID de conversación o hilo opcional. Si no se provee, se deriva del teléfono.",
    )

    @field_validator("raw_query")
    @classmethod
    def validate_raw_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("raw_query no puede estar vacío ni contener solo espacios en blanco.")
        return clean

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        clean = str(v).strip()
        if not clean:
            raise ValueError("phone_number no puede estar vacío.")
        return clean


class WebhookResponse(BaseModel):
    """Esquema de respuesta devuelto por el Webhook de Spark Agent."""

    status: str = Field(default="success", description="Estado de procesamiento de la petición.")
    user_id: int = Field(..., description="ID de usuario/vendedor en Odoo asignado al contacto.")
    phone_number: str = Field(..., description="Número de teléfono normalizado del remitente.")
    session_id: str = Field(..., description="Identificador único del hilo de conversación (thread_id).")
    intent: Optional[str] = Field(default=None, description="Intención detectada por el Router de Spark Agent.")
    response: str = Field(..., description="Respuesta final generada por el agente para el usuario.")
    is_topic_finished: bool = Field(
        default=False,
        description="Indica si la interacción o transacción actual ha concluido.",
    )
