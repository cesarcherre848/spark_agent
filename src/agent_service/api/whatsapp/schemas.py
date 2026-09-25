"""
src/agent_service/api/whatsapp/schemas.py - Modelos Pydantic v2 para WhatsApp Cloud API de Meta
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


# ==============================================================================
# 1. MODELOS DE WEBHOOK ENTRANTE DE META
# ==============================================================================

class MetaTextBody(BaseModel):
    """Contenido del mensaje de texto enviado por el usuario."""
    body: str


class MetaMessage(BaseModel):
    """Representación de un mensaje individual en el payload de Meta."""
    model_config = ConfigDict(extra="ignore")

    from_number: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str = "text"
    text: Optional[MetaTextBody] = None


class MetaContactProfile(BaseModel):
    """Perfil del contacto que interactúa."""
    name: Optional[str] = None


class MetaContact(BaseModel):
    """Contacto remitente en la plataforma de WhatsApp."""
    profile: Optional[MetaContactProfile] = None
    wa_id: str


class MetaStatus(BaseModel):
    """Notificación de estado de entrega de mensaje (sent, delivered, read, failed)."""
    model_config = ConfigDict(extra="ignore")

    id: str
    status: Literal["sent", "delivered", "read", "failed"]
    timestamp: str
    recipient_id: str
    errors: Optional[List[Dict[str, Any]]] = None


class MetaMetadata(BaseModel):
    """Metadatos de la cuenta corporativa receptora."""
    display_phone_number: Optional[str] = None
    phone_number_id: str


class MetaValue(BaseModel):
    """Contenido del cambio reportado por el webhook de WhatsApp."""
    model_config = ConfigDict(extra="ignore")

    messaging_product: str = "whatsapp"
    metadata: Optional[MetaMetadata] = None
    contacts: Optional[List[MetaContact]] = Field(default_factory=list)
    messages: Optional[List[MetaMessage]] = Field(default_factory=list)
    statuses: Optional[List[MetaStatus]] = Field(default_factory=list)


class MetaChange(BaseModel):
    """Cambio en la cuenta de WhatsApp Business."""
    field: str
    value: MetaValue


class MetaEntry(BaseModel):
    """Entrada en el payload de webhook de Meta."""
    id: str
    changes: List[MetaChange]


class MetaWebhookPayload(BaseModel):
    """Estructura raíz de la notificación push de Meta WhatsApp Cloud API."""
    object: str = "whatsapp_business_account"
    entry: List[MetaEntry] = Field(default_factory=list)


# ==============================================================================
# 2. MODELOS PARA MENSAJES SALIENTES Y RESPUESTAS
# ==============================================================================

class WhatsAppOutboundTextMessage(BaseModel):
    """Estructura de la solicitud HTTP saliente hacia Meta Graph API."""
    messaging_product: str = "whatsapp"
    recipient_type: str = "individual"
    to: str
    type: str = "text"
    text: Dict[str, Any]


class WhatsAppSendMessageResponse(BaseModel):
    """Respuesta devuelta por Meta Graph API al enviar un mensaje."""
    messaging_product: str = "whatsapp"
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
