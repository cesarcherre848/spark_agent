from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExtractedSKUItem(BaseModel):
    sku: str = Field(
        description="Código SKU exacto o referencia interna del producto (ej: '1', '11', 'SKU-101')."
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Atributos extraídos como cantidad requerida, dimensiones o especificaciones técnicas (ej: {'cantidad': 10}).",
    )
    partner_name: Optional[str] = Field(
        default=None,
        description="Nombre del proveedor o partner solicitado para este SKU, si fue mencionado por el usuario.",
    )


class ExtractionResult(BaseModel):
    items: List[ExtractedSKUItem] = Field(
        default_factory=list,
        description="Lista de productos y SKUs identificados en la consulta del usuario.",
    )
    is_complete: bool = Field(
        description="True si se identificó al menos un SKU con información suficiente para procesar. False si falta información crítica o no se identificaron SKUs.",
    )
    missing_reason: Optional[str] = Field(
        default=None,
        description="Explicación detallada de qué información falta si is_complete es False.",
    )


class SynthesizeResponse(BaseModel):
    response_text: str = Field(
        description="Respuesta comercial final estructurada y clara para el cliente, organizada por partner con cantidades, precios y subtotales.",
    )
