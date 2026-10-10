from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class ExtractedSKUItem(BaseModel):
    sku: str = Field(
        description="Código comercial SKU o código de referencia del producto (ej: '20013876', '47000', '20010996')."
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Atributos extraídos como cantidad requerida, dimensiones o especificaciones técnicas (ej: {'cantidad': 10}).",
    )
    partner_name: Optional[str] = Field(
        default=None,
        description="Nombre del proveedor o partner solicitado para este SKU, si fue mencionado por el usuario.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_sku_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "sku" not in data or not data["sku"]:
                candidate = (
                    data.get("code")
                    or data.get("codigo")
                    or data.get("product_sku")
                    or data.get("reference")
                    or data.get("product_name")
                    or data.get("name")
                    or ""
                )
                data["sku"] = str(candidate).strip()
        return data


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
