from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class CustomerItem(BaseModel):
    """Información individual de un cliente en solicitudes por lote o unitarias."""
    name: str = Field(
        description="Nombre completo o razón social del cliente.",
    )
    phones: List[str] = Field(
        default_factory=list,
        description="Lista de números telefónicos o celulares detectados para este cliente.",
    )
    contact_id: Optional[int] = Field(
        default=None,
        description="ID numérico de Odoo si se hace referencia a un cliente específico.",
    )


class CustomerExtractionResult(BaseModel):
    """Extracción estructurada de la acción y datos del cliente."""
    action: Literal["list", "upsert", "remove"] = Field(
        description=(
            "'list': para ver, listar o consultar clientes asignados al vendedor. "
            "'upsert': para registrar, crear o actualizar un cliente. "
            "'remove': para eliminar o archivar un cliente."
        )
    )
    name: Optional[str] = Field(
        default=None,
        description="Nombre completo o razón social del cliente principal (o único).",
    )
    phones: List[str] = Field(
        default_factory=list,
        description="Lista de números telefónicos o celulares proporcionados para el cliente principal.",
    )
    contact_id: Optional[int] = Field(
        default=None,
        description="ID numérico de Odoo en caso de hacer referencia a un cliente específico existente.",
    )
    customers: List[CustomerItem] = Field(
        default_factory=list,
        description="Lista completa de todos los clientes a registrar o consultar cuando se solicitan múltiples.",
    )


class DuplicateCheckResult(BaseModel):
    """Veredicto del auditor LLM sobre posibles duplicados en la cartera."""
    has_duplicates: bool = Field(
        description="True si algún candidato existente parece ser la misma persona o empresa según su nombre o teléfonos."
    )
    matched_customer_id: Optional[int] = Field(
        default=None,
        description="ID del cliente existente que coincide como duplicado potencial.",
    )
    reasoning: str = Field(
        description="Explicación clara de por qué es o no es un duplicado potencial.",
    )


class CustomerSynthesizeResponse(BaseModel):
    """Respuesta final sintetizada dirigida al usuario/vendedor."""
    response_text: str = Field(
        description="Respuesta profesional, concisa y cordial resumiendo el resultado de la operación."
    )
