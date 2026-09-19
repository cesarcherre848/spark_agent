"""
src/agent_service/graph/sub_graphs/sales_manage/schemas.py - Esquemas Pydantic estructurados para sales_manage
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class SalesSKUItem(BaseModel):
    """Ítem individual extraído de la consulta de ventas."""
    sku: str = Field(description="Código de referencia comercial / SKU del producto, o nombre comercial si no se especificó SKU numérico (ej: '6189', '5104', 'Delineador', 'Sexy Glam').")
    qty: float = Field(default=1.0, description="Cantidad de unidades referenciadas.")
    price_unit: Optional[float] = Field(default=None, description="Precio unitario acordado si se especificó.")
    action: Literal["add", "set", "remove", "subtract"] = Field(
        default="add",
        description=(
            "'add': Para agregar o sumar unidades al pedido (ej: 'agrega 2 del SKU 6189', '2 del código 5104'). "
            "'set': Para fijar o cambiar la cantidad a un número exacto (ej: 'cambia la cantidad a 5', 'deja solo 3'). "
            "'remove': Para eliminar o borrar completamente el producto de la orden (ej: 'elimina el producto 6189', 'quita el delineador'). "
            "'subtract': Para restar, quitar o reducir cierta cantidad de unidades (ej: 'elimina 2 items de 6189', 'resta 1 unidad', 'quita 2')."
        ),
    )



class SalesExtractionResult(BaseModel):
    """Extracción estructurada de la intención de ventas y sus atributos."""
    action: Literal["list", "view", "upsert", "add_items", "remove_items", "confirm", "edit_order", "remove"] = Field(
        description=(
            "'list': para listar o consultar órdenes y cotizaciones del vendedor en general. "
            "'view': para ver el detalle de una orden o cotización específica (ej: 'muestrame la cotizacion 03 de Adhara', 'ver orden S00003'). "
            "'upsert': para crear una cotización nueva desde cero o cotizar productos nuevos. "
            "'add_items': para agregar o sumar productos/unidades a una cotización activa o existente (ej: 'agrega 2 delineadores', 'suma 3 del SKU 5104'). "
            "'remove_items': para quitar, restar o eliminar unidades o productos de una cotización activa o existente (ej: 'quita 2 unidades del Delineador', 'elimina 2 items de 6189'). "
            "'confirm': para confirmar una cotización u orden pasando a venta oficial. "
            "'edit_order': para modificar cantidades o líneas de un pedido ya confirmado. "
            "'remove': para cancelar o anular una cotización o pedido."
        )
    )
    customer: Optional[str] = Field(
        default=None,
        description="Nombre del cliente o razón social de la empresa referenciada en la solicitud.",
    )
    items: List[SalesSKUItem] = Field(
        default_factory=list,
        description="Lista de productos/SKUs con sus cantidades asociadas a la transacción.",
    )
    status: Optional[str] = Field(
        default=None,
        description="Filtro de estado comercial solicitado (ej: 'draft' para cotizaciones, 'sale' para pedidos confirmados, 'cancel' para cancelados).",
    )
    period: Optional[str] = Field(
        default=None,
        description="Periodo o intervalo temporal mencionado (ej: 'hoy', 'este mes', 'última semana').",
    )
    order_name: Optional[str] = Field(
        default=None,
        description="Código comercial de la orden o cotización (ej: '03', 'SO001', 'S00003') si el usuario hace referencia a una específica.",
    )


class SalesDuplicateCheckResult(BaseModel):
    """Veredicto del auditor LLM sobre cotizaciones abiertas similares del cliente."""
    has_duplicates: bool = Field(
        description="True si alguna cotización abierta ('draft' o 'sent') contiene productos similares o parece ser la misma transacción."
    )
    matched_order_name: Optional[str] = Field(
        default=None,
        description="Código comercial de la cotización similar identificada (ej: 'SO002').",
    )
    reasoning: str = Field(
        default="",
        description="Breve análisis de por qué se considera o no un duplicado potencial.",
    )


class SalesSynthesizeResponse(BaseModel):
    """Respuesta final estructurada dirigida al usuario/vendedor."""
    response_text: str = Field(
        description="Respuesta profesional, cordial y clara resumiendo la operación comercial sin revelar IDs técnicos internos de base de datos."
    )


class SalesReflectionResult(BaseModel):
    """Resultado del autoanálisis crítico del agente sobre la intención y consistencia comercial."""
    is_intent_clear: bool = Field(
        description="True si la intención del usuario y la asignación cliente-orden es inequívoca. False si persiste ambigüedad, contradicción o conflicto que requiere confirmación humana (HITL)."
    )
    resolved_action: Literal["upsert", "add_items", "remove_items", "list", "view", "confirm", "edit_order", "remove"] = Field(
        description="Acción comercial rectificada tras reflexionar sobre el contexto de la conversación y el estado de órdenes en Odoo."
    )
    target_customer_name: Optional[str] = Field(
        default=None,
        description="Nombre del cliente objetivo verificado y libre de contaminación de turnos anteriores.",
    )
    target_order_name: Optional[str] = Field(
        default=None,
        description="Código de la orden objetivo (ej: 'S00003') si realmente pertenece al cliente solicitado. None si es una orden nueva o no existe cotización.",
    )
    reset_active_order: bool = Field(
        default=False,
        description="True si la orden previa en memoria pertenecía a otro cliente y debe ser purgada del estado.",
    )
    reasoning: str = Field(
        description="Razonamiento crítico: por qué se ajustó la acción, por qué se descartó la orden previa o qué inconsistencia se detectó."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Pregunta en lenguaje natural para formular al usuario si is_intent_clear es False.",
    )
    suggested_options: Optional[List[str]] = Field(
        default=None,
        description="Opciones concisas sugeridas para la aclaración en el interrupt HITL.",
    )

