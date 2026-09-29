"""
src/agent_service/graph/sub_graphs/sales_manage/schemas.py - Esquemas Pydantic estructurados para sales_manage
"""

from typing import Optional, List, Dict, Any, Literal
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


# ==============================================================================
# Esquemas Pydantic para el Patrón Planner - Executor con Reflexión
# ==============================================================================

class PlannedSalesTool(BaseModel):
    """Definición estructurada de una invocación de herramienta decidida por el planificador."""
    tool_name: Literal[
        "odoo_list_sales_orders",
        "odoo_view_quotation",
        "odoo_create_quotation",
        "odoo_update_quotation",
        "odoo_confirm_order",
        "odoo_remove_sale_order",
        "odoo_unlock_order",
        "odoo_update_order",
        "odoo_lock_order",
        "odoo_list_current_customers",
        "odoo_upsert_customer",
        "search_product_catalog",
        "get_product_odoo_details",
    ] = Field(description="Nombre exacto de la herramienta comercial a invocar.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Argumentos estructurados pasados a la herramienta.",
    )
    purpose: str = Field(description="Razón y objetivo comercial por el cual se invoca esta herramienta.")


class SalesPlan(BaseModel):
    """Plan de acción generado por el Sales Planner para resolver la consulta comercial."""
    reasoning: str = Field(
        description="Análisis estructurado de la intención, requerimientos, cliente activo y contexto multi-turno."
    )
    strategy: Literal[
        "list_orders",
        "view_order_details",
        "create_quotation",
        "update_quotation",
        "confirm_order",
        "cancel_order",
        "edit_confirmed_order",
        "resolve_customer",
    ] = Field(
        default="list_orders",
        description="Estrategia general adoptada para responder a la necesidad comercial.",
    )
    target_customer: Optional[str] = Field(
        default=None,
        description="Nombre del cliente o razón social de la empresa asociada a la orden.",
    )
    target_order_name: Optional[str] = Field(
        default=None,
        description="Código comercial de la orden o cotización (ej: 'SO001', 'S00003') si está referenciada.",
    )
    requires_hitl: bool = Field(
        default=False,
        description="True si la acción requiere confirmación humana obligatoria (confirmación definitiva, anulación, desbloqueo o ambigüedad).",
    )
    hitl_type: Optional[Literal["confirm_order", "remove_order", "unlock_order", "ambiguous_customer"]] = Field(
        default=None,
        description="Tipo específico de interruptor HITL requerido.",
    )
    hitl_question: Optional[str] = Field(
        default=None,
        description="Pregunta en lenguaje natural o resumen ejecutivo para presentar al usuario en el interrupt.",
    )
    tool_calls: List[PlannedSalesTool] = Field(
        default_factory=list,
        description="Lista ordenada de herramientas a ejecutar para obtener o mutar los datos necesarios en Odoo.",
    )


class SalesQualityRubricEvaluation(BaseModel):
    """Rúbrica de evaluación multi-criterio para evaluar la calidad, fidelidad y completitud de la respuesta."""
    relevance_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿La respuesta satisface directamente la consulta comercial pedida?",
    )
    grounding_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿Los códigos de orden, clientes, productos, cantidades y montos provienen 100% de las herramientas ejecutadas sin alucinación?",
    )
    detail_completeness_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: Si el usuario solicitó ver o incluir productos/ítems, ¿se listaron explícitamente sin preguntas perezosas de si desea verlos?",
    )
    whitelabel_and_safety_score: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Puntaje de 1 a 10: ¿Se evitaron IDs internos (partner_id, user_id), términos de backend (Odoo/ERP), y se aplicó formato comercial sobrio?",
    )
    is_approved: bool = Field(
        ...,
        description="True si todos los criterios son >= 7.0 y el promedio es >= 8.0.",
    )
    reflection_action: Literal["approve", "refine_synthesis", "replan_tools"] = Field(
        default="approve",
        description="Acción de reflexión: 'approve' si pasa; 'refine_synthesis' si los datos son válidos pero la redacción omitió detalles o fue perezosa; 'replan_tools' si faltan herramientas de Odoo.",
    )
    critique: Optional[str] = Field(
        default=None,
        description="Explicación detallada de los puntos débiles o fallos si la rúbrica es rechazada.",
    )
    remedy_suggestions: Optional[List[str]] = Field(
        default_factory=list,
        description="Acciones correctivas sugeridas para que el planificador o sintetizador ajuste en la siguiente iteración.",
    )
