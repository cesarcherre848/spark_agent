from typing import Literal, Optional
from pydantic import BaseModel, Field


class HITLBinaryDecision(BaseModel):
    """Interpretación estructurada de la respuesta del usuario ante una interrupción Human-in-the-Loop."""
    decision: Literal["accept", "reject"] = Field(
        description=(
            "'accept' si el usuario aprueba, consiente, confirma, autoriza o desea proceder con la acción propuesta. "
            "'reject' si el usuario declina, rechaza, cancela, prefiere no hacerlo o expresa una negativa."
        )
    )
    reasoning: str = Field(
        default="",
        description="Breve justificación de cómo se interpretó la respuesta del usuario.",
    )


class HITLEntitySelection(BaseModel):
    """Interpretación estructurada de la selección de una entidad (ej: cliente) en lenguaje natural."""
    action: Literal["select", "new", "cancel"] = Field(
        description=(
            "'select': si el usuario eligió una de las opciones presentadas. "
            "'new': si el usuario indica que ninguna coincide y prefiere registrar o crear una nueva entidad. "
            "'cancel': si el usuario prefiere abortar o cancelar la operación."
        )
    )
    selected_index: Optional[int] = Field(
        default=None,
        description="Índice numérico basado en 0 de la opción elegida en la lista presentada.",
    )
    selected_entity_name: Optional[str] = Field(
        default=None,
        description="Nombre o identificador textual de la entidad que el usuario seleccionó.",
    )
    reasoning: str = Field(
        default="",
        description="Explicación de la elección deducida a partir de la respuesta del usuario.",
    )


class HITLOrderChoice(BaseModel):
    """Interpretación estructurada ante disyuntiva de cotizaciones/pedidos existentes vs nueva orden."""
    choice: Literal["new", "selected", "cancel"] = Field(
        description=(
            "'new': si el usuario prefiere abrir o crear una nueva orden/cotización desde cero. "
            "'selected': si el usuario prefiere reutilizar, actualizar o confirmar una orden existente ya identificada. "
            "'cancel': si el usuario prefiere cancelar o detener la operación."
        )
    )
    selected_order_name: Optional[str] = Field(
        default=None,
        description="Nombre comercial de la orden seleccionada (ej: SO001, SO004) si se especificó.",
    )
    reasoning: str = Field(
        default="",
        description="Justificación de la opción deducida según lo expresado por el usuario.",
    )
