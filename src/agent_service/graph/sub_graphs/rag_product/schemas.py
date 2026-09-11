from typing import List, Optional
from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(
        description="True si los productos devueltos por pgvector cubren el requerimiento técnico, False si no."
    )
    critique: Optional[str] = Field(
        default=None,
        description="Diagnóstico técnico del fallo si is_sufficient es False.",
    )

class FinalAnswer(BaseModel):
    matched_skus: List[str] = Field(
        description="Lista de códigos SKU exactos seleccionados para el usuario."
    )
    response_text: str = Field(
        description="Respuesta técnica y comercial explicando la selección."
    )

