from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class ProductCandidate(BaseModel):
    index: int = Field(description="Índice numérico asignado al producto candidato.")
    sku: Optional[str] = Field(default=None, description="Código SKU comercial del producto.")
    name: str = Field(description="Nombre comercial del producto.")
    description: str = Field(description="Descripción y detalles técnicos del producto.")
    pagina: Optional[int] = Field(default=None, description="Número de página del catálogo.")
    edicion: Optional[str] = Field(default=None, description="Edición o campaña del catálogo (ej: 'C10').")
    marca: Optional[str] = Field(default=None, description="Marca comercial del producto.")

    def to_prompt_item(self) -> str:
        sku_str = f" [SKU: {self.sku}]" if self.sku else ""
        tags_parts = []
        if self.marca:
            tags_parts.append(self.marca)
        if self.edicion:
            tags_parts.append(f"Campaña {self.edicion}")
        if self.pagina is not None:
            tags_parts.append(f"Pág. {self.pagina}")
        tags_str = f" [{ ' | '.join(tags_parts) }]" if tags_parts else ""
        return f"[{self.index}]{sku_str} {self.name}{tags_str}\n    Detalle: {self.description}"


def format_candidates_for_prompt(docs: List[Document], max_desc_len: int = 300) -> str:
    """Extrae y formatea candidatos de forma limpia y reutilizable para el contexto del LLM."""
    if not docs:
        return "No se encontraron productos disponibles en el catálogo."
    candidates = [
        ProductCandidate(
            index=idx,
            sku=str(doc.metadata.get("sku")) if doc.metadata.get("sku") is not None else None,
            name=doc.metadata.get("name") or "Producto sin nombre",
            description=(doc.page_content or "")[:max_desc_len].strip(),
            pagina=doc.metadata.get("pagina"),
            edicion=doc.metadata.get("edicion"),
            marca=doc.metadata.get("marca"),
        )
        for idx, doc in enumerate(docs, start=1)
    ]
    return "\n\n".join(c.to_prompt_item() for c in candidates)


class NormalizedQuery(BaseModel):
    search_query: str = Field(
        description="Consulta depurada y optimizada para catálogo: sustantivos técnicos, tipo de producto y atributos comerciales clave, sin saludos ni rodeos conversacionales."
    )
    pagina: Optional[int] = Field(
        default=None,
        description="Número de página del catálogo si el usuario lo menciona explícitamente (ej: 'página 12', 'pág 124' -> 124).",
    )
    edicion: Optional[str] = Field(
        default=None,
        description="Campaña o edición del catálogo si el usuario la menciona explícitamente (ej: 'C10', 'C-15', 'campaña 10' -> 'C10').",
    )
    marca: Optional[str] = Field(
        default=None,
        description="Marca comercial específica si el usuario la menciona explícitamente.",
    )


class EvaluationResult(BaseModel):
    is_sufficient: bool = Field(
        description="True si los productos cubren el requerimiento técnico del cliente, False si no."
    )
    selected_indices: List[int] = Field(
        default_factory=list,
        description="Lista de índices numéricos de los productos que sí son relevantes (ej. [1, 2]). Dejar vacío si ninguno es relevante."
    )
    critique: Optional[str] = Field(
        default=None,
        description="Diagnóstico técnico del fallo si is_sufficient es False, o resumen de la coincidencia si es True."
    )


class QueryRefinementResult(BaseModel):
    refined_query: str = Field(
        description=(
            "Consulta reescrita y optimizada para búsqueda vectorial y léxica. "
            "Debe incorporar sinónimos técnicos, términos comerciales de catálogo "
            "o corregir ambigüedades señaladas por la crítica."
        )
    )


class FinalAnswer(BaseModel):
    response_text: str = Field(
        description="Respuesta comercial y técnica orientada al cliente explicando las opciones recomendadas o disculpándose cordialmente si no hay disponibilidad."
    )