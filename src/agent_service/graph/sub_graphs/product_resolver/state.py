from typing import Annotated, Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SKUItem(TypedDict, total=False):
    attributes: Dict[str, Any]                 # Ej: {"cantidad": 10, "diametro": "1/2 pulg"}
    is_owner: bool                             # True si pertenece al user_id
    partner_id: Optional[str]                  # ID único resuelto (None si hay ambigüedad)
    candidate_partner_ids: List[str]           # IDs posibles para detectar conflictos
    partner_metadata: Dict[str, str]           # Mapeo id -> nombre: {"p_101": "Siderperu"}


class ProductResolverState(TypedDict, total=False):
    messages: Annotated[list, add_messages]     # Historial conversacional y avisos HITL
    user_id: str                                # ID del usuario para validar pertenencia (ownership)
    raw_query: Optional[str]                    # Consulta original del usuario

    items: Dict[str, SKUItem]                   # Diccionario indexado por SKU: {"SKU-101": SKUItem, ...}

    is_extraction_complete: bool                # Flag que gobierna el rombo 'is success?'
    has_partner_conflicts: bool                 # Flag que gobierna el rombo 'has conflicts?'

    tool_raw_output: List[Dict[str, Any]]       # Respuesta cruda devuelta por get_product_by_skus
    grouped_products: Dict[str, List[Dict[str, Any]]]  # Agrupado por partner_id: {"p_101": [...]}
    final_response: Optional[str]               # Texto estructurado devuelto al output user