from typing import Optional, List, Dict, Any, Literal, Annotated, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ContactManageState(TypedDict, total=False):
    """Estado del subgrafo de gestión de clientes/contactos."""
    messages: Annotated[List[BaseMessage], add_messages]
    raw_query: Optional[str]
    user_id: Optional[Union[int, str]]
    session_id: Optional[str]

    # Extracción de intención y datos
    contact_action: Literal["list", "upsert", "remove"]
    extracted_name: Optional[str]
    extracted_phones: List[str]
    target_contact_id: Optional[int]

    # Rama List
    customers_list: List[Dict[str, Any]]

    # Rama Upsert
    current_candidates: List[Dict[str, Any]]
    has_possible_duplicates: bool
    duplicate_rationale: Optional[str]
    upsert_confirmed: bool

    # Rama Remove
    remove_confirmed: bool

    # Resultado y Síntesis final
    operation_result: Optional[Dict[str, Any]]
    final_response: Optional[str]
