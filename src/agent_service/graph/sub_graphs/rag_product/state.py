from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ProductRagState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    user_id: Optional[int]

    raw_query: str
    refined_query: Optional[str]

    metadata_filters: Optional[Dict[str, Any]]

    retrieved_products: List[Document]

    is_sufficient: bool
    selected_indices: List[int]
    critique: Optional[str]

    iteration_count: int
    max_iterations: int

    matched_skus: List[str]
    final_response: Optional[str]