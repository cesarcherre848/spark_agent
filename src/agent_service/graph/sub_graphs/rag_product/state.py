from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RagProductState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    raw_query: str
    refined_query: Optional[str]

    metadata_filters: Dict[str, Any]

    retrieved_products: List[Document]
    reranked_products: List[Document]

    is_sufficient: bool
    retry_count: int
    critique: Optional[str]

    matched_skus: List[str]
    final_response: Optional[str]
    critique: Optional[str]