import json
from typing import Any, Dict, List, Optional, Tuple
from psycopg import sql
from psycopg_pool import AsyncConnectionPool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

class ProductVectorStore(VectorStore):
    def __init__():
        pass