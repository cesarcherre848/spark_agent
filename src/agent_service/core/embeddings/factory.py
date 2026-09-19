import os
import logging
from typing import Optional
import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDINGS: Optional[Embeddings] = None


def get_embedding_service(model_name: str = "BAAI/bge-m3") -> Embeddings:
    """Retorna la instancia global compartida (singleton) del servicio de embeddings.
    
    Usa aceleración por hardware 'mps' en Apple Silicon (Mac) o 'cpu' como respaldo.
    Carga perezosa (lazy load) para evitar consumo innecesario de memoria en pruebas.
    Permite configurar un directorio persistente de caché mediante EMBEDDING_CACHE_DIR o HF_HOME.
    """
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        cache_dir = os.getenv("EMBEDDING_CACHE_DIR") or os.getenv("HF_HOME")
        logger.info(
            f"Cargando modelo de embeddings '{model_name}' en dispositivo '{device}' "
            f"(cache_folder: {cache_dir or 'default'})..."
        )
        _GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _GLOBAL_EMBEDDINGS

