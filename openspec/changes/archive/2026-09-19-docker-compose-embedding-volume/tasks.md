# Tasks

## 1. Docker Compose & Dockerfile Configuration

- [x] 1.1 Configurar variables de entorno `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME` y `EMBEDDING_CACHE_DIR` y mapeo del volumen `embedding_cache:/root/.cache/huggingface` en `docker_compose.dev.yml`.
- [x] 1.2 Declarar el volumen raíz `embedding_cache` con nombre `spark_agent_embedding_cache` en `docker_compose.dev.yml`.
- [x] 1.3 Establecer `HF_HOME=/root/.cache/huggingface` en `Dockerfile`.

## 2. Embeddings Factory Core

- [x] 2.1 Actualizar `get_embedding_service` en `src/agent_service/core/embeddings/factory.py` para leer `EMBEDDING_CACHE_DIR` o `HF_HOME` y pasar `cache_folder` a `HuggingFaceEmbeddings`.

## 3. Validación y Verificación

- [x] 3.1 Validar la sintaxis de Docker Compose con `docker compose -f docker_compose.dev.yml config`.
- [x] 3.2 Ejecutar `openspec validate docker-compose-embedding-volume --strict`.
- [x] 3.3 Ejecutar la suite de pruebas unitarias `.venv/bin/pytest tests/test_llm_factory.py`.
