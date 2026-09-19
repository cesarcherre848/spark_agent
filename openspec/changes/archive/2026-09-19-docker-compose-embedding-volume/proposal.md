# Proposal: Persistencia de Caché del Modelo de Embeddings en Docker Compose

## Why
Actualmente, el contenedor de Docker para desarrollo (`docker_compose.dev.yml`) solo monta el código de la aplicación (`.:/app`). Cada vez que el contenedor se recrea o la imagen se reconstruye, `HuggingFaceEmbeddings` descarga nuevamente los ~2.2 GB del modelo `BAAI/bge-m3` hacia `/root/.cache/huggingface`, lo que causa demoras en el arranque y consumo innecesario de red.

Se requiere incorporar un volumen de Docker dedicado y configurar las variables de entorno de caché (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `EMBEDDING_CACHE_DIR`) para persistir el modelo localmente a través de reinicios y recreaciones de contenedores.

## What Changes
- **`docker_compose.dev.yml`**:
  - Declarar variables de entorno de caché (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `EMBEDDING_CACHE_DIR` apuntando a `/root/.cache/huggingface`).
  - Mapear el volumen nombrado `embedding_cache:/root/.cache/huggingface` en el servicio `spark_agent`.
  - Declarar el volumen nombrado `embedding_cache` con nombre explícito `spark_agent_embedding_cache`.
- **`Dockerfile`**:
  - Establecer la variable de entorno por defecto `HF_HOME=/root/.cache/huggingface`.
- **`src/agent_service/core/embeddings/factory.py`**:
  - Soportar lectura de `EMBEDDING_CACHE_DIR` o `HF_HOME` y pasar `cache_folder` a `HuggingFaceEmbeddings`.

## Capabilities

### New Capabilities
*(Ninguna - Cambio de infraestructura y optimización de caché)*

### Modified Capabilities
*(Ninguna - Cambio de infraestructura y optimización de caché)*

## Impact
- **Desarrollo**: El tiempo de arranque del contenedor tras el primer inicio se reduce drásticamente, sin volver a descargar los pesos del modelo.
- **Docker**: Se crea un volumen gestionado `spark_agent_embedding_cache`.
- **Código**: Soporte opcional de carpeta de caché en `get_embedding_service`.
