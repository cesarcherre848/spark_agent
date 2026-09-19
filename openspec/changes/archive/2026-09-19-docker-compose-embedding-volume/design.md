# Design: Persistencia de Caché del Modelo de Embeddings en Docker Compose

## Context
Ver `proposal.md` para la motivación. El modelo `BAAI/bge-m3` utilizado por el servicio de embeddings es descargado por `sentence_transformers` y `huggingface_hub` en el directorio de caché definido por `HF_HOME`.

## Goals / Non-Goals

**Goals:**
- Configurar un volumen nombrado de Docker (`spark_agent_embedding_cache`) montado en `/root/.cache/huggingface`.
- Establecer `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME` y `EMBEDDING_CACHE_DIR` hacia `/root/.cache/huggingface`.
- Adaptar `src/agent_service/core/embeddings/factory.py` para usar `cache_folder` respetando estas variables de entorno.

**Non-Goals:**
- No se descargará el modelo durante el `docker build` (para mantener la imagen Docker liviana y rápida de compilar en CI/CD). La descarga ocurre la primera vez en runtime y queda preservada en el volumen nombrado.

## Decisions

### Decisión 1: Volumen nombrado de Docker en lugar de bind mount de host
- **Razón**: En Docker Desktop para macOS, los bind mounts sufren una penalización de rendimiento considerable al leer miles de archivos de pesos y tensores (`virtiofs` / `osxfs`). Los volúmenes nombrados residen directamente en la VM de Docker con rendimiento nativo de Linux I/O y no ensucian el repositorio local con 2.2 GB de binarios.

### Decisión 2: Variables de entorno estándar de Hugging Face y Torch
- `HF_HOME=/root/.cache/huggingface`
- `SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface`
- `EMBEDDING_CACHE_DIR=/root/.cache/huggingface`
- Ambas librerías convergen en el mismo directorio montado en el volumen persistente.

## Risks / Trade-offs

- **[Risk] Primera ejecución requiere descarga** → **Mitigación**: La primera vez que el contenedor arranca descargará el modelo (normal). Las ejecuciones subsecuentes arrancarán en segundos reutilizando el volumen.
- **[Risk] Pérdida de caché con `docker volume prune`** → **Mitigación**: El volumen se nombra explícitamente `spark_agent_embedding_cache` para fácil identificación y retención.
