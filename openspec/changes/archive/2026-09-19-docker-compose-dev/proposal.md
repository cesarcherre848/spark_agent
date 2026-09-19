# Proposal: Dockerization for Development Environment (`docker_compose.dev.yml`)

## Why

Para estandarizar la ejecución de Spark Agent en desarrollo, garantizar reproducibilidad de dependencias entre diferentes entornos y preparar el despliegue a producción, se requiere contenerizar la aplicación con Docker y `docker_compose.dev.yml`, inyectando las variables de entorno de `.env.dev` y habilitando recarga en caliente de código.

## What Changes

- **`.dockerignore`**: Excluir entornos virtuales, cachés y archivos no necesarios para la imagen.
- **`Dockerfile`**: Imagen base `python:3.11-slim`, instalación de dependencias del sistema (`curl`, `gcc`), instalación de `requirements.txt` y configuración del comando de arranque `uvicorn`.
- **`docker_compose.dev.yml`**: Configuración del servicio con montaje de volumen `.:/app`, mapeo de puertos `8000:8000`, inyección de `.env.dev` y healthcheck en `/health`.

## Capabilities

### New Capabilities

*(Ninguna - Cambio de infraestructura y despliegue)*

### Modified Capabilities

*(Ninguna - Cambio de infraestructura y despliegue)*

## Impact

- **Archivos nuevos**: `.dockerignore`, `Dockerfile`, `docker_compose.dev.yml`.
- No afecta lógica de negocio ni código en `src/`.
