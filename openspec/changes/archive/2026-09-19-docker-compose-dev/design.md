# Design: Dockerization for Development Environment (`docker_compose.dev.yml`)

## Context

Spark Agent corre actualmente mediante un entorno virtual local (`.venv`) ejecutando FastAPI + Uvicorn. Para facilitar el despliegue y desarrollo reproducible, se define la infraestructura en contenedores Docker.

## Goals / Non-Goals

**Goals:**
- Crear `Dockerfile` basado en `python:3.11-slim`.
- Instalar dependencias compiladas de `requirements.txt`.
- Configurar `docker_compose.dev.yml` con `env_file: .env.dev`, `ports: 8000:8000`, y volumen `.:/app` para desarrollo con recarga en vivo.
- Configurar healthcheck en `/health`.

**Non-Goals:**
- No desplegar contenedores de base de datos locales (la aplicación se conecta a las bases de datos remotas en `134.199.209.31`).

## Decisions

### Decisión 1: Python 3.11-slim
Provee soporte completo sin advertencias de deprecación para las librerías modernas de Google AI / LangChain, y menor consumo de memoria.

### Decisión 2: Recarga en caliente con volumen `.:/app`
Permite a los desarrolladores editar código en local y ver los cambios reflejados de inmediato dentro del contenedor sin reconstruir la imagen.
