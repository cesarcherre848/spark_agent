# Tasks: docker-compose-dev

## 1. Configuración de Archivos Docker

- [x] 1.1 Crear `.dockerignore` excluyendo `.venv`, `__pycache__`, tests y artefactos temporales
- [x] 1.2 Crear `Dockerfile` con `python:3.11-slim`, dependencias de sistema y `pip install -r requirements.txt`
- [x] 1.3 Crear `docker_compose.dev.yml` con servicio `spark_agent`, puerto `8000:8000`, volumen `.:/app`, `env_file: .env.dev` y healthcheck

## 2. Verificación de Compilación y Ejecución

- [x] 2.1 Construir la imagen de Docker con `docker compose -f docker_compose.dev.yml build`
- [x] 2.2 Validar la conformidad del cambio con `openspec validate docker-compose-dev --strict`
