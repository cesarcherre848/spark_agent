# Tasks: Traefik con Let's Encrypt para Entorno QA (`docker_compose.qa.yml`)

## 1. Configuración de Variables de Entorno de QA

- [x] 1.1 Crear plantilla `.env.qa.example` con variables para QA: `ENVIRONMENT=qa`, `DOMAIN=mia-dev.evaspark.com`, `ACME_EMAIL=cesarcherre@gmail.com`, y configuraciones de base de datos y WhatsApp.

## 2. Definición del Orquestador `docker_compose.qa.yml`

- [x] 2.1 Configurar el servicio `traefik` en `docker_compose.qa.yml` utilizando `traefik:v3.1`, puertos 80 y 443, redirección HTTP->HTTPS, proveedor Docker y resolver ACME de Let's Encrypt (`cesarcherre@gmail.com`) con volumen persistente `spark_agent_qa_letsencrypt`.
- [x] 2.2 Configurar el servicio `spark_agent` en `docker_compose.qa.yml` con nombre `spark_agent_qa`, `env_file: .env.qa`, etiquetas de Traefik para el router `Host(mia-dev.evaspark.com)`, entrypoint `websecure`, TLS con `letsencrypt` y puerto `8000`.
- [x] 2.3 Declarar la red bridge compartida `spark_agent_qa_net` y los volúmenes nombrados `spark_agent_qa_letsencrypt` y `spark_agent_embedding_cache`.

## 3. Validación y Pruebas de Sintaxis

- [x] 3.1 Validar la sintaxis del archivo compose mediante `docker compose -f docker_compose.qa.yml config`.
- [x] 3.2 Verificar que `docker_compose.dev.yml` se mantenga inalterado y funcional.
- [x] 3.3 Validar el cambio OpenSpec mediante `openspec validate traefik-letsencrypt --strict`.
