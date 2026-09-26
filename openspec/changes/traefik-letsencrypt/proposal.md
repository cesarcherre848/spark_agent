# Proposal: Traefik con Let's Encrypt para Entorno QA (`docker_compose.qa.yml`)

## Why

Para desplegar y validar Spark Agent en el entorno de pruebas/QA bajo el dominio `mia-dev.evaspark.com`, se requiere un proxy inverso seguro que:
1. Termine el tráfico TLS/HTTPS mediante certificados emitidos y renovados automáticamente vía Let's Encrypt (protocolo ACME) con cuenta asociada a `cesarcherre@gmail.com`.
2. Redirija de forma transparente todo el tráfico HTTP (puerto 80) hacia HTTPS (puerto 443).
3. Enrute las peticiones externas al contenedor de Spark Agent de forma desacoplada y segura sin exponer el puerto interno de la aplicación directamente a la red pública.
4. Mantenga aislada la configuración de QA en un archivo dedicado `docker_compose.qa.yml`, preservando la configuración de desarrollo local existente (`docker_compose.dev.yml`).

## What Changes

- **`docker_compose.qa.yml`**: Nuevo archivo de orquestación Docker Compose para QA que define:
  - Servicio `traefik`: Basado en `traefik:v3.1`, exponiendo puertos 80 y 443, con redirección HTTP->HTTPS, proveedor Docker y resolución de certificados Let's Encrypt (`cesarcherre@gmail.com`) con almacenamiento persistente en un volumen nombrado.
  - Servicio `spark_agent`: Contenedor QA con etiquetas (`labels`) de Traefik para enrutar `Host(`mia-dev.evaspark.com`)` con resolver TLS `letsencrypt`, volumen para caché de embeddings, healthcheck y carga de variables desde `.env.qa`.
  - Red dedicada `traefik_net` (`spark_agent_qa_net`) y volúmenes nombrados para certificados Let's Encrypt (`spark_agent_qa_letsencrypt`) y embeddings (`spark_agent_embedding_cache`).
- **`.env.qa.example`**: Archivo de plantilla con variables de entorno para QA, incluyendo `DOMAIN=mia-dev.evaspark.com`, `ACME_EMAIL=cesarcherre@gmail.com`, configuración de base de datos y WhatsApp.

## Capabilities

### New Capabilities
*(Ninguna - Cambio de infraestructura y despliegue)*

### Modified Capabilities
*(Ninguna - Cambio de infraestructura y despliegue)*

## Impact

- **Archivos nuevos**:
  - `docker_compose.qa.yml`
  - `.env.qa.example`
- No modifica código en `src/` ni tests en `tests/`.
- Totalmente retrocompatible: `docker_compose.dev.yml` permanece intacto para desarrollo local.
