# Proposal: Webhook API para recepción de consultas y resolución Odoo con Caché

## Why
Actualmente `spark_agent` opera como un grafo de agentes ejecutado localmente desde consola o scripts (`main.py`). Para integrarse con canales de mensajería externos (WhatsApp, plataformas omnicanal, CRM o clientes web), se requiere un endpoint HTTP Webhook que reciba consultas entrantes (`raw_query`) junto con el número de teléfono del remitente.

Dado que cada vendedor o cliente en Odoo ERP está asociado a un `user_id` (vendedor comercial asignado), es imperativo consultar Odoo para identificar dicho `user_id` y contextualizar la sesión del agente. Para evitar latencias repetitivas de red de 200-500ms en cada mensaje entrante, es crucial incorporar una capa de caché en memoria de alto rendimiento (`cachetools.TTLCache`) que resuelva el `user_id` de forma instantánea tras la primera consulta.

## What Changes
- **Nuevo Endpoint Webhook**: Implementación de servidor y rutas API HTTP POST (`/webhook` y `/api/v1/webhook`) que aceptan `raw_query`, `phone_number` y `session_id` opcional.
- **Resolución Odoo por Teléfono**: Función de consulta asíncrona contra la API JSON-RPC de Odoo sobre `res.partner` (buscando en `phone` y `mobile`) para extraer el `user_id` (salesperson) asociado, con fallback configurable (`DEFAULT_USER_ID=5`).
- **Caché en Memoria con `cachetools`**: Integración de `TTLCache` con tiempo de expiración (TTL configurable, ej. 1 hora) y tamaño máximo para resolver `phone -> user_id` en O(1) sin consultar a Odoo en cada mensaje subsecuente.
- **Invocación del Grafo Principal**: Orquestación del flujo entrante hacia `get_main_graph().ainvoke(...)` inyectando `raw_query`, `user_id` y `session_id`, retornando la respuesta generada por Spark Agent en formato JSON estructurado.
- **Nuevas dependencias**: Inclusión de `fastapi`, `uvicorn` y `cachetools` en `requirements.txt`.

## Capabilities

### New Capabilities
- `api-web-hook`: Endpoint webhook HTTP para recepción de consultas entrantes, resolución de `user_id` vía teléfono contra Odoo ERP con caché en memoria (`cachetools`), y ejecución orquestada del Grafo Principal de Spark Agent.

### Modified Capabilities
*(Ninguna. No existen especificaciones previas en `openspec/specs/`)*

## Impact
- **APIs**: Nuevo endpoint HTTP POST disponible para integraciones externas.
- **Dependencias**: Se incorporan `fastapi>=0.111.0`, `uvicorn>=0.30.0` y `cachetools>=5.3.0`.
- **Estructura del Proyecto**: Se crea el paquete `src/agent_service/api/` conteniendo schemas, servicio de resolución con caché y enrutador FastAPI.
- **Compatibilidad**: No altera los subgrafos existentes ni `MainGraphState`; reutiliza `get_main_graph()` existente.
