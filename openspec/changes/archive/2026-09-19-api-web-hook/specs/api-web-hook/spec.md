# Spec Delta: api-web-hook

## Purpose

Proporciona un punto de entrada HTTP Webhook para canales de mensajería externos, resolviendo la identidad comercial (`user_id`) del remitente a partir de su número telefónico mediante Odoo ERP y optimizado mediante caché en memoria con `cachetools`, permitiendo la ejecución fluida del agente conversacional Spark Agent.

## ADDED Requirements

### Requirement: Webhook HTTP Input Ingestion
The system SHALL expose an HTTP POST endpoint (`/api/v1/webhook` and `/webhook`) that accepts a JSON payload containing mandatory `raw_query` and `phone_number`, and optional `session_id`.

#### Scenario: Successful Webhook Payload Processing
- **WHEN** un cliente HTTP envía una solicitud POST con `raw_query="Hola, ¿tienen stock de serum facial?"` y `phone_number="+51987654321"`
- **THEN** el sistema responde con código HTTP 200 y un JSON estructurado con `status="success"`, `user_id`, `intent`, `response` y `session_id`.

#### Scenario: Rejection of Missing Mandatory Fields
- **WHEN** un cliente HTTP envía una solicitud POST sin el campo `raw_query` o sin `phone_number`
- **THEN** el sistema rechaza la petición con código de error HTTP 422 Unprocessable Entity indicando los campos faltantes o inválidos.

### Requirement: Odoo Salesperson Resolution by Phone
The system SHALL query Odoo ERP on `res.partner` matching by `phone` or `mobile` to resolve the assigned salesperson `user_id`, falling back to `DEFAULT_USER_ID` (default `5`) when no partner or salesperson is found.

#### Scenario: Existing Partner with Assigned Salesperson
- **WHEN** se recibe un número de teléfono que coincide en Odoo con un cliente asignado al vendedor con ID `5`
- **THEN** el resolvedor retorna `user_id=5`.

#### Scenario: Unregistered Phone Fallback
- **WHEN** se recibe un número de teléfono que no registra ninguna coincidencia en Odoo
- **THEN** el resolvedor retorna el `DEFAULT_USER_ID` predeterminado sin interrumpir la atención.

### Requirement: High-Performance In-Memory Cache with cachetools
The system SHALL maintain an in-memory `cachetools.TTLCache` mapping normalized phone numbers to their resolved `user_id` with configurable TTL and capacity.

#### Scenario: Cache Miss Triggers Odoo Lookup
- **WHEN** se recibe una consulta de un número telefónico que no se encuentra en el caché
- **THEN** el sistema invoca la API de Odoo, almacena el `user_id` en el `TTLCache` y lo utiliza en la sesión.

#### Scenario: Cache Hit Bypasses Odoo Lookup
- **WHEN** se recibe una consulta subsecuente de un número telefónico ya almacenado en el `TTLCache`
- **THEN** el sistema obtiene el `user_id` directamente del caché en memoria en O(1) sin realizar llamadas de red a Odoo.

### Requirement: Execution of Spark Agent Main Graph
The system SHALL asynchronously invoke the Spark Agent main graph (`get_main_graph().ainvoke`) injecting `raw_query`, resolved `user_id`, and `session_id`, returning the structured final response.

#### Scenario: Agent Graph Invocation and Structured Response
- **WHEN** el webhook procesa la consulta y resuelve el `user_id`
- **THEN** invoca el Grafo Principal pasando `{"raw_query": raw_query, "user_id": user_id}` configurando el hilo con `session_id`, y retorna la respuesta procesada en el campo `response`.

### Requirement: Non-Blocking Asynchronous Execution
The system SHALL process all webhook operations and external I/O asynchronously using non-blocking primitives (`async/await`, `httpx.AsyncClient`, `ainvoke`), ensuring the FastAPI event loop remains unblocked during concurrent requests.

#### Scenario: Concurrent Non-Blocking Request Handling
- **WHEN** múltiples solicitudes concurrentes ingresan al endpoint del webhook simultáneamente
- **THEN** el servidor procesa las peticiones de forma asíncrona y no bloqueante mediante el event loop.

