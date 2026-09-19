# Spec Delta: api-web-hook

## MODIFIED Requirements

### Requirement: Webhook HTTP Input Ingestion
The system SHALL expose an HTTP POST endpoint (`/api/v1/webhook` and `/webhook`) that accepts a JSON payload containing mandatory `raw_query` and `phone_number`, and optional `session_id`, rejecting unauthenticated or invalid requests before agent invocation.

#### Scenario: Successful Webhook Payload Processing
- **WHEN** un cliente HTTP envía una solicitud POST con `raw_query="Hola, ¿tienen stock de serum facial?"` y `phone_number="+51987654321"` registrado en Odoo
- **THEN** el sistema responde con código HTTP 200 y un JSON estructurado con `status="success"`, `user_id`, `intent`, `response` y `session_id`.

#### Scenario: Rejection of Missing Mandatory Fields
- **WHEN** un cliente HTTP envía una solicitud POST sin el campo `raw_query` o sin `phone_number`
- **THEN** el sistema rechaza la petición con código de error HTTP 422 Unprocessable Entity indicando los campos faltantes o inválidos.

#### Scenario: Rejection of Unauthenticated Phone Number
- **WHEN** un cliente HTTP envía una solicitud POST desde un número de teléfono que no está registrado en Odoo o no posee un usuario/vendedor asignado
- **THEN** el sistema rechaza inmediatamente la petición con código HTTP 401 Unauthorized sin invocar el grafo del agente.

### Requirement: Odoo Salesperson Resolution by Phone
The system SHALL query Odoo ERP on `res.partner` matching by `phone` or `mobile` to resolve the assigned salesperson `user_id`, returning `None` when no partner or salesperson is found, without silently assigning arbitrary default users.

#### Scenario: Existing Partner with Assigned Salesperson
- **WHEN** se recibe un número de teléfono que coincide en Odoo con un cliente asignado al vendedor con ID `5`
- **THEN** el resolvedor retorna `user_id=5`.

#### Scenario: Unregistered Phone Fallback
- **WHEN** se recibe un número de teléfono que no registra ninguna coincidencia en Odoo o cuyo cliente carece de comercial asignado
- **THEN** el resolvedor retorna `None` indicando la ausencia de autenticación sin asignar usuarios arbitrarios por defecto.
