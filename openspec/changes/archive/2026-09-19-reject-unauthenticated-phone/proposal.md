# Proposal: Rejection of Unauthenticated / Unregistered Phone Numbers

## Why

Actualmente, cuando una solicitud llega al Webhook desde un número de teléfono no registrado en Odoo (o sin vendedor/usuario asignado), el servicio recurre silenciosamente a un `DEFAULT_USER_ID` de respaldo (por defecto `5`), permitiendo que usuarios no autenticados o desconocidos interactúen con el agente suplantando a un usuario existente. Para garantizar la seguridad del canal y la correcta trazabilidad de los clientes, las solicitudes provenientes de números no registrados deben ser rechazadas de inmediato con un código HTTP 401 Unauthorized.

## What Changes

- **Resolución estricta de identidad**: `PhoneUserResolver.resolve_user_id` ya no asignará un usuario por defecto cuando el teléfono no exista en Odoo o no cuente con comercial asignado; en su lugar, retornará `None` (o lanzará error de no autenticación).
- **Rechazo HTTP 401 Unauthorized en Webhook**: Si el teléfono no puede ser autenticado en Odoo (`user_id is None`), el endpoint del webhook interrumpirá el procesamiento antes de invocar el grafo y responderá inmediatamente con HTTP 401 Unauthorized y un mensaje de error claro.
- **Caché segura**: La caché en memoria no registrará usuarios por defecto para números inexistentes, evitando enmascarar accesos no autorizados.

## Capabilities

### New Capabilities

*(Ninguna)*

### Modified Capabilities

- `api-web-hook`: Se modifica el requisito `Odoo Salesperson Resolution by Phone` para eliminar el fallback permisivo a `DEFAULT_USER_ID` ante números no registrados o sin comercial, y se actualiza el comportamiento del webhook para responder con HTTP 401 Unauthorized en lugar de proceder con la ejecución del agente.

## Impact

- **APIs afectadas**: `POST /api/v1/webhook` y `POST /webhook` ahora retornan HTTP 401 cuando el número de teléfono no está registrado en Odoo o no tiene usuario asociado.
- **Servicios afectados**: `src/agent_service/api/service.py` (`PhoneUserResolver`) y `src/agent_service/api/webhook.py`.
- **Pruebas**: Se actualizarán las pruebas en `tests/test_api_webhook.py` para validar el rechazo con HTTP 401 en casos no registrados.
