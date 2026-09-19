# Proposal: Odoo Phone Domain Correction & Salesperson Self-Resolution

## Why

Al realizar consultas reales a Odoo ERP mediante `res.partner`:
1. El campo `mobile` no existe en el esquema de este entorno de Odoo, provocando un fallo de servidor `ValueError: Invalid field res.partner.mobile in condition` que hacía fallar cualquier búsqueda telefónica.
2. Cuando el propio vendedor o asesor comercial (usuario interno de Odoo) escribe desde su teléfono (por ejemplo, Cesar Cherre con `+51983689215`), su registro en `res.partner` tiene `user_id=False` porque los comerciales no tienen un vendedor asignado a sí mismos. El sistema lo catalogaba erróneamente como no autenticado (`None`), provocando un rechazo HTTP 401.

## What Changes

- **Corrección de dominio Odoo**: Reemplazar `mobile` por los campos nativos existentes `phone` y `phone_mobile_search`.
- **Resolución bidireccional cliente/vendedor**:
  - Si el teléfono pertenece a un cliente con comercial asignado (`partner.user_id`), resolver el `user_id` de su vendedor.
  - Si el teléfono pertenece a un vendedor interno (`partner.user_id` es False), buscar su registro correspondiente en `res.users` mediante `partner_id = partner.id` y resolver su propio `user_id`.
  - Si no coincide con un cliente atendido ni con un vendedor interno, retornar `None` (rechazo seguro).

## Capabilities

### New Capabilities

*(Ninguna)*

### Modified Capabilities

- `api-web-hook`: Se actualiza el requisito `Odoo Salesperson Resolution by Phone` para documentar la búsqueda en `res.partner` (`phone` / `phone_mobile_search`) y la resolución tanto de clientes como de usuarios comerciales internos de Odoo.

## Impact

- **Módulos afectados**: [service.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/src/agent_service/api/service.py) (`PhoneUserResolver.resolve_user_id`).
- **Pruebas afectadas**: [test_api_webhook.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/tests/test_api_webhook.py) cubrirá la resolución de vendedores internos además de clientes y casos no registrados.
