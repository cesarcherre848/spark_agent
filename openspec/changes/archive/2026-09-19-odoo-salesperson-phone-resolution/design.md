# Design: Odoo Phone Domain Correction & Salesperson Self-Resolution

## Context

Al inspeccionar Odoo ERP con consultas en vivo:
1. `res.partner` no posee el campo `mobile` en este entorno de base de datos. Los campos reales son `phone` y `phone_mobile_search`.
2. Cuando un vendedor interno como Cesar Cherre (`+51983689215`) escribe, su `res.partner` tiene `id=6` y `user_id=False`. Su usuario comercial real en `res.users` tiene `id=5` con `partner_id=[6, "Cesar Cherre Vendedor"]`.

## Goals / Non-Goals

**Goals:**
- Modificar el dominio de búsqueda en `PhoneUserResolver.resolve_user_id` a:
  ```python
  domain = [
      ["active", "=", True],
      "|",
      ["phone", "ilike", significant],
      ["phone_mobile_search", "ilike", significant],
  ]
  ```
- Si `partner.get("user_id")` es un par `[id, name]`, extraer `int(user_id[0])`.
- Si `partner.get("user_id")` es `False`, consultar `res.users` filtrando por `[["partner_id", "=", partner["id"]]]` para extraer `int(user["id"])`.
- Si ninguno de los dos produce un usuario, retornar `None`.

**Non-Goals:**
- No alterar otros endpoints de la API.
- No alterar la lógica de rechazo HTTP 401 del Webhook ya implementada.

## Decisions

### Decisión 1: Búsqueda mediante `phone` y `phone_mobile_search`
En lugar de fallar con `mobile`, se usan los campos estándar compatibles con Odoo 16/17/18.

### Decisión 2: Consulta secundaria a `res.users` sólo cuando `partner.user_id` es False
Para clientes habituales, la consulta a `res.partner` resuelve el comercial de inmediato en 1 petición RPC. La consulta a `res.users` solo se ejecuta como fallback cuando el partner es un usuario interno. Ambas quedan cacheadas en `TTLCache` para O(1) subsecuente.

## Risks / Trade-offs

- [Riesgo] Una llamada adicional a `res.users` para asesores internos:
  - *Mitigación*: Se ejecuta de forma asíncrona no bloqueante y el resultado se almacena en memoria en `TTLCache`, eliminando sobrecostos en mensajes futuros.
