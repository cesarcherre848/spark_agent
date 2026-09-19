# Design: Restrict Webhook Authentication Strictly to Salespeople with Exact Phone Search

## Context

El agente conversacional Spark Agent es una herramienta de uso interno exclusivo para **vendedores**.
Anteriormente, el resolvedor admitía clientes comunes mediante `res.partner` y realizaba búsquedas con operador parcial `ilike`. Esto presentaba dos riesgos:
1. Clientes finales podían consumir el agente suplantando a sus vendedores asignados.
2. `ilike` con comodines podía emparejar números erróneos o fragmentos de números parecidos.

## Goals / Non-Goals

**Goals:**
- Consultar directamente el modelo `res.users` filtrando por:
  - `active = True`
  - `share = False` (usuarios internos, excluye usuarios de portal o clientes externos)
  - Coincidencia exacta de teléfono en `partner_id.phone` o `partner_id.phone_sanitized`.
- Construir una lista determinista de candidatos exactos de teléfono para comparar con igualdad estricta (`=` o `in`):
  - `phone_input.strip()`
  - `clean_phone` (dígitos normalizados)
  - `+{clean_phone}` (formato E.164 con signo `+`)
  - Si tiene 9 dígitos (ej. celular peruano `983689215`), incluir `+51{clean_phone}` y `51{clean_phone}`.
  - Si tiene 11 dígitos iniciando con `51` (ej. `51983689215`), incluir los 9 dígitos nacionales `clean_phone[2:]`.
- Si se encuentra coincidencia en `res.users`, retornar su `id` (ej. `5`).
- Si no se encuentra coincidencia exacta, retornar `None`.
- Almacenar en `TTLCache` únicamente identificadores válidos.

**Non-Goals:**
- No admitir clientes externos en el webhook.

## Decisions

### Decisión 1: Consulta directa a `res.users` mediante relación `partner_id`
En lugar de buscar en `res.partner` y luego verificar si es vendedor, consultamos directamente `res.users` con dominio relacional en una sola llamada RPC:
```python
domain = [
    ["active", "=", True],
    ["share", "=", False],
    "|",
    ["partner_id.phone_sanitized", "in", exact_candidates],
    ["partner_id.phone", "in", exact_candidates],
]
```
Esto reduce la latencia de red a un único llamado RPC no bloqueante y garantiza a nivel de base de datos que el usuario resultante es un vendedor interno activo.

### Decisión 2: Conjunto determinista de formatos exactos
Para evitar problemas si el número en Odoo está guardado con o sin código de país (`+51`), generamos la tupla de variantes exactas equivalentes del número sin usar comodines `%` ni búsquedas difusas.

## Risks / Trade-offs

- [Riesgo] Pruebas anteriores que simulaban clientes (`user_id` en `res.partner`) deben adaptarse al nuevo diseño exclusivo de vendedores en `res.users`.
  - *Mitigación*: Actualizar `tests/test_api_webhook.py` para reflejar la autenticación directa en `res.users`.
