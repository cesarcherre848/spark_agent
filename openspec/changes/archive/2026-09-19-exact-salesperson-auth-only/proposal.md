# Proposal: Restrict Webhook Authentication Strictly to Salespeople with Exact Phone Search

## Why

El agente conversacional Spark Agent está diseñado exclusivamente como una herramienta de asistencia y productividad para **vendedores** internos de la empresa (usuarios `res.users` activos con `share=False`). No debe procesar solicitudes de clientes finales ni de contactos que no sean asesores comerciales autenticados.
Asimismo, para prevenir suplantaciones y colisiones numéricas accidentales (donde un fragmento de dígitos coincida con otro número), la búsqueda de teléfonos en Odoo ERP debe ser **estrictamente exacta** (`=`), eliminando coincidencias parciales con comodines (`ilike`).

## What Changes

- **Autenticación Exclusiva de Vendedores**: Modificar `PhoneUserResolver.resolve_user_id` para consultar directamente `res.users` (`active=True`, `share=False`), asegurando que solo usuarios comerciales internos puedan autenticarse. Los clientes finales (`res.partner` con comercial asignado) ya no se autenticarán como usuarios del webhook.
- **Búsqueda Exacta sin `ilike`**: La búsqueda en Odoo se realizará mediante comparaciones de igualdad exacta (`in` sobre un conjunto finito de formatos exactos: número crudo, formato normalizado, prefijo internacional `+51` y 9 dígitos nacionales), descartando coincidencias difusas o parciales.
- **Respuesta No Autorizado (401)**: Todo número que no corresponda con exactitud a un vendedor interno activo de Odoo será rechazado de inmediato con HTTP 401 Unauthorized sin invocar el grafo del agente.

## Capabilities

### New Capabilities

*(Ninguna)*

### Modified Capabilities

- `api-web-hook`: Se modifica el requisito `Odoo Salesperson Resolution by Phone` para exigir coincidencia exacta y autenticación restringida exclusivamente a vendedores internos (`res.users`).

## Impact

- **Módulos afectados**: [service.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/src/agent_service/api/service.py) (`PhoneUserResolver.resolve_user_id`).
- **Pruebas afectadas**: [test_api_webhook.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/tests/test_api_webhook.py) validará que solo los vendedores internos sean resueltos con búsqueda exacta y que los clientes comunes o números no registrados retornen `None` / HTTP 401.
