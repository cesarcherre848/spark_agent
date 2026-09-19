# Tasks: exact-salesperson-auth-only

## 1. Modificaciones en PhoneUserResolver

- [x] 1.1 Actualizar `PhoneUserResolver.resolve_user_id` en `src/agent_service/api/service.py` para consultar directamente `res.users` (`active=True`, `share=False`) con coincidencia telefónica exacta sin `ilike`
- [x] 1.2 Actualizar y agregar pruebas unitarias en `tests/test_api_webhook.py` validando que únicamente vendedores en `res.users` sean autenticados y que clientes o números no registrados retornen `None`, verificando con `.venv/bin/pytest tests/test_api_webhook.py -k test_resolver`

## 2. Verificación y Validación Final

- [x] 2.1 Ejecutar toda la suite de pruebas unitarias del proyecto con `.venv/bin/pytest -m "not real_db"` y confirmar 100% de tests pasando
- [x] 2.2 Validar la conformidad del cambio con `openspec validate exact-salesperson-auth-only --strict`
