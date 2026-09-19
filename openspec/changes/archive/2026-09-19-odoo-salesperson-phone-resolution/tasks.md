# Tasks: odoo-salesperson-phone-resolution

## 1. Modificaciones en PhoneUserResolver

- [x] 1.1 Actualizar el dominio de búsqueda en `PhoneUserResolver.resolve_user_id` en `src/agent_service/api/service.py` para usar `phone` y `phone_mobile_search` en lugar de `mobile`
- [x] 1.2 Implementar en `PhoneUserResolver.resolve_user_id` la consulta a `res.users` por `partner_id` cuando `partner.user_id` sea False para resolver a vendedores internos
- [x] 1.3 Agregar pruebas unitarias en `tests/test_api_webhook.py` simulando la resolución de un vendedor interno (`partner.user_id=False` con `res.users` existente) y verificar con `.venv/bin/pytest tests/test_api_webhook.py -k test_resolver`

## 2. Verificación y Validación Final

- [x] 2.1 Ejecutar toda la suite de pruebas unitarias del proyecto con `.venv/bin/pytest -m "not real_db"` y confirmar 100% de tests pasando
- [x] 2.2 Validar la conformidad del cambio con `openspec validate odoo-salesperson-phone-resolution --strict`
