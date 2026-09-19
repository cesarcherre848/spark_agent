# Tasks: reject-unauthenticated-phone

## 1. Modificaciones en PhoneUserResolver

- [x] 1.1 Actualizar `PhoneUserResolver.resolve_user_id` en `src/agent_service/api/service.py` para retornar `Optional[int]` (`None` cuando el teléfono no exista en Odoo o no tenga comercial asignado) y no almacenar `None` en `_cache`
- [x] 1.2 Actualizar las pruebas unitarias de `PhoneUserResolver` en `tests/test_api_webhook.py` para validar que teléfonos no registrados retornen `None` y verificar con `.venv/bin/pytest tests/test_api_webhook.py -k test_resolver`

## 2. Rechazo HTTP 401 en Webhook API

- [x] 2.1 Modificar `process_webhook` en `src/agent_service/api/webhook.py` para verificar si `user_id is None` tras la resolución y levantar `HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)` antes de invocar el grafo
- [x] 2.2 Agregar pruebas unitarias en `tests/test_api_webhook.py` verificando que peticiones con teléfonos no registrados reciban HTTP 401 Unauthorized sin llamar a `ainvoke` y verificar con `.venv/bin/pytest tests/test_api_webhook.py -k test_unauthorized`

## 3. Verificación y Validación Final

- [x] 3.1 Ejecutar toda la suite de pruebas unitarias del proyecto con `.venv/bin/pytest -m "not real_db"` y confirmar 100% de tests pasando
- [x] 3.2 Validar la conformidad del cambio con `openspec validate reject-unauthenticated-phone --strict`
