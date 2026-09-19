# Tasks: api-web-hook

## 1. Dependencias y Estructura Base

- [x] 1.1 Agregar `fastapi`, `uvicorn` y `cachetools` a `requirements.txt` e instalar en `.venv`, verificando con `.venv/bin/pip list`
- [x] 1.2 Crear el paquete `src/agent_service/api/` con `__init__.py` y verificar su importabilidad


## 2. Esquemas y Servicio de Resolución con Caché

- [x] 2.1 Implementar esquemas Pydantic (`WebhookRequest`, `WebhookResponse`) en `src/agent_service/api/schemas.py` con validaciones de campos no vacíos y alias opcionales
- [x] 2.2 Implementar `PhoneUserResolver` en `src/agent_service/api/service.py` con normalización telefónica, consulta asíncrona a Odoo (`res.partner`) y almacenamiento en `cachetools.TTLCache` con fallback a `DEFAULT_USER_ID`
- [x] 2.3 Crear pruebas unitarias para el resolvedor y la caché en `tests/test_api_webhook.py` (validando Cache Hit, Cache Miss, normalización y fallback de error) y verificar con `.venv/bin/pytest tests/test_api_webhook.py -k test_resolver`



## 3. Endpoints HTTP Webhook y FastAPI App

- [x] 3.1 Implementar la aplicación FastAPI y los endpoints `/api/v1/webhook` y `/webhook` en `src/agent_service/api/webhook.py` integrando el resolvedor y la ejecución asíncrona de `get_main_graph()`
- [x] 3.2 Implementar pruebas de integración de los endpoints HTTP con `httpx.AsyncClient` en `tests/test_api_webhook.py` (validando código 200 con respuesta del agente, 422 para payload inválido y manejo de excepciones) y verificar con `.venv/bin/pytest tests/test_api_webhook.py`


## 4. Verificación y Validación Final

- [x] 4.1 Ejecutar toda la suite de pruebas unitarias del proyecto con `.venv/bin/pytest -m "not real_db"` y confirmar 100% de tests pasando sin regresiones
- [x] 4.2 Validar la conformidad del cambio con `openspec validate api-web-hook --strict`

