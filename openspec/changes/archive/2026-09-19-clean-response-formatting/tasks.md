# Tasks: clean-response-formatting

## 1. Mejoras en LLM Core Factory

- [x] 1.1 Implementar `clean_text_from_tool_call_artifacts` y lectura prioritaria de `tool_calls` en `ResilientStructuredOutputRunnable` en `src/agent_service/core/llms/factory.py`, verificando con tests de unidad
- [x] 1.2 Agregar pruebas unitarias para `clean_text_from_tool_call_artifacts` y recuperación de `tool_calls` con prefijos `default_api:` en `tests/test_llm_factory.py` y verificar con `.venv/bin/pytest tests/test_llm_factory.py`


## 2. Sanitización en Webhook API

- [x] 2.1 Aplicar `clean_text_from_tool_call_artifacts` en `src/agent_service/api/webhook.py` asegurando respuestas limpias sin sintaxis técnica
- [x] 2.2 Agregar prueba en `tests/test_api_webhook.py` validando que respuestas con artefactos `call:default_api:` sean entregadas limpias en HTTP 200 y verificar con `.venv/bin/pytest tests/test_api_webhook.py -k test_sanitized`

## 3. Verificación y Validación Final

- [x] 3.1 Ejecutar toda la suite de pruebas del proyecto con `.venv/bin/pytest -m "not real_db"` y confirmar 100% de tests pasando sin regresiones
- [x] 3.2 Validar la conformidad del cambio con `openspec validate clean-response-formatting --strict`
