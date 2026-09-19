# Tasks

## 1. Core LLM Utilities

- [x] 1.1 Implementar la función `extract_clean_text` en `src/agent_service/core/llms/factory.py` con soporte para strings, dicts de bloques Gemini (`type: text`), listas de bloques, descarte de pensamientos (`type: thought`), desempaquetado de cadenas serializadas (`ast.literal_eval`, `json.loads`, regex) y sanitización de llamadas técnicas (`call:default_api:`).
- [x] 1.2 Actualizar `clean_text_from_tool_call_artifacts` y `_clean_content_to_text` en `src/agent_service/core/llms/factory.py` para utilizar `extract_clean_text`.

## 2. Pydantic Schema Cortafuegos & Webhook API

- [x] 2.1 Agregar el validador `@field_validator("response", mode="before")` en `WebhookResponse` en `src/agent_service/api/schemas.py` garantizando la sanitización obligatoria hacia texto limpio.
- [x] 2.2 Actualizar `src/agent_service/api/webhook.py` para usar `extract_clean_text` en la extracción de la respuesta final del agente.

## 3. LangGraph Nodes & Prompts

- [x] 3.1 Actualizar `general_chat_node` en `src/agent_service/graph/main_graph.py` reemplazando `str(response.content)` por `extract_clean_text(response.content)` e incorporando la directiva de texto plano conversacional en el prompt del sistema.
- [x] 3.2 Actualizar nodos de síntesis en subgrafos (`product_recomender` y `sales_manage`) asegurando el uso de `extract_clean_text` ante respuestas directas del LLM.

## 4. Pruebas Unitarias y Validación

- [x] 4.1 Añadir pruebas unitarias en `tests/test_api_webhook.py` cubriendo la sanitización de diccionarios Gemini serializados con firmas `extras.signature`, listas de bloques, descarte de bloques `thought`, y texto plano íntegro.
- [x] 4.2 Ejecutar `openspec validate clean-gemini-content-blocks --strict` y verificar que el cambio cumpla con todas las especificaciones.
- [x] 4.3 Ejecutar `.venv/bin/pytest tests/test_api_webhook.py` y `.venv/bin/pytest -m "not real_db"` verificando que todas las pruebas pasen al 100%.
