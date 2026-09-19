# Proposal: Sanitización y Limpieza de Respuestas del Agente

## Why
Al invocar modelos de Gemini con Function Calling para salidas estructuradas (`with_structured_output`), en ocasiones `raw_msg.content` contiene una lista con fragmentos técnicos internos de la API de Google (ej. `['call:default_api:RecommendationSynthesisResponse{response_text:', '...', '}']`). Cuando el recuperador resiliente de salida estructurada extrae el texto, puede convertir dicha lista a un string literal, provocando una fuga de metadatos técnicos hacia el usuario final y plataformas externas como OpenClaw o WhatsApp.

Se requiere asegurar que tanto el extractor estructurado resiliente como el Webhook de Spark Agent extraigan directamente los argumentos de `tool_calls` o saniticen cualquier residuo de sintaxis de llamadas a herramientas, garantizando respuestas 100% legibles y limpias para el usuario comercial.

## What Changes
- **Extracción directa de `tool_calls` en `ResilientStructuredOutputRunnable`**: Priorizar la lectura de los argumentos en `raw_msg.tool_calls` (donde Gemini sitúa los argumentos estructurados limpios) antes de recurrir a la conversión de `raw_msg.content`.
- **Desempaquetado inteligente de listas en `raw_msg.content`**: Si `content` es una lista o contiene firmas de llamada como `call:default_api:`, extraer únicamente el valor textual útil descartando prefijos y corchetes técnicos.
- **Sanitizador en Webhook API**: En [src/agent_service/api/webhook.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/src/agent_service/api/webhook.py), aplicar una función de limpieza `sanitize_final_response` que garantice que ningún residuo de sintaxis de tool call (`call:default_api:...`) o corchetes de lista sea entregado en el JSON final.

## Capabilities

### New Capabilities
*(Ninguna)*

### Modified Capabilities
- `api-web-hook`: Se agrega el requerimiento de sanitización estricta de la respuesta final devuelta por el webhook para prevenir fugas de metadatos técnicos.

## Impact
- **APIs**: Las respuestas devueltas por `/api/v1/webhook` y `/webhook` son garantizadas 100% limpias de artefactos de Function Calling.
- **LLM Core**: Mejora en `ResilientStructuredOutputRunnable` en [src/agent_service/core/llms/factory.py](file:///Users/cesarcherre/Documents/proyects/spark_agent/src/agent_service/core/llms/factory.py).
- **Pruebas**: Pruebas unitarias de sanitización y recuperación ante listas y firmas de `call:default_api:`.
