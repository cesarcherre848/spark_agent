# Proposal: Regla Always Text y Sanitización de Bloques de Contenido Gemini

## Why
Al interactuar con modelos modernos de Google Gemini (como Gemini 2.0 / 3.0 via `langchain-google-genai`), las respuestas emitidas pueden encapsular bloques de contenido, firmas criptográficas y metadatos internos (`{'type': 'text', 'text': '...', 'extras': {'signature': '...'}}`) o pensamientos (`{'type': 'thought'}`). Cuando los nodos del grafo o el Webhook convierten estos objetos directamente a cadenas de texto (`str(content)`), dichos metadatos técnicos se filtran en la respuesta entregada a clientes externos y plataformas de mensajería (OpenClaw, WhatsApp).

Es indispensable formalizar una regla estricta a nivel de contrato de datos ("Always Text") y una estrategia de defensa en profundidad para garantizar que cualquier salida destinada al usuario sea 100% texto plano conversacional, libre de estructuras serializadas, firmas y bloques técnicos.

## What Changes
- **Regla en Contrato de Datos (Pydantic v2 Cortafuegos)**: Añadir un validador en `WebhookResponse` que garantice que el campo `response` sea siempre procesado y limpiado como texto conversacional plano antes de salir del servidor.
- **Extractor Universal Resiliente (`extract_clean_text`)**: Implementar en `src/agent_service/core/llms/factory.py` un extractor robusto capaz de desempaquetar diccionarios, listas de bloques y cadenas serializadas (`ast.literal_eval` / `json.loads` / regex fallback), descartando bloques `thought` y firmas `extras.signature`.
- **Normalización en Nodos LangGraph**: Sustituir el uso de `str(response.content)` por `extract_clean_text(response.content)` en `general_chat_node` de `main_graph.py` y subgrafos asociados.
- **Invariante en Prompts de Sistema**: Añadir directrices de formato explícitas que instruyan al LLM a emitir únicamente texto conversacional sin metadatos.

## Capabilities

### New Capabilities
*(Ninguna)*

### Modified Capabilities
- `api-web-hook`: Se amplía el requerimiento de sanitización para exigir que cualquier estructura de bloques de Gemini, firmas criptográficas (`extras.signature`), bloques de pensamiento (`thought`) o diccionarios serializados sean eliminados, entregando exclusivamente texto plano conversacional en `response`.

## Impact
- **APIs**: El endpoint del Webhook `/api/v1/webhook` y `/webhook` garantiza que el campo `response` nunca contiene fragmentos serializados de Python o Gemini.
- **Modelos Pydantic**: Actualización de `WebhookResponse` en `src/agent_service/api/schemas.py`.
- **Core LLM**: Inclusión de `extract_clean_text` en `src/agent_service/core/llms/factory.py`.
- **Nodos del Grafo**: Sanitización en `src/agent_service/graph/main_graph.py`.
- **Pruebas**: Pruebas unitarias en `tests/test_api_webhook.py` y validación estricta de OpenSpec.
