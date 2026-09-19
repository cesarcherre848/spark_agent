# Design: clean-response-formatting

## Context
Ver `proposal.md` para la motivación. Cuando los modelos de Google Gemini procesan llamadas de salida estructurada, LangChain almacena los fragmentos de llamada en `AIMessage.content` o `tool_calls`. Si el parser falla en parsear el esquema principal, el mecanismo de contingencia (`ResilientStructuredOutputRunnable`) convertía el contenido a string sin inspeccionar primero `tool_calls` ni limpiar los marcadores `call:default_api:`.

## Goals / Non-Goals

**Goals:**
- Priorizar la extracción de argumentos estructurados directamente desde `raw_msg.tool_calls` en `ResilientStructuredOutputRunnable`.
- Detectar y desempaquetar listas de fragmentos en `raw_msg.content` eliminando wrappers de `call:default_api:` y llaves de cierre.
- Proveer una función pura y reutilizable `clean_text_from_tool_call_artifacts(text: str) -> str` en `src/agent_service/core/llms/factory.py`.
- Integrar la sanitización en `src/agent_service/api/webhook.py` antes de serializar `WebhookResponse`.

**Non-Goals:**
- No altera los esquemas Pydantic de los subgrafos (`RecommendationSynthesisResponse`, etc.).
- No modifica el comportamiento de enrutamiento ni los prompts del sistema.

## Decisions

### 1. Inspección Prioritaria de `tool_calls`
- **Decisión**: En `ResilientStructuredOutputRunnable`, si `res.get("parsed")` es None pero `res.get("raw")` tiene `tool_calls`, iterar sobre `tool_calls` para extraer `args` y validarlos directamente con `self._schema.model_validate(args)`.
- **Justificación**: Los argumentos en `tool_calls` ya vienen como un diccionario Python limpio de clave-valor provisto por LangChain, evitando cualquier necesidad de parsing con regex o manipulación de strings.

### 2. Sanitizador Regex de Contingencia
- **Decisión**: Crear `clean_text_from_tool_call_artifacts(text: str) -> str` que:
  1. Si el texto coincide con el patrón `\['call:default_api:[^{]+(?:\{[^:]+:)?\s*'(.*)'\s*,\s*'\}'\]`, extrae el contenido entre comillas.
  2. Si el texto empieza con `call:default_api:`, remueve la cabecera técnica y la llave final `}`.
  3. Reemplaza saltos de línea escapados literales `\\n` por `\n` si el string proviene de una lista stringificada de Python.
- **Justificación**: Actúa como un escudo de defensa en profundidad para que bajo ninguna circunstancia se filtre sintaxis interna al cliente.

## Risks / Trade-offs
- **[Riesgo] Falsos positivos que remuevan texto legítimo que mencione "call:"** → *Mitigación*: La regex requiere estrictamente el prefijo específico de Google GenAI `call:default_api:` seguido del nombre del schema y corchetes, evitando colisiones con texto ordinario.
