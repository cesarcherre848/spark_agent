# Design: Regla Always Text y Sanitización de Bloques de Contenido Gemini

## Context
Ver `proposal.md` para la motivación. Actualmente, las respuestas generadas por `ChatGoogleGenerativeAI` pueden contener estructuras de bloques en `response.content` (tipo `list` o `dict` con campos `text` y `extras.signature`, o `thought`), las cuales, al ser convertidas a string con `str()`, se serializan como texto literal exponiendo firmas criptográficas al usuario.

## Goals / Non-Goals

**Goals:**
- Proporcionar una función centralizada y resiliente `extract_clean_text` en `src/agent_service/core/llms/factory.py` que soporte objetos en memoria (`dict`, `list`), strings serializados (`ast.literal_eval`, `json.loads`) y fallback regex.
- Implementar una regla de validación antes de la serialización (`@field_validator("response", mode="before")`) en `WebhookResponse` para actuar como cortafuegos infalible en la capa de transporte API.
- Normalizar la extracción de texto en `src/agent_service/graph/main_graph.py` y subgrafos para que `final_response` siempre contenga texto conversacional plano.
- Descartar bloques de razonamiento interno (`type == "thought"`) de los modelos Gemini.

**Non-Goals:**
- No se alterará el formato de llamadas a herramientas estructuradas (`with_structured_output`) que requieran Pydantic schemas para decisiones internas del grafo (como el enrutador o evaluadores).
- No se eliminará información contextual legítima (emojis, saltos de línea, markdown de presentación de productos).

## Decisions

### Decisión 1: Extractor multinivel resiliente `extract_clean_text`
- **Enfoque**: Evaluar progresivamente el tipo de dato:
  1. Si es `dict`: extraer `"text"`, `"content"` o `"response_text"`. Si `type == "thought"`, retornar cadena vacía.
  2. Si es `list`: procesar recursivamente cada elemento, omitiendo pensamientos y uniendo textos con salto de línea.
  3. Si es `str`: si inicia con `{` o `[`, intentar deserializar con `ast.literal_eval` o `json.loads`. Si falla por caracteres especiales o truncamiento, aplicar regex `r"['\"]text['\"]\s*:\s*['\"](.*?)['\"](?:\s*,\s*['\"]extras['\"]|\s*\}\s*$)"`.
  4. Remover residuos de llamadas `call:default_api:...`.
- **Alternativas consideradas**:
  - *Solo regex*: Frágil ante saltos de línea y caracteres de escape.
  - *Solo `ast.literal_eval`*: Falla si la cadena contiene objetos o sintaxis no estándar.
  - *La combinación multinivel* asegura 100% de tolerancia a fallos.

### Decisión 2: Cortafuegos Pydantic v2 en `WebhookResponse`
- **Enfoque**: Colocar un `@field_validator("response", mode="before")` en `WebhookResponse`.
- **Razón**: Aunque los nodos del grafo se actualicen, el modelo de datos de FastAPI es la última línea de defensa que garantiza el contrato de la API. Ninguna respuesta puede salir al exterior con metadatos técnicos si el schema lo previene a nivel de validación.

## Risks / Trade-offs

- **[Risk] Pérdida de formato o saltos de línea al deserializar** → **Mitigación**: `extract_clean_text` procesa adecuadamente secuencias de escape `\n` y preserva la integridad del mensaje comercial.
- **[Risk] Sobrecarga de cómputo en la validación** → **Mitigación**: Si la entrada ya es un string plano normal sin prefijos `{` o `[`, la función retorna inmediatamente en O(1).
