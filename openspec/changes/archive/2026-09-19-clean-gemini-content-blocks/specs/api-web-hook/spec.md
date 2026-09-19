# Spec Delta: api-web-hook

## MODIFIED Requirements

### Requirement: Clean User-Facing Response Sanitization
The system SHALL strictly sanitize final agent responses to ensure that internal tool-call artifacts (including `call:default_api:` prefixes and function call wrappers), Gemini content block dictionaries, serialized dict structures, cryptographic thought signatures (`extras.signature`), and internal thought blocks are completely stripped, delivering exclusively clean conversational plain text to the client.

#### Scenario: Tool Call Metadata Stripping
- **WHEN** el grafo del agente produce una respuesta conteniendo metadatos técnicos de tool call (como `['call:default_api:RecommendationSynthesisResponse{response_text:', 'Texto limpio...', '}']`)
- **THEN** el sistema extrae y entrega exclusivamente el texto limpio `Texto limpio...` sin prefijos ni corchetes.

#### Scenario: Gemini Block Dict and Thought Signature Stripping
- **WHEN** el modelo de lenguaje o el grafo produce una respuesta encapsulada en un diccionario serializado con metadatos técnicos (como `{'type': 'text', 'text': 'Hola, ¡muy bien!...', 'extras': {'signature': '...'}}`)
- **THEN** el sistema extrae y entrega en el campo `response` exclusivamente el texto de `text`, descartando cualquier firma y estructura de diccionario.

#### Scenario: Gemini Thought Block Elimination
- **WHEN** la respuesta del modelo contiene bloques de razonamiento interno (`type == "thought"`) junto con bloques de texto
- **THEN** el sistema omite completamente los bloques de pensamiento y entrega únicamente el texto dirigido al usuario.

#### Scenario: Plain Text Pass-Through
- **WHEN** la respuesta del agente es una cadena de texto conversacional sin metadatos técnicos
- **THEN** el sistema preserva íntegramente el contenido, puntuación y emojis del mensaje.
