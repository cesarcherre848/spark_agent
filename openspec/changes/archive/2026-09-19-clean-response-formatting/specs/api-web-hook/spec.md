# Spec Delta: api-web-hook

## ADDED Requirements

### Requirement: Clean User-Facing Response Sanitization
The system SHALL sanitize final agent responses to ensure that internal tool-call artifacts (including `call:default_api:` prefixes, function call wrappers, and stringified raw list representations) are stripped before delivering the final response text to the client.

#### Scenario: Tool Call Metadata Stripping
- **WHEN** el grafo del agente produce una respuesta conteniendo metadatos técnicos de tool call (como `['call:default_api:RecommendationSynthesisResponse{response_text:', 'Texto limpio...', '}']`)
- **THEN** el sistema extrae y entrega exclusivamente el texto limpio `Texto limpio...` sin prefijos ni corchetes.
