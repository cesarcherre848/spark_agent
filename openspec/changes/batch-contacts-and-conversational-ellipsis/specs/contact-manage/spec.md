# Delta Spec: Batch Contact Creation and Conversational Ellipsis

## Capability: `contact-manage`

### Scenario: Extracción y registro de múltiples clientes en un solo mensaje
- **GIVEN** que un vendedor envía una instrucción con múltiples clientes (ej. "Registra a Valeria Osores y a Francia")
- **WHEN** el subgrafo `contact_manage` ejecuta `extract_customer_info`
- **THEN** debe extraer una lista con ambos clientes con sus respectivos nombres y números
- **AND** `upsert_customer_node` debe procesar cada cliente de la lista contra Odoo
- **AND** la respuesta sintetizada debe confirmar el alta de todos los clientes procesados.

### Scenario: Retrocompatibilidad con cliente individual
- **GIVEN** que un vendedor envía "Registra el cliente Pedro con teléfono 999888777"
- **WHEN** el extractor procesa el mensaje
- **THEN** debe mapear adecuadamente tanto el campo individual `extracted_name` como la colección `extracted_customers`
- **AND** registrar a Pedro de manera normal sin regresiones.

### Scenario: Elipsis conversacional con nombres propios homónimos a países
- **GIVEN** que en el turno anterior se habló del registro o gestión de contactos
- **WHEN** el usuario pregunta inmediatamente "¿Y a Francia?"
- **THEN** la Capa 2 del Guardrail debe evaluar el turno como ALLOW/negocio
- **AND** el `ContextJudgeNode` debe marcar `is_standalone=False`
- **AND** el `Router` debe clasificar la intención como `contact` (o la intención comercial correspondiente) en lugar de `out_of_scope` por geografía.
