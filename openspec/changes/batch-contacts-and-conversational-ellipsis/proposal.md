# Proposal: Batch Contact Creation and Conversational Ellipsis Resolution

## Problem Statement
Durante las pruebas de usuario en vivo con el asistente comercial (ej. Gemma / `wa_51953320946`), se observaron dos fallas funcionales críticas:
1. **Pérdida de contactos en solicitudes múltiples (Batch):** Cuando un usuario solicita registrar varios clientes en un solo mensaje (ej. *"Registra a Valeria Osores y a Francia..."*), el subgrafo `contact_manage` solo extrae y procesa un único contacto porque el esquema `CustomerExtractionResult` y la topología del grafo únicamente soportan un nombre individual (`name: Optional[str]`). El segundo cliente queda ignorado silenciosamente.
2. **Falso bloqueo/desvío por Elipsis Conversacional (Nombres propios ambiguos):** Cuando el usuario pregunta de seguimiento *"¿Y a Francia?"*, el Evaluador de Guardrail y el Router clasifican la pregunta como geográfica/fuera de ámbito (`out_of_scope`) asumiendo envíos internacionales a la República Francesa, en lugar de reconocer que "Francia" es el nombre de la persona pendiente de registrar del turno previo.

## Proposed Changes
1. **Soporte Multi-Cliente (Batch Upsert) en `contact_manage`:**
   - Extender `CustomerExtractionResult` para admitir una lista de clientes `customers: List[CustomerInputItem]`. Mantener retrocompatibilidad transparente con campos planos (`name`, `phones`).
   - Actualizar los nodos de extracción, validación de duplicados y ejecución en `ContactManageNodes` para iterar y registrar a todos los clientes especificados en la consulta.
   - Devolver una síntesis consolidada reportando el resultado de cada cliente registrado o actualizado.
2. **Context-Aware Guardrail & Router para Elipsis Conversacionales:**
   - Enriquecer `aevaluate_input_guardrail` en Capa 2 para contemplar el contexto conversacional reciente (últimos 2 mensajes) antes de clasificar una pregunta como `out_of_scope`.
   - Instruir al prompt del Router y de Capa 2 que consultas elípticas cortas (ej. *"¿Y a [Nombre]?"*, *"¿Y [Nombre]?"*) tras una interacción comercial o de contactos deben clasificarse como `contact` o `sales`, permitiendo su resolución contextual.
3. **ContextJudgeNode con Detección de Elipsis:**
   - Asegurar que consultas que comiencen con conectores de continuidad elíptica (*"Y a..."*, *"¿Y qué hay de...?"*) mantengan `is_standalone=False` y rescaten el contexto relevante en lugar de descartarlo.

## Acceptance Criteria
- [x] Una instrucción con 2 o más clientes (ej. *"Agrega a Valeria Osores al 999111222 y a Francia al 988333444"*) extrae y registra a ambos clientes en Odoo.
- [x] Preguntas elípticas de seguimiento como *"¿Y a Francia?"* no son catalogadas como geografía/envíos internacionales si el turno previo discutía creación de contactos o pedidos, sino que son enrutadas a `contact_manage`.
- [x] Suite de pruebas exhaustiva que cubra extracción por lotes, retrocompatibilidad mono-cliente y resolución de elipsis.
