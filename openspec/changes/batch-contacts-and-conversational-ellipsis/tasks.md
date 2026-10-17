# Tasks: Batch Contacts and Conversational Ellipsis Resolution

## Implementation Tasks

- [x] 1. Extender esquemas y estado en `contact_manage`
  - [x] 1.1 Agregar `CustomerItem` y `customers: List[CustomerItem]` a `CustomerExtractionResult` en `src/agent_service/graph/sub_graphs/contact_manage/schemas.py`.
  - [x] 1.2 Agregar `extracted_customers` y `batch_operation_results` a `ContactManageState` en `src/agent_service/graph/sub_graphs/contact_manage/state.py`.
- [x] 2. Implementar extracción y upsert por lotes en `contact_manage/nodes.py`
  - [x] 2.1 Actualizar prompt y parseo en `extract_customer_info` para soportar múltiples clientes.
  - [x] 2.2 Actualizar `upsert_customer_node` para iterar y registrar todos los clientes encontrados.
  - [x] 2.3 Actualizar `synthesize_response` para reportar un resumen claro de todos los clientes procesados.
- [x] 3. Mejorar resolución de elipsis conversacionales en Guardrail, ContextJudge y Router
  - [x] 3.1 Añadir contexto previo al prompt de `llm_evaluator.py` para evitar falsos positivos por nombres geográficos.
  - [x] 3.2 Añadir soporte de conectores elípticos en `ContextJudgeNode` (`is_standalone=False`).
  - [x] 3.3 Calibrar prompt del Router para que preguntas como "¿Y a Francia?" hereden la intención de contactos.
- [x] 4. Pruebas y Validación
  - [x] 4.1 Crear `tests/test_batch_contacts_and_ellipsis.py` con pruebas unitarias para batch upsert y resolución de elipsis.
  - [x] 4.2 Ejecutar suite completa con pytest: `.venv/bin/pytest tests/test_batch_contacts_and_ellipsis.py tests/test_context_judge.py`.
  - [x] 4.3 Desplegar cambios en el contenedor `spark_agent_qa` y probar en vivo.
