# Design: Batch Contacts and Conversational Ellipsis Resolution

## Architecture Overview

```
                      +-----------------------------+
                      |   Input Query + History     |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Input Guardrail (Capa 2)   |
                      | Context-aware / Ellipsis ok |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      Context Judge Node     |
                      | Detects ellipsis:           |
                      | is_standalone = False       |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |         Router Node         |
                      | "Y a Francia?" -> contact   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |     contact_manage Graph    |
                      |   Batch Extractor & Upsert  |
                      +-----------------------------+
```

## Detailed Component Changes

### 1. `src/agent_service/graph/sub_graphs/contact_manage/schemas.py`
- Agregar `CustomerItem(BaseModel)` con `name: str`, `phones: List[str]`, `contact_id: Optional[int]`.
- En `CustomerExtractionResult`, agregar:
  ```python
  customers: List[CustomerItem] = Field(default_factory=list)
  ```
  manteniendo `name` y `phones` opcionales en el root para retrocompatibilidad 100%.

### 2. `src/agent_service/graph/sub_graphs/contact_manage/state.py`
- Agregar campos a `ContactManageState`:
  - `extracted_customers: Optional[List[Dict[str, Any]]]`
  - `batch_operation_results: Optional[List[Dict[str, Any]]]`

### 3. `src/agent_service/graph/sub_graphs/contact_manage/nodes.py`
- En `extract_customer_info`:
  - Si `customers` viene poblado, poblar `extracted_customers`. Si solo viene `name`, empaquetar un `CustomerItem`.
  - Si la consulta es una elipsis (ej. `"Y a Francia?"`), el extractor debe ser capaz de asociar `"Francia"` como nuevo cliente a registrar si en el turno previo la acción era un registro (`upsert`).
- En `upsert_customer_node`:
  - Iterar sobre `extracted_customers` (o fallback a `extracted_name`).
  - Ejecutar `_upsert_customer_tool` para cada contacto.
  - Guardar lista de resultados y consolidar respuesta para el sintetizador.

### 4. `src/agent_service/core/guardrails/llm_evaluator.py` & `main_graph.py`
- Pasar contexto conversacional a `aevaluate_input_guardrail` o considerar histórico en el prompt de Capa 2.
- En el prompt de `create_router_node`:
  - Agregar regla explícita: Preguntas cortas de seguimiento como *"¿Y a [Nombre]?"*, *"¿Y qué hay de [X]?"* cuando el contexto previo involucra gestión de clientes o pedidos deben clasificarse en la intención previa (`contact` o `sales`).
- En `create_context_judge_node`:
  - Patrones elípticos (`^([¿\s]*y\s+(a\s+|de\s+|con\s+|para\s+)|qué hay de)`) marcan `is_standalone=False` y rescatan el contexto relevante del turno anterior.
