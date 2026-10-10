# Design: Reranker de Candidatos con Feedback Loops en Product Advisor

## Architecture Overview
La solución implementa el flujo flexible acordado con **Herramienta Reranker Contextual y Bucle de Reflexión de 3 Niveles** dentro del subgrafo `product_advisor`:

```
┌─────────────────────────────────┐
│                                 │
│     1. plan_and_select_tools    │◄─────────────────────────────────────────────┐
│                                 │                                              │
└────────────────┬────────────────┘                                              │
                 │                                                               │
                 ▼                                                               │
┌─────────────────────────────────┐                                              │
│                                 │                                              │
│         2. execute_tools        │◄─────────────────────────────────────────────┤
│                                 │                                              │
└────────────────┬────────────────┘                                              │
  Ejecuta search + reranker_tool                            3. Slow Loop: Faltan productos nuevos en BD
                 ▼                                                               │
┌─────────────────────────────────┐                                              │
│                                 │                                              │
│       3. synthesize_draft       │◄─2.─Medium─Loop:─Error─de─selección─de─productos<br>Re-ejecuta reranker_tool con critique
│                                 │                                              │
└────────────────┬────────────────┘                                              │
                 │                                            1. Fast Loop: Solo error de redacción
                 ▼                                                               │
┌─────────────────────────────────┐                                              │
│                                 │                                              │
│    4. rubric_evaluator_judge    ├──────────────────────────────────────────────┘
│                                 │
└────────────────┬────────────────┘
             Aprobado
                 ▼
┌─────────────────────────────────┐
│                                 │
│       5. finalize_response      │
│                                 │
└─────────────────────────────────┘
```

## Data Structures & Schemas

### 1. `RerankAndFilterInput`
En `src/agent_service/tools/product_tools.py`:
- `candidates`: Lista de productos candidatos a clasificar y ordenar.
- `context_query`: Consulta o foco comercial del turno (ej. *"cuidado facial económico para 50 años"*).
- `target_item_count`: Número de items requeridos si fue explícito (Top-$K$).
- `feedback`: Instrucción o crítica en lenguaje natural de la rúbrica o turno previo para reordenar la lista de forma flexible sin esquemas matemáticos rígidos.

### 2. `QualityRubricEvaluation`
En `src/agent_service/graph/sub_graphs/product_advisor/schemas.py`:
- `relevance_score`, `grounding_score`, `constraints_score`, `presentation_score`, `clarity_and_unambiguity_score`.
- `intent_fulfillment_score`: ¿La respuesta cumplió con el cambio solicitado en el turno (ej. menor precio)?
- `criteria_clarity_score`: ¿El criterio de consulta está 100% claro? Si no, se activa `ask_clarification`.
- `reflection_action`: `approve`, `refine_synthesis`, `re_rerank_with_feedback`, `replan_tools`, `ask_clarification`.
- `critique`: Crítica detallada en lenguaje natural que alimenta directamente al Medium Loop.

## Componentes Técnicos

### 1. Herramienta `rerank_and_filter_candidates`
En `src/agent_service/tools/product_tools.py` y `src/agent_service/graph/sub_graphs/product_advisor/tools.py`:
- Reemplaza y unifica la funcionalidad de `filter_and_sort_products`.
- Integra `UniversalProductHierarchyResolver` para procesar conceptos y variantes de forma jerárquica.
- Aplica corte dinámico a los Top-$K$ solicitados.
- Admite feedback semántico libre para reordenar candidatos sin rigidez.

### 2. Enrutador Condicional `_route_after_rubric`
En `src/agent_service/graph/sub_graphs/product_advisor/graph.py`:
- Si `reflection_action == "re_rerank_with_feedback"` e `iteration_count < max_iterations`:
  - Enruta a `execute_tools` pasando la crítica del juez para un re-ranking en memoria (< 200 ms).
- Si `reflection_action == "ask_clarification"`:
  - Enruta a `synthesize_draft` para emitir una pregunta enfocada de calibración en lugar de enviar a ciegas.
- Si `reflection_action == "refine_synthesis"`:
  - Fast Loop directo a `synthesize_draft`.
- Si faltan datos en catálogo:
  - Slow Loop a `plan_and_select_tools`.
