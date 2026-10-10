# Tasks: Implementación de Reranker con Feedback Loops en Product Advisor

## Task Group 1: Esquemas de Rúbrica y Soporte de Acciones de Reflexión
- [x] 1.1 Actualizar `QualityRubricEvaluation` en `src/agent_service/graph/sub_graphs/product_advisor/schemas.py` añadiendo `criteria_clarity_score`, `intent_fulfillment_score` y las acciones `re_rerank_with_feedback` y `ask_clarification`.
- [x] 1.2 Añadir `RerankAndFilterInput` en `src/agent_service/tools/product_tools.py` con soporte para candidatos, contexto, target_item_count y feedback en lenguaje natural.
- [x] 1.3 Crear pruebas unitarias para validación de los esquemas en `tests/test_advisor_reranker_schemas.py`.

## Task Group 2: Herramienta Flexible `rerank_and_filter_candidates`
- [x] 2.1 Implementar `create_rerank_and_filter_candidates_tool` en `src/agent_service/tools/product_tools.py` con comprensión de contexto, soporte de feedback semántico, integración con `UniversalProductHierarchyResolver` y corte dinámico Top-$K$.
- [x] 2.2 Registrar la herramienta en `src/agent_service/graph/sub_graphs/product_advisor/tools.py` unificando `filter_and_sort_products`.
- [x] 2.3 Implementar pruebas unitarias en `tests/test_rerank_and_filter_tool.py` validando descarte anatómico, reordenamiento por precio, asimilación de feedback y truncado Top-$K$.

## Task Group 3: Integración del Reranker en el Planificador y Depuración de Estado
- [x] 3.1 Actualizar `plan_and_select_tools` en `src/agent_service/graph/sub_graphs/product_advisor/nodes.py` para planificar la invocación de `rerank_and_filter_candidates` tras la búsqueda.
- [x] 3.2 Limpiar la acumulación de candidatos obsoletos (`candidate_products`) al inicio de cada nuevo turno para evitar contaminación cruzada.
- [x] 3.3 Preservar los SKUs y precios del turno previo en el estado para auditoría de reducción de precio.

## Task Group 4: Medium Reflection Loop y Enrutamiento en el Grafo
- [x] 4.1 Modificar `rubric_evaluator_judge` en `nodes.py` para emitir la acción `re_rerank_with_feedback` con la crítica en lenguaje natural cuando la selección de productos deba ajustarse.
- [x] 4.2 Actualizar `_route_after_rubric` en `src/agent_service/graph/sub_graphs/product_advisor/graph.py` implementando el enrutamiento del Medium Loop hacia `execute_tools` y de `ask_clarification` hacia `synthesize_draft`.
- [x] 4.3 Actualizar `execute_tools` en `nodes.py` para re-ejecutar `rerank_and_filter_candidates` con la crítica recibida sin volver a consultar bases de datos externas.

## Task Group 5: Verificación Integral y Pruebas de Regresión (Escenario Silvana / User_id = 8)
- [x] 5.1 Ejecutar suite de pruebas unitarias (`.venv/bin/pytest -m "not real_db"`).
- [x] 5.2 Implementar prueba de regresión de extremo a extremo que simule la conversación de `user_id = 8`, verificando que ante *"algo aún más económico"* el sistema seleccione Totalist (S/. 40.00), descarte Biomilk y ajuste el ranking en el Medium Loop si es necesario.
