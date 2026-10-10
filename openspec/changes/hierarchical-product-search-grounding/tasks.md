# Tasks: Implementación de Búsqueda Jerárquica y Blindaje Anti-Alucinación de SKUs

## Task Group 1: Marco de Blindaje y Validación Anti-Alucinación de SKUs
- [x] 1.1 Crear `src/agent_service/core/guardrails/sku_grounding.py` con `SKUExtractor`, `SKUGroundingValidator` y `SKUFailSafeSanitizer`.
- [x] 1.2 Implementar pruebas unitarias en `tests/test_sku_grounding.py` para validar extracción multi-patrón, discriminación de precios/cantidades y sanitización.

## Task Group 2: Formateo Limpio de Contexto en Sintetizador Base
- [x] 2.1 Refactorizar `BaseSynthesizerNode.format_products_context` en `src/agent_service/graph/base_synthesizer.py` para eliminar corchetes en números de página y presentar el árbol de conceptos y variantes.
- [x] 2.2 Actualizar las directrices de `build_synthesizer_system_prompt` en `src/agent_service/graph/base_synthesizer.py` y `nodes.py` con reglas de emparejamiento de facetas y grounding estricto de SKUs.

## Task Group 3: Resolver Jerárquico Universal de Productos
- [x] 3.1 Crear `src/agent_service/core/stores/product/product_hierarchy.py` con `UniversalProductHierarchyResolver` para agrupar candidatos en conceptos canónicos y variantes autorizadas.
- [x] 3.2 Integrar el resolver en `src/agent_service/graph/sub_graphs/product_advisor/nodes.py` y `tools.py`.
- [x] 3.3 Implementar pruebas unitarias en `tests/test_product_hierarchy.py`.

## Task Group 4: Integración del Validador Determinista en la Rúbrica de Calidad
- [x] 4.1 Modificar `rubric_evaluator_judge` en `src/agent_service/graph/sub_graphs/product_advisor/nodes.py` para auditar deterministamente los SKUs antes de la decisión del LLM y forzar `refine_synthesis` ante discrepancias.
- [x] 4.2 Probar el ciclo de autorreflexión y corrección de SKUs.

## Task Group 5: Verificación Integral y Pruebas Rigurosas (Escenarios User_id = 8)
- [x] 5.1 Ejecutar suite completa de pruebas unitarias (`.venv/bin/pytest -m "not real_db"`).
- [x] 5.2 Ejecutar pruebas de regresión con los casos reales de `user_id = 8` (Caso Silvana, Caso Charito, Caso Renzo, Caso Dika) verificando que el 100% de los SKUs sean válidos y correspondan a Odoo.
