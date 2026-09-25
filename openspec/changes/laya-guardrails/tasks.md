# Tasks: Laya Guardrails Decision Engine

## 1. Dependencias y Modelos de Datos

- [x] 1.1 Agregar `laya` a `requirements.txt` e instalarlo en el entorno virtual (`.venv/bin/pip install laya`).
- [x] 1.2 Extender esquemas en `src/agent_service/core/guardrails/schemas.py` agregando `GuardrailAction` (`ALLOW`, `WARN`, `BLOCK`), y los campos `action`, `is_warning`, `warning_message` y `scores` en `GuardrailResult`.

## 2. Cliente Laya y Reglas Semánticas (Score / Noul)

- [x] 2.1 Crear cliente singleton con carga perezosa (lazy load) para Laya en `src/agent_service/core/guardrails/laya_client.py` con fallback resiliente y soporte para variable de entorno `LAYA_GUARDRAIL_MODEL`.
- [x] 2.2 Definir en `src/agent_service/core/guardrails/rules.py` el diccionario de preguntas Laya (`noul` para jailbreak/inyecciones/datos sensibles/fuera de contexto, y `score` para severidad de daño) y la función de mapeo de umbrales para clasificar en `ALLOW`, `WARN` o `BLOCK`.
- [x] 2.3 Actualizar `src/agent_service/core/guardrails/evaluator.py` para ejecutar la Capa 1 de Regex rápida (< 1ms) y, si es segura, delegar a la Capa 2 de Laya computando la acción de decisión final.

## 3. Integración en el Grafo Principal y Propagación de Advertencias

- [x] 3.1 Actualizar el estado `MainGraphState` en `src/agent_service/graph/main_graph.py` incorporando los campos `guardrail_action`, `is_warning` y `guardrail_warning`.
- [x] 3.2 Modificar `input_guardrail_node` en `src/agent_service/graph/main_graph.py` para enrutar: `BLOCK` hacia `guardrail_blocked`, y `WARN`/`ALLOW` hacia el flujo comercial (`retrieve_memory`).
- [x] 3.3 Asegurar que en el flujo comercial, si `is_warning` está activo, la advertencia comercial se incorpore en la interacción final con el usuario sin romper la salida del subgrafo correspondiente.

## 4. Pruebas y Validación OpenSpec

- [x] 4.1 Crear la suite de pruebas unitarias en `tests/test_guardrails_laya.py` validando las tres decisiones (`ALLOW`, `WARN`, `BLOCK`), el Fast-Path regex, la invocación de Laya (con mocks en pruebas unitarias) y el fallback de seguridad.
- [x] 4.2 Ejecutar las pruebas unitarias del proyecto con `.venv/bin/pytest -m "not real_db"` y certificar que pasen al 100%.
- [x] 4.3 Ejecutar `openspec validate laya-guardrails --strict` y verificar que la especificación sea completamente válida.
