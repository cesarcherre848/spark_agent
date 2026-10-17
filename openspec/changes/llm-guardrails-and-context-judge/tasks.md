# Tasks: Reemplazo de Guardrails Laya por LLM y Juez de Contexto Conversacional

## Fase 1: Guardrails LLM (Eliminación de Laya y Sustitución por LLM)
- [x] 1.1 Crear `src/agent_service/core/guardrails/llm_evaluator.py` implementando `evaluate_with_llm` con Pydantic estructurado (`LLMGuardrailDecision`) y prompt especializado en catálogo comercial y ventas.
- [x] 1.2 Modificar `src/agent_service/core/guardrails/evaluator.py` para desvincular Laya y conectar `evaluate_with_llm` como la Capa 2 tras la Capa 1 de Regex.
- [x] 1.3 Desarrollar suite de pruebas exhaustivas `tests/test_guardrails_llm.py` cubriendo:
  - Comandos operativos comerciales ("Agregar Cliente a Valeria Osores", "Quiero el ganamas nivel 2").
  - Preguntas de catálogo e inventario.
  - Ataques deliberados de jailbreak / prompt injection (deben ser bloqueados).
  - Consultas ambiguas o con errores tipográficos.

## Fase 2: Ventana de Corto Plazo (24h) y Juez de Contexto (Context Judge)
- [x] 2.1 Implementar ventana temporal de 24 horas (`SESSION_INACTIVITY_TTL_SECONDS = 86400`) para el historial de mensajes de corto plazo en `WhatsAppService` / `MainGraphState`, purgando turnos de días anteriores.
- [x] 2.2 Crear esquema `ContextJudgement` en `src/agent_service/graph/main_graph.py` (o módulo de schemas).
- [x] 2.3 Implementar el nodo `context_judge_node` en `src/agent_service/graph/main_graph.py` que analice la consulta entrante respecto al historial reciente de las últimas 24h y determine `is_standalone`, `context_usefulness` y `relevant_context_summary`.
- [x] 2.4 Incorporar `context_judge_node` en la topología del grafo principal entre `input_guardrail` y `router`.
- [x] 2.5 Actualizar el prompt de `router_node` para estructurar textualmente la consulta actual como prioridad máxima y limitar el uso del historial según el veredicto del juez.
- [x] 2.6 Crear pruebas unitarias `tests/test_context_judge.py` evaluando la ventana de 24h, cambios de cliente y preguntas de seguimiento válidas.

## Fase 3: Delimitación de Contexto en Subgrafos y Memoria de Largo Plazo
- [x] 3.1 Adaptar `BaseSynthesizerNode` y `sub_graphs/product_advisor/nodes.py` para ignorar menciones a clientes previos cuando el juez determine que el contexto es irrelevante/obsoleto.
- [x] 3.2 Verificar que el subgrafo `user_memory` mantenga la recuperación semántica de hechos a largo plazo (`user_memories` con embeddings) para personalizar antecedentes sin contaminar el diálogo de turnos.
- [x] 3.3 Asegurar que el subgrafo `contact_manage` se active limpiamente cuando el router detecte intenciones como "Agregar Cliente a Valeria Osores".

## Fase 4: Pruebas de Regresión y Validación en QA
- [x] 4.1 Ejecutar la suite completa de pruebas unitarias (`pytest -m "not real_db" --ignore=tests/whisper`).
- [x] 4.2 Sincronizar los archivos modificados con el contenedor `spark_agent_qa`.
- [x] 4.3 Ejecutar simulación end-to-end con la secuencia exacta de mensajes que falló en Gemma para certificar la resolución completa.
