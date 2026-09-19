# AGENTS.md (spark_agent)

Este repositorio utiliza **OpenSpec (Spec-Driven Development)**. Todo cambio sustancial debe seguir estrictamente el flujo SDD.

## 1. Reglas de Oro y Restricciones
- **Planning Boundary:** En las fases de exploración (`explore`), propuesta (`propose`) y actualización (`update`), **NUNCA** modifiques código en `src/` ni `tests/`. Solo puedes crear o editar artefactos en `openspec/changes/<change-name>/`.
- **Fase de Aplicación (`apply`):** La modificación de código fuente solo está permitida aquí, ejecutando iterativamente las tareas de `tasks.md` de forma atómica.
- **Validación Obligatoria:** Antes de finalizar, ejecuta `openspec validate <change-name> --strict` y la suite de pruebas unitarias pertinente.

## 2. Ciclo de un Cambio OpenSpec
1. **Crear:** `openspec new change <id>`
2. **Planificar:** Rellenar `proposal.md`, `specs/<capability>/spec.md` (formato GIVEN/WHEN/THEN), `design.md` y `tasks.md`.
3. **Validar:** `openspec validate <id> --strict`
4. **Implementar:** Invocar al agente para ejecutar `tasks.md` paso a paso.
5. **Archivar:** `openspec archive <id>` una vez completado y probado.

## 3. Contexto Técnico Clave (`spark_agent`)
- **Stack:** Python 3.9+, LangGraph (subgrafos modulares), LangChain, Pydantic v2.
- **Persistencia:** PostgreSQL, `psycopg-pool`, `langgraph-checkpoint-postgres`, pgvector.
- **Estructura:** 
  - `src/agent_service/graph/`: Grafo principal y subgrafos.
  - `src/agent_service/tools/`: Herramientas ejecutables.
  - `src/agent_service/core/`: Componentes transversales y vector stores.
- **Pruebas:** 
  - Unitarias (sin DB real): `.venv/bin/pytest -m "not real_db"`
  - Integración: `.venv/bin/pytest`