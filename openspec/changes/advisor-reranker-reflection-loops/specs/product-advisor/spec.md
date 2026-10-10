# Spec Delta: product-advisor

## Purpose
Provee capacidades de re-ranking contextual y flexible de candidatos jerárquicos con soporte nativo de feedback en lenguaje natural y bucles de reflexión de 3 niveles (Fast, Medium y Slow Loops), garantizando fidelidad a la intención, asertividad comercial y alta flexibilidad conversacional.

## ADDED Requirements

### Requirement: Flexible Hierarchical Candidate Reranking Tool with Feedback
El sistema SHALL implementar una herramienta de re-ranking contextual (`rerank_and_filter_candidates`) que agrupe candidatos en familias canónicas (`UniversalProductHierarchyResolver`), discrimine contexto anatómico y de público objetivo de forma flexible, y sea capaz de recibir retroalimentación semántica directa en lenguaje natural para reordenar o filtrar productos sin requerir esquemas rígidos predefinidos.

#### Scenario: Body lotions discarded for facial query
- **GIVEN** candidate products containing both Biomilk Sensitive (body) and Totalist Antiarrugas (facial)
- **WHEN** reranker executes for a facial care query
- **THEN** Biomilk is discarded by context mismatch, and Totalist is selected in the top candidates.

#### Scenario: Dynamic Top-K Truncation
- **GIVEN** retrieved candidates for facial care and colognes
- **WHEN** user requests "dame 2 opciones de cremas y 1 colonia"
- **THEN** reranker selects and outputs exactly 2 facial care products and 1 cologne.

#### Scenario: Semantic feedback guides reranking
- **GIVEN** a pool of candidates with varying prices and benefits
- **WHEN** reranker executes with feedback "priorizar opciones antiedad accesibles y colonias florales suaves para 50 años"
- **THEN** candidates aligning with mature skin and soft floral fragrance are ranked at the top.

### Requirement: Three-Tier Reflection Loop (Fast, Medium, Slow)
El nodo auditor `rubric_evaluator_judge` SHALL soportar 3 niveles de reflexión:
1. `refine_synthesis` (Fast Loop): Corrige redacción sin re-ejecutar herramientas.
2. `re_rerank_with_feedback` (Medium Loop): Re-ejecuta `rerank_and_filter_candidates` inyectando la crítica en lenguaje natural de la rúbrica (máximo 1 reintento por turno).
3. `replan_tools` (Slow Loop): Vuelve a `plan_and_select_tools` para buscar nuevos productos si el catálogo está vacío.

#### Scenario: Inadequate product selection resolved via Medium Loop
- **GIVEN** a draft recommending a S/. 70 product after an economic request
- **WHEN** rubric judge evaluates the draft
- **THEN** it rejects with `reflection_action = "re_rerank_with_feedback"` and a critique indicating that products must be cheaper than the previous turn, and the Medium Loop re-scores the pool to promote Totalist (S/. 40) in under 250 ms.

### Requirement: Asertive Clarification on Infeasibility or Ambiguity
El sistema SHALL activar la acción `ask_clarification` cuando los criterios de la consulta no estén 100% claros o cuando exista una disyuntiva comercial marcada en el catálogo, formulando una pregunta enfocada al usuario en lugar de adivinar o forzar recomendaciones sin criterio.

#### Scenario: Ambiguous criteria triggers focused clarification
- **GIVEN** a user request with ambiguous expectations where available catalog options involve marked trade-offs
- **WHEN** rubric evaluates criteria clarity
- **THEN** it activates `ask_clarification` and the synthesizer proposes the best available alternatives while asking a focused question to calibrate the user's intent.
