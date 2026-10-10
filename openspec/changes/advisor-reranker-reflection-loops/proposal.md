# Proposal: Reranker de Candidatos con Feedback Loops en Product Advisor

## Problem Statement
En sesiones comerciales reales de Spark Agent (`user_id = 8`), el recomendador de catálogo presentó deficiencias críticas:
1. Recomendación errónea de cremas corporales (`Biomilk Sensitive`) ante solicitudes de cuidado facial/cutis.
2. Reincidencia en productos costosos (S/. 65.00 a S/. 80.00) cuando la vendedora solicitó opciones económicas para su clienta Silvana.
3. Fallo ante la objeción *"algo aún más económico"*: el agente repitió la misma BB Cream (S/. 65.00) y agregó Supreme 8 en 1 (S/. 70.00), ignorando opciones accesibles en catálogo como Totalist (S/. 40.00).
4. Acumulación descontrolada de estado en LangGraph (hasta 99 candidatos acumulados de turnos previos, mezclando perfumes de hombre y maquillaje de otros clientes).
5. Falta de proactividad consultiva y aprobación ciega de la rúbrica (10/10) sin auditar la reducción de precio.

## Proposed Changes
1. **Herramienta Reranker Flexible (`rerank_and_filter_candidates`):**
   - Agrupación por familias canónicas (`UniversalProductHierarchyResolver`) antes del corte de variantes.
   - Comprensión contextual fluida de zona anatómica, género y preferencias comerciales.
   - Capacidad nativa de recibir feedback semántico del turno o de la rúbrica para re-ponderar candidatos sin esquemas matemáticos rígidos.
   - Truncado dinámico a los Top-$K$ solicitados (default Top-3 a 4).
2. **Ciclo de Reflexión de 3 Niveles:**
   - **Fast Loop:** Refinamiento directo de redacción en `synthesize_draft` ante problemas de formato o claridad.
   - **Medium Loop:** Re-ejecución inmediata de `rerank_and_filter_candidates` con la crítica emitida por la rúbrica (máx. 1 reintento, < 200 ms).
   - **Slow Loop:** Re-planificación hacia `plan_and_select_tools` si faltan productos en la base de datos.
3. **Proactividad Asertiva y Acción `ask_clarification`:**
   - Si los criterios no están 100% claros o hay disyuntiva marcada en catálogo, se activa `ask_clarification` para formular una pregunta enfocada al comercial en lugar de enviar a ciegas.
4. **Depuración de Candidatos Obsoletos por Turno:**
   - Se reinicia la lista de candidatos activos al inicio de cada nuevo turno para evitar contaminación cruzada entre clientes o categorías pasadas.

## Success Criteria
- 0% de productos corporales o masculinos recomendados ante consultas de rostro femenino.
- Ante la objeción *"aún más económico"*, el 100% de los productos recomendados tienen precio estrictamente menor al mínimo del turno previo.
- Si el usuario pide $K$ items (ej. *"2 cremas y 1 colonia"*), la respuesta contiene exactamente la cantidad solicitada.
- El Medium Loop resuelve re-rankeos guiados por crítica en < 250 ms sin llamadas redundantes a bases de datos.
- Cobertura completa de pruebas unitarias (`.venv/bin/pytest -m "not real_db"`).
