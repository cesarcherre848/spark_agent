# Proposal: Abstracción de Consultas, Búsqueda Jerárquica y Marco Anti-Alucinación de SKUs

## Problem Statement
En sesiones comerciales reales de Spark Agent (caso `user_id = 8`), se detectaron alucinaciones de códigos SKU (ej. `[SKU: 27]`, `[SKU: 86]`, `[SKU: 89]`) causadas por confusión entre números de página, índices y SKUs en el prompt, así como limitaciones de asesoría en productos multivariante por falta de abstracción jerárquica (Concepto Base $\rightarrow$ Facetas de Variación).

## Proposed Changes
1. **Abstracción de Consultas y Query Decomposer:** Descomponer dinámicamente consultas compuestas multi-intención en sub-búsquedas enfocadas sin taxonomías rígidas ni sesgos de categoría.
2. **Universal Product Hierarchy Resolver:** Agrupar los resultados de búsqueda en conceptos canónicos de producto (`product_template`) y sus facetas de variación activas en Odoo (`product_product`).
3. **Formateo Universal y Limpio de Contexto:** Reestructurar `BaseSynthesizerNode.format_products_context` eliminando corchetes ambiguos en números de página.
4. **Marco de Validación y Blindaje Anti-Alucinación de SKUs (4 Capas):**
   - Extractor regex multi-patrón de menciones de SKU.
   - Auditor determinista contra la vista `view_user_authorized_products_metadata`.
   - Interceptor de reflexión en `rubric_evaluator_judge` con tabla de reemplazos obligatorios.
   - Sanitizador fail-safe antes del envío al canal.

## Success Criteria
- 100% de SKUs mencionados en las respuestas del recomendador deben ser válidos y verificados contra la base de datos de Odoo.
- Cero alucinaciones de números de página como SKUs.
- Consultas multi-intención (como la de Silvana: *facial económico + colonia fresca*) son atendidas de forma estructurada con sus SKUs oficiales.
- Pruebas unitarias y de regresión pasan al 100%.
