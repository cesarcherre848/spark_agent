# Spec Delta: product-advisor

## Purpose
Provee capacidades avanzadas de abstracción de consultas, agrupación jerárquica de conceptos de producto y un marco determinista anti-alucinación de SKUs en 4 capas para garantizar recomendaciones comerciales verídicas, completas y 100% conectadas con el catálogo de Odoo.

## ADDED Requirements

### Requirement: Query Intent Abstraction & Sub-search Decomposition
El sistema SHALL analizar y descomponer consultas compuestas multi-intención en sub-consultas de búsqueda independientes sin encasillar en taxonomías rígidas ni sesgos de categoría.

#### Scenario: Multi-intent query decomposes cleanly
- **WHEN** user sends "Quiero productos para el rostro económicos para 50 años y también una colonia fresca"
- **THEN** planner decomposes into sub-searches for facial care and fresh colognes, retrieving candidates for both needs without noise overlap.

### Requirement: Universal Product Hierarchy Resolution
El sistema SHALL agrupar dinámicamente los candidatos recuperados en Conceptos Canónicos de Producto (`product_template`) y sus Facetas de Variación autorizadas (`product_product`), extrayendo diferenciadores de tamaño/gramaje, formulación/activo, tono/acabado, SKU y precio.

#### Scenario: Multi-variant product expanded with all active options
- **WHEN** user requests economic anti-aging products
- **THEN** system retrieves the canonical Totalist family and exposes its active facets (Antiarrugas 100g SKU 150, Control Brillo 100g SKU 81, Antiarrugas SPF 30 50g SKU 79) with exact prices.

### Requirement: Deterministic SKU Grounding & Anti-Hallucination Guard
El sistema SHALL auditar de forma determinista mediante expresiones regulares y validación en base de datos (`view_user_authorized_products_metadata`) que el 100% de los códigos SKU citados en el borrador de respuesta existan en el catálogo autorizado.

#### Scenario: Hallucinated SKU rejected and auto-corrected via reflection
- **WHEN** synthesizer draft mentions an invalid SKU or page number (e.g., "[SKU: 27]" or "[SKU: 86]")
- **THEN** rubric judge node immediately marks `meets_rubric = False`, `reflection_action = "refine_synthesis"`, and provides a mandatory replacement table with official SKUs.

### Requirement: Fail-Safe SKU Sanitization
El sistema SHALL post-procesar la respuesta final antes de emitirla al canal (WhatsApp/Web), garantizando que ningún código inválido o no autorizado sea expuesto al usuario.

#### Scenario: Persistent invalid SKU sanitized before transmission
- **WHEN** an ungrounded bracketed code persists after reflection limit
- **THEN** fail-safe sanitizer replaces it with the canonical product SKU or neutralizes the bracketed hallucination.
