# Design: Abstracción de Consultas, Jerarquía de Productos y Validación de SKUs

## Architecture Overview
La solución desacopla y fortalece el ciclo de asesoría comercial de producto en tres capas:

1. **Abstracción y Planificación (`ProductAdvisorPlanner`):**
   - Extracción de propósito funcional y modificadores.
   - Generación concurrente de planes de búsqueda con `asyncio.gather`.

2. **Resolución Jerárquica Universal (`UniversalProductHierarchyResolver`):**
   - Agrupación por `product_tmpl_id` y líneas canónicas.
   - Consulta a `view_user_authorized_products_metadata` para obtener variantes autorizadas para el `user_id`.
   - Extracción normalizada de facetas (tamaño, tono, beneficio, precio, SKU).

3. **Marco de Blindaje Anti-Alucinación (`SKUGroundingEngine`):**
   - `SKUExtractor`: Detección multi-patrón de códigos en texto.
   - `SKUGroundingValidator`: Comprobación determinista de pertenencia al catálogo.
   - `RubricJudge`: Rechazo con tabla de sustitución explícita en Fast Reflection Loop.
   - `SKUFailSafeSanitizer`: Sellado previo al canal.

## Data Structures & Schemas
- `ProductVariantView`: Modelo para representar una variante con su `sku`, `name`, `price`, `attribute_details`.
- `ProductCanonicalFamily`: Modelo para el concepto base con su lista de `variants`.
- `SKUGroundingResult`: Resultado de validación con `is_valid`, `invalid_skus`, `replacement_suggestions`.

## Formateo Universal de Contexto
Eliminación de números de página en corchetes dentro de `BaseSynthesizerNode.format_products_context` para prevenir confusión en la atención del LLM.
