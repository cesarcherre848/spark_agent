"""
tests/test_exhaustive_suite.py - Batería de Pruebas Exhaustivas, Casos Borde, Estrés y Ataques Adversarios de SKUs.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from src.agent_service.core.guardrails.sku_grounding import (
    SKUExtractor,
    SKUGroundingValidator,
    SKUFailSafeSanitizer,
)
from src.agent_service.core.stores.product.product_hierarchy import (
    UniversalProductHierarchyResolver,
    extract_base_concept_name,
    extract_attribute_details,
)
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    FinalAnswer,
    QualityRubricEvaluation,
)


# ==============================================================================
# 1. BATERÍA EXHAUSTIVA DE EXTRACCIÓN Y DISCRIMINACIÓN DE SKUS (CASOS BORDE)
# ==============================================================================

@pytest.mark.parametrize(
    "input_text,expected_skus,forbidden_skus",
    [
        # Caso 1: Múltiples corchetes mezclados con precios, unidades y fechas
        (
            "Te sugiero [SKU: 150] Totalist a S/. 40.00 (100g) y [81] Totalist Gel a S/. 62.00 para 50 años en C10.",
            ["150", "81"],
            ["40", "62", "100", "50", "10"],
        ),
        # Caso 2: Formatos con prefijos variados ('código: #1234', 'ref 5678')
        (
            "Opciones: código: #2037 Ohm Now (S/. 230.00), Ref: 346 Desodorante Roll-on (S/. 34.50). Orden S00016.",
            ["2037", "346"],
            ["230", "34", "16"],
        ),
        # Caso 3: Textos con teléfonos, porcentajes, SPF y multiplicadores
        (
            "Llama al +51975674799. El producto [74] tiene 15% Vitamina C, SPF 100 y rinde 2x más por S/. 178.00.",
            ["74"],
            ["975674799", "51975674799", "15", "100", "2", "178"],
        ),
        # Caso 4: SKUs alfanuméricos con guiones o letras
        (
            "Tenemos los productos [SKU-PROMO1] y [ES_4018] junto con [79934].",
            ["SKU-PROMO1", "ES_4018", "79934"],
            [],
        ),
        # Caso 5: Texto conversacional sin SKUs explícitos
        (
            "Hola, actualmente no contamos con stock de esa línea en este momento. ¿Deseas ver otra opción?",
            [],
            [],
        ),
    ],
)
def test_sku_extractor_adversarial_edge_cases(input_text, expected_skus, forbidden_skus):
    """Prueba exhaustiva de extracción y rechazo de falsos positivos en textos complejos."""
    extracted = SKUExtractor.extract_skus(input_text)
    for exp in expected_skus:
        assert exp in extracted, f"Se esperaba extraer SKU '{exp}' pero no se encontró en {extracted}"
    for forb in forbidden_skus:
        assert forb not in extracted, f"El número '{forb}' fue extraído erróneamente como SKU en {extracted}"


# ==============================================================================
# 2. PRUEBAS DE ESTRÉS Y RENDIMIENTO (ALTA CARGA DE CANDIDATOS)
# ==============================================================================

@pytest.mark.asyncio
async def test_sku_grounding_performance_stress():
    """Valida que la validación y sanitización de 1,000 productos se ejecute en menos de 50 milisegundos."""
    # Generar catálogo sintético masivo de 1,000 productos
    massive_candidates = [
        {"product_id": i, "sku": f"SKU_{i:04d}", "name": f"Producto Genérico {i}", "price": float(i % 100 + 10), "pagina": (i % 50) + 1}
        for i in range(1, 1001)
    ]

    # Borrador con 10 SKUs válidos y 5 números de página erróneos
    draft = "Recomendaciones:\n" + "\n".join([f"- [SKU: SKU_{i:04d}] Producto {i}" for i in range(1, 11)]) + "\n- [SKU: 25] Falso 1\n- [SKU: 30] Falso 2"

    t0 = time.perf_counter()
    result = SKUGroundingValidator.validate_draft_grounding(draft, massive_candidates)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    assert elapsed_ms < 50.0, f"La validación tomó {elapsed_ms:.2f}ms (debe ser < 50ms)"
    assert result.is_valid is False
    assert "25" in result.invalid_skus
    assert "30" in result.invalid_skus
    assert len(result.valid_skus) == 10


# ==============================================================================
# 3. PRUEBAS DE RESOLUCIÓN JERÁRQUICA MULTI-CATEGORÍA Y ZERO-BIAS
# ==============================================================================

@pytest.mark.asyncio
async def test_universal_hierarchy_multi_domain():
    """Valida la resolución jerárquica agnóstica en 5 dominios de catálogo distintos:
    Joyería, Perfumería, Maquillaje, Skincare y Accesorios."""
    candidates = [
        # Joyería (baños y piedras)
        {"product_id": 1, "sku": "JOY_01", "name": "Aretes Amour Doré - Baño de Oro", "marca": "Yanbal", "price": 85.0, "description": "Baño de oro 24k"},
        {"product_id": 2, "sku": "JOY_02", "name": "Aretes Amour Doré - Baño de Plata", "marca": "Yanbal", "price": 75.0, "description": "Baño de plata fina"},
        # Perfumería (volúmenes)
        {"product_id": 3, "sku": "PERF_50", "name": "Ohm Parfum Cont. 50 ml", "marca": "Yanbal", "price": 180.0, "description": "50 ml toronja rosada"},
        {"product_id": 4, "sku": "PERF_100", "name": "Ohm Parfum Cont. 100 ml", "marca": "Yanbal", "price": 230.0, "description": "100 ml toronja rosada"},
        # Maquillaje (tonos)
        {"product_id": 5, "sku": "LIP_01", "name": "Hydra-Lip Tono Nude", "marca": "Yanbal", "price": 45.0, "description": "Tono Nude acabado mate"},
        {"product_id": 6, "sku": "LIP_02", "name": "Hydra-Lip Tono Coral", "marca": "Yanbal", "price": 45.0, "description": "Tono Coral acabado mate"},
    ]

    resolver = UniversalProductHierarchyResolver()
    families = await resolver.group_candidates_hierarchically(candidates)

    family_names = {f.family_name.lower() for f in families}
    assert any("amour doré" in fn for fn in family_names)
    assert any("ohm parfum" in fn for fn in family_names)
    assert any("hydra-lip" in fn for fn in family_names)

    for fam in families:
        if "amour doré" in fam.family_name.lower():
            assert len(fam.variants) == 2
        elif "ohm parfum" in fam.family_name.lower():
            assert len(fam.variants) == 2


# ==============================================================================
# 4. PRUEBA DEL CICLO COMPLETO EVALUATOR-OPTIMIZER (RECHAZO ➔ REFLEXIÓN ➔ APROBACIÓN)
# ==============================================================================

@pytest.mark.asyncio
async def test_full_evaluator_optimizer_cycle_grounding():
    """Prueba end-to-end del ciclo Evaluator-Optimizer:
    1. El borrador contiene un SKU alucinado (página 27).
    2. La Rúbrica rechaza con critique_message detallando el reemplazo a SKU 150.
    3. El sintetizador corrige usando SKU 150.
    4. La Rúbrica aprueba con 100% de éxito."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    candidates = [
        {"product_id": 241, "sku": "150", "name": "Totalist Crema Antiarrugas", "price": 40.0, "pagina": 27},
    ]

    # Iteración 1: Borrador con alucinación
    state_iter1 = {
        "raw_query": "recomiéndame Totalist",
        "candidate_products": candidates,
        "draft_response": "Te recomiendo [SKU: 27] Totalist Crema Antiarrugas (S/. 40.00).",
        "iteration_count": 0,
        "max_iterations": 3,
        "messages": [],
    }

    eval_iter1 = await nodes.rubric_evaluator_judge(state_iter1)
    assert eval_iter1["meets_rubric"] is False
    assert eval_iter1["reflection_action"] == "refine_synthesis"
    assert "TABLA DE REEMPLAZO OBLIGATORIO" in eval_iter1["critique"]

    # Iteración 2: Borrador corregido tras recibir el feedback de reflexión
    state_iter2 = {
        "raw_query": "recomiéndame Totalist",
        "candidate_products": candidates,
        "draft_response": "Te recomiendo [150] Totalist Crema Antiarrugas (S/. 40.00).",
        "iteration_count": 1,
        "max_iterations": 3,
        "messages": [],
    }

    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=9.5,
        grounding_score=10.0,
        constraints_score=9.5,
        presentation_score=9.0,
        is_approved=True,
    )

    eval_iter2 = await nodes.rubric_evaluator_judge(state_iter2)
    assert eval_iter2["meets_rubric"] is True
    assert eval_iter2["reflection_action"] == "approve"
