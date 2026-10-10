"""
tests/test_user8_scenarios.py - Pruebas rigurosas de los escenarios reales de la consultora Giuliana Yañez (user_id=8).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorPlan,
    FinalAnswer,
    QualityRubricEvaluation,
)
from src.agent_service.core.guardrails.sku_grounding import (
    SKUGroundingValidator,
    SKUExtractor,
)


@pytest.mark.asyncio
async def test_user8_silvana_scenario_grounding():
    """Valida que la consulta de Silvana (50 años, facial económico + colonia fresca)
    genere una recomendación con SKUs 100% reales y rechace cualquier número de página."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    # Catálogo real disponible para el turno
    candidates = [
        {"product_id": 241, "sku": "150", "name": "Totalist Crema Antiarrugas", "price": 40.0, "pagina": 27},
        {"product_id": 242, "sku": "81", "name": "Totalist Gel Control Brillo", "price": 62.0, "pagina": 27},
        {"product_id": 510, "sku": "79", "name": "Totalist Crema SPF 30 Antiarrugas", "price": 40.0, "pagina": 27},
        {"product_id": 11, "sku": "2210", "name": "Soy Única Colonia", "price": 68.0, "pagina": 86},
        {"product_id": 41, "sku": "2195", "name": "Violeta Salvaje Colonia", "price": 68.0, "pagina": 89},
        {"product_id": 424, "sku": "15225", "name": "Freshing Colonia corporal", "price": 46.9, "pagina": 110},
    ]

    # 1. Caso Falla previa: LLM alucina páginas 27 y 86 como SKUs
    hallucinated_draft = (
        "Para Silvana te recomiendo:\n"
        "- *[SKU: 27] Totalist Crema SPF 30 Antiarrugas* (S/. 40.00)\n"
        "- *[SKU: 86] Soy Única Colonia* (S/. 68.00)\n"
    )
    grounding_fail = SKUGroundingValidator.validate_draft_grounding(hallucinated_draft, candidates)
    assert grounding_fail.is_valid is False
    assert "27" in grounding_fail.invalid_skus
    assert "86" in grounding_fail.invalid_skus
    assert grounding_fail.replacement_map["27"] in ("79", "150")
    assert grounding_fail.replacement_map["86"] == "2210"

    # 2. Caso Corregido: LLM utiliza los SKUs oficiales reales
    correct_draft = (
        "Para Silvana (50 años, piel mixta) te recomiendo estas excelentes opciones económicas:\n"
        "- *[150] Totalist Crema Antiarrugas 100g* (S/. 40.00): Con jalea real para líneas de expresión.\n"
        "- *[81] Totalist Gel Control Brillo 100g* (S/. 62.00): Con aloe vera y niacinamida para piel mixta.\n"
        "- *[15225] Freshing Colonia corporal 750ml* (S/. 46.90): Súper económica y refrescante.\n"
    )
    grounding_success = SKUGroundingValidator.validate_draft_grounding(correct_draft, candidates)
    assert grounding_success.is_valid is True
    assert set(grounding_success.valid_skus) == {"150", "81", "15225"}


@pytest.mark.asyncio
async def test_user8_charito_rubor_scenario():
    """Valida que la consulta de Charito (rubor para piel trigueña) exponga variantes de tono."""
    candidates = [
        {"product_id": 49, "sku": "520", "name": "Rubor Compacto Petal Rose", "price": 96.0},
        {"product_id": 111, "sku": "4018", "name": "Rubor Compacto Champagne", "price": 96.0},
        {"product_id": 436, "sku": "6361", "name": "Rubor Compacto Bronzage", "price": 96.0},
    ]

    draft = (
        "Para Charito con piel trigueña te recomiendo:\n"
        "- *[6361] Rubor Compacto Bronzage* (S/. 96.00): Tono cálido ideal para su tono de piel.\n"
        "- *[4018] Rubor Compacto Champagne* (S/. 96.00): Aporta luminosidad elegante.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"6361", "4018"}
