"""
tests/test_rerank_and_filter_tool.py - Pruebas unitarias para rerank_and_filter_candidates.
"""

import pytest
from src.agent_service.tools.product_tools import create_rerank_and_filter_candidates_tool


@pytest.fixture
def rerank_tool():
    return create_rerank_and_filter_candidates_tool()


def test_anatomical_gatekeeper_discards_body_products_for_facial_query(rerank_tool):
    """Verifica que para consultas de rostro/cutis se descarten lociones corporales como Biomilk."""
    candidates = [
        {"sku": "750", "name": "Biomilk Sensitive", "description": "Crema corporal hidratante", "price": 80.0},
        {"sku": "79", "name": "Totalist Crema Antiarrugas", "description": "Crema facial nutritiva", "price": 40.0},
        {"sku": "67", "name": "BB Cream Matificante", "description": "Crema facial matificante con color", "price": 65.0},
    ]

    result = rerank_tool.invoke({
        "candidates": candidates,
        "context_query": "Silvana, 50 años, busca crema para el cuidado del rostro",
        "target_item_count": 3,
    })

    skus = [p["sku"] for p in result]
    assert "750" not in skus, "Biomilk corporal debió ser descartado para consulta facial"
    assert "79" in skus
    assert "67" in skus


def test_price_reordering_for_economy_focus(rerank_tool):
    """Verifica que ante expresiones de economía los productos más baratos queden primero."""
    candidates = [
        {"sku": "21213", "name": "Supreme 8 en 1", "description": "Crema facial antiedad", "price": 70.0},
        {"sku": "67", "name": "BB Cream Matificante", "description": "Crema facial matificante", "price": 65.0},
        {"sku": "79", "name": "Totalist Antiarrugas", "description": "Crema facial antiarrugas", "price": 40.0},
    ]

    result = rerank_tool.invoke({
        "candidates": candidates,
        "context_query": "Silvana quisiera algo aún más económico para el cutis",
        "target_item_count": 3,
    })

    assert len(result) >= 2
    # El primer producto debe ser el más económico (Totalist S/. 40)
    assert result[0]["sku"] == "79"
    assert result[0]["price"] == 40.0


def test_semantic_feedback_assimilation(rerank_tool):
    """Verifica que la herramienta asimile feedback para excluir SKUs o recortar cotas de precio."""
    candidates = [
        {"sku": "67", "name": "BB Cream Matificante", "description": "Facial SPF 30", "price": 65.0},
        {"sku": "21213", "name": "Supreme 8 en 1", "description": "Facial antiedad", "price": 70.0},
        {"sku": "79", "name": "Totalist Antiarrugas", "description": "Facial antiarrugas", "price": 40.0},
    ]

    result = rerank_tool.invoke({
        "candidates": candidates,
        "context_query": "Silvana cremas rostro",
        "feedback": "Excluir SKU 67 y 21213 por superar el presupuesto. Buscar opciones menor a S/. 50",
        "target_item_count": 2,
    })

    skus = [p["sku"] for p in result]
    assert "67" not in skus
    assert "21213" not in skus
    assert "79" in skus


def test_dynamic_top_k_truncation(rerank_tool):
    """Verifica que si el usuario pide exactamente 2 items, se trunque a 2."""
    candidates = [
        {"sku": "1", "name": "Totalist Antiarrugas", "price": 40.0},
        {"sku": "2", "name": "Totalist Ultrahidratante", "price": 40.0},
        {"sku": "3", "name": "Crema Anti Imperfecciones", "price": 62.0},
        {"sku": "4", "name": "BB Cream Matificante", "price": 65.0},
    ]

    result = rerank_tool.invoke({
        "candidates": candidates,
        "context_query": "Dame 2 opciones para el rostro",
        "target_item_count": 2,
    })

    assert len(result) == 2
