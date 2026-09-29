"""
tests/test_incentive_tools.py - Pruebas unitarias para las herramientas de incentivos y tendencias Odoo
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import Response
from datetime import datetime

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.incentive_tools import (
    odoo_get_active_incentives,
    extract_sales_trends,
    calculate_incentive_gap,
    get_active_incentive_targets,
)


@pytest.fixture
def mock_odoo_client():
    client = MagicMock(spec=OdooClient)
    client._uid = 1
    client._client = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_odoo_get_active_incentives_rest_success(mock_odoo_client):
    """Valida que odoo_get_active_incentives consuma correctamente el endpoint REST /api/v1/sales_incentives."""
    mock_resp = MagicMock(spec=Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "count": 2,
        "total": 2,
        "data": [
            {
                "id": 42,
                "title": "Colonia Soy Sexy",
                "name": "[Colonia Soy Sexy] Unique S.A.",
                "supplier_id": 9,
                "supplier_name": "Unique S.A.",
                "partner_id": 9,
                "partner_name": "Unique S.A.",
                "description": "Bonificación de 1 colonia Soy Sexy gratis",
                "observations": "Por la compra acumulada de 3 unidades",
                "date_start": "2026-09-15 00:00:00",
                "date_end": None,
                "state": "active",
                "is_valid": True,
            },
            {
                "id": 44,
                "title": "Escala 30% Descuento",
                "name": "[Escala 30%] Unique S.A.",
                "supplier_id": 9,
                "supplier_name": "Unique S.A.",
                "partner_id": 9,
                "partner_name": "Unique S.A.",
                "description": "30% de descuento comercial desde S/ 720",
                "observations": "Aplicable para pedidos desde S/ 720",
                "date_start": "2026-09-15 00:00:00",
                "date_end": None,
                "state": "active",
                "is_valid": True,
            }
        ]
    }
    mock_odoo_client._client.get.return_value = mock_resp

    res = await odoo_get_active_incentives(supplier_id=9, client=mock_odoo_client)

    assert len(res) == 2
    assert res[0]["title"] == "Colonia Soy Sexy"
    assert res[1]["title"] == "Escala 30% Descuento"
    mock_odoo_client._client.get.assert_awaited_once_with(
        "/api/v1/sales_incentives",
        params={"limit": 50, "offset": 0, "supplier_id": 9, "only_active": "true"},
    )


@pytest.mark.asyncio
async def test_odoo_get_active_incentives_fallback_jsonrpc(mock_odoo_client):
    """Valida que ante fallo de REST, odoo_get_active_incentives use fallback a JSON-RPC."""
    mock_odoo_client._client.get.side_effect = Exception("REST endpoint down")

    mock_jsonrpc_resp = MagicMock(spec=Response)
    mock_jsonrpc_resp.status_code = 200
    mock_jsonrpc_resp.json.return_value = {
        "result": [
            {
                "id": 26,
                "name": "[Rappel Anual 5%] CETCO S.A.",
                "title": "Rappel Anual 5%",
                "partner_id": [17, "CETCO S.A."],
                "date_start": "2026-01-01 00:00:00",
                "date_end": False,
                "state": "active",
                "is_valid": True,
                "description": "5% rappel anual por cuota",
                "observations": "Liquidación anual",
            }
        ]
    }
    mock_odoo_client._client.post.return_value = mock_jsonrpc_resp

    res = await odoo_get_active_incentives(supplier_id=17, client=mock_odoo_client)

    assert len(res) == 1
    assert res[0]["title"] == "Rappel Anual 5%"
    assert res[0]["supplier_name"] == "CETCO S.A."
    mock_odoo_client._client.post.assert_awaited_once()


def test_extract_sales_trends():
    """Valida que extract_sales_trends ordene correctamente los productos más vendidos."""
    orders_data = {
        "sale": [
            {
                "id": 1,
                "lines": [
                    {"product_id": [11, "[2210] Soy Única Colonia"], "product_uom_qty": 3.0, "price_subtotal": 102.0},
                    {"product_id": [13, "[2035] Ccori Rubí Parfum"], "product_uom_qty": 1.0, "price_subtotal": 212.0},
                ],
            },
            {
                "id": 2,
                "lines": [
                    {"product_id": [11, "[2210] Soy Única Colonia"], "product_uom_qty": 2.0, "price_subtotal": 68.0},
                    {"product_id": [458, "[2218] Soy Poderosa Colonia"], "product_uom_qty": 4.0, "price_subtotal": 136.0},
                ],
            }
        ],
        "draft": []
    }

    trends = extract_sales_trends(orders_data, top_k=2)

    assert len(trends) == 2
    # SKU 2210 aparece en 2 órdenes distintas (order_count = 2)
    assert trends[0]["sku"] == "2210"
    assert trends[0]["total_qty"] == 5.0
    assert trends[0]["order_count"] == 2


def test_calculate_incentive_gap():
    """Valida el cálculo de brecha de ventas hacia la siguiente escala de descuento y promociones."""
    orders_data = {
        "sale": [
            {"id": 1, "amount_total": 350.0},
        ],
        "draft": [
            {"id": 2, "amount_total": 100.0},
        ]
    }
    incentive_plans = [
        {
            "id": 42,
            "title": "Vende y Gana: Colonia Soy Sexy Gratis",
            "description": "1 colonia gratis por 3 unidades",
            "observations": "Por la compra acumulada de 3 unidades de colonias Soy (Cód. 2210, 2218).",
            "date_end": "2026-10-31 23:59:59",
            "supplier_name": "Unique S.A.",
        }
    ]

    ref_date = datetime(2026, 10, 1)
    gap = calculate_incentive_gap(orders_data, incentive_plans, reference_date=ref_date)

    assert gap["confirmed_total_pen"] == 350.0
    assert gap["draft_total_pen"] == 100.0
    assert gap["potential_total_pen"] == 450.0
    # 350.0 supera 230 (25%), pero no alcanza 720 (30%)
    assert gap["current_discount_tier"] == "25% de Descuento"
    assert gap["next_discount_tier"] == "30% de Descuento"
    assert gap["gap_to_next_tier_pen"] == 370.0  # 720.0 - 350.0
    assert len(gap["product_promos"]) == 1
    assert "2210" in gap["product_promos"][0]["skus"]
    assert gap["days_remaining"] == 30


@pytest.mark.asyncio
async def test_get_active_incentive_targets_integration(mock_odoo_client):
    """Valida que la herramienta puente retorne la estructura completa esperada por product_advisor."""
    with patch("src.agent_service.tools.incentive_tools.odoo_list_sales_orders", new_callable=AsyncMock) as mock_list, \
         patch("src.agent_service.tools.incentive_tools.odoo_get_active_incentives", new_callable=AsyncMock) as mock_inc:
        mock_list.return_value = {
            "sale": [{"id": 1, "amount_total": 400.0, "lines": []}],
            "draft": [],
        }
        mock_inc.return_value = [
            {
                "id": 42,
                "title": "Colonia Soy Sexy",
                "observations": "Por 3 unidades de Cód. 2210",
                "description": "Colonia gratis",
            }
        ]

        res = await get_active_incentive_targets(user_id=5, supplier_id=9, client=mock_odoo_client)

        assert res["user_id"] == 5
        assert "2210" in res["bonus_skus"]
        assert res["current_discount_tier"] == "25% de Descuento"
        assert res["gap_to_next_tier_pen"] == 320.0  # 720 - 400
