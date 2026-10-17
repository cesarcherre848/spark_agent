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
    calculate_multibrand_incentive_gaps,
    resolve_supplier_sales_metrics,
    detect_target_supplier,
    get_active_incentive_targets,
    get_supplier_default_campaign,
    detect_campaign_from_query,
    normalize_incentive_metadata,
)


@pytest.fixture
def mock_odoo_client():
    client = MagicMock(spec=OdooClient)
    client._uid = 1
    client._client = AsyncMock()
    return client


def test_detect_target_supplier():
    """Valida la detección de marcas en consultas de usuario."""
    assert detect_target_supplier("¿Cómo voy en Yanbal?")["id"] == 9
    assert detect_target_supplier("cual es el incentiv9 mas grande que tiene yambal ahora ?")["id"] == 9
    assert detect_target_supplier("¿Cuánto me falta en Unique?")["id"] == 9
    assert detect_target_supplier("¿Cuáles son mis metas de Ésika?")["id"] == 17
    assert detect_target_supplier("¿Cómo voy en esica?")["id"] == 17
    assert detect_target_supplier("¿Cómo voy con Belcorp?")["id"] == 17
    assert detect_target_supplier("¿Y en Cyzone?")["id"] == 17
    assert detect_target_supplier("¿Cómo voy en L'Bel?")["id"] == 17
    assert detect_target_supplier("¿Cómo van mis avances de venta?") is None
    assert detect_target_supplier("Quiero ver mis pedidos generales") is None


def test_resolve_supplier_sales_metrics():
    """Valida que las líneas de pedido se particionen correctamente por proveedor."""
    orders_data = {
        "sale": [
            {
                "id": 1,
                "lines": [
                    {"product_name": "[5107] Base Antiedad Yanbal", "price_subtotal": 69.0, "quantity": 1.0},
                    {"product_name": "[35515] Set Max Supreme Ésika", "price_subtotal": 142.0, "quantity": 2.0},
                ],
            }
        ],
        "draft": [
            {
                "id": 2,
                "lines": [
                    {"product_name": "[5208] Labial Glow Unique", "price_subtotal": 33.0, "quantity": 1.0}
                ],
            }
        ]
    }
    sku_map = {
        "5107": 9,   # Unique S.A.
        "35515": 17, # CETCO S.A.
        "5208": 9,   # Unique S.A.
    }

    metrics = resolve_supplier_sales_metrics(orders_data, sku_supplier_map=sku_map)

    assert metrics[9]["confirmed_total_pen"] == 69.0
    assert metrics[9]["draft_total_pen"] == 33.0
    assert metrics[9]["potential_total_pen"] == 102.0
    assert metrics[17]["confirmed_total_pen"] == 142.0
    assert metrics[17]["draft_total_pen"] == 0.0
    assert metrics[17]["potential_total_pen"] == 142.0


def test_calculate_multibrand_incentive_gaps():
    """Valida el cálculo de brechas independiente para cada proveedor."""
    orders_data = {
        "sale": [
            {
                "id": 1,
                "lines": [
                    {"product_name": "[5107] Base Antiedad Yanbal", "price_subtotal": 250.0, "quantity": 1.0},
                    {"product_name": "[35515] Set Max Supreme Ésika", "price_subtotal": 500.0, "quantity": 1.0},
                ],
            }
        ],
        "draft": []
    }
    incentives = [
        {
            "id": 44,
            "title": "Escala 30% Descuento",
            "supplier_id": 9,
            "supplier_name": "Unique S.A.",
        },
        {
            "id": 26,
            "title": "Promoción Rappel Anual 5%",
            "supplier_id": 17,
            "supplier_name": "CETCO S.A.",
        }
    ]
    sku_map = {"5107": 9, "35515": 17}

    gaps = calculate_multibrand_incentive_gaps(orders_data, incentives, sku_supplier_map=sku_map)

    # Unique: 250 supera 230 (25%), pero no 720 (30%)
    assert gaps[9]["confirmed_total_pen"] == 250.0
    assert gaps[9]["current_discount_tier"] == "25% de Descuento"
    assert gaps[9]["next_discount_tier"] == "30% de Descuento"
    assert gaps[9]["gap_to_next_tier_pen"] == 470.0  # 720 - 250

    # CETCO: Rappel activo en la lista
    assert gaps[17]["confirmed_total_pen"] == 500.0
    assert gaps[17]["current_discount_tier"] == "Rappel Comercial Vigente"
    assert gaps[17]["has_active_incentives"] is True


def test_calculate_multibrand_incentive_gaps_inactive_cetco():
    """Valida que si CETCO tiene 0 planes activos en Odoo, no se inventen metas ficticias."""
    orders_data = {"sale": [], "draft": []}
    incentives = [
        {"id": 44, "title": "Escala 30% Descuento", "supplier_id": 9, "supplier_name": "Unique S.A."}
    ]
    gaps = calculate_multibrand_incentive_gaps(orders_data, incentives)
    assert gaps[17]["has_active_incentives"] is False
    assert gaps[17]["current_discount_tier"] == "Sin promociones activas"
    assert gaps[17]["gap_to_next_tier_pen"] == 0.0
    assert len(gaps[17]["product_promos"]) == 0


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
    assert trends[0]["sku"] == "2210"
    assert trends[0]["total_qty"] == 5.0
    assert trends[0]["order_count"] == 2


def test_calculate_incentive_gap():
    """Valida el cálculo de brecha de ventas hacia la siguiente escala de descuento y promociones."""
    orders_data = {
        "sale": [
            {"id": 1, "amount_total": 350.0, "lines": []},
        ],
        "draft": [
            {"id": 2, "amount_total": 100.0, "lines": []},
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
         patch("src.agent_service.tools.incentive_tools.odoo_get_active_incentives", new_callable=AsyncMock) as mock_inc, \
         patch("src.agent_service.tools.incentive_tools.get_sku_supplier_map", new_callable=AsyncMock) as mock_sku:
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
        mock_sku.return_value = {}

        res = await get_active_incentive_targets(user_id=5, supplier_id=9, client=mock_odoo_client)

        assert res["user_id"] == 5
        assert "2210" in res["bonus_skus"]
        assert res["current_discount_tier"] == "25% de Descuento"
        assert res["gap_to_next_tier_pen"] == 320.0  # 720 - 400


def test_detect_campaign_from_query():
    """Valida la detección semántica de campaña en consultas de usuario."""
    assert detect_campaign_from_query("¿Cuáles son los premios de la C11?") == "C11"
    assert detect_campaign_from_query("Quiero revisar la campaña 11") == "C11"
    assert detect_campaign_from_query("¿Hay beneficios en C-15?") == "C-15"
    assert detect_campaign_from_query("Ver catálogo C10") == "C10"
    assert detect_campaign_from_query("¿Cómo van mis ventas de Yanbal?") is None


def test_get_supplier_default_campaign():
    """Valida la obtención de la campaña por defecto por proveedor."""
    assert get_supplier_default_campaign(9) == "C11"
    assert get_supplier_default_campaign(17) == "C-15"
    assert get_supplier_default_campaign(999) is None
    assert get_supplier_default_campaign(None) is None


def test_normalize_incentive_metadata():
    """Valida la extracción y normalización de metadatos desde etiquetas y observaciones."""
    sample_record = {
        "id": 96,
        "title": "Vende y Gana Perfumes: 1 Ohm Parfum 100 ml Gratis (Cód. Catálogo 200)",
        "supplier_id": 9,
        "description": "Producto Bonificado: 1 Ohm Parfum (Cód. 200) gratis por pedidos S/ 690+",
        "observations": "Incluye Cód. 231, Cód. 2223, Cód. 200, Cód. 2041",
        "tag_ids": [77, 80, 82, 126],
        "tags": [
            {"id": 77, "name": "Yanbal"},
            {"id": 80, "name": "Pág 10"},
            {"id": 82, "name": "Pág 12"},
            {"id": 126, "name": "C11"},
        ],
    }
    meta = normalize_incentive_metadata(sample_record)

    assert meta["campaign"] == "C11"
    assert meta["brand"] == "Yanbal"
    assert meta["pages"] == [10, 12]
    assert meta["category"] == "vende_y_gana"
    assert "200" in meta["skus"]
    assert "231" in meta["skus"]
    assert "2223" in meta["skus"]
    assert "2041" in meta["skus"]


@pytest.mark.asyncio
async def test_odoo_get_active_incentives_with_campaign_and_metadata_filter(mock_odoo_client):
    """Valida la consulta con filtro de campaña y post-filtrado por metadata."""
    mock_resp = MagicMock(spec=Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "count": 2,
        "total": 2,
        "data": [
            {
                "id": 84,
                "title": "Escala Comercial 25%",
                "supplier_id": 9,
                "tags": [{"id": 126, "name": "C11"}, {"id": 82, "name": "Pág 12"}],
                "tag_ids": [126, 82],
                "observations": "Descuento en factura",
                "description": "25% de descuento",
            },
            {
                "id": 96,
                "title": "Vende y Gana Perfumes (Cód. 200)",
                "supplier_id": 9,
                "tags": [{"id": 126, "name": "C11"}, {"id": 80, "name": "Pág 10"}, {"id": 82, "name": "Pág 12"}],
                "tag_ids": [126, 80, 82],
                "observations": "Cód. 200 gratis",
                "description": "Perfume gratis",
            },
        ],
    }
    mock_odoo_client._client.get.return_value = mock_resp

    # Consulta filtrando por category="vende_y_gana"
    res = await odoo_get_active_incentives(
        supplier_id=9,
        campaign="C11",
        metadata_filters={"category": "vende_y_gana"},
        client=mock_odoo_client,
    )

    assert len(res) == 1
    assert res[0]["id"] == 96
    assert res[0]["metadata"]["category"] == "vende_y_gana"
    assert res[0]["metadata"]["campaign"] == "C11"
    mock_odoo_client._client.get.assert_awaited_once_with(
        "/api/v1/sales_incentives",
        params={"limit": 50, "offset": 0, "supplier_id": 9, "only_active": "true", "tag": "C11"},
    )

