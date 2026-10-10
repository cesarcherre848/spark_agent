import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import Response

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.sales_tools import (
    odoo_list_sales_orders,
    odoo_list_current_sales_orders,
    odoo_create_quotation,
    odoo_update_quotation,
    odoo_view_quotation,
    odoo_confirm_order,
    odoo_unlock_order,
    odoo_update_order,
    odoo_lock_order,
    odoo_remove_sale_order,
)


@pytest.fixture
def mock_odoo_client():
    client = MagicMock(spec=OdooClient)
    client._uid = 2
    client._client = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_odoo_list_sales_orders_grouped(mock_odoo_client):
    """Prueba que el listado agrupe por estados comerciales: draft, sale, cancel."""
    fake_orders = [
        {
            "id": 101,
            "name": "SO001",
            "partner_id": [14, "Empresa Alfa"],
            "date_order": "2026-09-18 10:00:00",
            "amount_total": 450.0,
            "amount_untaxed": 381.36,
            "amount_tax": 68.64,
            "state": "draft",
            "user_id": [5, "Vendedor"],
            "order_line": [1, 2],
        },
        {
            "id": 102,
            "name": "SO002",
            "partner_id": [14, "Empresa Alfa"],
            "date_order": "2026-09-17 09:00:00",
            "amount_total": 1200.0,
            "amount_untaxed": 1016.95,
            "amount_tax": 183.05,
            "state": "sale",
            "user_id": [5, "Vendedor"],
            "order_line": [3],
        },
        {
            "id": 103,
            "name": "SO003",
            "partner_id": [25, "Distribuidora Beta"],
            "date_order": "2026-09-15 15:00:00",
            "amount_total": 300.0,
            "amount_untaxed": 254.24,
            "amount_tax": 45.76,
            "state": "cancel",
            "user_id": [5, "Vendedor"],
            "order_line": [4],
        },
    ]

    mock_resp = MagicMock(spec=Response)
    mock_resp.json.return_value = {"result": fake_orders}
    mock_odoo_client._client.post.return_value = mock_resp

    grouped = await odoo_list_sales_orders(user_id=5, client=mock_odoo_client)

    assert "draft" in grouped
    assert "sale" in grouped
    assert "cancel" in grouped
    assert len(grouped["draft"]) == 1
    assert grouped["draft"][0]["name"] == "SO001"
    assert grouped["draft"][0]["customer_name"] == "Empresa Alfa"
    assert len(grouped["sale"]) == 1
    assert grouped["sale"][0]["name"] == "SO002"
    assert len(grouped["cancel"]) == 1
    assert grouped["cancel"][0]["name"] == "SO003"


@pytest.mark.asyncio
async def test_odoo_list_current_sales_orders(mock_odoo_client):
    """Prueba la consulta de cotizaciones activas de un cliente."""
    fake_quotes = [
        {
            "id": 101,
            "name": "SO001",
            "partner_id": [14, "Empresa Alfa"],
            "date_order": "2026-09-18 10:00:00",
            "amount_total": 450.0,
            "state": "draft",
            "order_line": [1, 2],
        }
    ]
    mock_resp = MagicMock(spec=Response)
    mock_resp.json.return_value = {"result": fake_quotes}
    mock_odoo_client._client.post.return_value = mock_resp

    res = await odoo_list_current_sales_orders(user_id=5, partner_id=14, client=mock_odoo_client)
    assert len(res) == 1
    assert res[0]["name"] == "SO001"
    assert res[0]["amount_total"] == 450.0


@pytest.mark.asyncio
async def test_odoo_create_quotation(mock_odoo_client):
    """Prueba la creación de una cotización en borrador."""
    # 1. create call returns order ID 105
    resp_create = MagicMock(spec=Response)
    resp_create.json.return_value = {"result": 105}

    # 2. view_quotation search_read call returns order 105
    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {
        "result": [
            {
                "id": 105,
                "name": "SO005",
                "partner_id": [14, "Empresa Alfa"],
                "date_order": "2026-09-18 11:00:00",
                "state": "draft",
                "amount_untaxed": 100.0,
                "amount_tax": 18.0,
                "amount_total": 118.0,
                "order_line": [10],
            }
        ]
    }

    # 3. view_quotation read lines call
    resp_lines = MagicMock(spec=Response)
    resp_lines.json.return_value = {
        "result": [
            {
                "id": 10,
                "product_id": [1, "SKU 1 Producto Genérico"],
                "product_uom_qty": 2.0,
                "price_unit": 50.0,
                "price_subtotal": 100.0,
            }
        ]
    }

    mock_odoo_client._client.post.side_effect = [resp_create, resp_search, resp_lines]

    items = [{"product_id": 1, "qty": 2.0, "price_unit": 50.0}]
    res = await odoo_create_quotation(user_id=5, partner_id=14, items=items, client=mock_odoo_client)

    assert res["success"] is True
    assert res["order_id"] == 105
    assert res["name"] == "SO005"
    assert res["amount_total"] == 118.0
    assert len(res["lines"]) == 1


@pytest.mark.asyncio
async def test_odoo_confirm_order(mock_odoo_client):
    """Prueba la confirmación de cotización a pedido."""
    resp_confirm = MagicMock(spec=Response)
    resp_confirm.json.return_value = {"result": True}

    resp_view = MagicMock(spec=Response)
    resp_view.json.return_value = {
        "result": [
            {
                "id": 105,
                "name": "SO005",
                "partner_id": [14, "Empresa Alfa"],
                "state": "sale",
                "amount_total": 118.0,
                "order_line": [],
            }
        ]
    }

    mock_odoo_client._client.post.side_effect = [resp_confirm, resp_view]

    res = await odoo_confirm_order(order_id=105, client=mock_odoo_client)
    assert res["success"] is True
    assert res["action"] == "confirmed"
    assert res["state"] == "sale"


@pytest.mark.asyncio
async def test_odoo_unlock_and_lock_order(mock_odoo_client):
    """Prueba el desbloqueo y re-bloqueo de orden."""
    resp_unlock = MagicMock(spec=Response)
    resp_unlock.json.return_value = {"result": True}
    resp_lock = MagicMock(spec=Response)
    resp_lock.json.return_value = {"result": True}

    mock_odoo_client._client.post.side_effect = [resp_unlock, resp_lock]

    res_unlock = await odoo_unlock_order(order_id=105, client=mock_odoo_client)
    assert res_unlock["success"] is True
    assert res_unlock["action"] == "unlocked"

    res_lock = await odoo_lock_order(order_id=105, client=mock_odoo_client)
    assert res_lock["success"] is True
    assert res_lock["action"] == "locked"


@pytest.mark.asyncio
async def test_odoo_remove_sale_order(mock_odoo_client):
    """Prueba la cancelación segura de orden."""
    resp_cancel = MagicMock(spec=Response)
    resp_cancel.json.return_value = {"result": True}
    mock_odoo_client._client.post.return_value = resp_cancel

    res = await odoo_remove_sale_order(order_id=105, client=mock_odoo_client)
    assert res["success"] is True
    assert res["action"] == "cancelled"


@pytest.mark.asyncio
async def test_odoo_create_quotation_with_skus_resolution(mock_odoo_client):
    """Prueba que odoo_create_quotation resuelva SKUs usando ProductOdooDetail correctamente."""
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput, ProductOdooDetail

    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[
                ProductOdooDetail(
                    product_id=117,
                    sku="6189",
                    name="Sexy Glam",
                    price=179.0,
                ),
                ProductOdooDetail(
                    product_id=127,
                    sku="5104",
                    name="Delineador Tattoo",
                    price=61.0,
                ),
            ],
            not_found_skus=[],
        )
    )

    resp_create = MagicMock(spec=Response)
    resp_create.json.return_value = {"result": 201}

    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {
        "result": [
            {
                "id": 201,
                "name": "SO009",
                "partner_id": [12, "Adhara Banda"],
                "date_order": "2026-09-18 11:30:00",
                "state": "draft",
                "amount_untaxed": 541.0,
                "amount_tax": 0.0,
                "amount_total": 541.0,
                "order_line": [11, 12],
            }
        ]
    }

    resp_lines = MagicMock(spec=Response)
    resp_lines.json.return_value = {
        "result": [
            {"id": 11, "product_id": [117, "Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0, "price_subtotal": 358.0},
            {"id": 12, "product_id": [127, "Delineador Tattoo"], "product_uom_qty": 3.0, "price_unit": 61.0, "price_subtotal": 183.0},
        ]
    }

    mock_odoo_client._client.post.side_effect = [resp_create, resp_search, resp_lines]

    items = [{"sku": "6189", "qty": 2.0}, {"sku": "5104", "qty": 3.0}]
    res = await odoo_create_quotation(user_id=5, partner_id=12, items=items, client=mock_odoo_client)

    assert res["success"] is True
    assert res["order_id"] == 201
    assert res["name"] == "SO009"
    assert res["amount_total"] == 541.0

    # Verificar que call_kw create recibió los product_ids mapeados
    create_call = mock_odoo_client._client.post.call_args_list[0]
    payload = create_call.kwargs["json"]["params"]["args"][0]
    lines = payload["order_line"]
    assert len(lines) == 2
    assert lines[0][2]["product_id"] == 117
    assert lines[0][2]["product_uom_qty"] == 2.0
    assert lines[0][2]["price_unit"] == 179.0
    assert lines[1][2]["product_id"] == 127
    assert lines[1][2]["product_uom_qty"] == 3.0
    assert lines[1][2]["price_unit"] == 61.0


@pytest.mark.asyncio
async def test_odoo_create_quotation_skus_not_found(mock_odoo_client):
    """Prueba que odoo_create_quotation retorne un error limpio si los SKUs no existen en Odoo."""
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput

    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[],
            not_found_skus=["INEXISTENTE999"],
        )
    )

    items = [{"sku": "INEXISTENTE999", "qty": 1.0}]
    res = await odoo_create_quotation(user_id=5, partner_id=12, items=items, client=mock_odoo_client)

    assert res["success"] is False
    assert "INEXISTENTE999" in res["error"]
    assert "INEXISTENTE999" in res["not_found_skus"]
    # No debe llamar a post de create
    mock_odoo_client._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_odoo_update_quotation_with_skus_resolution(mock_odoo_client):
    """Prueba que odoo_update_quotation resuelva SKUs usando ProductOdooDetail correctamente al añadir producto nuevo."""
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput, ProductOdooDetail

    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[
                ProductOdooDetail(
                    product_id=117,
                    sku="6189",
                    name="Sexy Glam",
                    price=179.0,
                )
            ],
            not_found_skus=[],
        )
    )

    resp_read_order = MagicMock(spec=Response)
    resp_read_order.json.return_value = {"result": [{"id": 201, "name": "SO009", "order_line": []}]}

    resp_write = MagicMock(spec=Response)
    resp_write.json.return_value = {"result": True}

    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {
        "result": [
            {
                "id": 201,
                "name": "SO009",
                "partner_id": [12, "Adhara Banda"],
                "date_order": "2026-09-18 11:30:00",
                "state": "draft",
                "amount_untaxed": 179.0,
                "amount_tax": 0.0,
                "amount_total": 179.0,
                "order_line": [13],
            }
        ]
    }

    resp_lines = MagicMock(spec=Response)
    resp_lines.json.return_value = {
        "result": [
            {"id": 13, "product_id": [117, "Sexy Glam"], "product_uom_qty": 1.0, "price_unit": 179.0, "price_subtotal": 179.0},
        ]
    }

    mock_odoo_client._client.post.side_effect = [resp_read_order, resp_write, resp_search, resp_lines]

    items = [{"sku": "6189", "qty": 1.0}]
    res = await odoo_update_quotation(order_id=201, items=items, client=mock_odoo_client)

    assert res["success"] is True
    assert res["order_id"] == 201
    assert res["action"] == "updated"

    # Verificar call_kw write con product_id 117 (inserción (0, 0, vals))
    write_call = mock_odoo_client._client.post.call_args_list[1]
    payload = write_call.kwargs["json"]["params"]["args"][1]
    lines = payload["order_line"]
    assert len(lines) == 1
    assert lines[0][0] == 0
    assert lines[0][2]["product_id"] == 117
    assert lines[0][2]["product_uom_qty"] == 1.0
    assert lines[0][2]["price_unit"] == 179.0


@pytest.mark.asyncio
async def test_odoo_update_quotation_subtract_and_deduplicate(mock_odoo_client):
    """Prueba que odoo_update_quotation reste cantidades, actualice la línea principal y elimine duplicados con (2, id, 0)."""
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput, ProductOdooDetail

    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[
                ProductOdooDetail(product_id=117, sku="6189", name="Sexy Glam", price=179.0),
                ProductOdooDetail(product_id=118, sku="5104", name="Delineador Plumón", price=61.0),
            ],
            not_found_skus=[],
        )
    )

    # Orden con duplicados: 6189 en line 5 y 7 (2+2=4 unidades), 5104 en line 6 y 8 (3+3=6 unidades)
    resp_read_order = MagicMock(spec=Response)
    resp_read_order.json.return_value = {"result": [{"id": 3, "name": "S00003", "order_line": [5, 6, 7, 8]}]}

    resp_read_lines = MagicMock(spec=Response)
    resp_read_lines.json.return_value = {
        "result": [
            {"id": 5, "product_id": [117, "[6189] Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0},
            {"id": 6, "product_id": [118, "[5104] Delineador Plumón"], "product_uom_qty": 3.0, "price_unit": 61.0},
            {"id": 7, "product_id": [117, "[6189] Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0},
            {"id": 8, "product_id": [118, "[5104] Delineador Plumón"], "product_uom_qty": 3.0, "price_unit": 61.0},
        ]
    }

    resp_write = MagicMock(spec=Response)
    resp_write.json.return_value = {"result": True}

    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {
        "result": [
            {
                "id": 3,
                "name": "S00003",
                "partner_id": [12, "Adhara Banda"],
                "date_order": "2026-09-18",
                "state": "draft",
                "amount_untaxed": 602.0,
                "amount_tax": 90.30,
                "amount_total": 692.30,
                "order_line": [5, 6],
            }
        ]
    }

    resp_view_lines = MagicMock(spec=Response)
    resp_view_lines.json.return_value = {
        "result": [
            {"id": 5, "product_id": [117, "[6189] Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0, "price_subtotal": 358.0},
            {"id": 6, "product_id": [118, "[5104] Delineador Plumón"], "product_uom_qty": 4.0, "price_unit": 61.0, "price_subtotal": 244.0},
        ]
    }

    mock_odoo_client._client.post.side_effect = [
        resp_read_order,
        resp_read_lines,
        resp_write,
        resp_search,
        resp_view_lines,
    ]

    items = [
        {"sku": "6189", "qty": 2.0, "action": "subtract"},
        {"sku": "5104", "qty": 2.0, "action": "subtract"},
    ]
    res = await odoo_update_quotation(order_id=3, items=items, client=mock_odoo_client)

    assert res["success"] is True
    assert res["order_id"] == 3

    # Verificar comandos ORM enviados a write
    write_call = mock_odoo_client._client.post.call_args_list[2]
    order_line_cmds = write_call.kwargs["json"]["params"]["args"][1]["order_line"]

    # Debe actualizar line 5 a 2.0 unidades y desvincular line 7
    # Debe actualizar line 6 a 4.0 unidades y desvincular line 8
    assert (1, 5, {"product_uom_qty": 2.0}) in order_line_cmds
    assert (2, 7, 0) in order_line_cmds
    assert (1, 6, {"product_uom_qty": 4.0}) in order_line_cmds
    assert (2, 8, 0) in order_line_cmds


@pytest.mark.asyncio
async def test_odoo_update_quotation_remove_item(mock_odoo_client):
    """Prueba que action='remove' elimine todas las líneas del producto vía (2, line_id, 0)."""
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput, ProductOdooDetail

    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[ProductOdooDetail(product_id=117, sku="6189", name="Sexy Glam", price=179.0)],
            not_found_skus=[],
        )
    )

    resp_read_order = MagicMock(spec=Response)
    resp_read_order.json.return_value = {"result": [{"id": 3, "name": "S00003", "order_line": [5, 7]}]}

    resp_read_lines = MagicMock(spec=Response)
    resp_read_lines.json.return_value = {
        "result": [
            {"id": 5, "product_id": [117, "[6189] Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0},
            {"id": 7, "product_id": [117, "[6189] Sexy Glam"], "product_uom_qty": 2.0, "price_unit": 179.0},
        ]
    }

    resp_write = MagicMock(spec=Response)
    resp_write.json.return_value = {"result": True}

    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {"result": [{"id": 3, "name": "S00003", "order_line": []}]}

    resp_view_lines = MagicMock(spec=Response)
    resp_view_lines.json.return_value = {"result": []}

    mock_odoo_client._client.post.side_effect = [
        resp_read_order,
        resp_read_lines,
        resp_write,
        resp_search,
        resp_view_lines,
    ]

    items = [{"sku": "6189", "action": "remove"}]
    res = await odoo_update_quotation(order_id=3, items=items, client=mock_odoo_client)

    assert res["success"] is True
    write_call = mock_odoo_client._client.post.call_args_list[2]
    order_line_cmds = write_call.kwargs["json"]["params"]["args"][1]["order_line"]
    assert (2, 5, 0) in order_line_cmds
    assert (2, 7, 0) in order_line_cmds


def test_normalize_order_name_and_extract_code():
    """Valida la normalización de códigos de órdenes y cotizaciones."""
    from src.agent_service.tools.sales_tools import normalize_order_name, extract_order_code

    # Números directos
    assert normalize_order_name("03") == "S00003"
    assert normalize_order_name("3") == "S00003"
    assert normalize_order_name("15") == "S00015"
    assert normalize_order_name("S00003") == "S00003"
    assert normalize_order_name("SO003") == "S00003"

    # Frases
    assert extract_order_code("muestrame la cotizacion 03 de Adhara") == "S00003"
    assert extract_order_code("ver cotización 3") == "S00003"
    assert extract_order_code("orden S00003") == "S00003"
    assert extract_order_code("pedido #15") == "S00015"
    assert extract_order_code("quita 2 unidades del Delineador") is None


@pytest.mark.asyncio
async def test_odoo_atomic_quotation_tools(mock_odoo_client):
    """Prueba las herramientas atómicas odoo_remove_quotation_items y odoo_add_quotation_items."""
    from src.agent_service.tools.sales_tools import odoo_remove_quotation_items, odoo_add_quotation_items
    from httpx import Response

    resp_read_order = MagicMock(spec=Response)
    resp_read_order.json.return_value = {"result": [{"id": 3, "name": "S00003", "order_line": [5]}]}

    resp_read_lines = MagicMock(spec=Response)
    resp_read_lines.json.return_value = {
        "result": [{"id": 5, "name": "[5104] Delineador Plumón Tattoo", "product_id": [118, "Delineador"], "product_uom_qty": 4.0, "price_unit": 61.0}]
    }

    resp_write = MagicMock(spec=Response)
    resp_write.json.return_value = {"result": True}

    resp_search = MagicMock(spec=Response)
    resp_search.json.return_value = {"result": [{"id": 3, "name": "S00003", "order_line": [5]}]}

    resp_view_lines = MagicMock(spec=Response)
    resp_view_lines.json.return_value = {
        "result": [{"id": 5, "name": "[5104] Delineador Plumón Tattoo", "product_id": [118, "Delineador"], "product_uom_qty": 2.0, "price_unit": 61.0}]
    }

    mock_odoo_client._client.post.side_effect = [
        resp_read_order,
        resp_read_lines,
        resp_write,
        resp_search,
        resp_view_lines,
    ]

    res = await odoo_remove_quotation_items(
        order_id=3,
        product="Delineador",
        qty=2.0,
        client=mock_odoo_client,
    )
    assert res["success"] is True
    write_call = mock_odoo_client._client.post.call_args_list[2]
    order_line_cmds = write_call.kwargs["json"]["params"]["args"][1]["order_line"]
    assert (1, 5, {"product_uom_qty": 2.0}) in order_line_cmds


@pytest.mark.asyncio
async def test_odoo_create_quotation_auto_resolves_customer_name(mock_odoo_client):
    """Valida que odoo_create_quotation resuelva automáticamente el cliente por customer_name
    en lugar de requerir o hacer fallback silencioso a partner_id=1.
    """
    resp_create = MagicMock(spec=Response)
    resp_create.json.return_value = {"result": 99}

    resp_view = MagicMock(spec=Response)
    resp_view.json.return_value = {
        "result": [{
            "id": 99,
            "name": "S00010",
            "amount_total": 200.0,
            "amount_untaxed": 200.0,
            "amount_tax": 0.0,
            "order_line": [],
            "partner_id": [20, "Carmen Saavedra"],
        }]
    }
    mock_odoo_client._client.post.side_effect = [resp_create, resp_view]

    with patch("src.agent_service.tools.contact_tools.odoo_list_current_customers", new_callable=AsyncMock) as mock_list_cust:
        mock_list_cust.return_value = [{"id": 20, "name": "Carmen Saavedra"}]

        res = await odoo_create_quotation(
            user_id=7,
            customer_name="Carmen Saavedra",
            items=[{"product_id": 11, "qty": 1.0, "price_unit": 200.0}],
            client=mock_odoo_client,
        )

        assert res["success"] is True
        assert res["name"] == "S00010"
        assert res["partner_id"] == 20
        create_call = mock_odoo_client._client.post.call_args_list[0]
        create_vals = create_call.kwargs["json"]["params"]["args"][0]
        assert create_vals["partner_id"] == 20
        assert create_vals["user_id"] == 7


@pytest.mark.asyncio
async def test_odoo_create_quotation_fails_without_partner_or_customer(mock_odoo_client):
    """Valida que odoo_create_quotation falle limpiamente sin asignar a partner 1 si no hay cliente."""
    res = await odoo_create_quotation(
        user_id=7,
        partner_id=None,
        customer_name=None,
        items=[{"product_id": 11, "qty": 1.0}],
        client=mock_odoo_client,
    )
    assert res["success"] is False
    assert "No se identificó" in res["error"]


def test_normalize_item_input_polymorphic():
    """Valida la normalización polimórfica de ítems asegurando que 'código' siempre sea SKU."""
    from src.agent_service.tools.sales_tools import normalize_item_input

    # 1. product_sku y cantidad
    it1 = normalize_item_input({"product_sku": "20013876", "cantidad": 3, "action": "set"})
    assert it1["sku"] == "20013876"
    assert it1["qty"] == 3.0
    assert it1["action"] == "set"

    # 2. code y unidades
    it2 = normalize_item_input({"code": "47000", "unidades": 2})
    assert it2["sku"] == "47000"
    assert it2["qty"] == 2.0
    assert it2["action"] == "add"

    # 3. codigo y quantity
    it3 = normalize_item_input({"codigo": "20010996", "quantity": 1})
    assert it3["sku"] == "20010996"
    assert it3["qty"] == 1.0

    # 4. Texto descriptivo en product_id (caso de falla en sesión de usuario 7)
    it4 = normalize_item_input({"product_id": "BB Cream antiedad", "qty": 2})
    assert it4["sku"] == "BB Cream antiedad"
    assert it4["product_id"] is None
    assert it4["qty"] == 2.0

    # 5. ID numérico válido
    it5 = normalize_item_input({"product_id": 42, "sku": "20013876", "qty": 1})
    assert it5["sku"] == "20013876"
    assert it5["product_id"] == 42


@pytest.mark.asyncio
async def test_odoo_create_quotation_partitions_by_supplier(mock_odoo_client):
    """Valida la regla de negocio: un pedido no puede tener 2 proveedores.
    Si se solicitan ítems de Yanbal y Belcorp, se particionan en órdenes separadas por proveedor.
    """
    from src.agent_service.core.stores.product.schemas import GetProductBySkusOutput, ProductOdooDetail

    # Mock productos resueltos
    mock_odoo_client.get_products_by_skus = AsyncMock(
        return_value=GetProductBySkusOutput(
            products=[
                ProductOdooDetail(sku="20013876", name="BB Cream Yanbal", price=50.0, product_id=101),
                ProductOdooDetail(sku="47000", name="Labial Ésika", price=30.0, product_id=202),
            ],
            not_found_skus=[],
        )
    )

    # 2 llamadas a create (una por cada proveedor)
    resp_create_1 = MagicMock(spec=Response)
    resp_create_1.json.return_value = {"result": 301}
    resp_create_2 = MagicMock(spec=Response)
    resp_create_2.json.return_value = {"result": 302}

    # 2 llamadas a view_quotation (search_read + read lines)
    resp_search_1 = MagicMock(spec=Response)
    resp_search_1.json.return_value = {
        "result": [{
            "id": 301,
            "name": "S00021",
            "amount_total": 50.0,
            "amount_untaxed": 50.0,
            "amount_tax": 0.0,
            "order_line": [1],
            "partner_id": [11, "Gemma"],
        }]
    }
    resp_lines_1 = MagicMock(spec=Response)
    resp_lines_1.json.return_value = {"result": [{"id": 1, "product_id": [101, "[20013876] BB Cream"], "product_uom_qty": 1.0, "price_unit": 50.0, "price_subtotal": 50.0}]}

    resp_search_2 = MagicMock(spec=Response)
    resp_search_2.json.return_value = {
        "result": [{
            "id": 302,
            "name": "S00022",
            "amount_total": 30.0,
            "amount_untaxed": 30.0,
            "amount_tax": 0.0,
            "order_line": [2],
            "partner_id": [11, "Gemma"],
        }]
    }
    resp_lines_2 = MagicMock(spec=Response)
    resp_lines_2.json.return_value = {"result": [{"id": 2, "product_id": [202, "[47000] Labial Ésika"], "product_uom_qty": 1.0, "price_unit": 30.0, "price_subtotal": 30.0}]}

    mock_odoo_client._client.post.side_effect = [
        resp_create_1, resp_search_1, resp_lines_1,
        resp_create_2, resp_search_2, resp_lines_2,
    ]

    with patch("src.agent_service.tools.incentive_tools.get_sku_supplier_map", new_callable=AsyncMock) as mock_sup_map:
        mock_sup_map.return_value = {"20013876": 9, "47000": 17}

        items = [
            {"product_sku": "20013876", "qty": 1.0},
            {"code": "47000", "qty": 1.0},
        ]
        res = await odoo_create_quotation(user_id=7, partner_id=11, items=items, client=mock_odoo_client)

        assert res["success"] is True
        assert res["partitioned_by_supplier"] is True
        assert len(res["orders"]) == 2
        suppliers = {o["supplier_id"] for o in res["orders"]}
        assert suppliers == {9, 17}
        assert res["amount_total"] == 80.0


@pytest.mark.asyncio
async def test_odoo_update_quotation_rejects_conflicting_supplier(mock_odoo_client):
    """Valida que odoo_update_quotation rechace agregar productos de un proveedor distinto
    a una cotización existente que ya contiene productos de otro proveedor.
    """
    # 1. Leer orden existente (S00009)
    resp_order = MagicMock(spec=Response)
    resp_order.json.return_value = {
        "result": [{
            "id": 99,
            "name": "S00009",
            "order_line": [10],
        }]
    }

    # 2. Leer líneas existentes (contiene producto de Yanbal: SKU 20013876)
    resp_lines = MagicMock(spec=Response)
    resp_lines.json.return_value = {
        "result": [{
            "id": 10,
            "name": "[20013876] BB Cream Yanbal",
            "product_id": [101, "[20013876] BB Cream Yanbal"],
            "product_uom_qty": 2.0,
            "price_unit": 50.0,
        }]
    }

    mock_odoo_client._client.post.side_effect = [resp_order, resp_lines]

    with patch("src.agent_service.tools.incentive_tools.get_sku_supplier_map", new_callable=AsyncMock) as mock_sup_map:
        mock_sup_map.return_value = {"20013876": 9, "47000": 17}

        # Intentar agregar producto de Belcorp (SKU 47000) a la orden de Yanbal
        items = [{"code": "47000", "qty": 1.0, "action": "add"}]
        res = await odoo_update_quotation(order_id=99, items=items, client=mock_odoo_client)

        assert res["success"] is False
        assert "un solo proveedor" in res["error"]
        assert res["conflicting_supplier"] == "Belcorp (Ésika / Cyzone / L'Bel)"





