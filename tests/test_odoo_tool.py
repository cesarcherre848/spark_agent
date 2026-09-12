import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from dotenv import load_dotenv

from src.agent_service.core.stores.product.schemas import (
    GetProductBySkusOutput,
    ProductOdooDetail,
)
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.product_tools import create_get_product_by_skus_tool

load_dotenv(".env.dev")


# ==============================================================================
# PRUEBAS UNITARIAS CON MOCKS (Sin requerir conexión a internet)
# ==============================================================================

@pytest.mark.asyncio
async def test_odoo_client_empty_skus_returns_immediately():
    mock_http = MagicMock()
    client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    result = await client.get_products_by_skus(skus=[])
    assert result.products == []
    assert result.not_found_skus == []
    mock_http.post.assert_not_called()


@pytest.mark.asyncio
async def test_odoo_client_fetches_and_maps_fields_correctly():
    mock_http = AsyncMock()

    # 1. Mock de autenticación
    auth_response = MagicMock()
    auth_response.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 42}}
    auth_response.raise_for_status = MagicMock()

    # 2. Mock de search_read en product.product
    call_response = MagicMock()
    call_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": [
            {
                "id": 100,
                "default_code": "SKU-001",
                "name": "Serum Facial Antiedad",
                "list_price": 55.0,
                "currency_id": [1, "PEN"],
                "description_sale": "Serum concentrado con ácido hialurónico.",
                "categ_id": [5, "Cuidado Facial"],
                "uom_name": "Units",
                "barcode": "775001002003",
            }
        ],
    }

    mock_http.post.side_effect = [auth_response, call_response]

    client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    # Consultar solicitando solo 'price', 'description' y 'category'
    result: GetProductBySkusOutput = await client.get_products_by_skus(
        skus=["SKU-001", "SKU-NOT-FOUND"],
        fields=["price", "description", "category"],
    )

    assert len(result.products) == 1
    prod = result.products[0]
    assert prod.sku == "SKU-001"
    assert prod.name == "Serum Facial Antiedad"
    assert prod.price == 55.0
    assert prod.currency == "PEN"
    assert prod.sales_description == "Serum concentrado con ácido hialurónico."
    assert prod.category == "Cuidado Facial"

    # Campos no solicitados deben ser None
    assert prod.barcode is None
    assert prod.taxes is None

    # El SKU no encontrado debe estar listado explícitamente
    assert result.not_found_skus == ["SKU-NOT-FOUND"]


@pytest.mark.asyncio
async def test_product_tool_langchain_interface():
    mock_odoo_client = AsyncMock(spec=OdooClient)
    mock_odoo_client.get_products_by_skus.return_value = GetProductBySkusOutput(
        products=[
            ProductOdooDetail(
                product_id=1,
                sku="1",
                name="Loción de Seda",
                price=210.0,
                currency="PEN",
            )
        ],
        not_found_skus=["999"],
    )

    tool = create_get_product_by_skus_tool(odoo_client=mock_odoo_client)

    # Ejecutar la tool de LangChain
    output = await tool.ainvoke({"skus": ["1", "999"], "fields": ["price"]})

    assert "products" in output
    assert "not_found_skus" in output
    assert len(output["products"]) == 1
    assert output["products"][0]["sku"] == "1"
    assert output["products"][0]["price"] == 210.0
    assert output["not_found_skus"] == ["999"]

    mock_odoo_client.get_products_by_skus.assert_awaited_once_with(
        skus=["1", "999"], fields=["price"]
    )


# ==============================================================================
# PRUEBAS DE INTEGRACIÓN REAL CONTRA EL SERVIDOR ODOO DEV
# ==============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_odoo_client_real_integration():
    """Prueba real contra el servidor Odoo dev en http://134.199.209.31:8069."""
    url = os.getenv("ODOO_URL")
    db = os.getenv("ODOO_DB")
    user = os.getenv("ODOO_USERNAME")
    pwd = os.getenv("ODOO_PASSWORD")

    if not all([url, db, user, pwd]):
        pytest.skip("Faltan variables ODOO_URL/ODOO_DB/ODOO_USERNAME/ODOO_PASSWORD para el test real.")

    client = OdooClient(
        base_url=url,
        db=db,
        username=user,
        password=pwd,
    )

    try:
        # Consultar productos reales comprobados en la BD y uno inexistente
        result = await client.get_products_by_skus(
            skus=["1", "11", "NONEXISTENT_99999"],
            fields=["price", "description", "uom"],
        )

        assert len(result.products) == 2, "Se esperaba encontrar exactamente los SKUs '1' y '11'"
        assert result.not_found_skus == ["NONEXISTENT_99999"]

        # Validar primer producto (Loción de Seda)
        prod1 = next(p for p in result.products if p.sku == "1")
        assert prod1.name == "Loción de Seda"
        assert prod1.price == 210.0
        assert prod1.currency == "PEN"
        assert prod1.uom == "Units"
        assert prod1.sales_description is not None
        assert "efecto perlado" in prod1.sales_description

        # Validar segundo producto (Desmaquillador Doble Fase)
        prod2 = next(p for p in result.products if p.sku == "11")
        assert prod2.name == "Desmaquillador Doble Fase"
        assert prod2.price == 39.5
        assert prod2.currency == "PEN"

    finally:
        await client.aclose()
