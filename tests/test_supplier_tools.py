import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent_service.core.stores.product.schemas import (
    SupplierInfo,
    SupplierSearchOutput,
)
from src.agent_service.tools.product_tools import (
    create_search_suppliers_tool,
    create_validate_product_ownership_tool,
)


@pytest.mark.asyncio
async def test_search_suppliers_tool():
    """Verifica la ejecución de la herramienta search_suppliers con cliente Odoo mockeado."""
    mock_odoo = AsyncMock()
    mock_odoo.search_suppliers.return_value = SupplierSearchOutput(
        suppliers=[
            SupplierInfo(id=9, name="Unique S.A.", display_name="Unique S.A.", vat="20100102413"),
        ],
        query="Unique",
    )

    tool = create_search_suppliers_tool(odoo_client=mock_odoo)
    result = await tool.ainvoke({"query": "Unique", "limit": 5})

    assert "suppliers" in result
    assert len(result["suppliers"]) == 1
    assert result["suppliers"][0]["id"] == 9
    assert result["suppliers"][0]["name"] == "Unique S.A."
    mock_odoo.search_suppliers.assert_awaited_once_with(query="Unique", limit=5)


@pytest.mark.asyncio
async def test_validate_product_ownership_tool():
    """Verifica la validación de ownership con mock del pool de base de datos."""
    mock_pool = MagicMock()
    mock_pool.closed = False
    
    # Context manager conn y cur
    mock_conn = MagicMock()
    mock_cur = AsyncMock()

    # Simulamos que para SKU 5110 devuelve proveedor 9
    # y para SKU 60117 devuelve múltiples proveedores (9 y 15)
    # y para SKU 99999 no devuelve nada
    mock_rows = [
        ("5110", 109, 9, "Unique S.A."),
        ("60117", 4, 9, "Unique S.A."),
        ("60117", 4, 15, "Siderperu S.A."),
    ]
    mock_cur.fetchall.return_value = mock_rows

    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=mock_cur)
    cursor_cm.__aexit__ = AsyncMock(return_value=None)
    mock_conn.cursor.return_value = cursor_cm

    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    mock_pool.connection.return_value = conn_cm

    tool = create_validate_product_ownership_tool(pool=mock_pool)
    result = await tool.ainvoke({"user_id": 5, "skus": ["5110", "60117", "99999"]})

    assert result["user_id"] == 5
    results = result["results"]

    # 1. SKU 5110: Proveedor único autorizado
    assert results["5110"]["is_owner"] is True
    assert results["5110"]["has_conflict"] is False
    assert len(results["5110"]["candidate_partners"]) == 1
    assert results["5110"]["candidate_partners"][0]["partner_id"] == "9"

    # 2. SKU 60117: Conflicto (múltiples proveedores autorizados)
    assert results["60117"]["is_owner"] is True
    assert results["60117"]["has_conflict"] is True
    assert len(results["60117"]["candidate_partners"]) == 2

    # 3. SKU 99999: No autorizado / no encontrado
    assert results["99999"]["is_owner"] is False
    assert results["99999"]["has_conflict"] is False
    assert len(results["99999"]["candidate_partners"]) == 0


@pytest.mark.integration
@pytest.mark.real_db
@pytest.mark.asyncio
async def test_real_supplier_tools_integration():
    """Verifica la conectividad real con Odoo JSON-RPC y PostgreSQL view_user_authorized_products."""
    import os
    from src.agent_service.tools.product_tools import search_suppliers, validate_product_ownership
    from src.agent_service.config.database import get_db_pool

    if not os.getenv("PG_HOST") or not os.getenv("ODOO_URL"):
        pytest.skip("Faltan variables en .env.dev para el test real de tools.")

    pool = get_db_pool()
    if pool.closed:
        await pool.open()

    # 1. Probar búsqueda de proveedores en Odoo API (Unique)
    supp_result = await search_suppliers.ainvoke({"query": "Unique", "limit": 5})
    assert "suppliers" in supp_result
    assert any(s["id"] == 9 for s in supp_result["suppliers"])

    # 2. Probar validación de pertenencia en PostgreSQL view_user_authorized_products
    own_result = await validate_product_ownership.ainvoke({
        "user_id": 5,
        "skus": ["5110", "114", "99999_NO_EXISTE"],
    })
    assert own_result["user_id"] == 5
    res = own_result["results"]
    assert res["5110"]["is_owner"] is True
    assert res["5110"]["candidate_partners"][0]["partner_id"] == "9"
    assert res["114"]["is_owner"] is True
    assert res["99999_NO_EXISTE"]["is_owner"] is False

