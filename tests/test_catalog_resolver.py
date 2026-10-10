"""
tests/test_catalog_resolver.py

Pruebas unitarias y de integración para VendorCatalogResolver.
Valida la detección de la última campaña vigente por proveedor,
el cálculo de variaciones de precio inter-catálogo y el comportamiento de la caché en memoria.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.agent_service.core.stores.product.catalog_resolver import (
    VendorCatalogResolver,
    get_vendor_catalog_resolver,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    pool.closed = False

    conn = MagicMock()
    cur = AsyncMock()

    cur_cm = MagicMock()
    cur_cm.__aenter__ = AsyncMock(return_value=cur)
    cur_cm.__aexit__ = AsyncMock(return_value=None)
    conn.cursor.return_value = cur_cm

    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=conn)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pool.connection.return_value = conn_cm

    pool._mock_cur = cur
    return pool


@pytest.mark.asyncio
async def test_get_latest_campaigns_per_vendor_and_caching(mock_pool):
    """Verifica que la resolución de campañas consulta BD la primera vez y cachea el resultado."""
    mock_cur = mock_pool._mock_cur
    now = datetime.now()
    mock_cur.fetchall.return_value = [
        (9, "Unique S.A.", "C11", 259, now),
        (17, "CETCO S.A.", "C-15", 97, now),
    ]

    resolver = VendorCatalogResolver(pool=mock_pool, ttl_seconds=60)
    
    # 1era llamada: consulta a la BD
    campaigns = await resolver.get_latest_campaigns_per_vendor()
    assert 9 in campaigns
    assert campaigns[9]["latest_edicion"] == "C11"
    assert campaigns[9]["product_count"] == 259
    assert 17 in campaigns
    assert campaigns[17]["latest_edicion"] == "C-15"
    assert mock_cur.execute.call_count == 1

    # 2da llamada: responde desde caché en memoria sin volver a ejecutar SQL
    cached_campaigns = await resolver.get_latest_campaigns_per_vendor()
    assert cached_campaigns == campaigns
    assert mock_cur.execute.call_count == 1

    # Con force_refresh=True: vuelve a consultar BD
    await resolver.get_latest_campaigns_per_vendor(force_refresh=True)
    assert mock_cur.execute.call_count == 2


@pytest.mark.asyncio
async def test_get_active_editions_for_user_and_marca(mock_pool):
    """Verifica el filtrado de campañas activas por marca, vendor_id y fallback."""
    mock_cur = mock_pool._mock_cur
    now = datetime.now()
    mock_cur.fetchall.return_value = [
        (9, "Unique S.A. (Yanbal)", "C11", 259, now),
        (17, "CETCO S.A. (Ésika)", "C-15", 97, now),
    ]

    resolver = VendorCatalogResolver(pool=mock_pool, ttl_seconds=60)

    # Por vendor_id
    editions_yanbal = await resolver.get_active_editions_for_user(vendor_id=9)
    assert editions_yanbal == ["C11"]

    # Por marca (substring match)
    editions_esika = await resolver.get_active_editions_for_user(marca="Ésika")
    assert editions_esika == ["C-15"]

    # Fallback general: todas las últimas campañas
    all_editions = await resolver.get_active_editions_for_user()
    assert "C11" in all_editions
    assert "C-15" in all_editions


@pytest.mark.asyncio
async def test_get_price_variation_history(mock_pool):
    """Verifica el cálculo de variaciones de precio inter-catálogo y detección de descuentos."""
    mock_cur = mock_pool._mock_cur
    # Simular que en la campaña previa (C10) el producto costaba 110.0
    mock_cur.fetchone.return_value = ("C10", 110.0, datetime.now())

    resolver = VendorCatalogResolver(pool=mock_pool, ttl_seconds=60)

    # Caso: En la campaña actual C11 cuesta 82.0 (descuento de 28 soles)
    variation = await resolver.get_price_variation_history(
        product_name="Ohm Black Parfum",
        current_price=82.0,
        current_edition="C11",
    )

    assert variation is not None
    assert variation["previous_edition"] == "C10"
    assert variation["previous_price"] == 110.0
    assert variation["current_price"] == 82.0
    assert variation["difference"] == -28.0
    assert variation["is_discount"] is True
    assert variation["percentage_change"] == -25.5

    # Verificación de caché: no ejecuta SQL de nuevo para el mismo producto y precio
    call_count_before = mock_cur.execute.call_count
    cached_var = await resolver.get_price_variation_history(
        product_name="Ohm Black Parfum",
        current_price=82.0,
        current_edition="C11",
    )
    assert cached_var == variation
    assert mock_cur.execute.call_count == call_count_before


def test_clear_cache():
    """Verifica que el método clear_cache vacíe las memorias caché."""
    resolver = VendorCatalogResolver()
    resolver._cache["test_key"] = "test_val"
    resolver._variations_cache["var_key"] = {"diff": 10}

    resolver.clear_cache()
    assert len(resolver._cache) == 0
    assert len(resolver._variations_cache) == 0


def test_singleton():
    """Verifica que get_vendor_catalog_resolver retorne la misma instancia."""
    r1 = get_vendor_catalog_resolver()
    r2 = get_vendor_catalog_resolver()
    assert r1 is r2
