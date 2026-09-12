from typing import List, Optional, Dict, Any
from langchain_core.tools import tool

from src.agent_service.core.stores.product.schemas import (
    ProductAttribute,
    GetProductBySkusInput,
    GetProductBySkusOutput,
)
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.config.odoo import get_odoo_settings


def create_get_product_by_skus_tool(odoo_client: Optional[OdooClient] = None):
    """Fábrica de la herramienta get_product_by_skus permitiendo inyección de dependencias."""
    client = odoo_client

    @tool("get_product_by_skus", args_schema=GetProductBySkusInput)
    async def get_product_by_skus_tool(
        skus: List[str],
        fields: Optional[List[ProductAttribute]] = None,
    ) -> Dict[str, Any]:
        """Consulta en tiempo real la API de Odoo ERP para obtener precios de venta, moneda,
        unidad de medida y especificaciones comerciales de una lista de códigos SKU de productos.

        Parámetros:
            - skus: Lista de códigos SKU o referencias internas exactas (ej: ['1', '11', '114']).
            - fields: Lista opcional de atributos a extraer:
                * 'price': Precio de venta y divisa (ej. 39.5 PEN).
                * 'description': Descripción comercial de venta.
                * 'uom': Unidad de medida (ej. 'Units', 'Litros').
                * 'category': Categoría del producto.
                * 'barcode': Código de barras.
                * 'taxes': Texto informativo de impuestos aplicables.
              Si se omite o es None, se devuelven los campos comerciales estándar (nombre, precio, moneda).
        """
        nonlocal client
        if client is None:
            settings = get_odoo_settings()
            client = OdooClient(
                base_url=settings.url,
                db=settings.db,
                username=settings.username,
                password=settings.password,
            )

        result: GetProductBySkusOutput = await client.get_products_by_skus(skus=skus, fields=fields)
        return result.model_dump()

    return get_product_by_skus_tool


# Instancia por defecto lista para ser usada por agentes
get_product_by_skus = create_get_product_by_skus_tool()
