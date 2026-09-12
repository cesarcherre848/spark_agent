from typing import List, Optional, Dict, Any
from collections import defaultdict
from langchain_core.tools import tool
from psycopg_pool import AsyncConnectionPool

from src.agent_service.core.stores.product.schemas import (
    ProductAttribute,
    GetProductBySkusInput,
    GetProductBySkusOutput,
    SupplierSearchInput,
    SupplierSearchOutput,
    ValidateProductOwnershipInput,
    ValidateProductOwnershipOutput,
    SkuOwnershipResult,
    ProductPartnerInfo,
)
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.config.odoo import get_odoo_settings
from src.agent_service.config.database import get_db_pool


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
            - fields: Lista opcional de atributos a extraer ('price', 'description', 'uom', 'category', 'barcode', 'taxes').
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


def create_search_suppliers_tool(odoo_client: Optional[OdooClient] = None):
    """Fábrica de la herramienta search_suppliers conectada a Odoo API (res.partner)."""
    client = odoo_client

    @tool("search_suppliers", args_schema=SupplierSearchInput)
    async def search_suppliers_tool(
        query: str,
        limit: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """Busca proveedores en Odoo ERP por nombre comercial, razón social o número de RUC/VAT.

        Parámetros:
            - query: Nombre o RUC del proveedor a buscar (ej: 'Unique', 'Siderperu', '20100102413').
            - limit: Cantidad máxima de coincidencias a retornar (por defecto 5).
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

        result: SupplierSearchOutput = await client.search_suppliers(query=query, limit=limit or 5)
        return result.model_dump()

    return search_suppliers_tool


def create_validate_product_ownership_tool(pool: Optional[AsyncConnectionPool] = None):
    """Fábrica de la herramienta validate_product_ownership contra view_user_authorized_products."""
    db_pool = pool

    @tool("validate_product_ownership", args_schema=ValidateProductOwnershipInput)
    async def validate_product_ownership_tool(
        user_id: int,
        skus: List[str],
    ) -> Dict[str, Any]:
        """Valida si una lista de SKUs pertenecen al catálogo comercial autorizado para un usuario
        en la base de datos relacional de Odoo (PostgreSQL: view_user_authorized_products).

        Parámetros:
            - user_id: ID numérico del usuario (res_users) asignado a los equipos comerciales CRM.
            - skus: Lista de códigos SKU a verificar (ej: ['5110', '114']).

        Retorna los SKUs autorizados, sus proveedores oficiales y si existen conflictos (múltiples proveedores).
        """
        nonlocal db_pool
        if db_pool is None:
            db_pool = get_db_pool()

        if db_pool.closed:
            await db_pool.open()

        clean_skus = [str(s).strip() for s in (skus or []) if str(s).strip()]
        if not clean_skus:
            output = ValidateProductOwnershipOutput(user_id=user_id, results={})
            return output.model_dump()

        query_sql = """
            SELECT 
                vuap.sku,
                vuap.product_id,
                vuap.vendor_id,
                rp.name AS vendor_name
            FROM view_user_authorized_products vuap
            JOIN res_partner rp ON rp.id = vuap.vendor_id
            WHERE vuap.sku = ANY(%(skus)s)
              AND vuap.user_id = %(user_id)s;
        """

        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query_sql, {"skus": clean_skus, "user_id": int(user_id)})
                rows = await cur.fetchall()

        # Agrupar filas por SKU: (sku, product_id, vendor_id, vendor_name)
        sku_to_rows = defaultdict(list)
        for r in rows:
            sku_to_rows[str(r[0])].append(r)

        results: Dict[str, SkuOwnershipResult] = {}
        for s in clean_skus:
            matched_rows = sku_to_rows.get(s, [])
            if not matched_rows:
                results[s] = SkuOwnershipResult(
                    sku=s,
                    product_id=None,
                    is_owner=False,
                    candidate_partners=[],
                    has_conflict=False,
                    conflict_reason=f"El SKU '{s}' no está autorizado o no pertenece al catálogo del usuario {user_id}.",
                )
            else:
                p_id = matched_rows[0][1]
                partners = [
                    ProductPartnerInfo(
                        partner_id=str(r[2]),
                        vendor_id=int(r[2]),
                        vendor_name=str(r[3]),
                    )
                    for r in matched_rows
                ]
                has_conf = len(partners) > 1
                results[s] = SkuOwnershipResult(
                    sku=s,
                    product_id=p_id,
                    is_owner=True,
                    candidate_partners=partners,
                    has_conflict=has_conf,
                    conflict_reason=(
                        f"El SKU '{s}' cuenta con {len(partners)} proveedores posibles."
                        if has_conf else None
                    ),
                )

        output = ValidateProductOwnershipOutput(user_id=user_id, results=results)
        return output.model_dump()

    return validate_product_ownership_tool


# Instancias por defecto listas para ser usadas por agentes
get_product_by_skus = create_get_product_by_skus_tool()
search_suppliers = create_search_suppliers_tool()
validate_product_ownership = create_validate_product_ownership_tool()
