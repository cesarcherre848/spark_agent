import logging
from typing import List, Optional, Set, Dict, Any
import httpx

from src.agent_service.core.stores.product.schemas import (
    ProductAttribute,
    ProductOdooDetail,
    GetProductBySkusOutput,
    SupplierInfo,
    SupplierSearchOutput,
)

logger = logging.getLogger(__name__)

ATTRIBUTE_TO_ODOO_FIELDS: Dict[ProductAttribute, List[str]] = {
    "price": ["list_price", "currency_id"],
    "description": ["description_sale"],
    "uom": ["uom_name"],
    "category": ["categ_id"],
    "barcode": ["barcode"],
    "taxes": ["tax_string"],
}

BASE_ODOO_FIELDS: List[str] = ["id", "default_code", "name"]


class OdooClient:
    """Cliente asíncrono para interactuar con la API JSON-RPC 2.0 de Odoo ERP."""

    def __init__(
        self,
        base_url: str,
        db: str,
        username: str,
        password: str,
        timeout: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._db = db
        self._username = username
        self._password = password
        self._timeout = timeout
        self._uid: Optional[int] = None

        # Si se inyecta un cliente HTTP (útil para pruebas unitarias con mocks), se reutiliza
        self._client = client or httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        self._owns_client = client is None

    async def authenticate(self) -> int:
        """Autentica contra Odoo mediante /web/session/authenticate vía JSON-RPC 2.0."""
        payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "db": self._db,
                "login": self._username,
                "password": self._password,
            },
            "id": 1,
        }

        try:
            response = await self._client.post("/web/session/authenticate", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error de red al conectar con Odoo: {e}")
            raise ConnectionError(f"No fue posible conectar con el servidor Odoo en {self._base_url}: {e}") from e

        if "error" in data:
            err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
            raise PermissionError(f"Error de autenticación en Odoo: {err_msg}")

        result = data.get("result")
        if not result or not result.get("uid"):
            raise PermissionError(f"Odoo no devolvió un UID válido. Credenciales incorrectas para '{self._username}'.")

        self._uid = result["uid"]
        logger.info(f"Autenticado exitosamente en Odoo. UID: {self._uid}")
        return self._uid

    async def get_products_by_skus(
        self,
        skus: List[str],
        fields: Optional[List[ProductAttribute]] = None,
    ) -> GetProductBySkusOutput:
        """Consulta productos en Odoo por referencia interna (default_code / SKU) proyectando atributos dinámicos."""
        # 1. Normalización y deduplicación de SKUs
        clean_skus: List[str] = []
        seen_skus: Set[str] = set()
        for raw_s in (skus or []):
            s = str(raw_s).strip()
            if s and s not in seen_skus:
                seen_skus.add(s)
                clean_skus.append(s)

        if not clean_skus:
            return GetProductBySkusOutput(products=[], not_found_skus=[])

        # 2. Asegurar sesión activa
        if not self._uid:
            await self.authenticate()

        # 3. Determinar qué campos de Odoo solicitar
        requested_attrs: Set[ProductAttribute] = set(fields) if fields else {"price"}
        
        odoo_fields_set: Set[str] = set(BASE_ODOO_FIELDS)
        for attr in requested_attrs:
            odoo_fields_set.update(ATTRIBUTE_TO_ODOO_FIELDS.get(attr, []))
        
        odoo_fields_list = sorted(list(odoo_fields_set))

        # 4. Construir payload JSON-RPC call_kw
        call_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "product.product",
                "method": "search_read",
                "args": [[
                    ["default_code", "in", clean_skus],
                    ["active", "=", True],
                ]],
                "kwargs": {
                    "fields": odoo_fields_list,
                },
            },
            "id": 2,
        }

        try:
            response = await self._client.post("/web/dataset/call_kw", json=call_payload)
            res_data = response.json()

            # Re-autenticación si la sesión expiró
            if "error" in res_data:
                logger.warning("Error en llamada JSON-RPC; re-autenticando sesión de Odoo...")
                await self.authenticate()
                response = await self._client.post("/web/dataset/call_kw", json=call_payload)
                res_data = response.json()
        except Exception as e:
            logger.error(f"Fallo al ejecutar search_read en Odoo: {e}")
            raise RuntimeError(f"Error al consultar productos en Odoo: {e}") from e

        if "error" in res_data:
            err_msg = res_data["error"].get("data", {}).get("message") or res_data["error"].get("message")
            raise RuntimeError(f"Odoo rechazó la consulta: {err_msg}")

        raw_records = res_data.get("result", [])

        # 5. Mapear registros devueltos a Pydantic
        products: List[ProductOdooDetail] = []
        found_skus: Set[str] = set()

        for rec in raw_records:
            sku = str(rec.get("default_code") or "")
            found_skus.add(sku)

            # Desempaquetar tupla de divisa si se solicitó precio
            price_val: Optional[float] = None
            curr_code: Optional[str] = None
            if "price" in requested_attrs:
                price_val = float(rec.get("list_price") or 0.0)
                curr_raw = rec.get("currency_id")
                curr_code = curr_raw[1] if isinstance(curr_raw, (list, tuple)) and len(curr_raw) > 1 else "PEN"

            # Desempaquetar categoría
            category_val: Optional[str] = None
            if "category" in requested_attrs:
                categ_raw = rec.get("categ_id")
                category_val = categ_raw[1] if isinstance(categ_raw, (list, tuple)) and len(categ_raw) > 1 else None

            # Campos directos (en Odoo los campos vacíos vienen como False en lugar de None)
            desc_raw = rec.get("description_sale")
            desc_val = str(desc_raw) if ("description" in requested_attrs and desc_raw) else None

            uom_raw = rec.get("uom_name")
            uom_val = str(uom_raw) if ("uom" in requested_attrs and uom_raw) else None

            barcode_raw = rec.get("barcode")
            barcode_val = str(barcode_raw) if ("barcode" in requested_attrs and barcode_raw) else None

            taxes_raw = rec.get("tax_string")
            taxes_val = str(taxes_raw) if ("taxes" in requested_attrs and taxes_raw) else None


            products.append(
                ProductOdooDetail(
                    product_id=rec["id"],
                    sku=sku,
                    name=rec.get("name") or "",
                    price=price_val,
                    currency=curr_code,
                    uom=uom_val,
                    sales_description=desc_val,
                    category=category_val,
                    barcode=barcode_val,
                    taxes=taxes_val,
                )
            )

        # Detectar SKUs no encontrados
        not_found_skus = [s for s in clean_skus if s not in found_skus]

        return GetProductBySkusOutput(products=products, not_found_skus=not_found_skus)

    async def search_suppliers(
        self,
        query: str,
        limit: int = 5,
    ) -> SupplierSearchOutput:
        """Busca proveedores en Odoo ERP por nombre comercial, razón social o RUC (res.partner)."""
        clean_query = (query or "").strip()
        if not clean_query:
            return SupplierSearchOutput(suppliers=[], query="")

        if not self._uid:
            await self.authenticate()

        # Dominio: busca por nombre o vat (RUC) donde sea proveedor (supplier_rank > 0)
        domain = [
            ["supplier_rank", ">", 0],
            "|",
            ["name", "ilike", clean_query],
            ["vat", "ilike", clean_query],
        ]

        call_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "res.partner",
                "method": "search_read",
                "args": [domain],
                "kwargs": {
                    "fields": ["id", "name", "display_name", "vat"],
                    "limit": limit,
                },
            },
            "id": 2,
        }

        try:
            response = await self._client.post("/web/dataset/call_kw", json=call_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error de red al consultar proveedores en Odoo: {e}")
            raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

        if "error" in data:
            err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
            logger.error(f"Error devuelto por Odoo al buscar proveedores: {err_msg}")
            raise RuntimeError(f"Error en Odoo API (search_suppliers): {err_msg}")

        records: List[Dict[str, Any]] = data.get("result", []) or []
        suppliers: List[SupplierInfo] = []
        for rec in records:
            suppliers.append(
                SupplierInfo(
                    id=rec["id"],
                    name=rec.get("name") or "",
                    display_name=rec.get("display_name") or rec.get("name") or "",
                    vat=rec.get("vat") if rec.get("vat") else None,
                )
            )

        return SupplierSearchOutput(suppliers=suppliers, query=clean_query)

    async def aclose(self):
        """Cierra el cliente HTTP si fue creado internamente."""
        if self._owns_client and self._client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

