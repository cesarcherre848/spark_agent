from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


class ProductCatalogFilter(BaseModel):
    user_id: Optional[int] = Field(
        default=None,
        description="ID del usuario para validar productos autorizados en la vista view_user_authorized_products.",
    )
    allowed_vendor_ids: Optional[List[int]] = Field(
        default=None,
        description="IDs de res_partner (proveedores) que el usuario tiene autorizados según sus equipos CRM.",
    )


ProductAttribute = Literal[
    "price",        # list_price + currency_id (ej. 39.5 PEN)
    "description",  # description_sale (descripción comercial)
    "uom",          # uom_name (ej. 'Units', 'Litros')
    "category",     # categ_id (nombre de categoría)
    "barcode",      # barcode (código de barras)
    "taxes",        # tax_string (impuestos aplicables)
]


class GetProductBySkusInput(BaseModel):
    skus: List[str] = Field(
        description="Lista de códigos SKU exactos o referencias internas a consultar (ej: ['1', '11'])."
    )
    fields: Optional[List[ProductAttribute]] = Field(
        default=None,
        description=(
            "Lista de atributos específicos a consultar: 'price', 'description', 'uom', "
            "'category', 'barcode', 'taxes'. Si se omite, se devuelven los atributos comerciales básicos "
            "(nombre, precio y moneda)."
        ),
    )


class ProductOdooDetail(BaseModel):
    product_id: int = Field(description="ID interno del producto en Odoo (product.product).")
    sku: str = Field(description="Código SKU o referencia interna (default_code).")
    name: str = Field(description="Nombre comercial del producto.")
    price: Optional[float] = Field(default=None, description="Precio de venta unitario.")
    currency: Optional[str] = Field(default=None, description="Moneda comercial (ej. 'PEN', 'USD').")
    uom: Optional[str] = Field(default=None, description="Unidad de medida (ej. 'Units', 'Litros').")
    sales_description: Optional[str] = Field(default=None, description="Descripción comercial de venta.")
    category: Optional[str] = Field(default=None, description="Categoría asignada en el catálogo.")
    barcode: Optional[str] = Field(default=None, description="Código de barras.")
    taxes: Optional[str] = Field(default=None, description="Información descriptiva de impuestos.")


class GetProductBySkusOutput(BaseModel):
    products: List[ProductOdooDetail] = Field(
        default_factory=list,
        description="Lista de productos encontrados con sus atributos comerciales solicitados.",
    )
    not_found_skus: List[str] = Field(
        default_factory=list,
        description="Lista de SKUs consultados que no fueron encontrados o no están activos en Odoo.",
    )


# --- Esquemas para Búsqueda Oficial de Proveedores (Odoo API) ---

class SupplierSearchInput(BaseModel):
    query: str = Field(
        description="Texto de búsqueda para el proveedor (nombre comercial, razón social o RUC/VAT)."
    )
    limit: Optional[int] = Field(
        default=5,
        description="Número máximo de proveedores a retornar (por defecto 5).",
    )


class SupplierInfo(BaseModel):
    id: int = Field(description="ID interno del proveedor en Odoo (res.partner).")
    name: str = Field(description="Nombre comercial o razón social del proveedor.")
    display_name: str = Field(description="Nombre completo mostrado en Odoo.")
    vat: Optional[str] = Field(default=None, description="Número de identificación fiscal o RUC.")


class SupplierSearchOutput(BaseModel):
    suppliers: List[SupplierInfo] = Field(
        default_factory=list,
        description="Lista de proveedores encontrados que coinciden con la búsqueda.",
    )
    query: str = Field(description="Texto de consulta utilizado.")


# --- Esquemas para Validación de Ownership de Productos (PostgreSQL) ---

class ValidateProductOwnershipInput(BaseModel):
    user_id: int = Field(
        description="ID del usuario en Odoo/CRM para comprobar permisos de catálogo."
    )
    skus: List[str] = Field(
        description="Lista de códigos SKU a validar contra la vista de productos autorizados."
    )


class ProductPartnerInfo(BaseModel):
    partner_id: str = Field(description="ID del proveedor en formato string (ej: '9').")
    vendor_id: int = Field(description="ID numérico de res_partner.")
    vendor_name: str = Field(description="Nombre comercial de res_partner.")


class SkuOwnershipResult(BaseModel):
    sku: str = Field(description="Código SKU consultado.")
    product_id: Optional[int] = Field(default=None, description="ID interno de product.product en Odoo.")
    is_owner: bool = Field(description="True si el producto está autorizado para el usuario.")
    candidate_partners: List[ProductPartnerInfo] = Field(
        default_factory=list,
        description="Proveedores autorizados para este producto y usuario.",
    )
    has_conflict: bool = Field(
        default=False,
        description="True si hay múltiples proveedores posibles para este SKU.",
    )
    conflict_reason: Optional[str] = Field(
        default=None,
        description="Detalle del conflicto o razón de no autorización.",
    )


class ValidateProductOwnershipOutput(BaseModel):
    user_id: int = Field(description="ID del usuario consultado.")
    results: Dict[str, SkuOwnershipResult] = Field(
        default_factory=dict,
        description="Resultados de validación indexados por código SKU.",
    )

