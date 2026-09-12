from typing import List, Optional, Literal
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
