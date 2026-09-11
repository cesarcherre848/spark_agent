from typing import List, Optional
from pydantic import BaseModel, Field


class ProductCatalogFilter(BaseModel):
    user_id: Optional[int] = Field(
        default=None,
        description="ID del usuario para validar productos autorizados en la vista view_user_authorized_products."
    )
    allowed_vendor_ids: Optional[List[int]] = Field(
        default=None, 
        description="IDs de res_partner (proveedores) que el usuario tiene autorizados según sus equipos CRM."
    )
