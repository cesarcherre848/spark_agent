"""
src/agent_service/graph/sub_graphs/sales_manage/__init__.py - Módulo de gestión de ventas y cotizaciones Odoo
"""

from src.agent_service.graph.sub_graphs.sales_manage.state import SalesManageState
from src.agent_service.graph.sub_graphs.sales_manage.schemas import (
    SalesExtractionResult,
    SalesDuplicateCheckResult,
    SalesSynthesizeResponse,
)
from src.agent_service.graph.sub_graphs.sales_manage.graph import (
    build_sales_manage_graph,
    get_sales_manage_graph,
)

__all__ = [
    "SalesManageState",
    "SalesExtractionResult",
    "SalesDuplicateCheckResult",
    "SalesSynthesizeResponse",
    "build_sales_manage_graph",
    "get_sales_manage_graph",
]
