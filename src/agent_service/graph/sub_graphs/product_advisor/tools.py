"""
src/agent_service/graph/sub_graphs/product_advisor/tools.py - Registro e inyección de dependencias de herramientas.
"""

import logging
from typing import Dict, Any, Optional
from psycopg_pool import AsyncConnectionPool

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.product_tools import (
    create_search_product_catalog_tool,
    create_get_product_by_skus_tool,
    create_get_cross_sell_recommendations_tool,
    create_filter_and_sort_products_tool,
    create_get_customer_purchase_history_tool,
)

logger = logging.getLogger(__name__)


def build_advisor_tools_registry(
    vector_store: Optional[ProductVectorStore] = None,
    odoo_client: Optional[OdooClient] = None,
    pool: Optional[AsyncConnectionPool] = None,
) -> Dict[str, Any]:
    """Crea e inyecta las instancias de herramientas disponibles para el planificador."""
    search_catalog = create_search_product_catalog_tool(vector_store=vector_store)
    get_details = create_get_product_by_skus_tool(odoo_client=odoo_client)
    cross_sell = create_get_cross_sell_recommendations_tool(
        vector_store=vector_store,
        odoo_client=odoo_client,
    )
    filter_sort = create_filter_and_sort_products_tool()
    purchase_history = create_get_customer_purchase_history_tool(odoo_client=odoo_client)

    return {
        "search_product_catalog": search_catalog,
        "get_product_odoo_details": get_details,
        "get_cross_sell_recommendations": cross_sell,
        "filter_and_sort_products": filter_sort,
        "get_customer_purchase_history": purchase_history,
    }


from langchain_core.tools import BaseTool


async def invoke_advisor_tool(
    tools_registry: Dict[str, Any],
    tool_name: str,
    arguments: Dict[str, Any],
) -> Any:
    """Invoca de manera segura y asíncrona una herramienta por su nombre comercial."""
    tool_fn = tools_registry.get(tool_name)
    if not tool_fn:
        logger.error(f"Herramienta desconocida solicitada por el planificador: '{tool_name}'")
        return {"error": f"Herramienta '{tool_name}' no disponible en el registro."}

    try:
        # BaseTool formal de LangChain
        if isinstance(tool_fn, BaseTool):
            return await tool_fn.ainvoke(arguments)
        # Callables o Mocks
        elif callable(tool_fn):
            try:
                res = tool_fn(**arguments) if isinstance(arguments, dict) else tool_fn(arguments)
            except TypeError:
                res = tool_fn(arguments)
            if hasattr(res, "__await__"):
                return await res
            return res
        elif hasattr(tool_fn, "ainvoke"):
            return await tool_fn.ainvoke(arguments)
        else:
            return {"error": f"La herramienta '{tool_name}' no es invocable."}
    except Exception as e:
        logger.exception(f"Error al ejecutar la herramienta '{tool_name}' con argumentos {arguments}: {e}")
        return {"error": str(e), "tool_name": tool_name}
