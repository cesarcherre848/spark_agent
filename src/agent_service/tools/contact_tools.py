"""
src/agent_service/tools/contact_tools.py - Herramientas externas para gestión de clientes (Customers) en Odoo ERP

Interactúa directamente con la API JSON-RPC de Odoo sobre el modelo res.partner.
Garantiza el filtro de cartera asignada por vendedor (Salesperson: user_id) y clasifica
a los contactos con customer_rank=1 y supplier_rank=0.
"""

import logging
from typing import List, Optional, Dict, Any

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.config.odoo import get_odoo_settings

logger = logging.getLogger(__name__)


def get_shared_odoo_client(client: Optional[OdooClient] = None) -> OdooClient:
    """Retorna el cliente de Odoo inyectado o inicializa uno nuevo usando la configuración."""
    if client is not None:
        return client
    settings = get_odoo_settings()
    return OdooClient(
        base_url=settings.url,
        db=settings.db,
        username=settings.username,
        password=settings.password,
    )


async def odoo_get_customers(
    user_id: int = 5,
    query: Optional[str] = None,
    limit: int = 15,
    client: Optional[OdooClient] = None,
) -> List[Dict[str, Any]]:
    """Consulta en tiempo real la lista de clientes (customers) en Odoo asignados al vendedor.

    Filtra estrictamente por:
      - user_id = user_id (Salesperson asignado)
      - supplier_rank = 0 (Excluye proveedores)
      - active = True (Clientes activos)
    """
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    domain: List[Any] = [
        ["user_id", "=", int(user_id)],
        ["supplier_rank", "=", 0],
    ]
    if query and query.strip():
        q = query.strip()
        domain.extend(["|", ["name", "ilike", q], ["phone", "ilike", q]])

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "res.partner",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": ["id", "name", "phone", "active", "user_id"],
                "limit": limit,
            },
        },
        "id": 201,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al consultar clientes en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al consultar clientes: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_get_customers): {err_msg}")

    return data.get("result", []) or []


async def odoo_list_current_customers(
    user_id: int = 5,
    name: Optional[str] = None,
    phones: Optional[List[str]] = None,
    limit: int = 5,
    client: Optional[OdooClient] = None,
) -> List[Dict[str, Any]]:
    """Busca en Odoo si ya existen clientes con nombre o teléfono similar para el vendedor."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    conditions: List[Any] = []
    if name and name.strip():
        conditions.append(["name", "ilike", name.strip()])
    if phones:
        for p in phones:
            clean_p = str(p).strip()
            if clean_p:
                conditions.append(["phone", "ilike", clean_p])

    if not conditions:
        return []

    domain: List[Any] = [
        ["supplier_rank", "=", 0],
    ]
    if user_id is not None:
        domain.insert(0, ["user_id", "=", int(user_id)])

    # Encadenar operadores '|' si hay múltiples condiciones de coincidencia
    for _ in range(len(conditions) - 1):
        domain.append("|")
    domain.extend(conditions)

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "res.partner",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": ["id", "name", "phone", "user_id"],
                "limit": limit,
            },
        },
        "id": 202,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al listar candidatos de clientes en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al listar candidatos: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_list_current_customers): {err_msg}")

    return data.get("result", []) or []


async def odoo_upsert_customer(
    user_id: int = 5,
    name: str = "",
    phones: Optional[List[str]] = None,
    contact_id: Optional[int] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Crea un nuevo Customer o actualiza uno existente en Odoo asignado al vendedor."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    clean_phones = [str(p).strip() for p in (phones or []) if p and str(p).strip()]
    phone_val = ", ".join(clean_phones) if clean_phones else False

    values: Dict[str, Any] = {
        "name": (name or "").strip(),
        "user_id": int(user_id),
        "supplier_rank": 0,
    }
    if phone_val:
        values["phone"] = phone_val

    if contact_id:
        # Actualización de cliente existente
        call_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "res.partner",
                "method": "write",
                "args": [[int(contact_id)], values],
                "kwargs": {},
            },
            "id": 203,
        }
        try:
            response = await c._client.post("/web/dataset/call_kw", json=call_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error de red al actualizar cliente en Odoo: {e}")
            raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

        if "error" in data:
            err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
            logger.error(f"Error devuelto por Odoo al actualizar cliente: {err_msg}")
            raise RuntimeError(f"Error en Odoo API (odoo_upsert_customer write): {err_msg}")

        return {"id": int(contact_id), "action": "updated", "success": True, **values}
    else:
        # Creación con customer_rank = 1 (Customer nativo de Odoo)
        values["customer_rank"] = 1
        call_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "res.partner",
                "method": "create",
                "args": [values],
                "kwargs": {},
            },
            "id": 204,
        }
        try:
            response = await c._client.post("/web/dataset/call_kw", json=call_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error de red al crear cliente en Odoo: {e}")
            raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

        if "error" in data:
            err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
            logger.error(f"Error devuelto por Odoo al crear cliente: {err_msg}")
            raise RuntimeError(f"Error en Odoo API (odoo_upsert_customer create): {err_msg}")

        new_id = data.get("result")
        return {"id": new_id, "action": "created", "success": True, **values}


async def odoo_remove_customer(
    contact_id: int,
    user_id: int = 5,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Archiva de forma segura el cliente (active=False) en Odoo ERP."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "res.partner",
            "method": "write",
            "args": [[int(contact_id)], {"active": False}],
            "kwargs": {},
        },
        "id": 205,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al archivar cliente en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al archivar cliente: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_remove_customer): {err_msg}")

    return {"id": int(contact_id), "action": "archived", "success": True}
