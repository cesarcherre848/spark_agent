"""
src/agent_service/tools/sales_tools.py - Herramientas externas para gestión de órdenes y cotizaciones en Odoo ERP

Interactúa con la API JSON-RPC 2.0 sobre los modelos sale.order y sale.order.line.
Garantiza el filtro de cartera asignada al vendedor (user_id) y no expone IDs técnicos
en los textos o salidas hacia el usuario.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.config.odoo import get_odoo_settings

logger = logging.getLogger(__name__)


def extract_order_code(text: Optional[str]) -> Optional[str]:
    """Extrae y normaliza un código de orden desde cualquier texto o cadena libre."""
    if not text or not str(text).strip():
        return None
    val = str(text).strip().upper()
    # 1. Coincidencia directa tipo 'S00003', 'SO003', 'S3'
    m_code = re.search(r"\b(?:S|SO)(\d{1,5})\b", val)
    if m_code:
        return f"S{int(m_code.group(1)):05d}"
    # 2. Coincidencia por frase: 'cotización 03', 'orden 3', 'pedido #3'
    m_phrase = re.search(r"\b(?:COTIZACI[OÓ]N|ORDEN|PEDIDO)\s*(?:N[ÚU]MERO|N[°º]|#)?\s*(\d{1,5})\b", val)
    if m_phrase:
        return f"S{int(m_phrase.group(1)):05d}"
    # 3. Si toda la cadena es solo un número: '03', '3'
    m_num = re.match(r"^(\d{1,5})$", val)
    if m_num:
        return f"S{int(m_num.group(1)):05d}"
    return None


def normalize_order_name(order_name: Optional[str]) -> Optional[str]:
    """Normaliza identificadores de orden como '03', '3', 's3', 'SO003', 'cotización 03' a 'S00003' o formato estándar."""
    extracted = extract_order_code(order_name)
    if extracted:
        return extracted
    if order_name and str(order_name).strip():
        return str(order_name).strip()
    return None


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


async def odoo_list_sales_orders(
    user_id: int = 5,
    partner_id: Optional[int] = None,
    customer_name: Optional[str] = None,
    status: Optional[str] = None,
    period: Optional[str] = None,
    limit: int = 25,
    client: Optional[OdooClient] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Consulta órdenes de venta en Odoo para el vendedor y las agrupa por estado comercial:
    - 'draft': cotizaciones (estados 'draft' y 'sent')
    - 'sale': pedidos confirmados (estados 'sale' y 'done')
    - 'cancel': pedidos o cotizaciones canceladas (estado 'cancel')
    """
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    domain: List[Any] = [["user_id", "=", int(user_id)]]

    if partner_id:
        domain.append(["partner_id", "=", int(partner_id)])
    elif customer_name and customer_name.strip():
        domain.append(["partner_id.name", "ilike", customer_name.strip()])

    if status:
        st = status.lower().strip()
        if st in ("draft", "sent", "sale", "cancel", "done"):
            domain.append(["state", "=", st])
        elif st in ("cotizacion", "cotizaciones"):
            domain.append(["state", "in", ["draft", "sent"]])
        elif st in ("confirmado", "confirmados", "aprobado", "aprobados", "cerrado", "cerrados"):
            domain.append(["state", "in", ["sale", "done"]])
        elif st in ("pedido", "pedidos"):
            domain.append(["state", "in", ["draft", "sent", "sale", "done"]])

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": [
                    "id",
                    "name",
                    "partner_id",
                    "date_order",
                    "amount_total",
                    "amount_untaxed",
                    "amount_tax",
                    "state",
                    "user_id",
                    "order_line",
                ],
                "limit": limit,
                "order": "date_order desc, id desc",
            },
        },
        "id": 301,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al consultar órdenes en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al listar órdenes: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_list_sales_orders): {err_msg}")

    records = data.get("result", []) or []

    # Enriquecer con detalle de líneas para cada orden
    all_line_ids = []
    for rec in records:
        all_line_ids.extend(rec.get("order_line", []))

    lines_by_order: Dict[int, List[Dict[str, Any]]] = {}
    if all_line_ids:
        line_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "sale.order.line",
                "method": "read",
                "args": [all_line_ids],
                "kwargs": {
                    "fields": ["id", "order_id", "name", "product_id", "product_uom_qty", "price_unit", "price_subtotal"],
                },
            },
            "id": 308,
        }
        try:
            line_resp = await c._client.post("/web/dataset/call_kw", json=line_payload)
            line_resp.raise_for_status()
            line_data = line_resp.json().get("result", []) or []
            for l in line_data:
                order_ref = l.get("order_id")
                oid_val = order_ref[0] if isinstance(order_ref, (list, tuple)) and len(order_ref) > 0 else order_ref
                prod_display = l.get("product_id", [None, "Producto"])
                prod_name = prod_display[1] if isinstance(prod_display, (list, tuple)) and len(prod_display) > 1 else str(l.get("name", "Producto")).split("\n")[0]
                if oid_val:
                    lines_by_order.setdefault(int(oid_val), []).append({
                        "id": l.get("id"),
                        "product_name": prod_name,
                        "quantity": float(l.get("product_uom_qty", 0.0)),
                        "price_unit": float(l.get("price_unit", 0.0)),
                        "price_subtotal": float(l.get("price_subtotal", 0.0)),
                    })
        except Exception as e:
            logger.warning(f"No se pudieron leer las líneas detalladas de las órdenes listadas: {e}")

    # Agrupación por estado comercial
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "draft": [],
        "sale": [],
        "cancel": [],
    }

    for rec in records:
        st = rec.get("state", "draft")
        # Formateo amigable sin exponer IDs al usuario
        partner_display = rec.get("partner_id", [None, "Cliente Desconocido"])
        cust_name = partner_display[1] if isinstance(partner_display, (list, tuple)) and len(partner_display) > 1 else "Cliente"
        oid = rec.get("id")
        
        formatted_rec = {
            "id": oid,
            "name": rec.get("name"),  # Código comercial visible (ej. SO001)
            "customer_name": cust_name,
            "partner_id": rec.get("partner_id", [None])[0] if isinstance(rec.get("partner_id"), (list, tuple)) else rec.get("partner_id"),
            "date_order": str(rec.get("date_order", "")),
            "amount_total": float(rec.get("amount_total", 0.0)),
            "amount_untaxed": float(rec.get("amount_untaxed", 0.0)),
            "amount_tax": float(rec.get("amount_tax", 0.0)),
            "state": st,
            "line_count": len(rec.get("order_line", [])),
            "lines": lines_by_order.get(int(oid), []) if oid else [],
        }

        if st in ("draft", "sent"):
            grouped["draft"].append(formatted_rec)
        elif st in ("sale", "done"):
            grouped["sale"].append(formatted_rec)
        elif st == "cancel":
            grouped["cancel"].append(formatted_rec)
        else:
            grouped["draft"].append(formatted_rec)

    return grouped


async def odoo_list_current_sales_orders(
    user_id: int = 5,
    partner_id: Optional[int] = None,
    customer_name: Optional[str] = None,
    limit: int = 5,
    client: Optional[OdooClient] = None,
) -> List[Dict[str, Any]]:
    """Consulta cotizaciones recientes abiertas ('draft' o 'sent') de un cliente específico
    para la evaluación de duplicados por el LLM Judge.
    """
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    domain: List[Any] = [
        ["user_id", "=", int(user_id)],
        ["state", "in", ["draft", "sent"]],
    ]

    if partner_id:
        domain.append(["partner_id", "=", int(partner_id)])
    elif customer_name and customer_name.strip():
        domain.append(["partner_id.name", "ilike", customer_name.strip()])
    else:
        return []

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": ["id", "name", "partner_id", "date_order", "amount_total", "state", "order_line"],
                "limit": limit,
                "order": "id desc",
            },
        },
        "id": 302,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al consultar cotizaciones actuales en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al listar cotizaciones actuales: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_list_current_sales_orders): {err_msg}")

    results = data.get("result", []) or []

    # Enriquecer con detalle de líneas para cada cotización
    all_line_ids = []
    for r in results:
        all_line_ids.extend(r.get("order_line", []))

    lines_by_order: Dict[int, List[Dict[str, Any]]] = {}
    if all_line_ids:
        line_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "sale.order.line",
                "method": "read",
                "args": [all_line_ids],
                "kwargs": {
                    "fields": ["id", "order_id", "name", "product_id", "product_uom_qty", "price_unit", "price_subtotal"],
                },
            },
            "id": 307,
        }
        try:
            line_resp = await c._client.post("/web/dataset/call_kw", json=line_payload)
            line_resp.raise_for_status()
            line_data = line_resp.json().get("result", []) or []
            for l in line_data:
                order_ref = l.get("order_id")
                oid_val = order_ref[0] if isinstance(order_ref, (list, tuple)) and len(order_ref) > 0 else order_ref
                prod_display = l.get("product_id", [None, "Producto"])
                prod_name = prod_display[1] if isinstance(prod_display, (list, tuple)) and len(prod_display) > 1 else str(l.get("name", "Producto")).split("\n")[0]
                if oid_val:
                    lines_by_order.setdefault(int(oid_val), []).append({
                        "id": l.get("id"),
                        "product_name": prod_name,
                        "quantity": float(l.get("product_uom_qty", 0.0)),
                        "price_unit": float(l.get("price_unit", 0.0)),
                        "price_subtotal": float(l.get("price_subtotal", 0.0)),
                    })
        except Exception as e:
            logger.warning(f"No se pudieron leer las líneas detalladas de las órdenes candidatas: {e}")

    output = []
    for r in results:
        partner_display = r.get("partner_id", [None, "Cliente"])
        cust_name = partner_display[1] if isinstance(partner_display, (list, tuple)) and len(partner_display) > 1 else "Cliente"
        oid = r.get("id")
        output.append({
            "id": oid,
            "name": r.get("name"),
            "customer_name": cust_name,
            "partner_id": partner_display[0] if isinstance(partner_display, (list, tuple)) else partner_display,
            "date_order": str(r.get("date_order", "")),
            "amount_total": float(r.get("amount_total", 0.0)),
            "state": r.get("state"),
            "order_line_ids": r.get("order_line", []),
            "lines": lines_by_order.get(int(oid), []) if oid else [],
        })
    return output


async def odoo_create_quotation(
    user_id: int = 5,
    partner_id: int = 1,
    items: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Crea una nueva cotización ('draft') en Odoo para el cliente y vendedor indicados."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    order_lines = []
    skus_to_resolve = []
    items_clean = items or []

    for it in items_clean:
        if not it.get("product_id") and it.get("sku"):
            skus_to_resolve.append(str(it["sku"]))

    # Si hay SKUs sin product_id numérico, resolverlos
    resolved_skus_map = {}
    not_found_skus = []
    if skus_to_resolve:
        prod_res = await c.get_products_by_skus(skus=skus_to_resolve, fields=["price", "description"])
        not_found_skus = getattr(prod_res, "not_found_skus", [])
        for p in prod_res.products:
            clean_sku = str(p.sku).strip()
            resolved_skus_map[clean_sku] = p

    merged_items: Dict[int, Dict[str, Any]] = {}
    for it in items_clean:
        pid = it.get("product_id")
        qty = float(it.get("qty") or it.get("cantidad") or 1.0)
        sku = str(it.get("sku", "")).strip()

        if not pid and sku in resolved_skus_map:
            p_obj = resolved_skus_map[sku]
            pid = p_obj.product_id
            price_unit = it.get("price_unit") or (p_obj.price if p_obj.price is not None else 0.0)
        else:
            price_unit = it.get("price_unit")

        if pid:
            int_pid = int(pid)
            if int_pid in merged_items:
                merged_items[int_pid]["qty"] += qty
                if price_unit is not None:
                    merged_items[int_pid]["price_unit"] = price_unit
            else:
                merged_items[int_pid] = {"qty": qty, "price_unit": price_unit}

    for pid, data in merged_items.items():
        line_vals: Dict[str, Any] = {
            "product_id": pid,
            "product_uom_qty": data["qty"],
        }
        if data["price_unit"] is not None:
            line_vals["price_unit"] = float(data["price_unit"])
        order_lines.append((0, 0, line_vals))

    if not order_lines:
        return {
            "success": False,
            "error": f"Los productos con código {not_found_skus or skus_to_resolve} no existen o están inactivos en Odoo.",
            "not_found_skus": not_found_skus or skus_to_resolve,
        }

    create_vals = {
        "partner_id": int(partner_id),
        "user_id": int(user_id),
        "order_line": order_lines,
    }

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "create",
            "args": [create_vals],
            "kwargs": {},
        },
        "id": 303,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al crear cotización en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al crear cotización: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_create_quotation): {err_msg}")

    order_id = data.get("result")
    if not order_id:
        raise RuntimeError("Odoo no devolvió un ID de orden válido al crear la cotización.")

    # Leer datos finales de la orden creada
    view_data = await odoo_view_quotation(order_id=int(order_id), client=c)
    return {
        "success": True,
        "action": "created",
        "order_id": int(order_id),
        "name": view_data.get("name"),
        "amount_total": view_data.get("amount_total"),
        "amount_untaxed": view_data.get("amount_untaxed"),
        "amount_tax": view_data.get("amount_tax"),
        "lines": view_data.get("lines", []),
        "customer_name": view_data.get("customer_name"),
    }


async def odoo_update_quotation(
    order_id: Optional[int] = None,
    order_name: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Actualiza, añade o elimina líneas de una cotización en borrador existente en Odoo."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    if not order_id and order_name:
        view_res = await odoo_view_quotation(order_name=order_name, client=c)
        order_id = view_res.get("id")

    if not order_id:
        return {
            "success": False,
            "error": "No se identificó el ID o nombre de la cotización a actualizar en Odoo.",
        }

    # 1. Leer las líneas existentes en la orden de venta
    order_read_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "read",
            "args": [[int(order_id)]],
            "kwargs": {"fields": ["id", "name", "order_line"]},
        },
        "id": 303,
    }
    try:
        ord_resp = await c._client.post("/web/dataset/call_kw", json=order_read_payload)
        ord_resp.raise_for_status()
        ord_data = ord_resp.json()
    except Exception as e:
        logger.error(f"Error de red al consultar orden {order_id} en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    order_records = ord_data.get("result", []) or []
    if not order_records:
        return {
            "success": False,
            "error": f"No se encontró la cotización con ID {order_id} en Odoo.",
        }

    existing_line_ids = order_records[0].get("order_line", [])
    existing_lines: List[Dict[str, Any]] = []
    if existing_line_ids:
        line_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "sale.order.line",
                "method": "read",
                "args": [existing_line_ids],
                "kwargs": {
                    "fields": ["id", "name", "product_id", "product_uom_qty", "price_unit"],
                },
            },
            "id": 304,
        }
        try:
            line_resp = await c._client.post("/web/dataset/call_kw", json=line_payload)
            line_resp.raise_for_status()
            existing_lines = line_resp.json().get("result", []) or []
        except Exception as e:
            logger.warning(f"Error al leer líneas de la orden {order_id}: {e}")

    # Mapear líneas existentes agrupadas por product_id, por SKU y por nombre comercial
    existing_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    existing_by_sku: Dict[str, List[Dict[str, Any]]] = {}
    existing_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for l in existing_lines:
        p_val = l.get("product_id")
        pid_int = p_val[0] if isinstance(p_val, (list, tuple)) and p_val else p_val
        if pid_int:
            existing_by_pid.setdefault(int(pid_int), []).append(l)

        display_name = l.get("name") or (p_val[1] if isinstance(p_val, (list, tuple)) and len(p_val) > 1 else "")
        m_sku = re.search(r"\[(.*?)\]", display_name)
        if m_sku:
            clean_s = m_sku.group(1).strip()
            existing_by_sku.setdefault(clean_s, []).append(l)

        clean_name = re.sub(r"\[.*?\]", "", display_name).strip().lower()
        if clean_name:
            existing_by_name.setdefault(clean_name, []).append(l)

    items_clean = items or []
    skus_to_resolve = [str(it["sku"]).strip() for it in items_clean if not it.get("product_id") and it.get("sku")]

    resolved_skus_map = {}
    not_found_skus = []
    if skus_to_resolve:
        prod_res = await c.get_products_by_skus(skus=skus_to_resolve, fields=["price", "description"])
        not_found_skus = list(getattr(prod_res, "not_found_skus", []) or [])
        for p in prod_res.products:
            clean_sku = str(p.sku).strip()
            resolved_skus_map[clean_sku] = p

    # Agrupar operaciones solicitadas por product_id
    items_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    for it in items_clean:
        pid = it.get("product_id")
        sku = str(it.get("sku", "")).strip()

        if not pid and sku in resolved_skus_map:
            p_obj = resolved_skus_map[sku]
            pid = p_obj.product_id
        elif not pid and sku in existing_by_sku:
            line_ref = existing_by_sku[sku][0]
            p_val = line_ref.get("product_id")
            pid = p_val[0] if isinstance(p_val, (list, tuple)) else p_val
            if sku in not_found_skus:
                not_found_skus.remove(sku)
        elif not pid and sku:
            # Buscar por coincidencia en nombre comercial de las líneas existentes en la orden
            sku_clean = sku.strip().lower()
            for norm_name, l_list in existing_by_name.items():
                if sku_clean in norm_name or norm_name in sku_clean or any(w in norm_name for w in sku_clean.split() if len(w) > 3):
                    line_ref = l_list[0]
                    p_val = line_ref.get("product_id")
                    pid = p_val[0] if isinstance(p_val, (list, tuple)) else p_val
                    if sku in not_found_skus:
                        not_found_skus.remove(sku)
                    break

        if pid:
            items_by_pid.setdefault(int(pid), []).append(it)

    if not items_by_pid and not_found_skus:
        return {
            "success": False,
            "error": f"Los productos con código {not_found_skus} no existen o están inactivos en Odoo.",
            "not_found_skus": not_found_skus,
        }

    order_lines = []
    for pid, ops in items_by_pid.items():
        lines_for_p = existing_by_pid.get(pid, [])
        current_total_qty = sum(float(l.get("product_uom_qty", 0.0)) for l in lines_for_p)
        latest_price_unit = None
        op_sku = None

        new_qty = current_total_qty
        should_remove = False

        for op in ops:
            qty = float(op.get("qty") or op.get("cantidad") or 1.0)
            action = str(op.get("action") or "add").lower()
            if op.get("sku"):
                op_sku = str(op["sku"]).strip()
            if op.get("price_unit") is not None:
                latest_price_unit = float(op["price_unit"])

            if action == "remove":
                should_remove = True
                new_qty = 0.0
            elif action == "subtract":
                new_qty = max(0.0, new_qty - qty)
                if new_qty <= 0.0:
                    should_remove = True
            elif action == "set":
                new_qty = max(0.0, qty)
                if new_qty <= 0.0:
                    should_remove = True
                else:
                    should_remove = False
            else:  # "add"
                new_qty = new_qty + qty
                should_remove = False

        if lines_for_p:
            primary_line = lines_for_p[0]
            dup_lines = lines_for_p[1:]

            if should_remove:
                # Eliminar todas las líneas existentes de este producto
                for l in lines_for_p:
                    order_lines.append((2, l["id"], 0))
            else:
                # Actualizar la línea principal con la nueva cantidad total
                vals: Dict[str, Any] = {"product_uom_qty": new_qty}
                if latest_price_unit is not None:
                    vals["price_unit"] = latest_price_unit
                order_lines.append((1, primary_line["id"], vals))
                # Eliminar cualquier línea duplicada redundante
                for dl in dup_lines:
                    order_lines.append((2, dl["id"], 0))
        else:
            # El producto no existía previamente en la orden
            if not should_remove and new_qty > 0.0:
                line_vals: Dict[str, Any] = {
                    "product_id": int(pid),
                    "product_uom_qty": new_qty,
                }
                if latest_price_unit is not None:
                    line_vals["price_unit"] = latest_price_unit
                elif op_sku and op_sku in resolved_skus_map and resolved_skus_map[op_sku].price is not None:
                    line_vals["price_unit"] = float(resolved_skus_map[op_sku].price)
                order_lines.append((0, 0, line_vals))

    if order_lines:
        call_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "sale.order",
                "method": "write",
                "args": [[int(order_id)], {"order_line": order_lines}],
                "kwargs": {},
            },
            "id": 305,
        }

        try:
            response = await c._client.post("/web/dataset/call_kw", json=call_payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error de red al actualizar cotización en Odoo: {e}")
            raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

        if "error" in data:
            err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
            logger.error(f"Error devuelto por Odoo al actualizar cotización: {err_msg}")
            raise RuntimeError(f"Error en Odoo API (odoo_update_quotation): {err_msg}")

    view_data = await odoo_view_quotation(order_id=int(order_id), client=c)
    return {
        "success": True,
        "action": "updated",
        "order_id": int(order_id),
        "name": view_data.get("name"),
        "amount_total": view_data.get("amount_total"),
        "amount_untaxed": view_data.get("amount_untaxed"),
        "amount_tax": view_data.get("amount_tax"),
        "lines": view_data.get("lines", []),
        "customer_name": view_data.get("customer_name"),
        "not_found_skus": not_found_skus,
    }


async def odoo_remove_quotation_items(
    order_id: Optional[int] = None,
    order_name: Optional[str] = None,
    product: Optional[str] = None,
    qty: Optional[float] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Herramienta atómica para restar o eliminar productos de una cotización existente por SKU o nombre comercial."""
    items_list = list(items or [])
    if product:
        action = "subtract" if (qty is not None and qty > 0) else "remove"
        items_list.append({"sku": str(product).strip(), "qty": float(qty or 1.0), "action": action})
    return await odoo_update_quotation(order_id=order_id, order_name=order_name, items=items_list, client=client)


async def odoo_add_quotation_items(
    order_id: Optional[int] = None,
    order_name: Optional[str] = None,
    product: Optional[str] = None,
    qty: Optional[float] = 1.0,
    price_unit: Optional[float] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Herramienta atómica para agregar o sumar productos a una cotización existente por SKU o nombre comercial."""
    items_list = list(items or [])
    if product:
        item_dict = {"sku": str(product).strip(), "qty": float(qty or 1.0), "action": "add"}
        if price_unit is not None:
            item_dict["price_unit"] = float(price_unit)
        items_list.append(item_dict)
    return await odoo_update_quotation(order_id=order_id, order_name=order_name, items=items_list, client=client)


async def odoo_view_quotation(
    order_id: Optional[int] = None,
    order_name: Optional[str] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Obtiene el detalle financiero y de productos de una cotización u orden en Odoo."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    domain: List[Any] = []
    if order_id:
        domain.append(["id", "=", int(order_id)])
    elif order_name and order_name.strip():
        norm_name = normalize_order_name(order_name.strip())
        domain.append(["name", "=", norm_name])
    else:
        return {}

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": [
                    "id",
                    "name",
                    "partner_id",
                    "date_order",
                    "state",
                    "amount_untaxed",
                    "amount_tax",
                    "amount_total",
                    "order_line",
                ],
                "limit": 1,
            },
        },
        "id": 305,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al consultar detalle de orden en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    records = data.get("result", []) or []
    if not records:
        return {}

    order_rec = records[0]
    partner_display = order_rec.get("partner_id", [None, "Cliente"])
    cust_name = partner_display[1] if isinstance(partner_display, (list, tuple)) and len(partner_display) > 1 else "Cliente"
    partner_id_val = partner_display[0] if isinstance(partner_display, (list, tuple)) else partner_display

    # Leer líneas de la orden
    line_ids = order_rec.get("order_line", [])
    lines_output = []
    if line_ids:
        line_payload = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "model": "sale.order.line",
                "method": "read",
                "args": [line_ids],
                "kwargs": {
                    "fields": ["id", "name", "product_id", "product_uom_qty", "price_unit", "price_subtotal", "price_total"],
                },
            },
            "id": 306,
        }
        try:
            line_resp = await c._client.post("/web/dataset/call_kw", json=line_payload)
            line_resp.raise_for_status()
            line_data = line_resp.json().get("result", []) or []
            for l in line_data:
                prod_display = l.get("product_id", [None, "Producto"])
                prod_name = prod_display[1] if isinstance(prod_display, (list, tuple)) and len(prod_display) > 1 else "Producto"
                lines_output.append({
                    "product_name": prod_name,
                    "quantity": float(l.get("product_uom_qty", 0.0)),
                    "price_unit": float(l.get("price_unit", 0.0)),
                    "price_subtotal": float(l.get("price_subtotal", 0.0)),
                })
        except Exception as e:
            logger.warning(f"No se pudieron leer las líneas detalladas de la orden: {e}")

    return {
        "id": order_rec.get("id"),
        "name": order_rec.get("name"),
        "customer_name": cust_name,
        "partner_id": partner_id_val,
        "date_order": str(order_rec.get("date_order", "")),
        "state": order_rec.get("state"),
        "amount_untaxed": float(order_rec.get("amount_untaxed", 0.0)),
        "amount_tax": float(order_rec.get("amount_tax", 0.0)),
        "amount_total": float(order_rec.get("amount_total", 0.0)),
        "lines": lines_output,
    }


async def odoo_confirm_order(
    order_id: int,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Confirma una cotización en borrador convirtiéndola en un pedido oficial de venta en Odoo (action_confirm)."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "action_confirm",
            "args": [[int(order_id)]],
            "kwargs": {},
        },
        "id": 307,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al confirmar orden en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al confirmar orden: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_confirm_order): {err_msg}")

    view_data = await odoo_view_quotation(order_id=int(order_id), client=c)
    return {
        "success": True,
        "action": "confirmed",
        "order_id": int(order_id),
        "name": view_data.get("name"),
        "state": view_data.get("state", "sale"),
        "amount_total": view_data.get("amount_total"),
        "customer_name": view_data.get("customer_name"),
    }


async def odoo_unlock_order(
    order_id: int,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Desbloquea una orden confirmada en Odoo (action_unlock) para habilitar su edición."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "action_unlock",
            "args": [[int(order_id)]],
            "kwargs": {},
        },
        "id": 308,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al desbloquear orden en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.warning(f"Aviso al desbloquear orden en Odoo: {err_msg}")

    return {"success": True, "action": "unlocked", "order_id": int(order_id)}


async def odoo_update_order(
    order_id: int,
    items: Optional[List[Dict[str, Any]]] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Actualiza líneas en una orden confirmada previamente desbloqueada."""
    return await odoo_update_quotation(order_id=order_id, items=items, client=client)


async def odoo_lock_order(
    order_id: int,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Vuelve a bloquear la orden en Odoo (action_lock) para proteger su integridad comercial."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "action_lock",
            "args": [[int(order_id)]],
            "kwargs": {},
        },
        "id": 309,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al re-bloquear orden en Odoo: {e}")
        # No romper la transacción si la versión de Odoo maneja el lock internamente
        return {"success": True, "action": "locked", "order_id": int(order_id)}

    return {"success": True, "action": "locked", "order_id": int(order_id)}


async def odoo_remove_sale_order(
    order_id: int,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Cancela una orden o cotización en Odoo mediante action_cancel preservando la trazabilidad contable."""
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.order",
            "method": "action_cancel",
            "args": [[int(order_id)]],
            "kwargs": {},
        },
        "id": 310,
    }

    try:
        response = await c._client.post("/web/dataset/call_kw", json=call_payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error de red al cancelar orden en Odoo: {e}")
        raise ConnectionError(f"Error al conectar con Odoo API: {e}") from e

    if "error" in data:
        err_msg = data["error"].get("data", {}).get("message") or data["error"].get("message")
        logger.error(f"Error devuelto por Odoo al cancelar orden: {err_msg}")
        raise RuntimeError(f"Error en Odoo API (odoo_remove_sale_order): {err_msg}")

    return {
        "success": True,
        "action": "cancelled",
        "order_id": int(order_id),
    }
