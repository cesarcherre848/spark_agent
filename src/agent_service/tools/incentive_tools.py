"""
src/agent_service/tools/incentive_tools.py - Herramientas para la consulta y cálculo de incentivos y tendencias en Odoo ERP

Consume la API REST /api/v1/sales_incentives y el modelo sale.incentive de Odoo ERP,
minando las tendencias de compra en órdenes recientes (sale.order.line) y calculando
las brechas hacia los objetivos comerciales por proveedor.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.sales_tools import (
    get_shared_odoo_client,
    odoo_list_sales_orders,
    odoo_list_current_sales_orders,
)

logger = logging.getLogger(__name__)


async def odoo_get_active_incentives(
    supplier_id: Optional[int] = None,
    supplier_name: Optional[str] = None,
    validity_date: Optional[str] = None,
    only_active: bool = True,
    limit: int = 50,
    offset: int = 0,
    client: Optional[OdooClient] = None,
) -> List[Dict[str, Any]]:
    """Consulta los planes de incentivo vigentes en Odoo ERP por proveedor.
    
    Prioriza el endpoint REST /api/v1/sales_incentives y utiliza fallback
    a JSON-RPC call_kw sobre el modelo sale.incentive si fuera necesario.
    """
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    # 1. Intentar vía endpoint REST oficial /api/v1/sales_incentives
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
    }
    if supplier_id is not None:
        params["supplier_id"] = int(supplier_id)
    if supplier_name and supplier_name.strip():
        params["supplier_name"] = supplier_name.strip()
    if only_active:
        params["only_active"] = "true"
    if validity_date:
        params["validity_date"] = validity_date

    try:
        response = await c._client.get("/api/v1/sales_incentives", params=params)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
    except Exception as exc:
        logger.warning(f"Error consultando REST /api/v1/sales_incentives: {exc}. Intentando fallback JSON-RPC.")

    # 2. Fallback vía JSON-RPC call_kw (sale.incentive)
    domain: List[Any] = []
    if only_active:
        domain.append(["state", "=", "active"])
        domain.append(["is_valid", "=", True])
    if supplier_id is not None:
        domain.append(["partner_id", "=", int(supplier_id)])
    elif supplier_name and supplier_name.strip():
        domain.append(["partner_id.name", "ilike", supplier_name.strip()])

    call_payload = {
        "jsonrpc": "2.0",
        "method": "call",
        "params": {
            "model": "sale.incentive",
            "method": "search_read",
            "args": [domain],
            "kwargs": {
                "fields": [
                    "id",
                    "name",
                    "title",
                    "partner_id",
                    "date_start",
                    "date_end",
                    "state",
                    "is_valid",
                    "description",
                    "observations",
                ],
                "limit": limit,
                "offset": offset,
            },
        },
        "id": 401,
    }

    try:
        res = await c._client.post("/web/dataset/call_kw", json=call_payload)
        res.raise_for_status()
        raw_records = res.json().get("result", []) or []
        normalized = []
        for r in raw_records:
            partner = r.get("partner_id") or [None, "Sin proveedor"]
            p_id = partner[0] if isinstance(partner, (list, tuple)) else partner
            p_name = partner[1] if isinstance(partner, (list, tuple)) and len(partner) > 1 else str(partner)
            normalized.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "title": r.get("title") or r.get("name"),
                "supplier_id": p_id,
                "supplier_name": p_name,
                "partner_id": p_id,
                "partner_name": p_name,
                "description": r.get("description"),
                "observations": r.get("observations"),
                "benefit_plan": r.get("title") or r.get("name"),
                "date_start": r.get("date_start"),
                "date_end": r.get("date_end"),
                "state": r.get("state"),
                "is_valid": r.get("is_valid", True),
            })
        return normalized
    except Exception as exc:
        logger.error(f"Error en fallback JSON-RPC sobre sale.incentive: {exc}")
        return []


def extract_sales_trends(
    orders_data: Any,
    supplier_id: Optional[int] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Extrae las tendencias reales de compra de la cartera a partir de las líneas
    de pedidos confirmados (sale) y cotizaciones activas (draft).
    
    Analiza la frecuencia de repetición, cantidades vendidas y volumen en PEN
    para alimentar la componente épsilon (demanda orgánica) en la estrategia de ventas.
    """
    if isinstance(orders_data, dict):
        confirmed_orders = orders_data.get("sale", []) or []
        draft_orders = orders_data.get("draft", []) or []
        all_orders = confirmed_orders + draft_orders
    elif isinstance(orders_data, list):
        all_orders = orders_data
    else:
        all_orders = []

    stats_by_sku: Dict[str, Dict[str, Any]] = {}

    for order in all_orders:
        lines = order.get("lines", []) or []
        for line in lines:
            prod = line.get("product_id")
            # prod puede ser [id, "[SKU] Nombre"] o id
            prod_name = ""
            sku = ""
            if isinstance(prod, (list, tuple)) and len(prod) > 1:
                full_label = str(prod[1])
                match_sku = re.search(r"\[(.*?)\]", full_label)
                if match_sku:
                    sku = match_sku.group(1).strip()
                    prod_name = full_label.replace(f"[{sku}]", "").strip()
                else:
                    prod_name = full_label
            elif line.get("name"):
                full_label = str(line.get("name"))
                match_sku = re.search(r"\[(.*?)\]", full_label)
                if match_sku:
                    sku = match_sku.group(1).strip()
                    prod_name = full_label.replace(f"[{sku}]", "").strip()
                else:
                    prod_name = full_label

            if not sku:
                sku = str(line.get("product_id") or "SKU-N/A")

            qty = float(line.get("product_uom_qty") or 1.0)
            subtotal = float(line.get("price_subtotal") or 0.0)

            if sku not in stats_by_sku:
                stats_by_sku[sku] = {
                    "sku": sku,
                    "name": prod_name or line.get("name") or f"Producto {sku}",
                    "total_qty": 0.0,
                    "order_count": 0,
                    "total_subtotal": 0.0,
                }
            stats_by_sku[sku]["total_qty"] += qty
            stats_by_sku[sku]["order_count"] += 1
            stats_by_sku[sku]["total_subtotal"] += subtotal

    # Ordenar por frecuencia de pedido y luego por cantidad total
    ranked = sorted(
        stats_by_sku.values(),
        key=lambda x: (x["order_count"], x["total_qty"], x["total_subtotal"]),
        reverse=True,
    )
    return ranked[:top_k]


def calculate_incentive_gap(
    orders_data: Any,
    incentive_plans: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Calcula el estado actual de ventas del vendedor y la brecha hacia las metas
    de incentivos activas (escalas de descuento y promociones de regalo).
    """
    if isinstance(orders_data, dict):
        confirmed_orders = orders_data.get("sale", []) or []
        draft_orders = orders_data.get("draft", []) or []
    elif isinstance(orders_data, list):
        confirmed_orders = [o for o in orders_data if o.get("state") in ("sale", "done")]
        draft_orders = [o for o in orders_data if o.get("state") in ("draft", "sent")]
    else:
        confirmed_orders, draft_orders = [], []

    confirmed_total = sum(float(o.get("amount_total") or 0.0) for o in confirmed_orders)
    draft_total = sum(float(o.get("amount_total") or 0.0) for o in draft_orders)
    potential_total = confirmed_total + draft_total

    # 1. Análisis de Escalas de Descuento (ej: 25% S/ 230, 30% S/ 720, 35% S/ 1,500)
    discount_tiers = [
        {"tier": "25% de Descuento", "min_amount": 230.0},
        {"tier": "30% de Descuento", "min_amount": 720.0},
        {"tier": "35% de Descuento", "min_amount": 1500.0},
    ]

    current_tier = "20% (Base)"
    next_tier = discount_tiers[0]["tier"]
    gap_to_next_tier_pen = max(0.0, discount_tiers[0]["min_amount"] - confirmed_total)
    next_tier_target_pen = discount_tiers[0]["min_amount"]

    for i, t in enumerate(discount_tiers):
        if confirmed_total >= t["min_amount"]:
            current_tier = t["tier"]
            if i + 1 < len(discount_tiers):
                next_tier = discount_tiers[i + 1]["tier"]
                next_tier_target_pen = discount_tiers[i + 1]["min_amount"]
                gap_to_next_tier_pen = max(0.0, discount_tiers[i + 1]["min_amount"] - confirmed_total)
            else:
                next_tier = "Escala Máxima Alcanzada"
                gap_to_next_tier_pen = 0.0
                next_tier_target_pen = t["min_amount"]

    # 2. Análisis de Promociones de Producto (Vende y Gana / Bonos específicos)
    product_promos = []
    for plan in incentive_plans:
        title = plan.get("title") or plan.get("name") or ""
        obs = plan.get("observations") or ""
        desc = plan.get("description") or ""
        combined_text = f"{title} {obs} {desc}"

        # Extraer códigos SKU participantes mencionados en texto (ej. Cód. 2210, 2218, 2035)
        codes_found = re.findall(r"\b(?:C[oó]d\.?|SKU)?\s*(\d{3,5})\b", combined_text, re.IGNORECASE)
        # Limpiar códigos que sean números de montos
        participating_skus = [c for c in set(codes_found) if c not in ("230", "470", "720", "1500", "2000", "3400", "3500")]

        # Extraer unidades mínimas requeridas (ej: '3 unidades', '4 o más unidades')
        qty_match = re.search(r"(\d+)\s*(?:o más)?\s*unidades", combined_text, re.IGNORECASE)
        req_units = int(qty_match.group(1)) if qty_match else None

        if "vende y gana" in combined_text.lower() or "gratis" in combined_text.lower():
            product_promos.append({
                "id": plan.get("id"),
                "title": title,
                "reward": plan.get("description") or title,
                "skus": participating_skus,
                "required_units": req_units or 3,
                "supplier_name": plan.get("supplier_name"),
            })

    # 3. Días restantes si existe date_end
    now = reference_date or datetime.now()
    days_remaining = None
    for plan in incentive_plans:
        date_end_str = plan.get("date_end")
        if date_end_str:
            try:
                dt_end = datetime.fromisoformat(str(date_end_str))
                delta = (dt_end - now).days
                if days_remaining is None or delta < days_remaining:
                    days_remaining = max(0, delta)
            except Exception:
                pass

    return {
        "confirmed_total_pen": round(confirmed_total, 2),
        "draft_total_pen": round(draft_total, 2),
        "potential_total_pen": round(potential_total, 2),
        "current_discount_tier": current_tier,
        "next_discount_tier": next_tier,
        "next_tier_target_pen": round(next_tier_target_pen, 2),
        "gap_to_next_tier_pen": round(gap_to_next_tier_pen, 2),
        "product_promos": product_promos,
        "days_remaining": days_remaining,
    }


async def get_active_incentive_targets(
    user_id: int = 5,
    supplier_id: Optional[int] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Herramienta puente para inyectar metas de incentivo y tendencias reales
    en el recomendador de productos (product_advisor).
    """
    c = get_shared_odoo_client(client)
    orders_data = await odoo_list_sales_orders(user_id=user_id, client=c)
    incentives = await odoo_get_active_incentives(supplier_id=supplier_id, only_active=True, client=c)
    trends = extract_sales_trends(orders_data, supplier_id=supplier_id, top_k=5)
    gap_info = calculate_incentive_gap(orders_data, incentives)

    # Recolectar SKUs participantes de bonos activos
    bonus_skus = []
    for promo in gap_info.get("product_promos", []):
        bonus_skus.extend(promo.get("skus", []))

    # Conocidos por catálogo en Odoo de Unique para Vende y Gana
    default_bonus_skus = ["2210", "2218", "2219", "2220", "897", "161", "152", "143", "2035", "233", "2041", "202"]
    for s in default_bonus_skus:
        if s not in bonus_skus:
            bonus_skus.append(s)

    trending_skus = [t["sku"] for t in trends if t.get("sku")]

    return {
        "user_id": user_id,
        "supplier_id": supplier_id,
        "bonus_skus": list(set(bonus_skus)),
        "trending_skus": trending_skus,
        "top_trends": trends,
        "current_discount_tier": gap_info["current_discount_tier"],
        "next_discount_tier": gap_info["next_discount_tier"],
        "gap_to_next_tier_pen": gap_info["gap_to_next_tier_pen"],
        "product_promos": gap_info["product_promos"][:3],
        "confirmed_total_pen": gap_info["confirmed_total_pen"],
        "draft_total_pen": gap_info["draft_total_pen"],
    }
