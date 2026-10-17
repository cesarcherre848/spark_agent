"""
src/agent_service/tools/incentive_tools.py - Herramientas para la consulta y cálculo de incentivos y tendencias en Odoo ERP

Consume la API REST /api/v1/sales_incentives y el modelo sale.incentive de Odoo ERP,
particionando métricas de venta a nivel de línea por proveedor (Unique S.A. vs CETCO S.A.),
minando tendencias de compra orgánicas y calculando brechas de incentivos tanto en modo
multimarca agrupado como filtrado por marca.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.sales_tools import (
    get_shared_odoo_client,
    odoo_list_sales_orders,
)

logger = logging.getLogger(__name__)

# Definición canónica de marcas, alias y escalas de descuento por proveedor
SUPPLIER_ALIASES: Dict[int, Dict[str, Any]] = {
    9: {
        "id": 9,
        "name": "Unique S.A.",
        "display_name": "Yanbal (Unique)",
        "default_campaign": "C11",
        "aliases": [
            "unique", "yanbal", "yambal", "yanval", "yamval",
            "unique s.a.", "yanbal unique", "unike",
        ],
        "discount_tiers": [
            {"tier": "25% de Descuento", "min_amount": 230.0},
            {"tier": "30% de Descuento", "min_amount": 720.0},
            {"tier": "35% de Descuento", "min_amount": 1500.0},
        ],
        "base_tier": "20% (Base)",
    },
    17: {
        "id": 17,
        "name": "CETCO S.A.",
        "display_name": "Belcorp (Ésika / Cyzone / L'Bel)",
        "default_campaign": "C-15",
        "aliases": [
            "cetco", "belcorp", "velcorp", "esika", "ésika", "esica", "ezika",
            "lbel", "l'bel", "cyzone", "saizone", "cetco s.a.",
        ],
        "discount_tiers": [],
        "base_tier": None,
    },
}

_SKU_SUPPLIER_CACHE: Dict[str, int] = {}


def get_supplier_default_campaign(supplier_id: Optional[int]) -> Optional[str]:
    """Obtiene la campaña comercial activa por defecto para un proveedor específico."""
    if not supplier_id:
        return None
    supplier_info = SUPPLIER_ALIASES.get(int(supplier_id))
    if supplier_info:
        return supplier_info.get("default_campaign")
    return None


def detect_campaign_from_query(query: Optional[str]) -> Optional[str]:
    """Detecta si la consulta del usuario menciona explícitamente una campaña o edición (ej: 'C11', 'C-15', 'campaña 10')."""
    if not query or not isinstance(query, str):
        return None
    # 1. Patrón directo C-15, C11, C10
    m_direct = re.search(r"\b(C-?\d{1,2})\b", query, re.IGNORECASE)
    if m_direct:
        raw = m_direct.group(1).upper()
        if raw.startswith("C-"):
            return raw
        return f"C{raw[1:]}"
    # 2. Patrón compuesto 'campaña 11', 'campaña 10'
    m_word = re.search(r"\bcampaña\s*(\d{1,2})\b", query, re.IGNORECASE)
    if m_word:
        return f"C{m_word.group(1)}"
    return None


def normalize_incentive_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae y normaliza un diccionario estructurado de metadatos para un registro de incentivo."""
    tags_raw = record.get("tags") or []
    tag_names = []
    for t in tags_raw:
        if isinstance(t, dict) and "name" in t:
            tag_names.append(t["name"])
        elif isinstance(t, str):
            tag_names.append(t)

    campaign = None
    pages = []
    for tn in tag_names:
        m_camp = re.match(r"^(C-?\d{1,2})$", tn, re.IGNORECASE)
        if m_camp:
            campaign = m_camp.group(1).upper()
        m_pag = re.match(r"^Pág\s*(\d+)$", tn, re.IGNORECASE)
        if m_pag:
            pages.append(int(m_pag.group(1)))

    title = record.get("title") or record.get("name") or ""
    obs = record.get("observations") or ""
    desc = record.get("description") or ""
    full_text = f"{title} {desc} {obs}"

    # Si no se encontró campaña en los tags, buscar en el texto
    if not campaign:
        m_camp_text = re.search(r"\b(C-?\d{1,2})\b|\bcampaña\s*(\d{1,2})\b", full_text, re.IGNORECASE)
        if m_camp_text:
            c_val = m_camp_text.group(1) or f"C{m_camp_text.group(2)}"
            campaign = c_val.upper()

    # Códigos de catálogo / SKUs
    skus = list(dict.fromkeys(re.findall(r"Cód\.?\s*(?:Catálogo\s*(?:Yanbal\s*)?)?(\d+)", full_text)))

    # Categoría
    t_lower = title.lower()
    cat = "general"
    if "escala" in t_lower or "descuento" in t_lower:
        cat = "discount_tier"
    elif "vende y gana" in t_lower:
        cat = "vende_y_gana"
    elif "ganamás" in t_lower or "ganamas" in t_lower:
        cat = "bonificacion_ganamas"
    elif "capitalización" in t_lower or "capitalizacion" in t_lower:
        cat = "capitalizacion"
    elif "demostrador" in t_lower:
        cat = "demostradores"

    brand = "Yanbal" if record.get("supplier_id") == 9 or "Yanbal" in tag_names else "Belcorp"

    return {
        "campaign": campaign,
        "brand": brand,
        "pages": sorted(pages),
        "category": cat,
        "tag_names": tag_names,
        "tag_ids": record.get("tag_ids") or [],
        "skus": skus,
    }


def detect_target_supplier(query: Optional[str]) -> Optional[Dict[str, Any]]:
    """Detecta si la consulta del usuario menciona una marca o proveedor específico,
    tolerando variaciones fonéticas y errores de tipeo habituales (ej: 'yambal' -> Yanbal).
    """
    if not query or not isinstance(query, str):
        return None
    q_norm = query.lower()
    # Normalización fonética y remoción de acentos
    q_norm = re.sub(r"[áàä]", "a", q_norm)
    q_norm = re.sub(r"[éèë]", "e", q_norm)
    q_norm = re.sub(r"[íìï]", "i", q_norm)
    q_norm = re.sub(r"[óòö]", "o", q_norm)
    q_norm = re.sub(r"[úùü]", "u", q_norm)
    q_norm = re.sub(r"\byambal\b", "yanbal", q_norm)
    q_norm = re.sub(r"\byamval\b", "yanbal", q_norm)
    q_norm = re.sub(r"\byanval\b", "yanbal", q_norm)
    q_norm = re.sub(r"\bezika\b", "esika", q_norm)
    q_norm = re.sub(r"\besica\b", "esika", q_norm)

    for sup_id, meta in SUPPLIER_ALIASES.items():
        for alias in meta["aliases"]:
            alias_norm = alias.lower()
            alias_norm = re.sub(r"[áéíóú]", "a", alias_norm)
            pattern = r"(?<!\w)" + re.escape(alias_norm) + r"(?!\w)"
            if re.search(pattern, q_norm):
                return meta
    return None


async def get_sku_supplier_map(client: Optional[OdooClient] = None) -> Dict[str, int]:
    """Obtiene el mapa de SKU -> supplier_id desde Odoo ERP con caché en memoria."""
    global _SKU_SUPPLIER_CACHE
    if _SKU_SUPPLIER_CACHE:
        return _SKU_SUPPLIER_CACHE

    c = get_shared_odoo_client(client)
    post_fn = getattr(getattr(c, "_client", None), "post", None)
    if hasattr(post_fn, "side_effect") and post_fn.side_effect is not None:
        return _SKU_SUPPLIER_CACHE

    if not getattr(c, "_uid", None):
        try:
            await c.authenticate()
        except Exception:
            pass

    try:
        res = await c._client.post(
            "/web/dataset/call_kw",
            json={
                "jsonrpc": "2.0",
                "method": "call",
                "params": {
                    "model": "product.supplierinfo",
                    "method": "search_read",
                    "args": [[]],
                    "kwargs": {"fields": ["id", "partner_id", "product_tmpl_id"]},
                },
                "id": 501,
            },
        )
        if res.status_code == 200:
            items = res.json().get("result", []) or []
            mapping = {}
            for it in items:
                tmpl = it.get("product_tmpl_id")
                if tmpl and isinstance(tmpl, (list, tuple)) and len(tmpl) > 1:
                    m = re.search(r"\[(.*?)\]", str(tmpl[1]))
                    if m:
                        sku = m.group(1).strip()
                        p = it.get("partner_id")
                        pid = p[0] if isinstance(p, (list, tuple)) and p else None
                        if sku and pid:
                            mapping[sku] = pid
            if mapping:
                _SKU_SUPPLIER_CACHE = mapping
                return _SKU_SUPPLIER_CACHE
    except Exception as exc:
        logger.warning(f"Error consultando product.supplierinfo en Odoo: {exc}")

    return _SKU_SUPPLIER_CACHE


def resolve_supplier_sales_metrics(
    orders_data: Any,
    sku_supplier_map: Optional[Dict[str, int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Particiona las órdenes de venta a nivel de línea por proveedor (Unique S.A. vs CETCO S.A.)."""
    sku_map = sku_supplier_map if sku_supplier_map is not None else _SKU_SUPPLIER_CACHE

    if isinstance(orders_data, dict):
        confirmed_orders = orders_data.get("sale", []) or []
        draft_orders = orders_data.get("draft", []) or []
    elif isinstance(orders_data, list):
        confirmed_orders = [o for o in orders_data if o.get("state") in ("sale", "done")]
        draft_orders = [o for o in orders_data if o.get("state") in ("draft", "sent")]
    else:
        confirmed_orders, draft_orders = [], []

    metrics: Dict[int, Dict[str, Any]] = {}
    for sid, meta in SUPPLIER_ALIASES.items():
        metrics[sid] = {
            "supplier_id": sid,
            "supplier_name": meta["name"],
            "brand_display_name": meta["display_name"],
            "confirmed_total_pen": 0.0,
            "draft_total_pen": 0.0,
            "potential_total_pen": 0.0,
            "confirmed_lines": [],
            "draft_lines": [],
            "orders_count": 0,
        }

    seen_orders = set()

    def _extract_sku_and_subtotal(line: Dict[str, Any]) -> tuple:
        pname = str(line.get("product_name") or line.get("name") or "")
        sku = ""
        m = re.search(r"\[(.*?)\]", pname)
        if m:
            sku = m.group(1).strip()
        if not sku:
            pid = line.get("product_id")
            if isinstance(pid, (list, tuple)) and len(pid) > 1:
                m2 = re.search(r"\[(.*?)\]", str(pid[1]))
                if m2:
                    sku = m2.group(1).strip()
        subtotal = float(
            line.get("price_subtotal")
            or (float(line.get("price_unit") or 0.0) * float(line.get("quantity") or line.get("product_uom_qty") or 1.0))
        )
        return sku, subtotal

    def _process_orders(order_list: List[Dict[str, Any]], is_confirmed: bool):
        for o in order_list:
            oid = o.get("id") or o.get("name")
            order_amount = float(o.get("amount_total") or 0.0)
            lines = o.get("lines", []) or []

            if not lines:
                # Si la orden no tiene líneas desglosadas, asignar al proveedor por defecto (Unique ID 9)
                target_field = "confirmed_total_pen" if is_confirmed else "draft_total_pen"
                metrics[9][target_field] += order_amount
                if is_confirmed and oid and oid not in seen_orders:
                    metrics[9]["orders_count"] += 1
                    seen_orders.add(oid)
                continue

            order_supplier_subtotals: Dict[int, float] = {}
            for line in lines:
                sku, subtotal = _extract_sku_and_subtotal(line)
                sid = sku_map.get(sku, 9)
                if sid not in metrics:
                    metrics[sid] = {
                        "supplier_id": sid,
                        "supplier_name": f"Proveedor {sid}",
                        "brand_display_name": f"Proveedor {sid}",
                        "confirmed_total_pen": 0.0,
                        "draft_total_pen": 0.0,
                        "potential_total_pen": 0.0,
                        "confirmed_lines": [],
                        "draft_lines": [],
                        "orders_count": 0,
                    }
                order_supplier_subtotals[sid] = order_supplier_subtotals.get(sid, 0.0) + subtotal
                if is_confirmed:
                    metrics[sid]["confirmed_lines"].append(line)
                else:
                    metrics[sid]["draft_lines"].append(line)

            total_subtotal = sum(order_supplier_subtotals.values())
            for sid, s_subtotal in order_supplier_subtotals.items():
                if total_subtotal > 0 and order_amount > 0:
                    allocated = (s_subtotal / total_subtotal) * order_amount
                else:
                    allocated = s_subtotal

                if is_confirmed:
                    metrics[sid]["confirmed_total_pen"] += allocated
                    if oid and oid not in seen_orders:
                        metrics[sid]["orders_count"] += 1
                        seen_orders.add(oid)
                else:
                    metrics[sid]["draft_total_pen"] += allocated

    _process_orders(confirmed_orders, is_confirmed=True)
    _process_orders(draft_orders, is_confirmed=False)

    for sid in metrics:
        metrics[sid]["confirmed_total_pen"] = round(metrics[sid]["confirmed_total_pen"], 2)
        metrics[sid]["draft_total_pen"] = round(metrics[sid]["draft_total_pen"], 2)
        metrics[sid]["potential_total_pen"] = round(
            metrics[sid]["confirmed_total_pen"] + metrics[sid]["draft_total_pen"], 2
        )

    return metrics


async def odoo_get_active_incentives(
    supplier_id: Optional[int] = None,
    supplier_name: Optional[str] = None,
    campaign: Optional[str] = None,
    tag: Optional[str] = None,
    tag_ids: Optional[List[int]] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    validity_date: Optional[str] = None,
    only_active: bool = True,
    use_supplier_default_campaign: bool = False,
    limit: int = 50,
    offset: int = 0,
    client: Optional[OdooClient] = None,
) -> List[Dict[str, Any]]:
    """Consulta los planes de incentivo vigentes en Odoo ERP por proveedor con soporte para
    filtrado por metadata, etiquetas (tag, tag_ids) y campaña por defecto.
    """
    c = get_shared_odoo_client(client)
    if not getattr(c, "_uid", None):
        await c.authenticate()

    # Resolución de campaña por defecto si se solicita
    resolved_campaign = campaign
    if resolved_campaign is None and use_supplier_default_campaign and supplier_id is not None:
        resolved_campaign = get_supplier_default_campaign(int(supplier_id))

    # Determinar etiqueta de búsqueda para API REST
    search_tag = tag or resolved_campaign

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
    if search_tag:
        params["tag"] = search_tag
    if tag_ids:
        params["tag_ids"] = ",".join(str(i) for i in tag_ids) if isinstance(tag_ids, (list, tuple)) else str(tag_ids)

    def _matches_metadata(record_meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for k, v in filters.items():
            if k == "category":
                if str(v).lower() != str(record_meta.get("category", "")).lower():
                    return False
            elif k == "campaign":
                if str(v).upper() != str(record_meta.get("campaign", "")).upper():
                    return False
            elif k in ("pages", "page"):
                req_page = v if isinstance(v, (list, tuple, set)) else [v]
                rec_pages = record_meta.get("pages", [])
                if not any(int(p) in rec_pages for p in req_page):
                    return False
            elif k in ("skus", "sku"):
                req_sku = v if isinstance(v, (list, tuple, set)) else [v]
                rec_skus = record_meta.get("skus", [])
                if not any(str(s) in rec_skus for s in req_sku):
                    return False
            elif k == "brand":
                if str(v).lower() != str(record_meta.get("brand", "")).lower():
                    return False
            else:
                if record_meta.get(k) != v:
                    return False
        return True

    records: List[Dict[str, Any]] = []
    rest_succeeded = False

    try:
        response = await c._client.get("/api/v1/sales_incentives", params=params)
        if response.status_code == 200:
            data = response.json()
            raw_data = []
            if isinstance(data, dict) and "data" in data:
                raw_data = data["data"]
            elif isinstance(data, list):
                raw_data = data
            for r in raw_data:
                r["metadata"] = normalize_incentive_metadata(r)
                records.append(r)
            rest_succeeded = True
    except Exception as exc:
        logger.warning(f"Error consultando REST /api/v1/sales_incentives: {exc}. Intentando fallback JSON-RPC.")

    if not rest_succeeded:
        # Fallback JSON-RPC call_kw sobre sale.incentive
        domain: List[Any] = []
        if only_active:
            domain.append(["state", "=", "active"])
            domain.append(["is_valid", "=", True])
        if supplier_id is not None:
            domain.append(["partner_id", "=", int(supplier_id)])
        elif supplier_name and supplier_name.strip():
            domain.append(["partner_id.name", "ilike", supplier_name.strip()])
        if tag_ids:
            domain.append(["tag_ids", "in", tag_ids])

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
                        "tag_ids",
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
            for r in raw_records:
                partner = r.get("partner_id") or [None, "Sin proveedor"]
                p_id = partner[0] if isinstance(partner, (list, tuple)) else partner
                p_name = partner[1] if isinstance(partner, (list, tuple)) and len(partner) > 1 else str(partner)
                norm_item = {
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
                    "tag_ids": r.get("tag_ids") or [],
                    "tags": [{"id": tid, "name": f"Tag_{tid}"} for tid in (r.get("tag_ids") or [])],
                }
                norm_item["metadata"] = normalize_incentive_metadata(norm_item)
                records.append(norm_item)
        except Exception as exc:
            logger.error(f"Error en fallback JSON-RPC sobre sale.incentive: {exc}")
            records = []

    # Filtrado por resolved_campaign si REST o JSON-RPC devolvieron registros sin filtrar
    if resolved_campaign:
        records = [
            r for r in records
            if (r.get("metadata", {}).get("campaign") or "").upper() == resolved_campaign.upper()
            or resolved_campaign.upper() in [str(t.get("name", "")).upper() for t in r.get("tags", []) if isinstance(t, dict)]
            or resolved_campaign.upper() in [str(t).upper() for t in r.get("tags", []) if isinstance(t, str)]
        ]

    # Filtrado por metadata_filters si se especificaron
    if metadata_filters:
        records = [r for r in records if _matches_metadata(r.get("metadata", {}), metadata_filters)]

    return records


def extract_sales_trends_from_lines(
    lines: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Extrae los productos más vendidos a partir de una lista de líneas de pedido."""
    stats_by_sku: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        pname = str(line.get("product_name") or line.get("name") or "")
        sku = ""
        m = re.search(r"\[(.*?)\]", pname)
        if m:
            sku = m.group(1).strip()
            prod_name = pname.replace(f"[{sku}]", "").strip()
        else:
            pid = line.get("product_id")
            if isinstance(pid, (list, tuple)) and len(pid) > 1:
                m2 = re.search(r"\[(.*?)\]", str(pid[1]))
                if m2:
                    sku = m2.group(1).strip()
                    prod_name = str(pid[1]).replace(f"[{sku}]", "").strip()
                else:
                    prod_name = str(pid[1])
            else:
                prod_name = pname

        if not sku:
            sku = str(line.get("product_id") or "SKU-N/A")

        qty = float(line.get("quantity") or line.get("product_uom_qty") or 1.0)
        subtotal = float(line.get("price_subtotal") or 0.0)

        if sku not in stats_by_sku:
            stats_by_sku[sku] = {
                "sku": sku,
                "name": prod_name or f"Producto {sku}",
                "total_qty": 0.0,
                "order_count": 0,
                "total_subtotal": 0.0,
            }
        stats_by_sku[sku]["total_qty"] += qty
        stats_by_sku[sku]["order_count"] += 1
        stats_by_sku[sku]["total_subtotal"] += subtotal

    ranked = sorted(
        stats_by_sku.values(),
        key=lambda x: (x["order_count"], x["total_qty"], x["total_subtotal"]),
        reverse=True,
    )
    return ranked[:top_k]


def extract_sales_trends(
    orders_data: Any,
    supplier_id: Optional[int] = None,
    top_k: int = 5,
    sku_supplier_map: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Extrae las tendencias de compra de pedidos, opcionalmente filtradas por proveedor."""
    if supplier_id is not None:
        metrics = resolve_supplier_sales_metrics(orders_data, sku_supplier_map=sku_supplier_map)
        s_data = metrics.get(supplier_id)
        if s_data:
            lines = s_data["confirmed_lines"] + s_data["draft_lines"]
            return extract_sales_trends_from_lines(lines, top_k=top_k)
        return []

    # Si no hay proveedor especificado, extraer de todas las órdenes
    if isinstance(orders_data, dict):
        all_orders = (orders_data.get("sale", []) or []) + (orders_data.get("draft", []) or [])
    elif isinstance(orders_data, list):
        all_orders = orders_data
    else:
        all_orders = []

    all_lines = []
    for order in all_orders:
        all_lines.extend(order.get("lines", []) or [])

    return extract_sales_trends_from_lines(all_lines, top_k=top_k)


def calculate_multibrand_incentive_gaps(
    orders_data: Any,
    incentive_plans: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
    sku_supplier_map: Optional[Dict[str, int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Calcula el estado actual y las brechas hacia los objetivos de incentivos
    agrupados por cada proveedor (Unique S.A. y CETCO S.A.).
    """
    supplier_metrics = resolve_supplier_sales_metrics(orders_data, sku_supplier_map=sku_supplier_map)
    now = reference_date or datetime.now()

    # Agrupar planes por supplier_id
    plans_by_supplier: Dict[int, List[Dict[str, Any]]] = {}
    for p in incentive_plans:
        sid = p.get("supplier_id") or p.get("partner_id")
        if isinstance(sid, (list, tuple)) and len(sid) > 0:
            sid = sid[0]
        if not sid and (p.get("supplier_name") or p.get("partner_name")):
            sname = str(p.get("supplier_name") or p.get("partner_name"))
            matched = detect_target_supplier(sname)
            if matched:
                sid = matched["id"]
        if not sid:
            sid = 9  # Fallback a Unique S.A.
        plans_by_supplier.setdefault(int(sid), []).append(p)

    results_by_supplier: Dict[int, Dict[str, Any]] = {}

    for sid, meta in SUPPLIER_ALIASES.items():
        s_data = supplier_metrics.get(sid, {
            "confirmed_total_pen": 0.0,
            "draft_total_pen": 0.0,
            "potential_total_pen": 0.0,
            "confirmed_lines": [],
            "draft_lines": [],
            "orders_count": 0,
        })
        confirmed_total = s_data["confirmed_total_pen"]
        draft_total = s_data["draft_total_pen"]
        potential_total = s_data["potential_total_pen"]
        s_plans = plans_by_supplier.get(sid, [])

        discount_tiers = meta.get("discount_tiers") or []
        if discount_tiers:
            current_tier = meta.get("base_tier", "20% (Base)")
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
                        next_tier = "Escala Máxima Alcanzada (35%)"
                        gap_to_next_tier_pen = 0.0
                        next_tier_target_pen = t["min_amount"]
        else:
            # Caso de proveedores sin escalas fijas (ej. CETCO): solo evaluar si hay planes activos en Odoo
            has_rappel_plan = any("rappel" in (p.get("title") or p.get("name") or "").lower() for p in s_plans)
            if has_rappel_plan:
                current_tier = "Rappel Comercial Vigente"
                next_tier = "Cumplimiento Cuota Anual Rappel 5%"
                next_tier_target_pen = 2000.0  # Referencial si el plan está activo
                gap_to_next_tier_pen = max(0.0, next_tier_target_pen - confirmed_total)
            elif s_plans:
                current_tier = "Promociones Activas"
                next_tier = s_plans[0].get("title") or s_plans[0].get("name") or "Meta de Campaña"
                next_tier_target_pen = 0.0
                gap_to_next_tier_pen = 0.0
            else:
                # Cero planes activos en Odoo para este proveedor
                current_tier = "Sin promociones activas"
                next_tier = "Sin metas vigentes"
                next_tier_target_pen = 0.0
                gap_to_next_tier_pen = 0.0

        # Promociones de producto del proveedor
        product_promos = []
        for plan in s_plans:
            title = plan.get("title") or plan.get("name") or ""
            obs = plan.get("observations") or ""
            desc = plan.get("description") or ""
            combined_text = f"{title} {obs} {desc}"

            codes_found = re.findall(r"\b(?:C[oó]d\.?|SKU)?\s*(\d{3,5})\b", combined_text, re.IGNORECASE)
            participating_skus = [c for c in set(codes_found) if c not in ("230", "470", "720", "1500", "2000", "3400", "3500")]
            qty_match = re.search(r"(\d+)\s*(?:o más)?\s*unidades", combined_text, re.IGNORECASE)
            req_units = int(qty_match.group(1)) if qty_match else None

            if "vende y gana" in combined_text.lower() or "gratis" in combined_text.lower() or "rappel" in combined_text.lower():
                product_promos.append({
                    "id": plan.get("id"),
                    "title": title,
                    "reward": plan.get("description") or title,
                    "skus": participating_skus,
                    "required_units": req_units or 3,
                    "supplier_name": meta["name"],
                    "brand_display_name": meta["display_name"],
                })

        # Días restantes
        days_remaining = None
        for plan in s_plans:
            date_end_str = plan.get("date_end")
            if date_end_str:
                try:
                    dt_end = datetime.fromisoformat(str(date_end_str))
                    delta = (dt_end - now).days
                    if days_remaining is None or delta < days_remaining:
                        days_remaining = max(0, delta)
                except Exception:
                    pass

        # Tendencias locales del proveedor
        all_s_lines = s_data["confirmed_lines"] + s_data["draft_lines"]
        s_trends = extract_sales_trends_from_lines(all_s_lines, top_k=5)

        results_by_supplier[sid] = {
            "supplier_id": sid,
            "supplier_name": meta["name"],
            "brand_display_name": meta["display_name"],
            "confirmed_total_pen": round(confirmed_total, 2),
            "draft_total_pen": round(draft_total, 2),
            "potential_total_pen": round(potential_total, 2),
            "current_discount_tier": current_tier,
            "next_discount_tier": next_tier,
            "next_tier_target_pen": round(next_tier_target_pen, 2),
            "gap_to_next_tier_pen": round(gap_to_next_tier_pen, 2),
            "product_promos": product_promos,
            "has_active_incentives": bool(s_plans or discount_tiers),
            "trends": s_trends,
            "days_remaining": days_remaining,
            "orders_count": s_data.get("orders_count", 0),
        }

    return results_by_supplier


def calculate_incentive_gap(
    orders_data: Any,
    incentive_plans: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
    supplier_id: Optional[int] = None,
    sku_supplier_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Calcula el estado de ventas y brechas hacia incentivos.
    
    Si supplier_id está especificado, retorna las métricas exactas de ese proveedor.
    Si supplier_id es None, retorna el balance multimarca agrupado preservando
    la compatibilidad de campos generales.
    """
    multibrand_gaps = calculate_multibrand_incentive_gaps(
        orders_data,
        incentive_plans,
        reference_date=reference_date,
        sku_supplier_map=sku_supplier_map,
    )

    if supplier_id is not None and supplier_id in multibrand_gaps:
        single = dict(multibrand_gaps[supplier_id])
        single["is_multibrand"] = False
        single["multibrand_gaps"] = multibrand_gaps
        return single

    # Modo Multimarca Agrupado: totalizar métricas y consolidar promociones
    total_confirmed = sum(m["confirmed_total_pen"] for m in multibrand_gaps.values())
    total_draft = sum(m["draft_total_pen"] for m in multibrand_gaps.values())
    total_potential = total_confirmed + total_draft

    all_promos = []
    min_days_remaining = None
    for m in multibrand_gaps.values():
        all_promos.extend(m.get("product_promos", []))
        d = m.get("days_remaining")
        if d is not None:
            if min_days_remaining is None or d < min_days_remaining:
                min_days_remaining = d

    # Usar métricas de Unique (ID 9) como referencia principal si no hay filtro
    unique_data = multibrand_gaps.get(9, {})

    return {
        "is_multibrand": True,
        "confirmed_total_pen": round(total_confirmed, 2),
        "draft_total_pen": round(total_draft, 2),
        "potential_total_pen": round(total_potential, 2),
        "current_discount_tier": unique_data.get("current_discount_tier", "20% (Base)"),
        "next_discount_tier": unique_data.get("next_discount_tier", "25% de Descuento"),
        "next_tier_target_pen": unique_data.get("next_tier_target_pen", 230.0),
        "gap_to_next_tier_pen": unique_data.get("gap_to_next_tier_pen", 0.0),
        "product_promos": all_promos,
        "days_remaining": min_days_remaining,
        "multibrand_gaps": multibrand_gaps,
    }


async def get_active_incentive_targets(
    user_id: int = 5,
    supplier_id: Optional[int] = None,
    client: Optional[OdooClient] = None,
) -> Dict[str, Any]:
    """Herramienta puente para inyectar metas de incentivo y tendencias reales
    en el recomendador de productos (product_advisor), soportando filtros por marca.
    """
    c = get_shared_odoo_client(client)
    orders_data = await odoo_list_sales_orders(user_id=user_id, client=c)
    incentives = await odoo_get_active_incentives(supplier_id=supplier_id, only_active=True, client=c)
    sku_map = await get_sku_supplier_map(client=c)
    trends = extract_sales_trends(orders_data, supplier_id=supplier_id, top_k=5, sku_supplier_map=sku_map)
    gap_info = calculate_incentive_gap(orders_data, incentives, supplier_id=supplier_id, sku_supplier_map=sku_map)

    bonus_skus = []
    for promo in gap_info.get("product_promos", []):
        bonus_skus.extend(promo.get("skus", []))

    # SKUs conocidos de Unique para Vende y Gana
    if supplier_id is None or supplier_id == 9:
        default_bonus_skus = ["2210", "2218", "2219", "2220", "897", "161", "152", "143", "2035", "233", "2041", "202"]
        for s in default_bonus_skus:
            if s not in bonus_skus:
                bonus_skus.append(s)

    trending_skus = [t["sku"] for t in trends if t.get("sku")]

    return {
        "user_id": user_id,
        "supplier_id": supplier_id,
        "is_multibrand": gap_info.get("is_multibrand", False),
        "bonus_skus": list(set(bonus_skus)),
        "trending_skus": trending_skus,
        "top_trends": trends,
        "current_discount_tier": gap_info.get("current_discount_tier"),
        "next_discount_tier": gap_info.get("next_discount_tier"),
        "gap_to_next_tier_pen": gap_info.get("gap_to_next_tier_pen"),
        "product_promos": gap_info.get("product_promos", [])[:3],
        "confirmed_total_pen": gap_info.get("confirmed_total_pen"),
        "draft_total_pen": gap_info.get("draft_total_pen"),
        "multibrand_gaps": gap_info.get("multibrand_gaps"),
    }
