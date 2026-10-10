"""
src/agent_service/core/stores/product/product_hierarchy.py - Resolver Jerárquico Universal de Productos y Variantes.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class ProductVariantView:
    """Representa una faceta o variante concreta autorizada de un producto."""
    product_id: int
    sku: str
    name: str
    price: Optional[float] = None
    currency: str = "PEN"
    attribute_details: str = ""
    pagina: Optional[int] = None
    edicion: Optional[str] = None
    marca: Optional[str] = None
    vendor_name: Optional[str] = None


@dataclass
class ProductCanonicalFamily:
    """Representa el concepto canónico de producto (familia/línea base) con sus facetas."""
    family_name: str
    marca: str = ""
    vendor_name: Optional[str] = None
    summary_description: str = ""
    template_id: Optional[int] = None
    variants: List[ProductVariantView] = field(default_factory=list)


def extract_base_concept_name(raw_name: str) -> str:
    """Extrae el nombre conceptual base de un producto eliminando sufijos de tono, gramaje o SPF.
    Opera de forma 100% genérica sin sesgos de categoría."""
    if not raw_name:
        return "Producto"
    s = str(raw_name).strip()
    
    # Remover traducciones JSON si las hay
    if s.startswith("{") and "en_US" in s:
        import json
        try:
            d = json.loads(s)
            s = d.get("es_PE") or d.get("es_ES") or d.get("en_US") or s
        except Exception:
            pass

    # Limpiar sufijos típicos de variantes de forma universal:
    s_clean = re.sub(r"\s+Cont\.\s*\d+.*$", "", s, flags=re.IGNORECASE)
    s_clean = re.sub(r"\s+[-–—]\s+.*$", "", s_clean)
    s_clean = re.sub(r"\s+Tono\s+.*$", "", s_clean, flags=re.IGNORECASE)
    return s_clean.strip() or s


def extract_attribute_details(name: str, desc: str) -> str:
    """Sintetiza los diferenciadores de atributo (tono, tamaño, activo, protección) de una variante."""
    parts = []
    
    # Extraer gramaje/volumen
    size_match = re.search(r"\b(?:Cont\.?\s*)?(\d+\s*(?:g|gr|ml|oz|fl\.?\s*oz))\b", f"{name} {desc}", re.IGNORECASE)
    if size_match:
        parts.append(size_match.group(1).strip())

    # Extraer SPF
    spf_match = re.search(r"\b(SPF\s*\d+|FPS\s*\d+)\b", f"{name} {desc}", re.IGNORECASE)
    if spf_match:
        parts.append(spf_match.group(1).strip())

    # Extraer activos o notas relevantes
    if desc:
        clean_desc = desc.split("\n")[0].strip()
        if len(clean_desc) > 60:
            clean_desc = clean_desc[:57] + "..."
        if clean_desc and clean_desc not in parts:
            parts.append(clean_desc)

    return " | ".join(parts) if parts else (desc or name)


class UniversalProductHierarchyResolver:
    """Resuelve y agrupa candidatos planos del catálogo en conceptos canónicos y variantes autorizadas."""

    def __init__(self, pool: Optional[AsyncConnectionPool] = None):
        self._pool = pool

    async def group_candidates_hierarchically(
        self,
        candidate_products: List[Dict[str, Any]],
        user_id: Optional[int] = None,
    ) -> List[ProductCanonicalFamily]:
        """Agrupa los productos candidatos por concepto canónico preservando y enriqueciendo variantes."""
        if not candidate_products:
            return []

        # 1. Agrupar candidatos iniciales por concepto base o template_id
        families_map: Dict[str, ProductCanonicalFamily] = {}
        seen_skus: Set[str] = set()

        for p in candidate_products:
            if not isinstance(p, dict):
                continue
            
            p_id = int(p.get("product_id") or p.get("id") or 0)
            sku = str(p.get("sku") or p.get("default_code") or "").strip()
            name = str(p.get("name") or "Producto").strip()
            marca = str(p.get("marca") or "").strip()
            vendor = p.get("vendor_name")
            desc = str(p.get("description") or p.get("description_sale") or "").strip()
            price = p.get("price")
            currency = p.get("currency") or "PEN"
            pagina = p.get("pagina")
            edicion = p.get("edicion")
            tmpl_id = p.get("product_tmpl_id")

            # Determinar clave de familia
            base_name = extract_base_concept_name(name)
            family_key = f"{marca}::{base_name}".lower()

            if family_key not in families_map:
                families_map[family_key] = ProductCanonicalFamily(
                    family_name=base_name,
                    marca=marca,
                    vendor_name=vendor,
                    summary_description=desc,
                    template_id=tmpl_id,
                    variants=[],
                )

            if sku and sku not in seen_skus:
                seen_skus.add(sku)
                attr_detail = extract_attribute_details(name, desc)
                variant = ProductVariantView(
                    product_id=p_id,
                    sku=sku,
                    name=name,
                    price=float(price) if price is not None else None,
                    currency=currency,
                    attribute_details=attr_detail,
                    pagina=pagina,
                    edicion=edicion,
                    marca=marca,
                    vendor_name=vendor,
                )
                families_map[family_key].variants.append(variant)

        return list(families_map.values())
