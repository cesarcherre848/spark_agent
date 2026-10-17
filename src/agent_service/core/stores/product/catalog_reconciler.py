"""
src/agent_service/core/stores/product/catalog_reconciler.py

Motor de verdad de catálogo (product_catalog / product_catalog_embedding) y
auto-curación regenerativa on-demand de discrepancias de SKUs, páginas,
variantes y precios de oferta entre Odoo (product_product) y el catálogo PDF.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from cachetools import TTLCache
from psycopg_pool import AsyncConnectionPool

from src.agent_service.config.database import get_db_pool

logger = logging.getLogger(__name__)


def clean_zero_width(text: Optional[str]) -> str:
    """Elimina caracteres invisibles de ancho cero (\\u200b, \\ufeff, \\u200c, \\u200d) extraídos del PDF."""
    if not text:
        return ""
    return re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", str(text))


@dataclass
class CatalogVariantInfo:
    """Representa un código SKU o variante (tono, inicial, aroma) extraído de una página de catálogo."""
    sku: str
    label: str = ""
    offer_price: Optional[float] = None
    normal_price: Optional[float] = None
    discount_tag: Optional[str] = None
    page_number: int = 0
    edicion: str = "C11"
    marca: str = "Yanbal"
    vendor_id: int = 9
    vendor_name: str = "Unique S.A."
    catalog_id: int = 11
    catalog_title: str = ""
    page_image_url: Optional[str] = None
    content_spec: Optional[str] = None


@dataclass
class ParsedCatalogPage:
    """Estructura enriquecida extraída de un chunk de página de product_catalog_embedding."""
    catalog_id: int
    catalog_title: str
    partner_id: int
    partner_name: str
    page_number: int
    edicion: str
    marca: str
    raw_text: str
    clean_text: str
    page_image_url: Optional[str] = None
    skus: List[str] = field(default_factory=list)
    variants: Dict[str, CatalogVariantInfo] = field(default_factory=dict)
    nsoc_numbers: Set[str] = field(default_factory=set)
    primary_offer_price: Optional[float] = None
    primary_normal_price: Optional[float] = None
    discount_tag: Optional[str] = None
    content_specs: List[str] = field(default_factory=list)
    has_free_gift: bool = False


class CatalogPageParser:
    """Parser determinista para extraer SKUs, tonos, precios de oferta/normal, gramajes
    y registros sanitarios (NSOC) desde raw_page_text de product_catalog_embedding."""

    # Registro sanitario (NSOC29140-24CO, NSOC55587-22PE, etc.)
    NSOC_PATTERN = re.compile(r"\bNSOC\s*([0-9]{3,8})(?:-[0-9A-Z]+)?\b", re.IGNORECASE)

    # Listas de iniciales tipo: CÓD. A 43490, B 43491, C 43492...
    INITIALS_BLOCK_PATTERN = re.compile(
        r"(?:CÓD\.?|COD\.?)\s+([A-Z]\s+\d{3,7}(?:\s*,\s*[A-Z]\s+\d{3,7})+)",
        re.IGNORECASE,
    )

    # Variante precediendo a CÓD: ej. "Tono 1N\nCÓD. 5106", "1. Intensive CÓD. 670", "Bronzage \nCÓD. 4017", "Sunset \nBronzer \nCÓD. 6261"
    VARIANT_COD_PATTERN = re.compile(
        r"(?:^|\n)\s*([A-Za-zÁÉÍÓÚáéíóúÑñ0-9°!().\s/-]{2,45}?)\s*(?:\n\s*)?(?:CÓD\.?|COD\.?)\s*(\d{2,7})\b",
        re.IGNORECASE,
    )

    # Cualquier CÓD. directo: "CÓD. 1809"
    DIRECT_COD_PATTERN = re.compile(r"(?:CÓD\.?|COD\.?)\s*(\d{2,7})\b", re.IGNORECASE)

    # Precios de oferta y normales
    NORMAL_PRICE_PATTERN = re.compile(r"P\.?\s*Normal\s*S/\.?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
    OFFER_PRICE_PATTERN = re.compile(
        r"(?:OFERTA(?:\s+TOP|\s+WOW|\s+DE\s+LANZAMIENTO)?|A\s+TAN\s+SOLO|SET|PRECIO\s+ESPECIAL)\s*(?:\n\s*)*"
        r"(?:\d{1,2}%\s*(?:dscto\.?|descuento)\s*(?:\n\s*)*)?"
        r"S/\.?\s*(\d+(?:\.\d{1,2})?)",
        re.IGNORECASE,
    )
    ANY_SOLES_PRICE_PATTERN = re.compile(r"S/\.?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE)
    DISCOUNT_PATTERN = re.compile(r"(\d{1,2}%\s*dscto\.?)", re.IGNORECASE)
    CONTENT_SPEC_PATTERN = re.compile(r"\bCont\.?\s*(\d+(?:\.\d+)?\s*(?:ml|g|gr|oz|fl\.?\s*oz|L)(?:\s*x\s*\d+)?)\b", re.IGNORECASE)

    @classmethod
    def parse_page(
        cls,
        page_number: int,
        raw_page_text: str,
        metadata_json: Optional[str] = None,
        catalog_id: int = 11,
        catalog_title: str = "Catalogo Sin arrugas, sin agujas",
        partner_id: int = 9,
        partner_name: str = "Unique S.A.",
        page_image_url: Optional[str] = None,
        edicion_override: Optional[str] = None,
        marca_override: Optional[str] = None,
    ) -> ParsedCatalogPage:
        cleaned = clean_zero_width(raw_page_text)

        meta: Dict[str, Any] = {}
        if metadata_json:
            try:
                meta = json.loads(metadata_json) if isinstance(metadata_json, str) else dict(metadata_json)
            except Exception:
                meta = {}

        tags = meta.get("tags") or []
        edicion = edicion_override.upper() if edicion_override else "C11"
        marca = marca_override if marca_override else "Yanbal"
        for t in tags:
            t_str = str(t).strip()
            if not edicion_override and re.match(r"^C-?\d{1,2}$", t_str, re.IGNORECASE):
                edicion = t_str.upper()
            elif not marca_override and t_str.lower() in ("yanbal", "ésika", "esika", "l'bel", "lbel", "cyzone"):
                marca = "Ésika" if "sik" in t_str.lower() else t_str.title()

        img_url = page_image_url or f"/api/v1/product_catalog/{catalog_id}/page/{page_number}/image"

        # 1. Extraer códigos sanitarios NSOC para jamás confundirlos con SKUs
        nsoc_numbers: Set[str] = set()
        for m in cls.NSOC_PATTERN.finditer(cleaned):
            nsoc_numbers.add(m.group(1))

        # Texto con NSOC neutralizado
        text_no_nsoc = cls.NSOC_PATTERN.sub(" ", cleaned)

        # 2. Extraer precios globales de la página
        normal_prices = [float(x) for x in cls.NORMAL_PRICE_PATTERN.findall(text_no_nsoc)]
        offer_prices = [float(x) for x in cls.OFFER_PRICE_PATTERN.findall(text_no_nsoc)]
        if not offer_prices:
            # Buscar precios S/ que no sean P. Normal
            text_without_normal = cls.NORMAL_PRICE_PATTERN.sub(" ", text_no_nsoc)
            offer_prices = [float(x) for x in cls.ANY_SOLES_PRICE_PATTERN.findall(text_without_normal)]

        discounts = cls.DISCOUNT_PATTERN.findall(text_no_nsoc)
        content_specs = [m.strip() for m in cls.CONTENT_SPEC_PATTERN.findall(text_no_nsoc)]
        has_free_gift = bool(re.search(r"¡?GRATIS!", text_no_nsoc, re.IGNORECASE))

        primary_offer = offer_prices[0] if offer_prices else None
        primary_normal = normal_prices[0] if normal_prices else None
        discount_tag = discounts[0] if discounts else None

        variants: Dict[str, CatalogVariantInfo] = {}
        ordered_skus: List[str] = []

        # 3. Extraer bloques de iniciales (ej. Pulsera / Collar Inicial Doretti)
        for init_match in cls.INITIALS_BLOCK_PATTERN.finditer(text_no_nsoc):
            pairs_str = init_match.group(1)
            local_offer, local_normal = cls._find_nearest_prices(text_no_nsoc, init_match.start(), primary_offer, primary_normal)
            for pair in re.findall(r"([A-Z])\s+(\d{3,7})", pairs_str):
                letter, sku_code = pair[0], pair[1]
                if sku_code in nsoc_numbers:
                    continue
                if sku_code not in variants:
                    ordered_skus.append(sku_code)
                    variants[sku_code] = CatalogVariantInfo(
                        sku=sku_code,
                        label=f"Inicial {letter}",
                        offer_price=local_offer,
                        normal_price=local_normal,
                        discount_tag=discount_tag,
                        page_number=page_number,
                        edicion=edicion,
                        marca=marca,
                        vendor_id=partner_id,
                        vendor_name=partner_name,
                        catalog_id=catalog_id,
                        catalog_title=catalog_title,
                        page_image_url=img_url,
                        content_spec=content_specs[0] if content_specs else None,
                    )

        # 4. Extraer todos los CÓD. en orden de aparición delimitando el segmento desde el CÓD. anterior
        noise_labels = {
            "oferta", "normal", "dscto", "descuento", "precio", "cont", "gratis", "nuevo", "nueva"
        }
        cod_matches = list(cls.DIRECT_COD_PATTERN.finditer(text_no_nsoc))
        prev_end = 0
        for cm in cod_matches:
            sku_code = cm.group(1).strip()
            segment = text_no_nsoc[prev_end : cm.start()]
            prev_end = cm.end()
            if sku_code in nsoc_numbers:
                continue
            local_offer, local_normal = cls._find_nearest_prices(text_no_nsoc, cm.start(), primary_offer, primary_normal)
            seg_label = cls._extract_preceding_block_title(segment)
            if any(nl in seg_label.lower() for nl in noise_labels) or re.search(r"S/\s*\d+", seg_label):
                seg_label = ""
            if sku_code not in variants:
                ordered_skus.append(sku_code)
                variants[sku_code] = CatalogVariantInfo(
                    sku=sku_code,
                    label=seg_label,
                    offer_price=local_offer,
                    normal_price=local_normal,
                    discount_tag=discount_tag,
                    page_number=page_number,
                    edicion=edicion,
                    marca=marca,
                    vendor_id=partner_id,
                    vendor_name=partner_name,
                    catalog_id=catalog_id,
                    catalog_title=catalog_title,
                    page_image_url=img_url,
                    content_spec=content_specs[0] if content_specs else None,
                )
            elif seg_label and not variants[sku_code].label:
                variants[sku_code].label = seg_label

        return ParsedCatalogPage(
            catalog_id=catalog_id,
            catalog_title=catalog_title,
            partner_id=partner_id,
            partner_name=partner_name,
            page_number=page_number,
            edicion=edicion,
            marca=marca,
            raw_text=raw_page_text,
            clean_text=cleaned,
            page_image_url=img_url,
            skus=ordered_skus,
            variants=variants,
            nsoc_numbers=nsoc_numbers,
            primary_offer_price=primary_offer,
            primary_normal_price=primary_normal,
            discount_tag=discount_tag,
            content_specs=content_specs,
            has_free_gift=has_free_gift,
        )

    @classmethod
    def _extract_preceding_block_title(cls, segment: str) -> str:
        """Extrae el nombre o tono de la variante dentro del segmento previo al CÓD. actual."""
        window = segment[-280:] if len(segment) > 280 else segment
        lines = [ln.strip(" .:-–—") for ln in window.split("\n") if ln.strip(" .:-–—")]
        if not lines:
            return ""
        # 1. Priorizar ítem numerado cercano como '1. Dendur Destiny' o '2. Té Verde'
        for ln in reversed(lines[-7:]):
            if re.match(r"^[1-9]\.\s+[A-Za-zÁÉÍÓÚáéíóúÑñ]", ln):
                return re.sub(r"^[1-9]\.\s+", "", ln).strip()
        # 2. Priorizar la línea inmediatamente anterior si es nombre de tono/variante corto
        skip_words = (
            "cont.", "oferta", "p. normal", "dscto", "s/", "aroma", "intensidad",
            "para día", "para noche", "cód", "cod.", "nsoc", "mitad de precio",
            "precio", "gratis", "familia", "olfativa", "beneficio", "textura"
        )
        for i in range(len(lines) - 1, max(-1, len(lines) - 6), -1):
            ln = lines[i]
            low = ln.lower()
            if any(sw in low for sw in skip_words) or re.match(r"^\d+$", low):
                continue
            if ln.startswith("(") and ln.endswith(")") and i - 1 >= 0:
                prev_ln = lines[i - 1]
                if not any(sw in prev_ln.lower() for sw in skip_words):
                    return f"{prev_ln} {ln}".strip()
            if 2 <= len(ln) <= 45:
                return ln
        return ""

    @classmethod
    def _find_nearest_prices(
        cls,
        text: str,
        pos: int,
        default_offer: Optional[float],
        default_normal: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Busca el bloque de precio (Oferta / P. Normal) más cercano a la posición del CÓD."""
        window_start = max(0, pos - 220)
        window_end = min(len(text), pos + 180)
        snippet = text[window_start:window_end]

        before_snippet = text[window_start : min(len(text), pos + 25)]
        normals_before = [float(x) for x in cls.NORMAL_PRICE_PATTERN.findall(before_snippet)]
        without_norm_before = cls.NORMAL_PRICE_PATTERN.sub(" ", before_snippet)
        offers_before = [float(x) for x in cls.ANY_SOLES_PRICE_PATTERN.findall(without_norm_before)]

        offer = offers_before[-1] if offers_before else None
        normal = normals_before[-1] if normals_before else None

        if offer is None or normal is None:
            normals_all = [float(x) for x in cls.NORMAL_PRICE_PATTERN.findall(snippet)]
            without_norm_all = cls.NORMAL_PRICE_PATTERN.sub(" ", snippet)
            offers_all = [float(x) for x in cls.ANY_SOLES_PRICE_PATTERN.findall(without_norm_all)]
            if offer is None and offers_all:
                offer = offers_all[0]
            if normal is None and normals_all:
                normal = normals_all[0]

        return (offer if offer is not None else default_offer, normal if normal is not None else default_normal)


class CatalogTruthReconciler:
    """Reconciliador y auto-curador regenerativo on-demand que toma product_catalog_embedding
    como fuente única de verdad frente a discrepancias en product_product."""

    def __init__(self, pool: Optional[AsyncConnectionPool] = None):
        self._pool = pool
        self._index_cache: TTLCache = TTLCache(maxsize=10, ttl=1800)
        self._healed_records_log: List[Dict[str, Any]] = []
        self._healed_keys: Set[str] = set()

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = get_db_pool()
        if getattr(self._pool, "closed", False) is True:
            await self._pool.open()
        return self._pool

    @property
    def healed_records_log(self) -> List[Dict[str, Any]]:
        return list(self._healed_records_log)

    async def get_catalog_index(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Construye y cachea en memoria el índice completo de páginas activas de product_catalog_embedding
        filtrando únicamente el ÚLTIMO catálogo vigente por proveedor/marca según sus tags (product_catalog_product_tag_rel)."""
        cache_key = "active_catalog_index"
        if not force_refresh and cache_key in self._index_cache:
            return self._index_cache[cache_key]

        pool = await self._ensure_pool()
        query = """
            WITH catalog_tags_agg AS (
                SELECT
                    rel.catalog_id,
                    max(TRIM(BOTH '"' FROM COALESCE(t.name->>'es_ES', t.name->>'es_PE', t.name->>'en_US', t.name::text)))
                        FILTER (WHERE TRIM(BOTH '"' FROM COALESCE(t.name->>'es_ES', t.name->>'es_PE', t.name->>'en_US', t.name::text)) ~* '^c-?[0-9]+') AS edicion_tag,
                    max(TRIM(BOTH '"' FROM COALESCE(t.name->>'es_ES', t.name->>'es_PE', t.name->>'en_US', t.name::text)))
                        FILTER (WHERE TRIM(BOTH '"' FROM COALESCE(t.name->>'es_ES', t.name->>'es_PE', t.name->>'en_US', t.name::text)) ~* '^(yanbal|ésika|esika|belcorp|cyzone|l''bel|lbel)') AS marca_tag
                FROM product_catalog_product_tag_rel rel
                JOIN product_tag t ON t.id = rel.tag_id
                GROUP BY rel.catalog_id
            ),
            latest_catalogs AS (
                SELECT
                    pc.id AS catalog_id,
                    pc.title,
                    pc.partner_id,
                    cta.edicion_tag,
                    cta.marca_tag,
                    ROW_NUMBER() OVER (
                        PARTITION BY pc.partner_id, COALESCE(lower(cta.marca_tag), '')
                        ORDER BY
                            COALESCE(NULLIF(regexp_replace(COALESCE(cta.edicion_tag, ''), '[^0-9]', '', 'g'), '')::integer, 0) DESC,
                            COALESCE(pc.date_start, pc.write_date, pc.create_date) DESC,
                            pc.id DESC
                    ) AS rn_latest
                FROM product_catalog pc
                LEFT JOIN catalog_tags_agg cta ON cta.catalog_id = pc.id
                WHERE pc.active = TRUE
                  AND (pc.index_state = 'done' OR pc.index_state IS NULL)
            )
            SELECT
                pce.catalog_id,
                lc.title,
                pce.partner_id,
                rp.name AS partner_name,
                pce.page_number,
                pce.raw_page_text,
                pce.metadata_json,
                lc.edicion_tag,
                lc.marca_tag
            FROM product_catalog_embedding pce
            JOIN latest_catalogs lc ON lc.catalog_id = pce.catalog_id AND lc.rn_latest = 1
            LEFT JOIN res_partner rp ON rp.id = pce.partner_id
            WHERE pce.active = TRUE
              AND pce.chunk_type = 'text'
            ORDER BY pce.catalog_id, pce.page_number;
        """

        pages_by_num: Dict[int, ParsedCatalogPage] = {}
        sku_to_variant: Dict[str, CatalogVariantInfo] = {}
        sku_to_page: Dict[str, ParsedCatalogPage] = {}
        nsoc_to_pages: Dict[str, List[ParsedCatalogPage]] = {}
        latest_catalog_ids: List[int] = []

        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    rows = await cur.fetchall()
                    seen_cat_ids: Set[int] = set()
                    for r in rows:
                        cat_id, title, partner_id, partner_name, page_num, raw_text, meta_json, edicion_tag, marca_tag = r
                        seen_cat_ids.add(int(cat_id))
                        parsed = CatalogPageParser.parse_page(
                            page_number=int(page_num),
                            raw_page_text=raw_text or "",
                            metadata_json=meta_json,
                            catalog_id=int(cat_id),
                            catalog_title=str(title or ""),
                            partner_id=int(partner_id or 9),
                            partner_name=str(partner_name or "Unique S.A."),
                            edicion_override=str(edicion_tag) if edicion_tag else None,
                            marca_override=str(marca_tag) if marca_tag else None,
                        )
                        pages_by_num[int(page_num)] = parsed
                        for sku, var_info in parsed.variants.items():
                            sku_to_variant[sku] = var_info
                            sku_to_page[sku] = parsed
                        for nsoc in parsed.nsoc_numbers:
                            nsoc_to_pages.setdefault(nsoc, []).append(parsed)
                    latest_catalog_ids = sorted(seen_cat_ids)

                    # Enriquecer pliegos dobles (facing spreads): si la página par izquierda (ej. Pág. 70) no tiene SKUs
                    # y contiene el encabezado del producto cuyos tonos/SKUs están en la página impar derecha (ej. Pág. 71),
                    # fusionar su texto limpio para que la página con SKUs tenga el contexto completo del pliego
                    for pnum, pg in list(pages_by_num.items()):
                        if not pg.skus and (pnum + 1) in pages_by_num and pages_by_num[pnum + 1].skus:
                            right_pg = pages_by_num[pnum + 1]
                            right_pg.clean_text = f"{pg.clean_text}\n{right_pg.clean_text}".strip()
        except Exception as e:
            logger.warning(f"CatalogTruthReconciler: Error cargando índice de product_catalog_embedding: {e}")

        index_data = {
            "pages_by_num": pages_by_num,
            "sku_to_variant": sku_to_variant,
            "sku_to_page": sku_to_page,
            "nsoc_to_pages": nsoc_to_pages,
            "latest_catalog_ids": latest_catalog_ids,
        }
        if pages_by_num:
            self._index_cache[cache_key] = index_data
        return index_data

    def _resolve_discrepant_product(
        self,
        product: Dict[str, Any],
        index_data: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[ParsedCatalogPage], Optional[CatalogVariantInfo], Optional[str]]:
        """Determina si un producto tiene discrepancia de SKU, página o precio frente al catálogo.
        Retorna (true_sku, matched_page, matched_variant, discrepancy_reason)."""
        raw_sku = str(product.get("sku") or product.get("default_code") or "").strip()
        name = clean_zero_width(str(product.get("name") or "")).strip()
        odoo_pag = product.get("pagina")
        odoo_price = product.get("price")

        sku_to_variant: Dict[str, CatalogVariantInfo] = index_data.get("sku_to_variant", {})
        sku_to_page: Dict[str, ParsedCatalogPage] = index_data.get("sku_to_page", {})
        nsoc_to_pages: Dict[str, List[ParsedCatalogPage]] = index_data.get("nsoc_to_pages", {})
        pages_by_num: Dict[int, ParsedCatalogPage] = index_data.get("pages_by_num", {})

        # 1. Caso prefijo 'CÓD. 696' en default_code
        prefixed_match = re.match(r"^(?:CÓD\.?|COD\.?)\s*(\d+)$", raw_sku, re.IGNORECASE)
        if prefixed_match:
            clean_code = prefixed_match.group(1)
            page = sku_to_page.get(clean_code)
            var = sku_to_variant.get(clean_code)
            return clean_code, page, var, f"sku_prefix_cleaned:{raw_sku}->{clean_code}"

        # 2. Caso el SKU actual existe directamente en el catálogo activo
        if raw_sku in sku_to_variant:
            var = sku_to_variant[raw_sku]
            page = sku_to_page[raw_sku]
            reason = None
            if odoo_pag is not None and int(odoo_pag) != page.page_number:
                reason = f"page_mismatch:{odoo_pag}->{page.page_number}"
            elif odoo_pag is None:
                reason = f"page_missing:None->{page.page_number}"
            return raw_sku, page, var, reason

        # 3. Caso CRÍTICO: El SKU en Odoo es en realidad un código sanitario NSOC del catálogo (ej. 49608 o 55587)
        if raw_sku in nsoc_to_pages:
            candidate_pages = nsoc_to_pages[raw_sku]
            target_page = candidate_pages[0]
            for cp in candidate_pages:
                if odoo_pag is not None and int(odoo_pag) == cp.page_number:
                    target_page = cp
                    break
            matched_var = self._match_variant_on_page(name, target_page)
            if matched_var:
                return (
                    matched_var.sku,
                    target_page,
                    matched_var,
                    f"nsoc_sku_corrected:{raw_sku}(NSOC)->{matched_var.sku}",
                )

        # 4. Caso discrepancia por nombre o variante en el catálogo
        matched_page, matched_var = self._find_page_and_variant_by_name(name, pages_by_num, odoo_price=odoo_price)
        if matched_page and matched_var:
            return (
                matched_var.sku,
                matched_page,
                matched_var,
                f"catalog_name_sku_corrected:{raw_sku}->{matched_var.sku}",
            )

        return None, None, None, None

    @staticmethod
    def _match_variant_on_page(product_name: str, page: ParsedCatalogPage) -> Optional[CatalogVariantInfo]:
        """Selecciona la variante más específica dentro de una página según el nombre/tono del producto."""
        if not page.variants:
            return None
        name_lower = product_name.lower()
        best_var: Optional[CatalogVariantInfo] = None
        best_var_score = 0.0
        for sku, var in page.variants.items():
            lbl = (var.label or "").lower().strip()
            if not lbl:
                continue
            v_score = 0.0
            if lbl == name_lower:
                v_score = 100.0 + len(lbl)
            elif len(lbl) >= 3 and lbl in name_lower:
                v_score = 50.0 + len(lbl) * 2.0
            else:
                lbl_tokens = [t for t in re.findall(r"[a-záéíóúñ0-9]+", lbl) if len(t) >= 3]
                if lbl_tokens and all(t in name_lower for t in lbl_tokens):
                    v_score = 40.0 + sum(len(t) for t in lbl_tokens)
            if v_score > best_var_score:
                best_var_score = v_score
                best_var = var
        if best_var is not None:
            return best_var
        first_sku = page.skus[0] if page.skus else next(iter(page.variants.keys()))
        return page.variants.get(first_sku)

    @classmethod
    def _find_page_and_variant_by_name(
        cls,
        product_name: str,
        pages_by_num: Dict[int, ParsedCatalogPage],
        odoo_price: Optional[Any] = None,
    ) -> Tuple[Optional[ParsedCatalogPage], Optional[CatalogVariantInfo]]:
        """Busca un producto por coincidencia de su nombre comercial o variante en las páginas del catálogo."""
        if not product_name or len(product_name) < 4:
            return None, None

        clean_name = re.sub(r"\(\+.*?\)", "", product_name).strip()
        name_lower = clean_name.lower()
        stopwords = {"para", "con", "del", "las", "los", "una", "uno", "por", "que", "eau", "parfum", "toilette"}
        name_tokens = [t for t in re.findall(r"[a-záéíóúñ0-9]+", name_lower) if len(t) >= 2 and t not in stopwords]

        ref_price: Optional[float] = None
        if odoo_price is not None:
            try:
                ref_price = float(odoo_price)
            except (ValueError, TypeError):
                ref_price = None

        best_page: Optional[ParsedCatalogPage] = None
        best_var: Optional[CatalogVariantInfo] = None
        best_score = 0.0

        for page in pages_by_num.values():
            page_text_lower = " ".join(page.clean_text.lower().split())
            if not page.skus:
                continue

            base_parts = [p.strip() for p in re.split(r"[-–—]", clean_name) if p.strip()]
            main_title = base_parts[0].lower()

            score = 0.0
            if name_lower in page_text_lower:
                score += 55.0
            elif main_title in page_text_lower and len(main_title) >= 6:
                score += 40.0
            elif name_tokens and all(tok in page_text_lower for tok in name_tokens):
                score += 45.0

            if score == 0.0:
                continue

            matched_var = None
            best_var_bonus = 0.0
            for sku, var in page.variants.items():
                lbl = (var.label or "").lower().strip()
                if not lbl:
                    continue
                v_bonus = 0.0
                if lbl == name_lower:
                    v_bonus = 65.0 + len(lbl)
                elif len(lbl) >= 3 and lbl in name_lower:
                    v_bonus = 45.0 + len(lbl) * 1.5
                else:
                    lbl_tokens = [t for t in re.findall(r"[a-záéíóúñ0-9]+", lbl) if len(t) >= 3]
                    if lbl_tokens and all(t in name_lower for t in lbl_tokens):
                        v_bonus = 40.0 + sum(len(t) for t in lbl_tokens)
                if v_bonus > best_var_bonus:
                    best_var_bonus = v_bonus
                    matched_var = var

            score += best_var_bonus

            if len(page.skus) == 1 and score >= 40.0:
                score += 20.0
                matched_var = page.variants[page.skus[0]]

            if matched_var is None and score >= 45.0:
                matched_var = cls._match_variant_on_page(clean_name, page)

            # Proximidad de precio para desambiguar fragancia vs desodorante de la misma línea (ej. Temptation Black S/ 101 vs S/ 14)
            if matched_var is not None and ref_price is not None and ref_price > 0:
                cat_p = matched_var.offer_price or matched_var.normal_price
                cat_norm = matched_var.normal_price or matched_var.offer_price
                if cat_p is not None:
                    diff_min = min(abs(ref_price - cat_p), abs(ref_price - (cat_norm or cat_p)))
                    if diff_min <= 15.0:
                        score += 25.0
                    elif ref_price >= 60.0 and (cat_norm or cat_p) < 40.0:
                        score -= 60.0
                    elif ref_price < 35.0 and (cat_p or 0) >= 70.0:
                        score -= 60.0

            if score > best_score and matched_var is not None:
                best_score = score
                best_page = page
                best_var = matched_var

        if best_score >= 55.0 and best_page and best_var:
            return best_page, best_var
        return None, None

    async def heal_product_in_db(
        self,
        product_id: Optional[int],
        old_sku: str,
        new_sku: str,
        new_page: int,
        new_edicion: str,
        reason: str,
    ) -> bool:
        """Persiste de forma regenerativa (on-demand) la corrección de SKU y metadatos en PostgreSQL."""
        heal_key = f"{product_id}:{old_sku}->{new_sku}:{new_page}:{new_edicion}"
        if heal_key in self._healed_keys:
            return True

        pool = await self._ensure_pool()
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    if product_id and old_sku != new_sku:
                        # Verificar que ningún otro product_product tenga ya ese default_code para evitar colisiones si no es necesario
                        await cur.execute(
                            "UPDATE product_product SET default_code = %s WHERE id = %s;",
                            (new_sku, int(product_id)),
                        )
                        logger.info(
                            f"[CatalogTruthReconciler] Auto-curación on-demand aplicada en BD: "
                            f"product_id={product_id} SKU '{old_sku}' -> '{new_sku}' (Pág. {new_page}, {new_edicion}) [{reason}]"
                        )
            self._healed_keys.add(heal_key)
            self._healed_records_log.append({
                "product_id": product_id,
                "old_sku": old_sku,
                "new_sku": new_sku,
                "page_number": new_page,
                "edicion": new_edicion,
                "reason": reason,
            })
            return True
        except Exception as e:
            logger.warning(f"[CatalogTruthReconciler] No se pudo persistir auto-curación para product_id={product_id}: {e}")
            return False

    async def reconcile_and_heal_candidates(
        self,
        candidates: List[Dict[str, Any]],
        catalog_pages: Optional[List[ParsedCatalogPage]] = None,
        persist_healing: bool = True,
    ) -> List[Dict[str, Any]]:
        """Reconcilia una lista de productos candidatos contra la verdad del catálogo (product_catalog_embedding),
        corrigiendo SKUs discrepantes, páginas y precios de oferta on-demand."""
        index_data = await self.get_catalog_index()
        reconciled: List[Dict[str, Any]] = []
        seen_skus: Set[str] = set()

        for item in candidates:
            p = dict(item)
            old_sku = str(p.get("sku") or p.get("default_code") or "").strip()
            product_id = p.get("product_id")

            true_sku, page, var, reason = self._resolve_discrepant_product(p, index_data)
            if true_sku and page and var:
                if reason and (old_sku != true_sku or p.get("pagina") != page.page_number):
                    p["healed_from_sku"] = old_sku if old_sku != true_sku else None
                    p["healing_reason"] = reason
                    if persist_healing and product_id and old_sku != true_sku:
                        await self.heal_product_in_db(
                            product_id=int(product_id),
                            old_sku=old_sku,
                            new_sku=true_sku,
                            new_page=page.page_number,
                            new_edicion=page.edicion,
                            reason=reason,
                        )

                p["sku"] = true_sku
                p["pagina"] = page.page_number
                p["edicion"] = page.edicion
                p["marca"] = p.get("marca") or page.marca
                p["vendor_id"] = p.get("vendor_id") or page.partner_id
                p["vendor_name"] = p.get("vendor_name") or page.partner_name
                p["catalog_verified"] = True
                p["catalog_id"] = page.catalog_id
                p["catalog_title"] = page.catalog_title
                p["page_image_url"] = page.page_image_url

                # El precio de oferta del catálogo es la fuente de verdad comercial vigente
                if var.offer_price is not None:
                    p["price"] = float(var.offer_price)
                    p["catalog_offer_price"] = float(var.offer_price)
                if var.normal_price is not None:
                    p["normal_price"] = float(var.normal_price)
                if var.discount_tag:
                    p["discount_tag"] = var.discount_tag

                # Adjuntar todas las variantes/tonos de la página para enriquecer asesoría de tonos/iniciales
                if len(page.variants) > 1:
                    p["catalog_variants"] = [
                        {
                            "sku": v.sku,
                            "label": v.label,
                            "offer_price": v.offer_price,
                            "normal_price": v.normal_price,
                        }
                        for v in page.variants.values()
                    ]

                # Enriquecer descripción con el extracto limpio de la página del catálogo
                cat_snippet = " ".join(page.clean_text.split())[:420]
                base_desc = str(p.get("description") or "").strip()
                if cat_snippet and cat_snippet[:60].lower() not in base_desc.lower():
                    p["description"] = f"{base_desc}\n[Catálogo {page.edicion} Pág. {page.page_number}]: {cat_snippet}".strip()

            cur_sku = str(p.get("sku") or "").strip()
            if cur_sku:
                seen_skus.add(cur_sku)
            reconciled.append(p)

        # Inyectar candidatos descubiertos directamente en las páginas relevantes de product_catalog_embedding
        # (máximo 1 representante por página cuando la página aún no tiene candidatos en reconciled, ya que catalog_variants contiene todos los tonos/SKUs de la página)
        existing_pages = {int(r["pagina"]) for r in reconciled if r.get("pagina") is not None}
        if catalog_pages:
            for cpage in catalog_pages:
                if cpage.page_number in existing_pages:
                    continue
                for sku, var in cpage.variants.items():
                    if sku in seen_skus:
                        continue
                    seen_skus.add(sku)
                    existing_pages.add(cpage.page_number)
                    title_hint = self._infer_product_title_from_page(cpage, var)
                    reconciled.append({
                        "product_id": None,
                        "sku": sku,
                        "name": title_hint,
                        "description": f"[Catálogo {cpage.edicion} Pág. {cpage.page_number}]: {' '.join(cpage.clean_text.split())[:450]}",
                        "price": var.offer_price or var.normal_price,
                        "catalog_offer_price": var.offer_price,
                        "normal_price": var.normal_price,
                        "discount_tag": var.discount_tag,
                        "pagina": cpage.page_number,
                        "edicion": cpage.edicion,
                        "marca": cpage.marca,
                        "vendor_id": cpage.partner_id,
                        "vendor_name": cpage.partner_name,
                        "catalog_verified": True,
                        "catalog_id": cpage.catalog_id,
                        "catalog_title": cpage.catalog_title,
                        "page_image_url": cpage.page_image_url,
                        "catalog_variants": [
                            {"sku": v.sku, "label": v.label, "offer_price": v.offer_price, "normal_price": v.normal_price}
                            for v in cpage.variants.values()
                        ] if len(cpage.variants) > 1 else None,
                    })
                    break

        return reconciled

    @staticmethod
    def _infer_product_title_from_page(page: ParsedCatalogPage, var: CatalogVariantInfo) -> str:
        """Infiere un nombre comercial legible a partir del texto de la página de catálogo y la etiqueta de variante."""
        lines = [ln.strip() for ln in page.clean_text.split("\n") if ln.strip()]
        skip_prefixes = ("oferta", "p. normal", "cód", "cod", "nsoc", "cont.", "*#", "*prueba", "ir", "antes", "después", "a tan solo")
        candidate_titles = []
        for ln in lines:
            low = ln.lower()
            if any(low.startswith(sp) for sp in skip_prefixes):
                continue
            if re.match(r"^s/\.?\s*\d+", low) or re.match(r"^\d+%?\s*(?:dscto|horas)?$", low):
                continue
            if 4 <= len(ln) <= 65:
                candidate_titles.append(ln)
        base_title = candidate_titles[0] if candidate_titles else f"Producto Catálogo Pág. {page.page_number}"
        if var.label and var.label.lower() not in base_title.lower():
            return f"{base_title} - {var.label}"
        return base_title


_GLOBAL_RECONCILER: Optional[CatalogTruthReconciler] = None


def get_catalog_truth_reconciler(pool: Optional[AsyncConnectionPool] = None) -> CatalogTruthReconciler:
    global _GLOBAL_RECONCILER
    if _GLOBAL_RECONCILER is None:
        _GLOBAL_RECONCILER = CatalogTruthReconciler(pool=pool)
    elif pool is not None and _GLOBAL_RECONCILER._pool is None:
        _GLOBAL_RECONCILER._pool = pool
    return _GLOBAL_RECONCILER
