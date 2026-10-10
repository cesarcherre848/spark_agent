"""
src/agent_service/core/stores/product/catalog_resolver.py

Módulo de resolución de últimas campañas vigentes por proveedor
basado en la fecha de modificación reciente en Odoo ERP (product_template.write_date)
con caché en memoria TTL para optimización de consultas.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from cachetools import TTLCache
from psycopg_pool import AsyncConnectionPool

from src.agent_service.config.database import get_db_pool

logger = logging.getLogger(__name__)

DEFAULT_CAMPAIGN_CACHE_TTL = int(os.getenv("CATALOG_CAMPAIGN_CACHE_TTL_SECONDS", "3600"))
DEFAULT_CAMPAIGN_CACHE_MAXSIZE = int(os.getenv("CATALOG_CAMPAIGN_CACHE_MAXSIZE", "500"))


class VendorCatalogResolver:
    """Resuelve la última campaña/edición vigente por cada proveedor a partir
    de la fecha más reciente de actualización de productos en Odoo ERP (product_template.write_date).
    Mantiene caché en memoria usando `cachetools.TTLCache`.
    """

    def __init__(
        self,
        pool: Optional[AsyncConnectionPool] = None,
        ttl_seconds: int = DEFAULT_CAMPAIGN_CACHE_TTL,
        maxsize: int = DEFAULT_CAMPAIGN_CACHE_MAXSIZE,
    ):
        self._pool = pool
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._variations_cache: TTLCache = TTLCache(maxsize=1000, ttl=ttl_seconds)

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = get_db_pool()
        if getattr(self._pool, "closed", False) is True:
            await self._pool.open()
        return self._pool

    async def get_latest_campaigns_per_vendor(
        self,
        force_refresh: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """Retorna un diccionario mapeando `vendor_id` -> info de la última campaña vigente:
        {
            vendor_id: {
                "vendor_id": 9,
                "vendor_marca": "Yanbal",
                "latest_edicion": "C11",
                "product_count": 259,
                "last_catalog_update": datetime(...)
            }
        }
        """
        cache_key = "latest_campaigns_all_vendors"
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]

        pool = await self._ensure_pool()
        query = """
            WITH edicion_vendor_stats AS (
                SELECT 
                    vuap.vendor_id,
                    COALESCE(vuap.marca, rp.name) as vendor_marca,
                    vuap.edicion,
                    count(DISTINCT vuap.product_id) as product_count,
                    max(pt.write_date) as last_catalog_update
                FROM view_user_authorized_products_metadata vuap
                JOIN product_product pp ON pp.id = vuap.product_id
                JOIN product_template pt ON pt.id = pp.product_tmpl_id
                LEFT JOIN res_partner rp ON rp.id = vuap.vendor_id
                WHERE vuap.edicion IS NOT NULL
                GROUP BY vuap.vendor_id, COALESCE(vuap.marca, rp.name), vuap.edicion
            ),
            ranked_editions AS (
                SELECT 
                    vendor_id,
                    vendor_marca,
                    edicion,
                    product_count,
                    last_catalog_update,
                    ROW_NUMBER() OVER (
                        PARTITION BY vendor_id 
                        ORDER BY last_catalog_update DESC
                    ) as rank_recency
                FROM edicion_vendor_stats
            )
            SELECT vendor_id, vendor_marca, edicion, product_count, last_catalog_update
            FROM ranked_editions
            WHERE rank_recency = 1;
        """

        results: Dict[int, Dict[str, Any]] = {}
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query)
                    rows = await cur.fetchall()
                    for r in rows:
                        vid = int(r[0])
                        results[vid] = {
                            "vendor_id": vid,
                            "vendor_marca": str(r[1] or ""),
                            "latest_edicion": str(r[2] or ""),
                            "product_count": int(r[3] or 0),
                            "last_catalog_update": r[4],
                        }
            self._cache[cache_key] = results
            logger.info(f"VendorCatalogResolver: detectadas {len(results)} últimas campañas vigentes.")
        except Exception as e:
            logger.error(f"Error resolviendo últimas campañas por vendor: {e}")
            if cache_key in self._cache:
                return self._cache[cache_key]
            return {}

        return results

    async def get_active_editions_for_user(
        self,
        user_id: Optional[int] = None,
        vendor_id: Optional[int] = None,
        marca: Optional[str] = None,
    ) -> List[str]:
        """Calcula las campañas activas para un contexto de usuario o marca.
        Si no hay restricciones, devuelve todas las últimas campañas vigentes."""
        campaigns_map = await self.get_latest_campaigns_per_vendor()
        if not campaigns_map:
            return []

        active_editions: List[str] = []

        # Si se especificó vendor_id puntual
        if vendor_id and vendor_id in campaigns_map:
            return [campaigns_map[vendor_id]["latest_edicion"]]

        # Si se especificó marca puntual
        if marca:
            clean_marca = marca.strip().lower()
            for vid, data in campaigns_map.items():
                v_marca = data.get("vendor_marca", "").strip().lower()
                if clean_marca in v_marca or v_marca in clean_marca:
                    active_editions.append(data["latest_edicion"])
            if active_editions:
                return list(dict.fromkeys(active_editions))

        # Si hay user_id, determinar a qué proveedores está autorizado el usuario
        if user_id:
            user_cache_key = f"user_active_editions_{user_id}"
            if user_cache_key in self._cache:
                return self._cache[user_cache_key]

            pool = await self._ensure_pool()
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT DISTINCT vendor_id FROM view_user_authorized_products_metadata WHERE user_id = %s;",
                            (int(user_id),),
                        )
                        u_vendors = [int(r[0]) for r in await cur.fetchall() if r[0] is not None]
                        for vid in u_vendors:
                            if vid in campaigns_map:
                                active_editions.append(campaigns_map[vid]["latest_edicion"])
            except Exception as e:
                logger.warning(f"Error detectando proveedores para user_id {user_id}: {e}")

            if active_editions:
                deduped = list(dict.fromkeys(active_editions))
                self._cache[user_cache_key] = deduped
                return deduped

        # Fallback general: todas las últimas campañas detectadas
        all_latest = [v["latest_edicion"] for v in campaigns_map.values() if v.get("latest_edicion")]
        return list(dict.fromkeys(all_latest))

    async def get_price_variation_history(
        self,
        product_name: str,
        current_price: float,
        current_edition: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Consulta si el producto existía en una edición/campaña anterior y calcula si hubo variación de precio."""
        if not product_name:
            return None

        clean_name = product_name.strip()
        cache_key = f"var_{clean_name.lower()}_{current_price}"
        if cache_key in self._variations_cache:
            return self._variations_cache[cache_key]

        pool = await self._ensure_pool()
        query = """
            SELECT 
                vuap.edicion,
                pt.list_price,
                pt.write_date
            FROM view_user_authorized_products_metadata vuap
            JOIN product_product pp ON pp.id = vuap.product_id
            JOIN product_template pt ON pt.id = pp.product_tmpl_id
            WHERE (
                lower(pt.name->>'es_PE') = lower(%s)
                OR lower(pt.name->>'en_US') = lower(%s)
                OR lower(pt.name->>'es_ES') = lower(%s)
                OR position(lower(%s) in lower(COALESCE(pt.name->>'es_PE', pt.name->>'en_US', ''))) > 0
            )
            AND vuap.edicion IS NOT NULL
            AND (%(current_ed)s::text IS NULL OR vuap.edicion != %(current_ed)s::text)
            ORDER BY pt.write_date DESC
            LIMIT 1;
        """
        params = (clean_name, clean_name, clean_name, clean_name)
        named_params = {
            "current_ed": current_edition,
        }

        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT 
                            vuap.edicion,
                            pt.list_price,
                            pt.write_date
                        FROM view_user_authorized_products_metadata vuap
                        JOIN product_product pp ON pp.id = vuap.product_id
                        JOIN product_template pt ON pt.id = pp.product_tmpl_id
                        WHERE (
                            lower(COALESCE(pt.name->>'es_PE', pt.name->>'en_US', '')) = lower(%s)
                        )
                        AND vuap.edicion IS NOT NULL
                        AND (%s::text IS NULL OR vuap.edicion != %s::text)
                        ORDER BY pt.write_date DESC
                        LIMIT 1;
                        """,
                        (clean_name, current_edition, current_edition),
                    )
                    row = await cur.fetchone()
                    if row:
                        prev_edition = str(row[0])
                        prev_price = float(row[1] or 0.0)
                        diff = round(current_price - prev_price, 2)
                        result = {
                            "previous_edition": prev_edition,
                            "previous_price": prev_price,
                            "current_price": current_price,
                            "difference": diff,
                            "is_discount": diff < 0,
                            "percentage_change": round((diff / prev_price) * 100, 1) if prev_price > 0 else 0.0,
                        }
                        self._variations_cache[cache_key] = result
                        return result
        except Exception as e:
            logger.debug(f"Error consultando variación para '{clean_name}': {e}")

        self._variations_cache[cache_key] = None
        return None

    def clear_cache(self) -> None:
        self._cache.clear()
        self._variations_cache.clear()


# Singleton global
_GLOBAL_CATALOG_RESOLVER: Optional[VendorCatalogResolver] = None


def get_vendor_catalog_resolver() -> VendorCatalogResolver:
    global _GLOBAL_CATALOG_RESOLVER
    if _GLOBAL_CATALOG_RESOLVER is None:
        _GLOBAL_CATALOG_RESOLVER = VendorCatalogResolver()
    return _GLOBAL_CATALOG_RESOLVER
