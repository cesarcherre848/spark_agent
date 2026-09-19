"""
src/agent_service/api/service.py - Servicio de resolución de vendedor (user_id) por teléfono con caché TTL en memoria
"""

import os
import re
import logging
from typing import Optional, List, Any
from cachetools import TTLCache

from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.tools.contact_tools import get_shared_odoo_client

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_USER_ID: Optional[int] = (
    int(os.getenv("DEFAULT_USER_ID")) if os.getenv("DEFAULT_USER_ID") else None
)
DEFAULT_CACHE_TTL = int(os.getenv("PHONE_CACHE_TTL_SECONDS", "3600"))
DEFAULT_CACHE_MAXSIZE = int(os.getenv("PHONE_CACHE_MAXSIZE", "10000"))


def normalize_phone(raw_phone: str) -> str:
    """Limpia el teléfono extrayendo únicamente los caracteres numéricos."""
    if not raw_phone:
        return ""
    digits = re.sub(r"[^\d]", "", str(raw_phone))
    return digits


class PhoneUserResolver:
    """Resuelve el ID de usuario/vendedor en Odoo ERP a partir del número telefónico.

    Mantiene una caché en memoria usando `cachetools.TTLCache` para garantizar
    resoluciones en O(1) tras la primera consulta, minimizando llamadas de red a Odoo.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_CACHE_TTL,
        maxsize: int = DEFAULT_CACHE_MAXSIZE,
        default_user_id: Optional[int] = DEFAULT_FALLBACK_USER_ID,
        odoo_client: Optional[OdooClient] = None,
    ):
        self.default_user_id = default_user_id
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._odoo_client = odoo_client

    def is_cached(self, phone: str) -> bool:
        """Indica si el teléfono ya reside en la caché activa."""
        clean_phone = normalize_phone(phone)
        return clean_phone in self._cache

    def get_cached(self, phone: str) -> Optional[int]:
        """Retorna el user_id en caché o None si no existe/expiró."""
        clean_phone = normalize_phone(phone)
        return self._cache.get(clean_phone)

    def set_cache(self, phone: str, user_id: int) -> None:
        """Almacena manualmente un mapeo de teléfono a user_id en caché."""
        clean_phone = normalize_phone(phone)
        if clean_phone and user_id is not None:
            self._cache[clean_phone] = int(user_id)

    def clear_cache(self) -> None:
        """Limpia todas las entradas en memoria de la caché."""
        self._cache.clear()

    async def resolve_user_id(
        self,
        phone: str,
        client: Optional[OdooClient] = None,
    ) -> Optional[int]:
        """Resuelve el user_id asignado al teléfono buscando en Odoo res.partner con caché.

        1. Verifica si el número normalizado ya está en `TTLCache`.
        2. Si no está en caché (Cache Miss), consulta la API de Odoo (`res.partner`).
        3. Si encuentra un partner con vendedor asignado (`user_id`), extrae su ID numérico.
        4. Si no encuentra coincidencia o ocurre un error, recurre a `default_user_id` (None por defecto).
        5. Almacena en `TTLCache` únicamente identificadores válidos (evita cachear None).
        """
        clean_phone = normalize_phone(phone)
        if not clean_phone:
            logger.warning("Teléfono vacío recibido para resolución.")
            return self.default_user_id

        # 1. Búsqueda en Caché
        if clean_phone in self._cache:
            cached_val = self._cache[clean_phone]
            logger.debug(f"Cache Hit para teléfono '{clean_phone}' -> user_id: {cached_val}")
            return cached_val

        logger.debug(f"Cache Miss para teléfono '{clean_phone}'. Consultando Odoo ERP...")

        # 2. Consulta en Odoo: Exclusivamente vendedores internos (res.users con share=False)
        resolved_uid: Optional[int] = None
        c = client or self._odoo_client or get_shared_odoo_client()

        try:
            if not getattr(c, "_uid", None):
                await c.authenticate()

            # Lista determinista de formatos exactos equivalentes (búsqueda estricta sin ilike)
            candidates = [phone.strip(), clean_phone, f"+{clean_phone}"]
            if len(clean_phone) == 9:
                candidates.extend([f"+51{clean_phone}", f"51{clean_phone}"])
            elif len(clean_phone) == 11 and clean_phone.startswith("51"):
                candidates.extend([clean_phone[2:], f"+{clean_phone}"])

            exact_candidates = list(dict.fromkeys([c for c in candidates if c]))

            domain: List[Any] = [
                ["active", "=", True],
                ["share", "=", False],
                "|",
                ["partner_id.phone_sanitized", "in", exact_candidates],
                ["partner_id.phone", "in", exact_candidates],
            ]

            call_payload = {
                "jsonrpc": "2.0",
                "method": "call",
                "params": {
                    "model": "res.users",
                    "method": "search_read",
                    "args": [domain],
                    "kwargs": {
                        "fields": ["id", "name", "login", "partner_id"],
                        "limit": 1,
                    },
                },
                "id": 901,
            }

            response = await c._client.post("/web/dataset/call_kw", json=call_payload)
            response.raise_for_status()
            data = response.json()

            if "result" in data and data["result"]:
                user = data["result"][0]
                resolved_uid = int(user["id"])
                logger.info(
                    f"Teléfono '{clean_phone}' autenticado exactamente como vendedor Odoo '{user.get('name')}' -> user_id: {resolved_uid}"
                )
            else:
                logger.warning(
                    f"Teléfono '{clean_phone}' no corresponde a ningún vendedor interno activo en Odoo; acceso denegado."
                )
                resolved_uid = self.default_user_id

        except Exception as e:
            logger.warning(
                f"Fallo al consultar Odoo para el teléfono '{clean_phone}': {e}."
            )
            resolved_uid = self.default_user_id

        # 3. Almacenar en Caché únicamente si resolved_uid es un usuario válido
        if resolved_uid is not None:
            self._cache[clean_phone] = resolved_uid

        return resolved_uid


_GLOBAL_RESOLVER: Optional[PhoneUserResolver] = None


def get_phone_user_resolver() -> PhoneUserResolver:
    """Retorna la instancia singleton de PhoneUserResolver."""
    global _GLOBAL_RESOLVER
    if _GLOBAL_RESOLVER is None:
        _GLOBAL_RESOLVER = PhoneUserResolver()
    return _GLOBAL_RESOLVER
