import re
from typing import List, Optional, Dict, Any, Union
from collections import defaultdict
from langchain_core.tools import tool
from psycopg_pool import AsyncConnectionPool

from src.agent_service.core.stores.product.product_hierarchy import extract_base_concept_name
from src.agent_service.core.stores.product.schemas import (
    ProductAttribute,
    GetProductBySkusInput,
    GetProductBySkusOutput,
    SupplierSearchInput,
    SupplierSearchOutput,
    ValidateProductOwnershipInput,
    ValidateProductOwnershipOutput,
    SkuOwnershipResult,
    ProductPartnerInfo,
    ProductCatalogFilter,
    SearchProductCatalogInput,
    GetCrossSellRecommendationsInput,
    FilterAndSortProductsInput,
    RerankAndFilterInput,
    CustomerPurchaseHistoryInput,
)
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.config.odoo import get_odoo_settings
from src.agent_service.config.database import get_db_pool


def create_get_product_by_skus_tool(odoo_client: Optional[OdooClient] = None):
    """Fábrica de la herramienta get_product_by_skus permitiendo inyección de dependencias."""
    client = odoo_client

    @tool("get_product_by_skus", args_schema=GetProductBySkusInput)
    async def get_product_by_skus_tool(
        skus: List[str],
        fields: Optional[List[ProductAttribute]] = None,
    ) -> Dict[str, Any]:
        """Consulta en tiempo real la API de Odoo ERP para obtener precios de venta, moneda,
        unidad de medida y especificaciones comerciales de una lista de códigos SKU de productos.

        Parámetros:
            - skus: Lista de códigos SKU o referencias internas exactas (ej: ['1', '11', '114']).
            - fields: Lista opcional de atributos a extraer ('price', 'description', 'uom', 'category', 'barcode', 'taxes').
        """
        nonlocal client
        if client is None:
            settings = get_odoo_settings()
            client = OdooClient(
                base_url=settings.url,
                db=settings.db,
                username=settings.username,
                password=settings.password,
            )

        result: GetProductBySkusOutput = await client.get_products_by_skus(skus=skus, fields=fields)
        return result.model_dump()

    return get_product_by_skus_tool


def create_search_suppliers_tool(odoo_client: Optional[OdooClient] = None):
    """Fábrica de la herramienta search_suppliers conectada a Odoo API (res.partner)."""
    client = odoo_client

    @tool("search_suppliers", args_schema=SupplierSearchInput)
    async def search_suppliers_tool(
        query: str,
        limit: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """Busca proveedores en Odoo ERP por nombre comercial, razón social o número de RUC/VAT.

        Parámetros:
            - query: Nombre o RUC del proveedor a buscar (ej: 'Unique', 'Siderperu', '20100102413').
            - limit: Cantidad máxima de coincidencias a retornar (por defecto 5).
        """
        nonlocal client
        if client is None:
            settings = get_odoo_settings()
            client = OdooClient(
                base_url=settings.url,
                db=settings.db,
                username=settings.username,
                password=settings.password,
            )

        result: SupplierSearchOutput = await client.search_suppliers(query=query, limit=limit or 5)
        return result.model_dump()

    return search_suppliers_tool


def create_validate_product_ownership_tool(pool: Optional[AsyncConnectionPool] = None):
    """Fábrica de la herramienta validate_product_ownership contra view_user_authorized_products."""
    db_pool = pool

    @tool("validate_product_ownership", args_schema=ValidateProductOwnershipInput)
    async def validate_product_ownership_tool(
        user_id: int,
        skus: List[str],
    ) -> Dict[str, Any]:
        """Valida si una lista de SKUs pertenecen al catálogo comercial autorizado para un usuario
        en la base de datos relacional de Odoo (PostgreSQL: view_user_authorized_products).

        Parámetros:
            - user_id: ID numérico del usuario (res_users) asignado a los equipos comerciales CRM.
            - skus: Lista de códigos SKU a verificar (ej: ['5110', '114']).

        Retorna los SKUs autorizados, sus proveedores oficiales y si existen conflictos (múltiples proveedores).
        """
        nonlocal db_pool
        if db_pool is None:
            db_pool = get_db_pool()

        if db_pool.closed:
            await db_pool.open()

        clean_skus = [str(s).strip() for s in (skus or []) if str(s).strip()]
        if not clean_skus:
            output = ValidateProductOwnershipOutput(user_id=user_id, results={})
            return output.model_dump()

        query_sql = """
            SELECT 
                vuap.sku,
                vuap.product_id,
                vuap.vendor_id,
                rp.name AS vendor_name
            FROM view_user_authorized_products vuap
            JOIN res_partner rp ON rp.id = vuap.vendor_id
            WHERE vuap.sku = ANY(%(skus)s)
              AND vuap.user_id = %(user_id)s;
        """

        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query_sql, {"skus": clean_skus, "user_id": int(user_id)})
                rows = await cur.fetchall()

        # Agrupar filas por SKU: (sku, product_id, vendor_id, vendor_name)
        sku_to_rows = defaultdict(list)
        for r in rows:
            sku_to_rows[str(r[0])].append(r)

        results: Dict[str, SkuOwnershipResult] = {}
        for s in clean_skus:
            matched_rows = sku_to_rows.get(s, [])
            if not matched_rows:
                results[s] = SkuOwnershipResult(
                    sku=s,
                    product_id=None,
                    is_owner=False,
                    candidate_partners=[],
                    has_conflict=False,
                    conflict_reason=f"El SKU '{s}' no está autorizado o no pertenece al catálogo del usuario {user_id}.",
                )
            else:
                p_id = matched_rows[0][1]
                partners = [
                    ProductPartnerInfo(
                        partner_id=str(r[2]),
                        vendor_id=int(r[2]),
                        vendor_name=str(r[3]),
                    )
                    for r in matched_rows
                ]
                has_conf = len(partners) > 1
                results[s] = SkuOwnershipResult(
                    sku=s,
                    product_id=p_id,
                    is_owner=True,
                    candidate_partners=partners,
                    has_conflict=has_conf,
                    conflict_reason=(
                        f"El SKU '{s}' cuenta con {len(partners)} proveedores posibles."
                        if has_conf else None
                    ),
                )

        output = ValidateProductOwnershipOutput(user_id=user_id, results=results)
        return output.model_dump()

    return validate_product_ownership_tool


def create_search_product_catalog_tool(vector_store: Optional[ProductVectorStore] = None):
    """Fábrica de la herramienta search_product_catalog conectada a PGVector híbrido."""
    vs = vector_store

    @tool("search_product_catalog", args_schema=SearchProductCatalogInput)
    async def search_product_catalog_tool(
        query: str,
        marca: Optional[str] = None,
        pagina: Optional[int] = None,
        edicion: Optional[str] = None,
        ediciones_activas: Optional[List[str]] = None,
        limit: Optional[int] = 8,
        user_id: Optional[int] = None,
        vendor_id: Optional[int] = None,
        vendor_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Busca productos en el catálogo comercial utilizando búsqueda semántica híbrida (RRF)
        y filtros de metadatos (marca comercial, página de catálogo y edición/campaña)."""
        nonlocal vs
        if vs is None:
            pool = get_db_pool()
            embeddings = get_embedding_service()
            vs = ProductVectorStore(pool=pool, embedding_service=embeddings)

        catalog_filter = (
            ProductCatalogFilter(
                user_id=user_id,
                pagina=pagina,
                edicion=edicion,
                ediciones_activas=ediciones_activas,
                marca=marca,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
            )
            if (user_id is not None or any(x is not None for x in [pagina, edicion, ediciones_activas, marca, vendor_id, vendor_name]))
            else None
        )

        docs = await vs.ahybrid_search(
            query=query,
            k=limit or 8,
            filters=catalog_filter,
        )

        from src.agent_service.core.stores.product.catalog_resolver import get_vendor_catalog_resolver
        catalog_resolver = get_vendor_catalog_resolver()

        results = []
        for doc in docs:
            meta = doc.metadata or {}
            p_name = meta.get("name") or "Producto sin nombre"
            p_edicion = meta.get("edicion")
            
            # Consultar si el producto tiene historial o variación frente a edición previa
            price_variation = None
            try:
                # El precio se obtiene del metadata o de la primera aproximación
                raw_price = meta.get("price") or meta.get("list_price")
                if raw_price is not None:
                    price_variation = await catalog_resolver.get_price_variation_history(
                        product_name=p_name,
                        current_price=float(raw_price),
                        current_edition=p_edicion,
                    )
            except Exception:
                price_variation = None

            results.append({
                "product_id": meta.get("product_id"),
                "sku": meta.get("sku"),
                "name": p_name,
                "description": doc.page_content,
                "pagina": meta.get("pagina"),
                "edicion": p_edicion,
                "marca": meta.get("marca"),
                "vendor_id": meta.get("vendor_id"),
                "vendor_name": meta.get("vendor_name"),
                "rrf_score": meta.get("rrf_score"),
                "price_variation": price_variation,
            })
        return results

    return search_product_catalog_tool


def create_get_cross_sell_recommendations_tool(
    vector_store: Optional[ProductVectorStore] = None,
    odoo_client: Optional[OdooClient] = None,
):
    """Fábrica de la herramienta get_cross_sell_recommendations para venta cruzada."""
    vs = vector_store
    client = odoo_client

    @tool("get_cross_sell_recommendations", args_schema=GetCrossSellRecommendationsInput)
    async def get_cross_sell_recommendations_tool(
        base_sku: Optional[str] = None,
        category: Optional[str] = None,
        marca: Optional[str] = None,
        limit: Optional[int] = 5,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Busca productos complementarios para venta cruzada (cross-sell) o sugerencias comerciales
        en base a una categoría, marca o producto de referencia."""
        nonlocal vs, client
        if vs is None:
            pool = get_db_pool()
            embeddings = get_embedding_service()
            vs = ProductVectorStore(pool=pool, embedding_service=embeddings)
        if client is None:
            from src.agent_service.tools.sales_tools import get_shared_odoo_client
            client = get_shared_odoo_client()

        search_terms = category or "productos recomendados complementarios"
        if base_sku and not category:
            try:
                details = await client.get_products_by_skus([base_sku], fields=["category", "description"])
                if details.products:
                    p = details.products[0]
                    search_terms = f"{p.category or ''} complementario {p.name}".strip()
            except Exception:
                pass

        catalog_filter = (
            ProductCatalogFilter(user_id=user_id, marca=marca)
            if (user_id is not None or marca)
            else None
        )

        docs = await vs.ahybrid_search(
            query=search_terms,
            k=(limit or 5) + (1 if base_sku else 0),
            filters=catalog_filter,
        )

        results = []
        for doc in docs:
            meta = doc.metadata or {}
            sku = meta.get("sku")
            if base_sku and str(sku) == str(base_sku):
                continue
            results.append({
                "product_id": meta.get("product_id"),
                "sku": sku,
                "name": meta.get("name") or "Producto sin nombre",
                "description": doc.page_content,
                "pagina": meta.get("pagina"),
                "edicion": meta.get("edicion"),
                "marca": meta.get("marca"),
                "vendor_id": meta.get("vendor_id"),
                "vendor_name": meta.get("vendor_name"),
                "relation_type": "cross_sell",
            })
            if len(results) >= (limit or 5):
                break
        return results

    return get_cross_sell_recommendations_tool


def create_filter_and_sort_products_tool():
    """Fábrica de la herramienta determinista filter_and_sort_products."""
    @tool("filter_and_sort_products", args_schema=FilterAndSortProductsInput)
    def filter_and_sort_products_tool(
        products: List[Dict[str, Any]],
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        marca: Optional[str] = None,
        vendor_name: Optional[str] = None,
        pagina: Optional[int] = None,
        edicion: Optional[str] = None,
        sort_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aplica filtros deterministas rigurosos (presupuesto mínimo/máximo, marca, proveedor, página, campaña)
        y ordenamiento por precio ('price_asc', 'price_desc') o relevancia sobre una lista de productos."""
        items = list(products or [])
        filtered = []
        target_brand = (marca or "").strip().lower()
        target_vendor = (vendor_name or "").strip().lower()
        target_edition = (edicion or "").strip().lower()

        for p in items:
            p_price = p.get("price")
            # 1. Filtro precio máximo
            if max_price is not None and p_price is not None:
                try:
                    if float(p_price) > float(max_price):
                        continue
                except (ValueError, TypeError):
                    pass

            # 2. Filtro precio mínimo
            if min_price is not None and p_price is not None:
                try:
                    if float(p_price) < float(min_price):
                        continue
                except (ValueError, TypeError):
                    pass

            # 3. Filtro marca
            if target_brand:
                p_brand = str(p.get("marca") or "").strip().lower()
                if p_brand and (target_brand not in p_brand and p_brand not in target_brand):
                    continue

            # 4. Filtro proveedor / vendor_name
            if target_vendor:
                p_vendor = str(p.get("vendor_name") or "").strip().lower()
                if p_vendor and (target_vendor not in p_vendor and p_vendor not in target_vendor):
                    continue

            # 5. Filtro página
            if pagina is not None and p.get("pagina") is not None:
                try:
                    if int(p.get("pagina")) != int(pagina):
                        continue
                except (ValueError, TypeError):
                    pass

            # 6. Filtro edición
            if target_edition:
                p_ed = str(p.get("edicion") or "").strip().lower()
                if p_ed and p_ed != target_edition:
                    continue

            filtered.append(p)

        # Ordenamiento
        if sort_by == "price_asc":
            filtered.sort(key=lambda x: float(x.get("price") or 0.0))
        elif sort_by == "price_desc":
            filtered.sort(key=lambda x: float(x.get("price") or 0.0), reverse=True)
        elif sort_by == "relevance":
            filtered.sort(key=lambda x: float(x.get("rrf_score") or 0.0), reverse=True)

        return filtered

    return filter_and_sort_products_tool


def create_rerank_and_filter_candidates_tool():
    """Fábrica de la herramienta contextual rerank_and_filter_candidates con soporte de feedback semántico e intent_spec."""
    @tool("rerank_and_filter_candidates", args_schema=RerankAndFilterInput)
    def rerank_and_filter_candidates_tool(
        candidates: List[Dict[str, Any]],
        context_query: Optional[str] = None,
        target_item_count: Optional[int] = None,
        feedback: Optional[str] = None,
        sort_by: Optional[str] = "relevance",
        user_id: Optional[int] = None,
        intent_spec: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Reordena, filtra y depura candidatos aplicando discriminación anatómica, demográfica y de género,
        asimilación de feedback semántico (crítica de la rúbrica) y corte dinámico Top-K."""
        items = list(candidates or [])
        if not items:
            return []

        ctx_str = f"{context_query or ''} {feedback or ''}".lower()
        spec = intent_spec or {}

        # 1. Extracción semántica de perfiles desde intent_spec con fallback al contexto
        target_gender = spec.get("target_gender")
        if not target_gender:
            if any(w in ctx_str for w in ["mujer", "femenin", "dama", "para ella"]):
                target_gender = "female"
            elif any(w in ctx_str for w in ["hombre", "masculin", "caballero", "para él"]):
                target_gender = "male"

        # Inferencia natural de edad: a partir de target_age numérico o contexto conversacional
        target_age = spec.get("target_age")
        if target_age is None:
            age_match = re.search(r"\b(\d{1,2})\s*(?:años|anos|de edad)\b", ctx_str)
            if not age_match:
                age_match = re.search(r"\b(?:tiene|edad(?:\s*de)?)\s*(\d{1,2})\b", ctx_str)
            if age_match:
                try:
                    target_age = int(age_match.group(1))
                except (ValueError, TypeError):
                    pass

        target_age_group = spec.get("target_age_group")
        # Inferencia dinámica de etapa de vida basada en edad numérica
        if target_age is not None:
            if target_age >= 45:
                target_age_group = "mature"
            elif 25 <= target_age < 45:
                target_age_group = "adult"
            elif 13 <= target_age < 25:
                target_age_group = "teen_young"
            else:
                target_age_group = "kids"
        elif not target_age_group:
            if any(w in ctx_str for w in ["madura", "piel madura", "adulta mayor", "señora"]):
                target_age_group = "mature"
            elif any(w in ctx_str for w in ["joven", "chico joven", "chica joven", "adolescente", "teen", "juvenil"]):
                target_age_group = "teen_young"
            elif any(w in ctx_str for w in ["niña", "niño", "bebe", "bebé", "infantil", "kids"]):
                target_age_group = "kids"

        is_mature = (target_age_group == "mature") or (target_age is not None and target_age >= 45)
        is_adult = is_mature or (target_age_group == "adult") or (target_age is not None and target_age >= 25)
        is_teen = (target_age_group == "teen_young") or (target_age is not None and 13 <= target_age < 25)

        anatomical_focus = spec.get("anatomical_focus")
        is_facial_focus = (anatomical_focus in ["facial", "face", "rostro"]) or any(
            w in ctx_str for w in ["rostro", "facial", "cutis", "cara", "antiarrugas", "ojos"]
        )
        is_fragrance_focus = (anatomical_focus in ["fragrance", "fragancia", "perfume", "colonia"]) or any(
            w in ctx_str for w in ["colonia", "perfume", "fragancia", "eau de", "aroma"]
        )

        preferred_brands = [str(b).strip().lower() for b in spec.get("preferred_brands") or []]
        excluded_brands = [str(b).strip().lower() for b in spec.get("excluded_brands") or []]
        excluded_profiles = [str(p).strip().lower() for p in spec.get("excluded_profiles") or []]

        excluded_targets = set(spec.get("excluded_targets") or [])
        if is_mature:
            excluded_targets.add("juvenil_teen")
            excluded_targets.add("kids")
        elif is_adult:
            excluded_targets.add("kids")

        is_economy_focus = any(
            w in ctx_str for w in ["económic", "economico", "barat", "menor precio", "menos precio", "aún más económico", "aun mas economico"]
        ) or bool(spec.get("max_price"))

        # Detección de límite de precio explícito en spec, feedback o contexto
        max_price_limit = spec.get("max_price")
        if max_price_limit is None:
            price_match = re.search(r"(?:menor|menos|<)\s*(?:a\s*)?(?:s\/\.?\s*)?(\d+(?:\.\d+)?)", ctx_str)
            if price_match:
                try:
                    max_price_limit = float(price_match.group(1))
                except (ValueError, TypeError):
                    pass

        # Detección de SKUs a excluir en feedback
        excluded_skus = set()
        if feedback:
            f_lower = feedback.lower()
            if "excluir" in f_lower or "rechazar" in f_lower or "no recomendar" in f_lower:
                for match in re.finditer(r"\b([0-9]{2,8})\b", feedback):
                    excluded_skus.add(match.group(1))

        # Descriptores de concepto etario universal (conceptos taxonómicos puros, sin submarcas ni nombres comerciales)
        concept_teen_descriptors = [
            "escarcha", "brillantina", "golosina", "chicle", "para adolescentes",
            "adolescente", "juvenil", "chicas jóvenes", "teen"
        ]
        concept_kids_descriptors = [
            "infantil", "para niñas", "para niños", "colonia niñas", "colonia niños",
            "kids", "baby", "para bebés", "pediátrico", "pediatrico"
        ]

        # Tokens significativos de la consulta para match semántico (ej: 'trigueña', 'bronzage', etc.)
        stop_words = {
            "para", "con", "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
            "que", "en", "por", "y", "o", "a", "al", "del", "mi", "mis", "tu", "tus",
            "su", "sus", "se", "lo", "le", "les", "me", "nos", "te", "es", "son",
            "tono", "ideal", "clienta", "cliente", "señora", "chico", "chica", "quien", "cual"
        }
        query_tokens = [w for w in re.findall(r"\b\w{3,}\b", ctx_str) if w not in stop_words]

        filtered = []
        for p in items:
            sku = str(p.get("sku") or p.get("default_code") or "").strip()
            name = str(p.get("name") or "").lower()
            desc = str(p.get("description") or p.get("description_sale") or "").lower()
            p_brand = str(p.get("marca") or "").strip().lower()
            p_price = p.get("price")
            p_text = f"{name} {desc}"

            # A. Exclusión de SKUs explícitos
            if sku and sku in excluded_skus:
                continue

            # B. Gatekeeper de marcas (preferred y excluded)
            if excluded_brands and any(eb in p_brand or p_brand in eb for eb in excluded_brands if eb):
                continue
            if preferred_brands and not any(pb in p_brand or p_brand in pb for pb in preferred_brands if pb):
                continue

            # C. Exclusión por perfiles anatómicos o funcionales excluidos
            if excluded_profiles and any(ep in p_text for ep in excluded_profiles if ep):
                continue

            # D. Gatekeeper anatómico genérico: Si es foco facial, descartar cremas corporales, de manos y capilares
            if is_facial_focus:
                if any(w in p_text for w in [
                    "crema corporal", "loción corporal", "locion corporal", "para el cuerpo",
                    "hidratante corporal", "manos y cuerpo", "crema de manos", "para manos",
                    "manos resecas", "moldeadora de rizos", "shampoo", "acondicionador"
                ]):
                    if not any(fw in name for fw in ["facial", "rostro", "cutis", "antiarrugas"]):
                        continue

            # E. Gatekeeper de género
            if target_gender == "female":
                if any(w in p_text for w in ["para hombre", "hombre", "masculino", "eau de parfum hombre", "eau de toilette hombre"]):
                    continue
            elif target_gender == "male":
                if any(w in p_text for w in ["para mujer", "femenina", "dama", "eau de parfum mujer", "eau de toilette mujer"]):
                    continue

            # F. Gatekeeper demográfico de edad conceptual (Descartar infantil y juvenil cuando el target es adulto maduro)
            if "kids" in excluded_targets or is_adult:
                if any(kd in p_text for kd in concept_kids_descriptors):
                    continue

            if "juvenil_teen" in excluded_targets or is_mature:
                if is_fragrance_focus:
                    if any(jd in p_text for jd in concept_teen_descriptors):
                        continue

            # G. Límite de precio si fue detectado explícitamente
            if max_price_limit is not None and p_price is not None:
                try:
                    if float(p_price) > max_price_limit:
                        continue
                except (ValueError, TypeError):
                    pass

            filtered.append(p)

        # Si el filtro fue demasiado estricto y vació todo, recuperar items originales como respaldo
        if not filtered and items:
            filtered = list(items)

        # 2. Scoring y Reordenamiento Dinámico
        def calculate_item_score(item: Dict[str, Any]) -> float:
            score = 50.0
            p_price = item.get("price")
            name = str(item.get("name") or "").lower()
            desc = str(item.get("description") or "").lower()
            combined = f"{name} {desc}"

            # Sensibilidad a precio
            if p_price is not None:
                try:
                    val = float(p_price)
                    if is_economy_focus:
                        score += max(0.0, (150.0 - val))
                    else:
                        score += 10.0
                except (ValueError, TypeError):
                    pass

            # Afinidad semántica con tokens significativos de la consulta
            matched_tokens_count = sum(1 for tok in query_tokens if tok in combined)
            if matched_tokens_count > 0:
                score += (matched_tokens_count * 15.0)

            # Afinidad dermatológica y de estilo inferida para adultos maduros (edad >= 45 o mature)
            if is_mature:
                # Facial: descriptores funcionales dermatológicos universales
                if any(w in combined for w in ["antiarrugas", "antiedad", "lifting", "firmeza", "regeneradora", "resveratrol", "nutritiva", "madura"]):
                    score += 30.0
                # Fragancias: descriptores conceptuales universales de estilo clásico, fresco y elegante
                if any(w in combined for w in ["elegante", "clasic", "clásic", "fresca", "floral", "suave", "diario", "delicada", "sobria"]):
                    score += 20.0
            elif is_teen:
                if any(jd in combined for jd in concept_teen_descriptors):
                    score += 20.0

            # Afinidad para piel mixta/grasa
            if any(w in ctx_str for w in ["mixta", "grasa", "sebo", "brillo", "matificante"]):
                if any(w in combined for w in ["matificante", "control de grasa", "sin brillo", "ligera", "oil control"]):
                    score += 20.0

            # Penalización si no tiene precio asignado
            if p_price is None:
                score -= 15.0

            return score

        if sort_by == "price_asc":
            filtered.sort(key=lambda x: (x.get("price") is None, float(x.get("price") or 999999.0), -calculate_item_score(x)))
        elif sort_by == "price_desc":
            filtered.sort(key=lambda x: (x.get("price") is None, -float(x.get("price") or 0.0)))
        else:
            filtered.sort(key=calculate_item_score, reverse=True)

        # 3. Deduplicación por concepto canónico preservando la mejor variante
        deduped = []
        seen_concepts = set()
        for it in filtered:
            c_name = extract_base_concept_name(it.get("name") or "")
            if c_name not in seen_concepts:
                seen_concepts.add(c_name)
                deduped.append(it)
            elif target_item_count and len(deduped) < target_item_count:
                deduped.append(it)

        # 4. Truncado dinámico (Top-K)
        limit_k = target_item_count if (target_item_count and target_item_count > 0) else 4
        return deduped[:limit_k]

    return rerank_and_filter_candidates_tool



def create_get_customer_purchase_history_tool(odoo_client: Optional[OdooClient] = None):
    """Fábrica de la herramienta get_customer_purchase_history conectada a sale.order de Odoo."""
    client = odoo_client

    @tool("get_customer_purchase_history", args_schema=CustomerPurchaseHistoryInput)
    async def get_customer_purchase_history_tool(
        partner_id: Optional[int] = None,
        customer_name: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: Optional[int] = 5,
    ) -> Dict[str, Any]:
        """Consulta el historial de compras previas del cliente en Odoo ERP para identificar
        productos recurrentes, marcas y categorías frecuentes para personalizar recomendaciones."""
        nonlocal client
        from src.agent_service.tools.sales_tools import odoo_list_sales_orders, get_shared_odoo_client

        c = client or get_shared_odoo_client()
        orders_dict = await odoo_list_sales_orders(
            user_id=user_id or 5,
            partner_id=partner_id,
            customer_name=customer_name,
            status="sale",
            limit=limit or 5,
            client=c,
        )

        all_orders = orders_dict.get("sale", []) + orders_dict.get("draft", [])

        products_frequency = defaultdict(int)
        recent_items = []
        total_spent = 0.0

        for ord in all_orders:
            total_spent += float(ord.get("amount_total") or 0.0)
            lines = ord.get("lines", [])
            for line in lines:
                name = line.get("name") or ""
                qty = float(line.get("product_uom_qty") or 1.0)
                if name:
                    clean_name = name.split("] ")[-1] if "]" in name else name
                    products_frequency[clean_name] += int(qty)
                    recent_items.append({
                        "product_name": clean_name,
                        "qty": qty,
                        "price_unit": line.get("price_unit"),
                        "date_order": ord.get("date_order"),
                    })

        sorted_freq = sorted(products_frequency.items(), key=lambda x: x[1], reverse=True)
        top_products = [{"name": item[0], "total_qty": item[1]} for item in sorted_freq[:5]]

        return {
            "partner_id": partner_id,
            "customer_name": customer_name,
            "orders_count": len(all_orders),
            "total_spent": round(total_spent, 2),
            "top_purchased_products": top_products,
            "recent_items": recent_items[:10],
            "preference_summary": (
                f"Cliente con {len(all_orders)} pedidos analizados. Productos más comprados: "
                + ", ".join([f"{p['name']} ({p['total_qty']} uds)" for p in top_products])
                if top_products else "Sin compras previas registradas para este cliente."
            ),
        }

    return get_customer_purchase_history_tool


# Instancias por defecto listas para ser usadas por agentes
get_product_by_skus = create_get_product_by_skus_tool()
search_suppliers = create_search_suppliers_tool()
validate_product_ownership = create_validate_product_ownership_tool()
search_product_catalog = create_search_product_catalog_tool()
get_cross_sell_recommendations = create_get_cross_sell_recommendations_tool()
filter_and_sort_products = create_filter_and_sort_products_tool()
rerank_and_filter_candidates = create_rerank_and_filter_candidates_tool()
get_customer_purchase_history = create_get_customer_purchase_history_tool()

