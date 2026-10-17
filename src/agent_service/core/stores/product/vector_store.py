import re
from typing import Any, Dict, List, Optional
from psycopg_pool import AsyncConnectionPool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.agent_service.core.stores.product.schemas import ProductCatalogFilter


def normalize_brand(brand: Optional[str]) -> Optional[str]:
    """Normaliza variantes comunes, acentos y errores ortográficos a las marcas canónicas del ERP."""
    if not brand:
        return None
    b = brand.strip().lower()
    if re.search(r"\b(es+ika|ésika)\b", b):
        return "Ésika"
    if re.search(r"\byanbal\b", b):
        return "Yanbal"
    if re.search(r"\b(belcorp|cyzone|l'?bel)\b", b):
        return b.title()
    return brand.strip()


_HYBRID_QUERY_SQL = """
    WITH semantic_search AS (
        SELECT 
            emb.id,
            ROW_NUMBER() OVER (ORDER BY emb.{embedding_column} <=> %(query_embedding)s::vector) AS rank_sem
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        WHERE emb.active = TRUE
          AND (%(user_id)s::integer IS NULL OR vuap.user_id IS NOT NULL)
          AND (%(vendor_id)s::integer IS NULL OR vuap.vendor_id = %(vendor_id)s::integer)
          AND (%(pagina)s::integer IS NULL OR vuap.pagina = %(pagina)s::integer)
          AND (
              (%(edicion)s::text IS NOT NULL AND lower(vuap.edicion) = lower(%(edicion)s::text))
              OR (%(edicion)s::text IS NULL AND %(ediciones_activas)s::text[] IS NOT NULL AND vuap.edicion = ANY(%(ediciones_activas)s::text[]))
              OR (%(edicion)s::text IS NULL AND %(ediciones_activas)s::text[] IS NULL)
          )
          AND (
              %(marca)s::text IS NULL 
              OR lower(vuap.marca) = lower(%(marca)s::text)
              OR position(translate(lower(%(marca)s::text), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou') in translate(lower(vuap.marca), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou')) > 0
              OR (%(marca)s::text ~* '^es+ika' AND vuap.marca ~* '^es+ika|^ésika')
          )
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
        ORDER BY emb.{embedding_column} <=> %(query_embedding)s::vector
        LIMIT %(k_candidates)s
    ),
    lexical_search AS (
        SELECT 
            emb.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(
                    to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')),
                    CASE 
                        WHEN to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')) @@ websearch_to_tsquery('spanish', %(query_text)s)
                        THEN websearch_to_tsquery('spanish', %(query_text)s)
                        ELSE to_tsquery('spanish', %(lexical_tsquery)s)
                    END
                ) DESC
            ) AS rank_lex
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        WHERE emb.active = TRUE
          AND (%(user_id)s::integer IS NULL OR vuap.user_id IS NOT NULL)
          AND (%(vendor_id)s::integer IS NULL OR vuap.vendor_id = %(vendor_id)s::integer)
          AND (%(pagina)s::integer IS NULL OR vuap.pagina = %(pagina)s::integer)
          AND (
              (%(edicion)s::text IS NOT NULL AND lower(vuap.edicion) = lower(%(edicion)s::text))
              OR (%(edicion)s::text IS NULL AND %(ediciones_activas)s::text[] IS NOT NULL AND vuap.edicion = ANY(%(ediciones_activas)s::text[]))
              OR (%(edicion)s::text IS NULL AND %(ediciones_activas)s::text[] IS NULL)
          )
          AND (
              %(marca)s::text IS NULL 
              OR lower(vuap.marca) = lower(%(marca)s::text)
              OR position(translate(lower(%(marca)s::text), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou') in translate(lower(vuap.marca), 'áéíóúÁÉÍÓÚ', 'aeiouaeiou')) > 0
              OR (%(marca)s::text ~* '^es+ika' AND vuap.marca ~* '^es+ika|^ésika')
          )
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
          AND (
              to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')) @@ websearch_to_tsquery('spanish', %(query_text)s)
              OR to_tsvector('spanish', COALESCE(emb.name, '') || ' ' || COALESCE(emb.source_text, '')) @@ to_tsquery('spanish', %(lexical_tsquery)s)
          )
        LIMIT %(k_candidates)s
    ),
    fused_candidates AS (
        SELECT DISTINCT ON (emb.id)
            emb.id,
            emb.product_id,
            emb.product_tmpl_id,
            emb.name,
            emb.source_text,
            vuap.vendor_id,
            vuap.sku,
            vuap.pagina,
            vuap.edicion,
            vuap.marca,
            rp.name AS vendor_name,
            (
                COALESCE(%(alpha)s / (60.0 + sem.rank_sem), 0.0) +
                COALESCE((1.0 - %(alpha)s) / (60.0 + lex.rank_lex), 0.0)
            ) AS rrf_score
        FROM {table_name} emb
        LEFT JOIN view_user_authorized_products_metadata vuap 
          ON vuap.product_id = emb.product_id 
         AND (%(user_id)s::integer IS NULL OR vuap.user_id = %(user_id)s::integer)
        LEFT JOIN res_partner rp ON rp.id = vuap.vendor_id
        LEFT JOIN semantic_search sem ON sem.id = emb.id
        LEFT JOIN lexical_search lex ON lex.id = emb.id
        WHERE (sem.id IS NOT NULL OR lex.id IS NOT NULL)
          AND emb.ai_provider = %(ai_provider)s
          AND emb.ai_model = %(ai_model)s
        ORDER BY emb.id
    )
    SELECT 
        id,
        product_id,
        product_tmpl_id,
        name,
        source_text,
        vendor_id,
        sku,
        rrf_score,
        pagina,
        edicion,
        marca,
        vendor_name
    FROM fused_candidates
    ORDER BY rrf_score DESC
    LIMIT %(k)s;
"""


def _build_lexical_tsquery(query: str) -> str:
    """Construye una expresión tsquery en español usando raíces léxicas y disyunción (|)
    para evitar que búsquedas largas o con adjetivos específicos se queden vacías."""
    # Stopwords comunes en español y palabras vacías para catálogo
    stopwords = {
        "para", "como", "pero", "este", "esta", "estos", "estas", "cual", "quien", "donde",
        "tengo", "necesito", "necesita", "busca", "busco", "quiero", "quiere", "recomienda",
        "bueno", "buena", "mejor", "mejores", "unos", "unas", "sobre", "entre", "desde",
        "hacia", "hasta", "anos", "años", "favor", "hola", "buenos", "dias", "tardes"
    }
    # Mapeo morfológico común para enriquecer raíces cosméticas/cuidado personal
    morph_synonyms = {
        "hidrat": ["hidrat", "ultrahidrat"],
        "hidratante": ["hidrat", "ultrahidrat"],
        "arruga": ["arrug", "antiarrug"],
        "antiarruga": ["arrug", "antiarrug"],
        "arrugas": ["arrug", "antiarrug"],
        "antiarrugas": ["arrug", "antiarrug"],
        "edad": ["edad", "arrug", "antiarrug", "antiedad"],
        "antiedad": ["arrug", "antiarrug", "antiedad"],
        "rostr": ["rostr", "facial"],
        "rostro": ["rostr", "facial"],
        "cara": ["rostr", "facial", "cara"],
        "facial": ["rostr", "facial"],
        "piel": ["piel", "cutis"],
        "aclarante": ["aclar", "concha", "manch"],
        "aclarar": ["aclar", "concha", "manch"],
        "economica": ["crem", "accesible", "facial"],
        "economico": ["crem", "accesible", "facial"],
    }

    clean_words = re.findall(r"[a-záéíóúñA-ZÁÉÍÓÚÑ0-9]+", (query or "").lower())
    terms = set()
    for w in clean_words:
        if len(w) <= 2 or w in stopwords:
            continue
        terms.add(w)
        for stem_key, expansions in morph_synonyms.items():
            if stem_key in w or w.startswith(stem_key[:4]):
                terms.update(expansions)

    if not terms:
        return "crema | facial | hidrat"
    return " | ".join(sorted(terms))


class ProductVectorStore(VectorStore):
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_service: Embeddings,
        table_name: str = "product_vector_embedding",
        embedding_column: str = "embedding_1024",
        ai_provider: Optional[str] = "huggingface",
        ai_model: Optional[str] = "BAAI/bge-m3",
    ):
        self._pool = pool
        self._embedding_service = embedding_service
        self._table_name = table_name
        self._embedding_column = embedding_column
        self._ai_provider = ai_provider
        self._ai_model = ai_model

        # Extraer dimensión de la columna (ej. '1536' de 'embedding_1536')
        dim_str = "".join(filter(str.isdigit, embedding_column))
        self._embedding_dim = int(dim_str) if dim_str else 1024

        # Compilación segura usando la plantilla definida
        self._compiled_query = _HYBRID_QUERY_SQL.format(
            table_name=self._table_name,
            embedding_column=self._embedding_column,
        )

    async def ahybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.55,
        filters: Optional[ProductCatalogFilter] = None,
    ) -> List[Document]:
        # Normalizar consulta a minúsculas y espacios limpios para evitar fragmentación de sub-tokens
        clean_query = " ".join((query or "").lower().split())
        lexical_tsquery = _build_lexical_tsquery(clean_query)

        # Generar embedding o crear vector dummy con dimensión exacta si alpha == 0.0
        if alpha > 0.0:
            query_embedding = await self._embedding_service.aembed_query(clean_query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
        else:
            embedding_str = f"[{','.join(['0.0'] * self._embedding_dim)}]"

        target_user_id = filters.user_id if filters else None
        target_pagina = (
            filters.pagina
            if (filters and filters.pagina is not None)
            else (filters.metadata.pagina if (filters and filters.metadata) else None)
        )
        target_edicion = (
            filters.edicion
            if (filters and filters.edicion is not None)
            else (filters.metadata.edicion if (filters and filters.metadata) else None)
        )
        target_ediciones_activas = (
            filters.ediciones_activas
            if (filters and getattr(filters, "ediciones_activas", None) is not None)
            else None
        )

        # Si el usuario NO especificó una edición puntual ni una lista explícita,
        # resolver dinámicamente las últimas campañas vigentes por proveedor
        raw_marca = (
            filters.marca
            if (filters and filters.marca is not None)
            else (filters.metadata.marca if (filters and filters.metadata) else None)
        )
        target_marca = normalize_brand(raw_marca)
        target_vendor_id = filters.vendor_id if (filters and filters.vendor_id is not None) else None

        if target_edicion is None and target_ediciones_activas is None:
            try:
                from src.agent_service.core.stores.product.catalog_resolver import get_vendor_catalog_resolver
                resolver = get_vendor_catalog_resolver()
                target_ediciones_activas = await resolver.get_active_editions_for_user(
                    user_id=target_user_id,
                    vendor_id=target_vendor_id,
                    marca=target_marca,
                )
            except Exception:
                target_ediciones_activas = None

        params: Dict[str, Any] = {
            "query_embedding": embedding_str,
            "query_text": clean_query,
            "lexical_tsquery": lexical_tsquery,
            "user_id": target_user_id,
            "vendor_id": target_vendor_id,
            "pagina": target_pagina,
            "edicion": target_edicion,
            "ediciones_activas": target_ediciones_activas if target_ediciones_activas else None,
            "marca": target_marca,
            "alpha": alpha,
            "k_candidates": max(k * 8, 60),
            "k": max(k * 2, 12),
            "ai_provider": self._ai_provider,
            "ai_model": self._ai_model,
        }

        if getattr(self._pool, "closed", False) is True:
            await self._pool.open()

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self._compiled_query, params)
                records = await cur.fetchall()

        raw_candidates: List[Dict[str, Any]] = []
        for row in records:
            pagina_val = row[8] if len(row) > 8 else None
            edicion_val = row[9] if len(row) > 9 else None
            marca_val = row[10] if len(row) > 10 else None
            vendor_name_val = row[11] if len(row) > 11 else None
            raw_candidates.append({
                "embedding_id": row[0],
                "product_id": row[1],
                "product_tmpl_id": row[2],
                "name": row[3],
                "source_text": row[4] or row[3] or "",
                "description": row[4] or row[3] or "",
                "vendor_id": row[5],
                "vendor_name": vendor_name_val,
                "sku": row[6],
                "rrf_score": float(row[7]),
                "pagina": pagina_val,
                "edicion": edicion_val,
                "marca": marca_val,
            })

        # 2. Búsqueda Híbrida Multimodal en product_catalog_embedding (Texto + Imagen 768d + FTS)
        #    y Reconciliación Regenerativa On-Demand (el catálogo siempre es la fuente de verdad)
        catalog_pages_matched = []
        page_scores: Dict[int, Dict[str, Any]] = {}
        try:
            catalog_pages_matched, page_scores = await self.acatalog_hybrid_search(
                query=clean_query,
                lexical_tsquery=lexical_tsquery,
                k=max(k, 8),
                alpha=alpha,
                pagina=target_pagina,
                vendor_id=target_vendor_id,
            )
        except Exception:
            catalog_pages_matched, page_scores = [], {}

        try:
            from src.agent_service.core.stores.product.catalog_reconciler import get_catalog_truth_reconciler
            reconciler = get_catalog_truth_reconciler(pool=self._pool)
            reconciled_candidates = await reconciler.reconcile_and_heal_candidates(
                candidates=raw_candidates,
                catalog_pages=catalog_pages_matched,
                persist_healing=True,
            )
        except Exception:
            reconciled_candidates = raw_candidates

        # 3. Fusión de scores (Product RRF + Catalog Multimodal RRF + Multimodal Image+Text Page Boost + Title Token Overlap)
        query_words = [w for w in re.findall(r"[a-záéíóúñ0-9]+", clean_query) if len(w) >= 4]
        fused_docs: List[Document] = []
        for cand in reconciled_candidates:
            p_num = cand.get("pagina")
            base_rrf = float(cand.get("rrf_score") or 0.0)
            cat_info = page_scores.get(int(p_num)) if p_num is not None else None
            cat_rrf = float(cat_info["catalog_rrf"]) if cat_info else 0.0
            multimodal_hit = bool(cat_info["multimodal_hit"]) if cat_info else False

            cand_name_low = str(cand.get("name") or "").lower()
            title_hits = sum(1 for qw in query_words if qw in cand_name_low)
            title_bonus = title_hits * 0.008 if title_hits >= 2 else (0.003 if title_hits == 1 else 0.0)

            if base_rrf > 0.0 and cat_rrf > 0.0:
                combined_rrf = base_rrf + (cat_rrf * 0.85) + (0.006 if multimodal_hit else 0.002) + title_bonus
            elif base_rrf > 0.0:
                combined_rrf = base_rrf + (0.002 if cand.get("catalog_verified") else 0.0) + title_bonus
            else:
                # Candidato descubierto directamente desde product_catalog_embedding
                combined_rrf = cat_rrf * 1.15 + (0.005 if multimodal_hit else 0.0) + title_bonus

            if target_pagina is not None and p_num is not None and int(p_num) != int(target_pagina):
                continue

            pagina_val = cand.get("pagina")
            edicion_val = cand.get("edicion")
            marca_val = cand.get("marca")
            vendor_id_val = cand.get("vendor_id")
            vendor_name_val = cand.get("vendor_name")

            fused_docs.append(
                Document(
                    page_content=cand.get("description") or cand.get("source_text") or cand.get("name") or "",
                    metadata={
                        "embedding_id": cand.get("embedding_id"),
                        "product_id": cand.get("product_id"),
                        "product_tmpl_id": cand.get("product_tmpl_id"),
                        "name": cand.get("name"),
                        "vendor_id": vendor_id_val,
                        "vendor_name": vendor_name_val,
                        "sku": cand.get("sku"),
                        "healed_from_sku": cand.get("healed_from_sku"),
                        "healing_reason": cand.get("healing_reason"),
                        "price": cand.get("price"),
                        "catalog_offer_price": cand.get("catalog_offer_price"),
                        "normal_price": cand.get("normal_price"),
                        "discount_tag": cand.get("discount_tag"),
                        "catalog_verified": bool(cand.get("catalog_verified")),
                        "multimodal_hit": multimodal_hit,
                        "catalog_variants": cand.get("catalog_variants"),
                        "page_image_url": cand.get("page_image_url"),
                        "rrf_score": round(combined_rrf, 6),
                        "pagina": pagina_val,
                        "edicion": edicion_val,
                        "marca": marca_val,
                        "tags": {
                            "pagina": pagina_val,
                            "edicion": edicion_val,
                            "marca": marca_val,
                            "vendor_id": vendor_id_val,
                            "vendor_name": vendor_name_val,
                        },
                    },
                )
            )

        fused_docs.sort(key=lambda d: float(d.metadata.get("rrf_score") or 0.0), reverse=True)

        # Garantizar diversidad de conceptos y páginas de catálogo en el corte Top-K
        # (como cada representante de página ya incluye todos sus tonos en catalog_variants, 1 representante por página/concepto maximiza la cobertura)
        from src.agent_service.core.stores.product.product_hierarchy import extract_base_concept_name
        diverse_docs: List[Document] = []
        overflow_docs: List[Document] = []
        page_concept_counts: Dict[str, int] = {}

        for doc in fused_docs:
            m = doc.metadata or {}
            c_name = extract_base_concept_name(str(m.get("name") or "")).lower().strip()
            p_num = m.get("pagina")
            group_key = f"{p_num}:{c_name}"
            cnt = page_concept_counts.get(group_key, 0)
            max_per_group = 1 if target_pagina is None else k
            if cnt < max_per_group:
                page_concept_counts[group_key] = cnt + 1
                diverse_docs.append(doc)
            else:
                overflow_docs.append(doc)

        if len(diverse_docs) < k:
            diverse_docs.extend(overflow_docs[: k - len(diverse_docs)])

        return diverse_docs[:k]

    async def acatalog_hybrid_search(
        self,
        query: str,
        lexical_tsquery: str,
        k: int = 8,
        alpha: float = 0.55,
        pagina: Optional[int] = None,
        vendor_id: Optional[int] = None,
    ) -> tuple[List[Any], Dict[int, Dict[str, Any]]]:
        """Ejecuta búsqueda híbrida sobre product_catalog_embedding combinando:
        1. Búsqueda semántica multimodal 768d (google/embeddinggemma-2) en chunks 'text' e 'image' vía Odoo REST API.
        2. Búsqueda léxica FTS (ts_rank_cd) en PostgreSQL sobre raw_page_text + source_text de product_catalog_embedding.
        Retorna (top_parsed_pages, page_scores_dict)."""
        from src.agent_service.core.stores.product.catalog_reconciler import get_catalog_truth_reconciler
        reconciler = get_catalog_truth_reconciler(pool=self._pool)
        idx = await reconciler.get_catalog_index()
        pages_by_num = idx.get("pages_by_num", {})
        latest_catalog_ids: List[int] = idx.get("latest_catalog_ids") or []
        if not pages_by_num:
            return [], {}

        # Si el usuario pidió una página puntual del catálogo, devolverla directamente con prioridad máxima
        if pagina is not None and int(pagina) in pages_by_num:
            pg = pages_by_num[int(pagina)]
            return [pg], {int(pagina): {"catalog_rrf": 0.05, "multimodal_hit": True}}

        sem_ranks_text: Dict[int, int] = {}
        sem_ranks_image: Dict[int, int] = {}

        if alpha > 0.0:
            try:
                from src.agent_service.config.odoo import get_odoo_settings
                from src.agent_service.core.stores.product.odoo_client import OdooClient
                settings = get_odoo_settings()
                primary_cat_id = latest_catalog_ids[0] if len(latest_catalog_ids) == 1 else None
                async with OdooClient(
                    base_url=settings.url,
                    db=settings.db,
                    username=settings.username,
                    password=settings.password,
                    timeout=6.0,
                ) as odoo_client:
                    api_results = await odoo_client.search_product_catalog_embeddings(
                        query=query,
                        limit=max(k * 3, 20),
                        catalog_id=primary_cat_id,
                        partner_id=vendor_id,
                        chunk_types=["text", "image"],
                    )
                t_rank = 1
                i_rank = 1
                for item in api_results:
                    pnum = item.get("page_number")
                    ctype = item.get("chunk_type")
                    cat_id_item = item.get("catalog_id")
                    if latest_catalog_ids and cat_id_item is not None and int(cat_id_item) not in latest_catalog_ids:
                        continue
                    if pnum is None:
                        continue
                    pnum_int = int(pnum)
                    if ctype == "text" and pnum_int not in sem_ranks_text:
                        sem_ranks_text[pnum_int] = t_rank
                        t_rank += 1
                    elif ctype == "image" and pnum_int not in sem_ranks_image:
                        sem_ranks_image[pnum_int] = i_rank
                        i_rank += 1
            except Exception:
                pass

        # Búsqueda léxica FTS en product_catalog_embedding (filtrada a los últimos catálogos vigentes por tags)
        lex_ranks: Dict[int, int] = {}
        catalog_lex_sql = """
            SELECT
                pce.page_number,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(
                        to_tsvector('spanish', COALESCE(pce.raw_page_text, '') || ' ' || COALESCE(pce.source_text, '')),
                        CASE
                            WHEN to_tsvector('spanish', COALESCE(pce.raw_page_text, '') || ' ' || COALESCE(pce.source_text, '')) @@ websearch_to_tsquery('spanish', %(query_text)s)
                            THEN websearch_to_tsquery('spanish', %(query_text)s)
                            ELSE to_tsquery('spanish', %(lexical_tsquery)s)
                        END
                    ) DESC
                ) AS rank_lex
            FROM product_catalog_embedding pce
            JOIN product_catalog pc ON pc.id = pce.catalog_id
            WHERE pc.active = TRUE
              AND pce.active = TRUE
              AND pce.chunk_type = 'text'
              AND (%(latest_catalog_ids)s::int[] IS NULL OR pce.catalog_id = ANY(%(latest_catalog_ids)s::int[]))
              AND (%(vendor_id)s::integer IS NULL OR pce.partner_id = %(vendor_id)s::integer)
              AND (
                  to_tsvector('spanish', COALESCE(pce.raw_page_text, '') || ' ' || COALESCE(pce.source_text, '')) @@ websearch_to_tsquery('spanish', %(query_text)s)
                  OR to_tsvector('spanish', COALESCE(pce.raw_page_text, '') || ' ' || COALESCE(pce.source_text, '')) @@ to_tsquery('spanish', %(lexical_tsquery)s)
              )
            LIMIT %(limit)s;
        """
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        catalog_lex_sql,
                        {
                            "query_text": query,
                            "lexical_tsquery": lexical_tsquery,
                            "latest_catalog_ids": latest_catalog_ids if latest_catalog_ids else None,
                            "vendor_id": vendor_id,
                            "limit": max(k * 3, 24),
                        },
                    )
                    rows = await cur.fetchall()
                    for r in rows:
                        lex_ranks[int(r[0])] = int(r[1])
        except Exception:
            pass

        all_candidate_pages = set(sem_ranks_text.keys()) | set(sem_ranks_image.keys()) | set(lex_ranks.keys())
        page_scores: Dict[int, Dict[str, Any]] = {}
        for pnum in all_candidate_pages:
            if pnum not in pages_by_num:
                continue
            r_text = sem_ranks_text.get(pnum)
            r_img = sem_ranks_image.get(pnum)
            r_lex = lex_ranks.get(pnum)

            sem_score = 0.0
            if r_text is not None:
                sem_score += 0.70 / (60.0 + r_text)
            if r_img is not None:
                sem_score += 0.30 / (60.0 + r_img)
            lex_score = (1.0 / (60.0 + r_lex)) if r_lex is not None else 0.0

            multimodal_hit = (r_text is not None and r_img is not None) or (r_img is not None and r_lex is not None)
            rrf_val = (alpha * sem_score) + ((1.0 - alpha) * lex_score) + (0.004 if multimodal_hit else 0.0)
            page_scores[pnum] = {
                "catalog_rrf": rrf_val,
                "multimodal_hit": multimodal_hit,
                "rank_text": r_text,
                "rank_image": r_img,
                "rank_lex": r_lex,
            }

            # Si una página izquierda de doble pliego (spread) no tiene SKUs (ej. Pág. 70 foto de portada de Base Resveratrol)
            # pero su página derecha contigua (Pág. 71) tiene los tonos y SKUs, propagar el score multimodal al pliego hermano
            if not pages_by_num[pnum].skus and (pnum + 1) in pages_by_num and pages_by_num[pnum + 1].skus:
                next_p = pnum + 1
                prev_entry = page_scores.get(next_p)
                if not prev_entry or rrf_val > prev_entry["catalog_rrf"]:
                    page_scores[next_p] = {
                        "catalog_rrf": rrf_val,
                        "multimodal_hit": True,
                        "rank_text": r_text,
                        "rank_image": r_img,
                        "rank_lex": r_lex,
                    }

        sorted_pnums = sorted(page_scores.keys(), key=lambda pn: page_scores[pn]["catalog_rrf"], reverse=True)
        top_pages = [pages_by_num[pn] for pn in sorted_pnums if pages_by_num[pn].skus][: max(k // 2, 4)]
        return top_pages, page_scores

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return await self.ahybrid_search(
            query=query, 
            k=k, 
            alpha=1.0, 
            filters=kwargs.get("filters")
        )

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        raise NotImplementedError("Usar métodos asíncronos (ahybrid_search o asimilarity_search).")

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, **kwargs: Any):
        raise NotImplementedError("Poblado mediante pipeline dedicado.")