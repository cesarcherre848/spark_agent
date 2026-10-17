import pytest
from langchain_core.documents import Document

from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.core.stores.product.catalog_reconciler import (
    CatalogPageParser,
    CatalogTruthReconciler,
    get_catalog_truth_reconciler,
)
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.tools.product_tools import (
    rerank_and_filter_candidates,
    validate_product_ownership,
)
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode


@pytest.mark.asyncio
async def test_catalog_page_parser_strips_nsoc_and_extracts_true_skus():
    """Verifica que CatalogPageParser jamás confunda un código sanitario NSOC con un SKU
    y extraiga todos los tonos/SKUs y precios de oferta de la página."""
    reconciler = get_catalog_truth_reconciler()
    idx = await reconciler.get_catalog_index(force_refresh=True)
    pages = idx["pages_by_num"]

    assert len(pages) == 132

    # Página 71: Base Antiedad con Resveratrol y Péptidos (NSOC49608-26CO vs SKUs 5106..5115)
    p71 = pages[71]
    assert "49608" in p71.nsoc_numbers
    assert "49608" not in p71.skus
    assert "5108" in p71.skus
    assert p71.primary_offer_price == 73.5
    assert p71.primary_normal_price == 134.0

    # Página 93: Ya! Barra Multiusos 3 en 1 Sunset Bronzer (NSOC55587-22PE vs SKUs 6140, 5441..5443, 6261)
    p93 = pages[93]
    assert "55587" in p93.nsoc_numbers
    assert "55587" not in p93.skus
    assert "6261" in p93.skus
    assert "Bronzer" in p93.variants["6261"].label
    assert p93.primary_offer_price == 29.0
    assert p93.primary_normal_price == 58.0

    # Página 76: Rubor Compacto (múltiples tonos: Petal Rose 4012, Bronzage 4017, Sunglow 6361, etc.)
    p76 = pages[76]
    assert "4012" in p76.variants
    assert p76.variants["4012"].label == "Petal Rose"
    assert "4017" in p76.variants
    assert p76.variants["4017"].label == "Bronzage"
    assert "6361" in p76.variants
    assert "Sunglow" in p76.variants["6361"].label


@pytest.mark.asyncio
async def test_regenerative_on_demand_sku_healing():
    """Verifica que CatalogTruthReconciler detecte y corrija on-demand:
    1. Códigos sanitarios NSOC guardados por error como SKU (49608 -> 5108, 55587 -> 6261)
    2. Prefijos sucios ('CÓD. 696' -> '696')
    3. SKUs desactualizados respecto al catálogo C11 (Petal Rose 520 -> 4012, Zentro 79896 -> 212,
       Temptation Black 114 -> 2192, Dendur Destiny 724 -> 2201, Té Verde 262 -> 79901)."""
    reconciler = CatalogTruthReconciler(pool=get_db_pool())
    discrepant_samples = [
        {"product_id": 599, "sku": "49608", "name": "Base Antiedad con Resveratrol y Péptidos", "pagina": 71, "price": 134.0},
        {"product_id": 608, "sku": "55587", "name": "Ya! Barra Multiusos 3 en 1 - Sunset Bronzer", "pagina": 93, "price": 58.0},
        {"product_id": 249, "sku": "CÓD. 696", "name": "Bálsamo Labial Restaurador", "pagina": None, "price": 55.0},
        {"product_id": 49, "sku": "520", "name": "Rubor Compacto Petal Rose", "pagina": 82, "price": 96.0},
        {"product_id": 293, "sku": "79896", "name": "Zentro Eau de Parfum", "pagina": 122, "price": 220.0},
        {"product_id": 296, "sku": "114", "name": "Temptation Black", "pagina": 128, "price": 202.0},
        {"product_id": 313, "sku": "724", "name": "Dendur Destiny", "pagina": None, "price": 192.0},
        {"product_id": 47, "sku": "262", "name": "Té Verde", "pagina": 66, "price": 104.0},
    ]

    healed = await reconciler.reconcile_and_heal_candidates(
        candidates=discrepant_samples,
        persist_healing=True,
    )
    by_id = {h["product_id"]: h for h in healed}

    assert by_id[599]["sku"] == "5108"
    assert by_id[599]["pagina"] == 71
    assert by_id[599]["price"] == 73.5

    assert by_id[608]["sku"] == "6261"
    assert by_id[608]["pagina"] == 93
    assert by_id[608]["price"] == 29.0

    assert by_id[249]["sku"] == "696"
    assert by_id[249]["pagina"] == 29

    assert by_id[49]["sku"] == "4012"
    assert by_id[49]["pagina"] == 76
    assert by_id[49]["price"] == 57.5

    assert by_id[293]["sku"] == "212"
    assert by_id[293]["pagina"] == 113
    assert by_id[293]["price"] == 121.0

    assert by_id[296]["sku"] == "2192"
    assert by_id[296]["pagina"] == 117
    assert by_id[296]["price"] == 101.0

    assert by_id[313]["sku"] == "2201"
    assert by_id[313]["pagina"] == 120
    assert by_id[313]["price"] == 96.0

    assert by_id[47]["sku"] == "79901"
    assert by_id[47]["pagina"] == 51
    assert by_id[47]["price"] == 46.5


@pytest.mark.asyncio
async def test_hybrid_multimodal_search_and_reranking():
    """Verifica la búsqueda híbrida fusionada (product_vector_embedding + product_catalog_embedding multimodal 768d)
    y el reranker contextual enriquecido con precios de oferta y variantes de catálogo."""
    pool = get_db_pool()
    emb = get_embedding_service()
    vs = ProductVectorStore(pool=pool, embedding_service=emb)

    docs = await vs.ahybrid_search("Rubor Compacto para piel trigueña Bronzage", k=6)
    assert len(docs) > 0

    candidates = [
        {
            "sku": d.metadata.get("sku"),
            "name": d.metadata.get("name"),
            "description": d.page_content,
            "price": d.metadata.get("price"),
            "normal_price": d.metadata.get("normal_price"),
            "pagina": d.metadata.get("pagina"),
            "edicion": d.metadata.get("edicion"),
            "marca": d.metadata.get("marca"),
            "catalog_verified": d.metadata.get("catalog_verified"),
            "multimodal_hit": d.metadata.get("multimodal_hit"),
            "catalog_variants": d.metadata.get("catalog_variants"),
        }
        for d in docs
    ]

    reranked = rerank_and_filter_candidates.invoke({
        "candidates": candidates,
        "context_query": "Rubor Compacto para piel trigueña Bronzage",
        "target_item_count": 3,
    })
    assert len(reranked) > 0
    top = reranked[0]
    assert top["pagina"] == 76
    assert top["price"] == 57.5
    assert top.get("catalog_variants") is not None

    ctx_text = BaseSynthesizerNode.format_products_context(reranked)
    assert "4017" in ctx_text
    assert "Bronzage" in ctx_text

    # Verificar que validate_product_ownership autoriza los SKUs del catálogo
    own_res = await validate_product_ownership.ainvoke({"user_id": 8, "skus": ["4017", "6261", "5108"]})
    for s in ("4017", "6261", "5108"):
        assert own_res["results"][s]["is_owner"] is True
