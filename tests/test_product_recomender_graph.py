"""
tests/test_product_recomender_graph.py - Pruebas completas del subgrafo product_recomender

Valida:
1. Extracción de intención, top-k dinámico y restricciones de presupuesto/categoría.
2. Regla de sobre-recuperación 2x (retrieval_k >= max(top_k * 2, 6)).
3. Enriquecimiento oficial de precios, categorías y unidades desde Odoo ERP.
4. Filtrado determinista de usuario (presupuesto máx/mín, exclusión del producto base).
5. Evaluación con Rúbrica multi-criterio y ciclo de reflexión (Evaluator-Optimizer Loop).
6. Límite de seguridad contra ciclos infinitos (max_iterations fallback).
7. Síntesis ejecutiva comercial y no exposición de IDs técnicos de base de datos.
8. Enrutamiento del grafo principal (main_graph) hacia product_recomender.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.core.stores.product.odoo_client import OdooClient
from src.agent_service.core.stores.product.schemas import (
    ProductOdooDetail,
    GetProductBySkusOutput,
    ProductCatalogFilter,
)
from src.agent_service.graph.sub_graphs.product_recomender.graph import (
    build_product_recomender_graph,
    _route_after_rubric,
)
from src.agent_service.graph.sub_graphs.product_recomender.nodes import ProductRecomenderNodes
from src.agent_service.graph.sub_graphs.product_recomender.state import ProductRecomenderState
from src.agent_service.graph.sub_graphs.product_recomender.schemas import (
    RecommendationIntentExtraction,
    RubricCriteriaScores,
    RubricEvaluationResult,
    RecommendationSynthesisResponse,
)
from src.agent_service.graph.main_graph import (
    MainGraphState,
    build_main_graph,
    RouterDecision,
)


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=ProductVectorStore)
    store.ahybrid_search = AsyncMock()
    return store


@pytest.fixture
def mock_odoo_client():
    client = MagicMock(spec=OdooClient)
    client.get_products_by_skus = AsyncMock()
    return client


@pytest.fixture
def sample_vector_docs():
    """Colección de 6 candidatos para probar sobre-recuperación y filtrado."""
    return [
        Document(
            page_content="Labial mate de larga duración color rojo pasión.",
            metadata={
                "product_id": 101,
                "sku": "LAB-ROJO-101",
                "name": "Labial Mate Rojo Pasión",
                "rrf_score": 0.95,
            },
        ),
        Document(
            page_content="Delineador retráctil para labios color rojo intenso.",
            metadata={
                "product_id": 102,
                "sku": "DEL-ROJO-102",
                "name": "Delineador de Labios Rojo",
                "rrf_score": 0.90,
            },
        ),
        Document(
            page_content="Brillo labial hidratante con vitamina E efecto gloss.",
            metadata={
                "product_id": 103,
                "sku": "GLOSS-VIT-103",
                "name": "Brillo Labial Gloss Hidratante",
                "rrf_score": 0.85,
            },
        ),
        Document(
            page_content="Esmalte de secado rápido color rojo cereza.",
            metadata={
                "product_id": 104,
                "sku": "ESM-CEREZA-104",
                "name": "Esmalte Gel Rojo Cereza",
                "rrf_score": 0.80,
            },
        ),
        Document(
            page_content="Bálsamo labial reparador con manteca de karité.",
            metadata={
                "product_id": 105,
                "sku": "BAL-KARITE-105",
                "name": "Bálsamo Labial Reparador Karité",
                "rrf_score": 0.75,
            },
        ),
        Document(
            page_content="Set exclusivo de brochas profesionales para maquillaje facial y labios.",
            metadata={
                "product_id": 106,
                "sku": "SET-BROCHAS-106",
                "name": "Set Brochas Profesionales Luxury",
                "rrf_score": 0.70,
            },
        ),
    ]


@pytest.fixture
def sample_odoo_details():
    prods = [
        ProductOdooDetail(
            product_id=101,
            sku="LAB-ROJO-101",
            name="Labial Mate Rojo Pasión",
            price=45.0,
            currency="PEN",
            category="Maquillaje / Labios",
            uom="Unidades",
            sales_description="Labial mate textura terciopelo.",
        ),
        ProductOdooDetail(
            product_id=102,
            sku="DEL-ROJO-102",
            name="Delineador de Labios Rojo",
            price=25.0,
            currency="PEN",
            category="Maquillaje / Labios",
            uom="Unidades",
            sales_description="Delineador de alta precisión.",
        ),
        ProductOdooDetail(
            product_id=103,
            sku="GLOSS-VIT-103",
            name="Brillo Labial Gloss Hidratante",
            price=30.0,
            currency="PEN",
            category="Maquillaje / Labios",
            uom="Unidades",
            sales_description="Brillo hidratante con efecto volumen.",
        ),
        ProductOdooDetail(
            product_id=104,
            sku="ESM-CEREZA-104",
            name="Esmalte Gel Rojo Cereza",
            price=18.0,
            currency="PEN",
            category="Uñas",
            uom="Unidades",
            sales_description="Esmalte de larga duración efecto gel.",
        ),
        ProductOdooDetail(
            product_id=105,
            sku="BAL-KARITE-105",
            name="Bálsamo Labial Reparador Karité",
            price=15.0,
            currency="PEN",
            category="Maquillaje / Labios",
            uom="Unidades",
            sales_description="Cuidado intensivo para labios resecos.",
        ),
        ProductOdooDetail(
            product_id=106,
            sku="SET-BROCHAS-106",
            name="Set Brochas Profesionales Luxury",
            price=120.0,  # Producto caro para probar filtro de presupuesto
            currency="PEN",
            category="Accesorios",
            uom="Set",
            sales_description="Set de 12 brochas de pelo sintético suave.",
        ),
    ]
    return GetProductBySkusOutput(products=prods, not_found_skus=[])


# ==============================================================================
# TESTS UNITARIOS DE NODOS Y REGLAS DE NEGOCIO
# ==============================================================================
@pytest.mark.asyncio
async def test_extract_recommendation_intent_dynamic_top_k():
    """Valida la extracción estructurada de intención, presupuesto y top-k dinámico."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = RecommendationIntentExtraction(
        base_product="LAB-ROJO-101",
        relation_type="cross_sell",
        top_k=2,
        min_price=10.0,
        max_price=35.0,
        category="Maquillaje",
        required_attributes=["mate"],
        search_query="delineador o brillo complementario para labial rojo",
    )
    mock_llm.with_structured_output.return_value = mock_extractor

    nodes = ProductRecomenderNodes(llm=mock_llm, default_top_k=3)
    state: ProductRecomenderState = {
        "raw_query": "Recomiéndame 2 productos complementarios para el labial rojo LAB-ROJO-101 con presupuesto de hasta 35 soles"
    }

    result = await nodes.extract_recommendation_intent(state)

    assert result["relation_type"] == "cross_sell"
    assert result["top_k"] == 2
    assert result["base_product"] == "LAB-ROJO-101"
    assert result["filters"]["max_price"] == 35.0
    assert result["filters"]["min_price"] == 10.0
    assert result["search_query"] == "delineador o brillo complementario para labial rojo"


@pytest.mark.asyncio
async def test_retrieval_3x_overfetch_rule(mock_vector_store, sample_vector_docs):
    """Valida que la fase de recuperación aplique la regla de 3x sobre-recuperación:
    retrieval_k = max(target_top_k * 3, 6), y por defecto top_k=15.
    """
    mock_vector_store.ahybrid_search.return_value = sample_vector_docs
    mock_llm = MagicMock(spec=BaseChatModel)

    nodes = ProductRecomenderNodes(
        llm=mock_llm,
        vector_store=mock_vector_store,
        default_top_k=15,
    )

    # Caso A: top_k = 2 -> 2 * 3 = 6 => retrieval_k = 6
    state_a: ProductRecomenderState = {
        "search_query": "labiales complementarios",
        "top_k": 2,
    }
    await nodes.retrieve_candidate_products(state_a)
    mock_vector_store.ahybrid_search.assert_awaited_with(query="labiales complementarios", k=6, filters=None)

    # Caso B: top_k = 4 -> 4 * 3 = 12 => retrieval_k = 12
    state_b: ProductRecomenderState = {
        "search_query": "sombras y delineadores",
        "top_k": 4,
    }
    await nodes.retrieve_candidate_products(state_b)
    mock_vector_store.ahybrid_search.assert_awaited_with(query="sombras y delineadores", k=12, filters=None)

    # Caso C: por defecto top_k = 15 -> 15 * 3 = 45 => retrieval_k = 45
    state_c: ProductRecomenderState = {
        "search_query": "productos cuidado facial",
    }
    await nodes.retrieve_candidate_products(state_c)
    mock_vector_store.ahybrid_search.assert_awaited_with(query="productos cuidado facial", k=45, filters=None)


@pytest.mark.asyncio
async def test_enrich_product_details_with_odoo(mock_odoo_client, sample_vector_docs, sample_odoo_details):
    """Valida la consulta cruzada con Odoo ERP para enriquecer candidatos con precios y ficha técnica."""
    mock_odoo_client.get_products_by_skus.return_value = sample_odoo_details
    mock_llm = MagicMock(spec=BaseChatModel)

    nodes = ProductRecomenderNodes(
        llm=mock_llm,
        odoo_client=mock_odoo_client,
    )

    state: ProductRecomenderState = {
        "candidate_documents": sample_vector_docs,
    }

    result = await nodes.enrich_product_details(state)
    enriched = result["enriched_products"]

    assert len(enriched) == 6
    mock_odoo_client.get_products_by_skus.assert_awaited_once()

    # Verificar que el producto tiene los datos de Odoo asociados correctamente
    item_delineador = next(p for p in enriched if p["sku"] == "DEL-ROJO-102")
    assert item_delineador["price"] == 25.0
    assert item_delineador["currency"] == "PEN"
    assert item_delineador["category"] == "Maquillaje / Labios"


@pytest.mark.asyncio
async def test_apply_user_filters_and_exclude_base_product(sample_odoo_details):
    """Valida el filtrado determinista:
    1. Excluye el producto base para no auto-recomendar el mismo producto.
    2. Excluye productos con precio mayor a max_price_budget (SET-BROCHAS-106 a 120 > 50).
    3. Excluye productos con precio menor a min_price_budget si existiera.
    """
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductRecomenderNodes(llm=mock_llm)

    enriched_list = [
        {
            "product_id": detail.product_id,
            "sku": detail.sku,
            "name": detail.name,
            "description": detail.sales_description,
            "price": detail.price,
            "currency": detail.currency,
            "category": detail.category,
            "uom": detail.uom,
        }
        for detail in sample_odoo_details.products
    ]

    state: ProductRecomenderState = {
        "enriched_products": enriched_list,
        "base_product": "LAB-ROJO-101",  # Se debe excluir
        "filters": {
            "max_price": 50.0,  # Excluye SET-BROCHAS-106 (120.0)
            "min_price": 16.0,  # Excluye BAL-KARITE-105 (15.0)
        },
    }

    result = await nodes.apply_user_filters(state)
    filtered = result["filtered_products"]
    remaining_skus = [p["sku"] for p in filtered]

    # LAB-ROJO-101 excluido por base product
    assert "LAB-ROJO-101" not in remaining_skus
    # SET-BROCHAS-106 excluido por precio > 50.0
    assert "SET-BROCHAS-106" not in remaining_skus
    # BAL-KARITE-105 excluido por precio < 16.0
    assert "BAL-KARITE-105" not in remaining_skus
    # DEL-ROJO-102 (25.0), GLOSS-VIT-103 (30.0), ESM-CEREZA-104 (18.0) deben estar incluidos
    assert "DEL-ROJO-102" in remaining_skus
    assert "GLOSS-VIT-103" in remaining_skus
    assert "ESM-CEREZA-104" in remaining_skus


# ==============================================================================
# TESTS DEL FLUJO COMPILADO CON PATRÓN DE REFLEXIÓN (EVALUATOR-OPTIMIZER)
# ==============================================================================
@pytest.mark.asyncio
async def test_product_recomender_happy_path(
    mock_vector_store, mock_odoo_client, sample_vector_docs, sample_odoo_details
):
    """Test Happy Path:
    START -> extract -> retrieve (2x) -> enrich (Odoo) -> filter -> judge (meets_rubric=True) -> synthesize -> END
    """
    mock_vector_store.ahybrid_search.return_value = sample_vector_docs
    mock_odoo_client.get_products_by_skus.return_value = sample_odoo_details

    # Mocks para structured output
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = RecommendationIntentExtraction(
        base_product="LAB-ROJO-101",
        relation_type="cross_sell",
        top_k=2,
        search_query="complementos de maquillaje labial rojo",
        max_price=40.0,
    )

    mock_judge = AsyncMock()
    mock_judge.ainvoke.return_value = RubricEvaluationResult(
        meets_rubric=True,
        scores=RubricCriteriaScores(
            relevance=5.0,
            filter_compliance=5.0,
            diversity=4.0,
            data_completeness=5.0,
        ),
        critique="Los productos DEL-ROJO-102 y GLOSS-VIT-103 son altamente relevantes y cumplen el presupuesto.",
        selected_skus=["DEL-ROJO-102", "GLOSS-VIT-103"],
        suggested_refinement=None,
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = RecommendationSynthesisResponse(
        response_text=(
            "Para acompañar tu Labial Mate Rojo, te sugerimos dos excelentes complementos:\n"
            "- [DEL-ROJO-102] Delineador de Labios Rojo: $25.00 (PEN)\n"
            "- [GLOSS-VIT-103] Brillo Labial Gloss Hidratante: $30.00 (PEN)\n"
            "¿Deseas que agreguemos alguna de estas opciones a tu cotización?"
        )
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind.return_value = mock_llm

    def structured_side_effect(schema, **kwargs):
        if schema == RecommendationIntentExtraction:
            return mock_extractor
        elif schema == RubricEvaluationResult:
            return mock_judge
        elif schema == RecommendationSynthesisResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = structured_side_effect

    app = build_product_recomender_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        odoo_client=mock_odoo_client,
        default_top_k=2,
    )

    initial_state = {
        "raw_query": "Recomiéndame 2 productos para complementar mi labial rojo LAB-ROJO-101 por menos de 40 soles",
        "user_id": 1,
    }

    final_state = await app.ainvoke(initial_state)

    # Aserciones
    assert final_state["meets_rubric"] is True
    assert final_state["iteration_count"] == 0
    assert len(final_state["final_recommended_products"]) == 2
    assert "Delineador de Labios Rojo" in final_state["final_response"]
    assert "25.00" in final_state["final_response"]
    assert "product_id" not in final_state["final_response"]  # No filtra IDs técnicos

    # Sobre-recuperación 2x ejecutada
    mock_vector_store.ahybrid_search.assert_awaited_once_with(
        query="complementos de maquillaje labial rojo", k=6, filters=ProductCatalogFilter(user_id=1)
    )
    mock_judge.ainvoke.assert_awaited_once()
    mock_synthesizer.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_product_recomender_reflection_loop(
    mock_vector_store, mock_odoo_client, sample_vector_docs, sample_odoo_details
):
    """Test Ciclo de Auto-Reflexión (Evaluator-Optimizer Loop):
    1a iteración: Rúbrica rechaza (meets_rubric=False) -> reflection_optimizer refina query.
    2a iteración: Recupera candidatos con nueva query -> Rúbrica aprueba (meets_rubric=True) -> synthesize.
    """
    mock_vector_store.ahybrid_search.return_value = sample_vector_docs
    mock_odoo_client.get_products_by_skus.return_value = sample_odoo_details

    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = RecommendationIntentExtraction(
        base_product="Perfume Floral",
        relation_type="cross_sell",
        top_k=2,
        search_query="algo aromático",
    )

    # El juez rechaza en iteración 1 y aprueba en iteración 2
    mock_judge = AsyncMock()
    mock_judge.ainvoke.side_effect = [
        RubricEvaluationResult(
            meets_rubric=False,
            scores=RubricCriteriaScores(
                relevance=2.0,
                filter_compliance=3.0,
                diversity=2.0,
                data_completeness=4.0,
            ),
            critique="Los productos recuperados son maquillaje de labios en lugar de fragancias o cuidado corporal aromático.",
            selected_skus=[],
            suggested_refinement="Buscar específicamente lociones corporales o cremas perfumadas florales.",
        ),
        RubricEvaluationResult(
            meets_rubric=True,
            scores=RubricCriteriaScores(
                relevance=5.0,
                filter_compliance=5.0,
                diversity=5.0,
                data_completeness=5.0,
            ),
            critique="Excelente complementariedad olfativa y corporal.",
            selected_skus=["DEL-ROJO-102", "GLOSS-VIT-103"],
        ),
    ]

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = RecommendationSynthesisResponse(
        response_text="Te recomendamos estas alternativas aromáticas complementarias:\n- [DEL-ROJO-102] Delineador de Labios Rojo: $25.00"
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind.return_value = mock_llm
    # Mock para el LLM de reformulación (ainvoke directo)
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="lociones corporales y cremas perfumadas florales complementarias")
    )

    def structured_side_effect(schema, **kwargs):
        if schema == RecommendationIntentExtraction:
            return mock_extractor
        elif schema == RubricEvaluationResult:
            return mock_judge
        elif schema == RecommendationSynthesisResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = structured_side_effect

    app = build_product_recomender_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        odoo_client=mock_odoo_client,
        default_top_k=2,
        default_max_iterations=2,
    )

    initial_state = {
        "raw_query": "Recomiéndame algo para complementar mi perfume floral",
        "user_id": 1,
    }

    final_state = await app.ainvoke(initial_state)

    # Verificaciones
    assert final_state["iteration_count"] == 1
    assert final_state["meets_rubric"] is True
    assert mock_vector_store.ahybrid_search.await_count == 2
    assert mock_judge.ainvoke.await_count == 2
    mock_synthesizer.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_product_recomender_max_iterations_fallback(
    mock_vector_store, mock_odoo_client, sample_vector_docs, sample_odoo_details
):
    """Test Límite de Iteraciones (evita bucles infinitos):
    Si la Rúbrica rechaza persistentemente, al alcanzar max_iterations (2)
    se enruta a synthesize_recommendations para presentar las opciones más cercanas.
    """
    mock_vector_store.ahybrid_search.return_value = sample_vector_docs
    mock_odoo_client.get_products_by_skus.return_value = sample_odoo_details

    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = RecommendationIntentExtraction(
        base_product=None,
        relation_type="substitute",
        top_k=2,
        search_query="producto inexistente en catálogo espacial",
    )

    # El juez rechaza siempre
    mock_judge = AsyncMock()
    mock_judge.ainvoke.return_value = RubricEvaluationResult(
        meets_rubric=False,
        scores=RubricCriteriaScores(
            relevance=1.0,
            filter_compliance=2.0,
            diversity=1.0,
            data_completeness=3.0,
        ),
        critique="No hay productos del catálogo que satisfagan esta solicitud.",
        selected_skus=[],
        suggested_refinement="Ampliar la categoría a productos generales.",
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = RecommendationSynthesisResponse(
        response_text="No encontramos productos con todas las características exactas solicitadas, pero te compartimos las opciones más cercanas de nuestro catálogo."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind.return_value = mock_llm
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="productos cosméticos generales"))

    def structured_side_effect(schema, **kwargs):
        if schema == RecommendationIntentExtraction:
            return mock_extractor
        elif schema == RubricEvaluationResult:
            return mock_judge
        elif schema == RecommendationSynthesisResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = structured_side_effect

    app = build_product_recomender_graph(
        llm=mock_llm,
        vector_store=mock_vector_store,
        odoo_client=mock_odoo_client,
        default_top_k=2,
        default_max_iterations=2,
    )

    initial_state = {
        "raw_query": "Recomiéndame 2 productos espaciales",
        "user_id": 1,
    }

    final_state = await app.ainvoke(initial_state)

    # Verificaciones: alcanzó el límite de 2 iteraciones y sintetizó
    assert final_state["iteration_count"] == 2
    assert final_state["meets_rubric"] is False
    assert mock_vector_store.ahybrid_search.await_count == 3  # initial + 2 reflections
    mock_synthesizer.ainvoke.assert_awaited_once()
    assert "No encontramos productos con todas las características" in final_state["final_response"]


def test_route_after_rubric_helper():
    """Valida el enrutador condicional de la rúbrica."""
    # Aprobado en primera vuelta
    assert _route_after_rubric({"meets_rubric": True, "iteration_count": 0, "max_iterations": 2}) == "synthesize_recommendations"
    # Rechazado en primera vuelta -> reflexionar
    assert _route_after_rubric({"meets_rubric": False, "iteration_count": 0, "max_iterations": 2}) == "reflection_optimizer"
    # Rechazado en segunda vuelta (alcanza max_iterations) -> sintetizar fallback
    assert _route_after_rubric({"meets_rubric": False, "iteration_count": 2, "max_iterations": 2}) == "synthesize_recommendations"


# ==============================================================================
# TEST DE ENRUTAMIENTO EN MAIN GRAPH
# ==============================================================================
@pytest.mark.asyncio
async def test_main_graph_routes_to_product_recomender():
    """Valida que el Router clasifique consultas de recomendación cruzada como 'recommender'
    y despache la ejecución al subgrafo product_recomender.
    """
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind = MagicMock(return_value=mock_llm)

    # Router clasifica como 'recommender'
    router_decision = RouterDecision(
        intent="recommender",
        reasoning="El usuario solicita recomendaciones de venta cruzada con presupuesto específico",
    )
    mock_llm.with_structured_output = MagicMock(
        return_value=AsyncMock(ainvoke=AsyncMock(return_value=router_decision))
    )

    # Subgrafo product_recomender simulado con StateGraph
    builder_recom = StateGraph(MainGraphState)
    builder_recom.add_node(
        "recom_exec",
        lambda s: {
            "final_response": "Te recomendamos el Delineador Rojo a PEN 25.0 como complemento perfecto.",
            "recommended_products": [{"sku": "DEL-ROJO-102", "name": "Delineador de Labios Rojo"}],
        },
    )
    builder_recom.add_edge(START, "recom_exec")
    builder_recom.add_edge("recom_exec", END)
    dummy_recom = builder_recom.compile()

    main_app = build_main_graph(
        llm=mock_llm,
        recommender_graph=dummy_recom,
        memory_store=None,
    )

    config = {"configurable": {"thread_id": "test-thread-recom"}}
    result = await main_app.ainvoke(
        {
            "raw_query": "¿Qué producto puedo recomendarle a una clienta que compró el labial rojo por menos de 30 soles?",
            "user_id": 99,
        },
        config=config,
    )

    assert result["intent"] == "recommender"
    assert "Delineador Rojo" in result["final_response"]


@pytest.mark.asyncio
async def test_extract_recommendation_intent_with_sort_by_price_asc():
    """Valida la extracción estructurada del criterio sort_by ('price_asc') ante superlativos de precio."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = RecommendationIntentExtraction(
        base_product=None,
        relation_type="general_recommendation",
        top_k=15,
        category="labiales",
        sort_by="price_asc",
        search_query="labiales mas baratos del catalogo",
    )
    mock_llm.with_structured_output.return_value = mock_extractor

    nodes = ProductRecomenderNodes(llm=mock_llm)
    state: ProductRecomenderState = {
        "raw_query": "recomiendame los labiales mas baratos que hay en el catalogo"
    }

    result = await nodes.extract_recommendation_intent(state)

    assert result["relation_type"] == "general_recommendation"
    assert result["top_k"] == 15
    assert result["sort_by"] == "price_asc"
    assert result["filters"]["category"] == "labiales"


@pytest.mark.asyncio
async def test_apply_user_filters_deterministic_price_sorting():
    """Valida que apply_user_filters ordene matemáticamente de menor a mayor precio con sort_by='price_asc'."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductRecomenderNodes(llm=mock_llm)

    unordered_candidates = [
        {"sku": "LIP-81", "name": "BB Lips Natural", "price": 81.50, "category": "General"},
        {"sku": "LIP-19", "name": "Hydra-Lip Barra", "price": 19.50, "category": "General"},
        {"sku": "LIP-127", "name": "Labial Elixir Lujo", "price": 127.00, "category": "General"},
        {"sku": "LIP-33", "name": "Hydra-Lip Brillo Gloss", "price": 33.00, "category": "General"},
    ]

    state_asc: ProductRecomenderState = {
        "enriched_products": unordered_candidates,
        "sort_by": "price_asc",
    }
    result_asc = await nodes.apply_user_filters(state_asc)
    prices_asc = [p["price"] for p in result_asc["filtered_products"]]
    assert prices_asc == [19.50, 33.00, 81.50, 127.00]

    state_desc: ProductRecomenderState = {
        "enriched_products": unordered_candidates,
        "sort_by": "price_desc",
    }
    result_desc = await nodes.apply_user_filters(state_desc)
    prices_desc = [p["price"] for p in result_desc["filtered_products"]]
    assert prices_desc == [127.00, 81.50, 33.00, 19.50]


@pytest.mark.asyncio
async def test_apply_user_filters_category_stem_and_semantic_matching():
    """Valida que candidatos claramente discrepantes (perfumes, acondicionador) sean descartados si la categoría es 'labiales'."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductRecomenderNodes(llm=mock_llm)

    mixed_candidates = [
        {"sku": "PERF-01", "name": "Dendur Perfume Masculino", "price": 124.0, "category": "General", "description": "Aroma oriental amaderado."},
        {"sku": "LIP-01", "name": "Hydra-Lip Brillo Labial", "price": 33.0, "category": "General", "description": "Brillo traslúcido para labios."},
        {"sku": "LIP-02", "name": "BB Lips Natural Balm", "price": 81.5, "category": "General", "description": "Bálsamo hidratante reparador de labios."},
        {"sku": "HAIR-01", "name": "Bio Milk Acondicionador", "price": 40.0, "category": "General", "description": "Acondicionador nutritivo para cabello."},
    ]

    state: ProductRecomenderState = {
        "enriched_products": mixed_candidates,
        "filters": {"category": "labiales"},
        "sort_by": "price_asc",
    }

    result = await nodes.apply_user_filters(state)
    filtered = result["filtered_products"]
    remaining_skus = [p["sku"] for p in filtered]

    assert "LIP-01" in remaining_skus
    assert "LIP-02" in remaining_skus
    assert "PERF-01" not in remaining_skus
    assert "HAIR-01" not in remaining_skus
    assert filtered[0]["sku"] == "LIP-01"  # 33.00 < 81.50


@pytest.mark.asyncio
async def test_resilient_structured_output_recovers_markdown_json():
    """Valida que ResilientStructuredOutputRunnable rescate instancias de Pydantic cuando el LLM emite markdown json."""
    from src.agent_service.core.llms.factory import ResilientStructuredOutputRunnable

    mock_runnable = AsyncMock()
    # Simula el comportamiento de Gemini 3 con razonamiento: LangChain devuelve None o dict con raw.content
    mock_raw_msg = MagicMock()
    mock_raw_msg.content = 'Pensamiento del modelo...\n```json\n{"base_product": "SKU-99", "relation_type": "cross_sell", "top_k": 5, "search_query": "complementos", "sort_by": "price_asc"}\n```'
    mock_runnable.ainvoke.return_value = {"raw": mock_raw_msg, "parsed": None}

    resilient = ResilientStructuredOutputRunnable(mock_runnable, RecommendationIntentExtraction)
    output = await resilient.ainvoke("consulta")

    assert isinstance(output, RecommendationIntentExtraction)
    assert output.base_product == "SKU-99"
    assert output.relation_type == "cross_sell"
    assert output.top_k == 5
    assert output.sort_by == "price_asc"

