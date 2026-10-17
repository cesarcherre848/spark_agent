"""
tests/test_nested_product_advisor_complex.py - Suite exhaustiva de pruebas anidadas y multi-turno
para el Asesor de Productos (Product Advisor) con validación de categorías funcionales,
preservación demográfica y auto-reflexión con rúbrica.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agent_service.tools.product_tools import (
    create_rerank_and_filter_candidates_tool,
    create_filter_and_sort_products_tool,
)
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    CustomerIntentSpec,
    AdvisorPlan,
    AdvisorToolCall,
    QualityRubricEvaluation,
    FinalAnswer,
)


# ==============================================================================
# 1. SUITE MULTI-TURNO: CUIDADO FACIAL (SKINCARE) VS MAQUILLAJE Y PEDIÁTRICO
# ==============================================================================

class TestSkincareMultiTurnNested:
    """Valida que consultas sobre cuidado facial / piel mixta descarten sistemáticamente
    maquillaje (bases, polvos) y líneas pediátricas/bebés a lo largo de múltiples turnos."""

    def test_turn1_skincare_mixed_skin_discards_makeup(self):
        """Turno 1: 'productos para piel mixta' no debe recomendar bases ni polvos matificantes,
        sino cremas/geles de tratamiento facial."""
        reranker = create_rerank_and_filter_candidates_tool()

        candidates = [
            {
                "sku": "5197",
                "name": "Base Líquida Antiedad Control Brillo Piel Mixta",
                "description": "Base de maquillaje con acabado mate y niacinamida para piel mixta",
                "price": 58.0,
                "marca": "Yanbal",
            },
            {
                "sku": "6232",
                "name": "Polvos Compactos Matificantes Piel Mixta",
                "description": "Polvos compactos para sellar el maquillaje y reducir brillo",
                "price": 45.0,
                "marca": "Yanbal",
            },
            {
                "sku": "101",
                "name": "Crema Hidratante Facial Matificante Piel Mixta",
                "description": "Tratamiento facial hidratante ligero libre de grasa con ácido hialurónico",
                "price": 62.0,
                "marca": "Yanbal",
            },
            {
                "sku": "102",
                "name": "Gel Limpiador Purificante Rostro",
                "description": "Limpiador facial diario para poros y control de sebo",
                "price": 38.0,
                "marca": "Yanbal",
            },
        ]

        result = reranker.invoke({
            "candidates": candidates,
            "context_query": "hola, que productos para piel mixta me recomiendas",
            "target_item_count": 2,
            "intent_spec": {
                "functional_category": "skincare_treatment",
                "anatomical_focus": "facial",
                "target_stage": "adult",
                "skin_type": "mixta",
            },
        })

        assert len(result) == 2
        selected_skus = [p["sku"] for p in result]
        # Prohibido maquillaje
        assert "5197" not in selected_skus, "Base Líquida no debe recomendarse para cuidado de piel mixta"
        assert "6232" not in selected_skus, "Polvos Compactos no deben recomendarse para cuidado de piel mixta"
        # Debe incluir cremas/geles de tratamiento
        assert "101" in selected_skus
        assert "102" in selected_skus

    def test_turn2_skincare_cheaper_discards_baby_cream_and_hygiene(self):
        """Turno 2: 'algo más económico para el rostro' no debe recomendar crema de bebés (Yanbal Baby)
        ni desodorantes ni crema corporal, sino la opción facial de adulto más económica."""
        reranker = create_rerank_and_filter_candidates_tool()

        candidates = [
            {
                "sku": "999",
                "name": "Crema Hidratante Rostro y Cuerpo Yanbal Baby",
                "description": "Crema pediátrica suave con manzanilla para bebés y recién nacidos",
                "price": 25.0,
                "marca": "Yanbal",
            },
            {
                "sku": "888",
                "name": "Desodorante Antitranspirante Roll On",
                "description": "Protección 48h aroma neutro",
                "price": 14.0,
                "marca": "Yanbal",
            },
            {
                "sku": "777",
                "name": "Crema Corporal Nutritiva Manos y Cuerpo",
                "description": "Hidratación corporal intensiva para codos y cuerpo",
                "price": 28.0,
                "marca": "Yanbal",
            },
            {
                "sku": "102",
                "name": "Gel Limpiador Purificante Rostro",
                "description": "Limpiador facial diario para poros y control de sebo",
                "price": 38.0,
                "marca": "Yanbal",
            },
            {
                "sku": "103",
                "name": "Crema Facial Hidratante Noche",
                "description": "Crema facial ligera para cuidado nocturno del rostro",
                "price": 45.0,
                "marca": "Yanbal",
            },
        ]

        result = reranker.invoke({
            "candidates": candidates,
            "context_query": "tienes algo más económico para el rostro?",
            "sort_by": "price_asc",
            "target_item_count": 2,
            "intent_spec": {
                "functional_category": "skincare_treatment",
                "anatomical_focus": "facial",
                "target_stage": "adult",
            },
        })

        assert len(result) == 2
        selected_skus = [p["sku"] for p in result]
        # Prohibido bebé para adulto
        assert "999" not in selected_skus, "Crema de bebé no debe recomendarse para rostro de adulto"
        # Prohibido higiene y corporal
        assert "888" not in selected_skus, "Desodorante no debe recomendarse para rostro"
        assert "777" not in selected_skus, "Crema corporal no debe recomendarse para rostro"
        # Debe seleccionar opciones faciales legítimas ordenadas por precio
        assert result[0]["sku"] == "102", "Gel Limpiador S/ 38 debe ser la primera opción económica"
        assert result[1]["sku"] == "103", "Crema Facial S/ 45 debe ser la segunda opción económica"


# ==============================================================================
# 2. SUITE MULTI-TURNO: PERFUMERÍA MASCULINA VS HIGIENE PERSONAL (DESODORANTES)
# ==============================================================================

class TestMensFragranceMultiTurnNested:
    """Valida que consultas de fragancias masculinas descarten colonias femeninas,
    desodorantes y jabones, preservando la categoría de perfumería aun al pedir menor precio."""

    def test_turn1_masculine_fragrance_discards_feminine_and_hygiene(self):
        """Turno 1: 'perfume para hombre joven de 27 años' debe descartar fragancias femeninas,
        infantiles y desodorantes."""
        reranker = create_rerank_and_filter_candidates_tool()

        candidates = [
            {
                "sku": "F-01",
                "name": "Colonia Femenina Dulce Rosa",
                "description": "Fragancia floral para mujer con notas de jazmín",
                "price": 68.0,
                "marca": "Yanbal",
            },
            {
                "sku": "K-01",
                "name": "Colonia Niños Aventurero",
                "description": "Aroma infantil para niños y bebés",
                "price": 42.0,
                "marca": "Yanbal",
            },
            {
                "sku": "H-01",
                "name": "Eau de Toilette Masculino Cítrico Hombre",
                "description": "Fragancia fresca masculina con notas de bergamota y madera",
                "price": 75.0,
                "marca": "Yanbal",
            },
            {
                "sku": "H-02",
                "name": "Perfume Masculino Herbal Fresco Hombre",
                "description": "Perfume elegante de alta intensidad para caballero",
                "price": 110.0,
                "marca": "Yanbal",
            },
            {
                "sku": "D-01",
                "name": "Desodorante Roll-on Hombre",
                "description": "Antitranspirante masculino en roll-on protección diaria",
                "price": 15.0,
                "marca": "Yanbal",
            },
        ]

        result = reranker.invoke({
            "candidates": candidates,
            "context_query": "un perfume para hombre joven de 27 años",
            "target_item_count": 2,
            "intent_spec": {
                "target_gender": "male",
                "target_age": 27,
                "functional_category": "fragrance_perfumery",
                "anatomical_focus": "fragrance",
            },
        })

        assert len(result) == 2
        selected_skus = [p["sku"] for p in result]
        assert "F-01" not in selected_skus, "Colonia femenina no debe ser recomendada para hombre"
        assert "K-01" not in selected_skus, "Colonia de niños no debe ser recomendada a joven de 27 años"
        assert "D-01" not in selected_skus, "Desodorante no debe ser recomendado como perfume"
        assert "H-01" in selected_skus
        assert "H-02" in selected_skus

    def test_turn2_masculine_fragrance_cheaper_does_not_substitute_with_deodorant(self):
        """Turno 2: Al pedir 'algo menos' o 'más barato', el recomendador NO debe sustituir
        la fragancia con desodorantes en roll-on ni jabones. Debe priorizar la fragancia más
        accesible y, si supera el presupuesto, marcar budget_exceeded=True."""
        reranker = create_rerank_and_filter_candidates_tool()

        candidates = [
            {
                "sku": "H-02",
                "name": "Perfume Masculino Herbal Fresco Hombre",
                "description": "Perfume elegante de alta intensidad para caballero",
                "price": 110.0,
                "marca": "Yanbal",
            },
            {
                "sku": "H-01",
                "name": "Eau de Toilette Masculino Cítrico Hombre",
                "description": "Fragancia fresca masculina con notas de bergamota y madera",
                "price": 75.0,
                "marca": "Yanbal",
            },
            {
                "sku": "D-01",
                "name": "Desodorante Antitranspirante Roll-on Hombre",
                "description": "Protección antitranspirante masculina para axilas",
                "price": 14.0,
                "marca": "Yanbal",
            },
            {
                "sku": "J-01",
                "name": "Jabón en barra Hombre Purificante",
                "description": "Jabón de tocador masculino limpieza corporal",
                "price": 12.0,
                "marca": "Yanbal",
            },
        ]

        # Supongamos que el usuario pide "algo menos de 50 soles"
        result = reranker.invoke({
            "candidates": candidates,
            "context_query": "tienes alguna colonia o perfume de hombre por menos de 50 soles?",
            "sort_by": "price_asc",
            "target_item_count": 2,
            "intent_spec": {
                "target_gender": "male",
                "functional_category": "fragrance_perfumery",
                "anatomical_focus": "fragrance",
                "max_price": 50.0,
            },
        })

        selected_skus = [p["sku"] for p in result]
        # NUNCA sustituir con desodorante ni jabón
        assert "D-01" not in selected_skus, "Desodorante no debe sustituir a perfume"
        assert "J-01" not in selected_skus, "Jabón no debe sustituir a perfume"

        # Debe preservar las fragancias masculinas y marcar budget_exceeded=True
        assert len(result) > 0
        assert result[0]["sku"] == "H-01"
        assert result[0].get("budget_exceeded") is True, "Debe marcar budget_exceeded=True para explicación consultiva"


# ==============================================================================
# 3. SUITE MULTI-TURNO: CAMBIO DINÁMICO DE CATEGORÍA ENTRE TURNOS
# ==============================================================================

class TestCategorySwitchingStateReset:
    """Valida transiciones limpias entre categorías en una misma sesión:
    Turno 1: Maquillaje -> Turno 2: Cuidado Facial -> Turno 3: Perfumería."""

    def test_seamless_category_switching_across_turns(self):
        reranker = create_rerank_and_filter_candidates_tool()

        all_catalog = [
            {"sku": "LIP-1", "name": "Labial Mate Larga Duración", "description": "Color intenso mate labios", "price": 35.0, "marca": "Yanbal"},
            {"sku": "CREAM-1", "name": "Crema Facial Antiarrugas Noche", "description": "Tratamiento facial nutritivo", "price": 75.0, "marca": "Yanbal"},
            {"sku": "PERF-1", "name": "Eau de Parfum Masculino Fougère", "description": "Fragancia sofisticada hombre", "price": 95.0, "marca": "Yanbal"},
        ]

        # Turno 1: Maquillaje
        t1 = reranker.invoke({
            "candidates": all_catalog,
            "context_query": "muéstrame labiales",
            "target_item_count": 1,
            "intent_spec": {"functional_category": "makeup_cosmetics", "anatomical_focus": "makeup"},
        })
        assert len(t1) == 1
        assert t1[0]["sku"] == "LIP-1"

        # Turno 2: Cambio a Cuidado Facial
        t2 = reranker.invoke({
            "candidates": all_catalog,
            "context_query": "ahora recomiéndame cremas faciales de tratamiento",
            "target_item_count": 1,
            "intent_spec": {"functional_category": "skincare_treatment", "anatomical_focus": "facial", "target_stage": "adult"},
        })
        assert len(t2) == 1
        assert t2[0]["sku"] == "CREAM-1"

        # Turno 3: Cambio a Perfumería Hombre
        t3 = reranker.invoke({
            "candidates": all_catalog,
            "context_query": "y qué perfumes para hombre tienes?",
            "target_item_count": 1,
            "intent_spec": {"functional_category": "fragrance_perfumery", "target_gender": "male", "anatomical_focus": "fragrance"},
        })
        assert len(t3) == 1
        assert t3[0]["sku"] == "PERF-1"


# ==============================================================================
# 4. SUITE DE EVALUACIÓN POR RÚBRICA Y JUEZ AUDITOR
# ==============================================================================

class TestRubricEvaluatorCategoryFidelity:
    """Valida que el Juez de Rúbrica desapruebe borradores que cometan errores de categoría
    (maquillaje para skincare, desodorante para perfume, o bebé para adulto)."""

    @pytest.mark.asyncio
    async def test_judge_rejects_deodorant_for_perfume_query(self):
        """Verifica que el juez rechace una respuesta que recomiende desodorantes cuando se pidió perfume."""
        mock_judge_llm = MagicMock()
        nodes = ProductAdvisorNodes(llm=mock_judge_llm)

        # Mocking el retorno del juez estructurado con desaprobación por categoría
        mock_eval = QualityRubricEvaluation(
            relevance_score=4.0,
            grounding_score=9.0,
            constraints_score=3.0,
            presentation_score=8.0,
            clarity_and_unambiguity_score=7.0,
            intent_fulfillment_score=4.0,
            criteria_clarity_score=9.0,
            is_approved=False,
            reflection_action="re_rerank_with_feedback",
            critique="El usuario solicitó fragancias masculinas económicas y se le recomendó un desodorante en roll-on en lugar de una fragancia.",
            remedy_suggestions=["Filtrar productos de higiene y priorizar fragancias masculinas disponibles."],
        )
        nodes._rubric_judge = MagicMock()
        nodes._rubric_judge.ainvoke = AsyncMock(return_value=mock_eval)

        state: ProductAdvisorState = {
            "raw_query": "algo menos en perfume de hombre",
            "draft_response": "Te recomiendo el [20013876] Desodorante Roll-on Hombre a S/. 15.00.",
            "candidate_products": [{"sku": "20013876", "name": "Desodorante Roll-on Hombre", "price": 15.0}],
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = await nodes.rubric_evaluator_judge(state)

        assert result["meets_rubric"] is False
        assert result["reflection_action"] == "re_rerank_with_feedback"
        assert "desodorante" in result["critique"].lower()

    @pytest.mark.asyncio
    async def test_judge_rejects_baby_cream_for_adult_skincare(self):
        """Verifica que el juez rechace una respuesta que recomiende Yanbal Baby para el rostro de un adulto."""
        mock_judge_llm = MagicMock()
        nodes = ProductAdvisorNodes(llm=mock_judge_llm)

        mock_eval = QualityRubricEvaluation(
            relevance_score=5.0,
            grounding_score=9.0,
            constraints_score=3.5,
            presentation_score=8.0,
            clarity_and_unambiguity_score=7.0,
            intent_fulfillment_score=4.0,
            criteria_clarity_score=9.0,
            is_approved=False,
            reflection_action="re_rerank_with_feedback",
            critique="Se recomendó la crema pediátrica Yanbal Baby para una consulta de cuidado facial de adulto.",
            remedy_suggestions=["Descartar productos de líneas de bebé e infantiles para clientes adultos."],
        )
        nodes._rubric_judge = MagicMock()
        nodes._rubric_judge.ainvoke = AsyncMock(return_value=mock_eval)

        state: ProductAdvisorState = {
            "raw_query": "tienes algo más económico para el rostro?",
            "draft_response": "Te sugiero la [999] Crema Hidratante Rostro y Cuerpo Yanbal Baby a S/. 25.00.",
            "candidate_products": [{"sku": "999", "name": "Crema Yanbal Baby", "price": 25.0}],
            "iteration_count": 0,
            "max_iterations": 3,
        }

        result = await nodes.rubric_evaluator_judge(state)

        assert result["meets_rubric"] is False
        assert result["reflection_action"] == "re_rerank_with_feedback"
        assert "bebé" in result["critique"].lower() or "pediátrica" in result["critique"].lower()

    @pytest.mark.asyncio
    async def test_judge_approves_consultative_price_boundary_explanation(self):
        """Verifica que el juez apruebe una respuesta que honestamente explica el piso de precio
        de fragancias masculinas cuando no hay opciones bajo el presupuesto pedido."""
        mock_judge_llm = MagicMock()
        nodes = ProductAdvisorNodes(llm=mock_judge_llm)

        mock_eval = QualityRubricEvaluation(
            relevance_score=9.0,
            grounding_score=9.5,
            constraints_score=8.5,
            presentation_score=9.0,
            clarity_and_unambiguity_score=9.5,
            intent_fulfillment_score=8.5,
            criteria_clarity_score=9.0,
            is_approved=True,
            reflection_action="approve",
            critique=None,
            remedy_suggestions=[],
        )
        nodes._rubric_judge = MagicMock()
        nodes._rubric_judge.ainvoke = AsyncMock(return_value=mock_eval)

        state: ProductAdvisorState = {
            "raw_query": "perfumes de hombre por menos de 50",
            "draft_response": "Actualmente en catálogo nuestras fragancias masculinas inician desde S/. 65.00 con la opción [H-01] Eau de Toilette Masculino Cítrico...",
            "candidate_products": [{"sku": "H-01", "name": "Eau de Toilette Masculino", "price": 65.0, "budget_exceeded": True}],
            "iteration_count": 1,
            "max_iterations": 3,
        }

        result = await nodes.rubric_evaluator_judge(state)

        assert result["meets_rubric"] is True
        assert result["reflection_action"] == "approve"


# ==============================================================================
# 5. INTEGRACIÓN DE NODO EXECUTE_TOOLS CON CANDIDATOS FILTRADOS
# ==============================================================================

class TestExecuteToolsCandidateIntegrity:
    """Valida que execute_tools en ProductAdvisorNodes filtre candidatos y nunca resucite
    artículos incompatibles a través de fallbacks."""

    @pytest.mark.asyncio
    async def test_execute_tools_preserves_only_valid_functional_category(self):
        mock_llm = MagicMock()
        nodes = ProductAdvisorNodes(llm=mock_llm)

        real_reranker = create_rerank_and_filter_candidates_tool()
        nodes._tools_registry = {
            "rerank_and_filter_candidates": real_reranker,
            "get_product_odoo_details": AsyncMock(return_value={"products": []}),
        }

        initial_candidates = [
            {"sku": "5197", "name": "Base Líquida Antiedad Piel Mixta", "price": 58.0},
            {"sku": "6232", "name": "Polvos Compactos Matificantes Piel Mixta", "price": 45.0},
            {"sku": "101", "name": "Crema Hidratante Facial Matificante Piel Mixta", "price": 62.0},
        ]

        state: ProductAdvisorState = {
            "raw_query": "productos para piel mixta",
            "candidate_products": initial_candidates,
            "planned_tools": [
                {
                    "tool_name": "rerank_and_filter_candidates",
                    "arguments": {
                        "context_query": "productos para piel mixta",
                        "intent_spec": {
                            "functional_category": "skincare_treatment",
                            "anatomical_focus": "facial",
                        },
                    },
                }
            ],
        }

        result = await nodes.execute_tools(state)
        retained = result["candidate_products"]

        assert len(retained) == 1
        assert retained[0]["sku"] == "101"
        assert retained[0]["name"] == "Crema Hidratante Facial Matificante Piel Mixta"
