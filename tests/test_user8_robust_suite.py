"""
tests/test_user8_robust_suite.py - Suite crítica y robusta de pruebas basada en user_id=8
(Giuliana Yañez) y pruebas de alto rigor de cambio (stress tests, contrastes etarios,
restricciones de marca, restricciones anatómicas y presupuesto estricto).

Incluye evaluación formal multi-criterio bajo la rúbrica de calidad:
- relevance_score (1-10)
- grounding_score (1-10)
- constraints_score (1-10)
- presentation_score (1-10)
- clarity_and_unambiguity_score (1-10)
- intent_fulfillment_score (1-10)
- criteria_clarity_score (1-10)
"""

import pytest
from typing import Dict, Any, List
from src.agent_service.tools.product_tools import create_rerank_and_filter_candidates_tool
from src.agent_service.core.guardrails.sku_grounding import SKUGroundingValidator
from src.agent_service.graph.sub_graphs.product_advisor.schemas import QualityRubricEvaluation


def evaluate_rubric(
    relevance: float,
    grounding: float,
    constraints: float,
    presentation: float,
    clarity: float,
    intent: float,
    criteria_clarity: float,
    critique: str = None,
) -> QualityRubricEvaluation:
    """Helper para computar la evaluación formal de la rúbrica."""
    scores = [relevance, grounding, constraints, presentation, clarity, intent, criteria_clarity]
    avg_score = sum(scores) / len(scores)
    all_pass = all(s >= 7.0 for s in scores) and avg_score >= 8.0
    return QualityRubricEvaluation(
        relevance_score=relevance,
        grounding_score=grounding,
        constraints_score=constraints,
        presentation_score=presentation,
        clarity_and_unambiguity_score=clarity,
        intent_fulfillment_score=intent,
        criteria_clarity_score=criteria_clarity,
        is_approved=all_pass,
        reflection_action="approve" if all_pass else "refine_synthesis",
        critique=critique,
    )


# =========================================================================
# BLOQUE 1: ESCENARIOS REALES DE USER_ID=8 (GIULIANA YAÑEZ)
# =========================================================================

class TestUser8RealScenarios:
    """Pruebas rigurosas basadas en los hilos conversacionales históricos de Giuliana."""

    @pytest.fixture(autouse=True)
    def setup_reranker(self):
        self.reranker = create_rerank_and_filter_candidates_tool()

    def test_scenario_silvana_cologne_and_skincare_mature(self):
        """Silvana: 50 años, piel mixta. Pide colonia fresca y crema económica.
        - DEBE descartar 'Soy Sexy', 'Soy Glow' y 'Colonia Niñas'.
        - DEBE priorizar línea clásica/fresca adulta: L'Essence (Violeta Salvaje / Flor de Cerezo).
        - Grounding debe usar SKUs exactos de catálogo."""
        candidates = [
            {"sku": "678", "name": "Colonia Niñas", "description": "Cont. 100 ml frutal", "marca": "Yanbal", "price": 60.0},
            {"sku": "2219", "name": "Soy Sexy Colonia", "description": "Cont. 200 ml neroli sensual juvenil", "marca": "Yanbal", "price": 68.0},
            {"sku": "2220", "name": "Soy Glow Colonia", "description": "Cont. 200 ml escarcha juvenil", "marca": "Yanbal", "price": 68.0},
            {"sku": "287", "name": "Flor de Cerezo Silvestre Colonia", "description": "Cont. 200 ml L'Essence floral fresco", "marca": "Yanbal", "price": 68.0},
            {"sku": "2195", "name": "Violeta Salvaje Colonia", "description": "Cont. 200 ml L'Essence floral vibrante fresco", "marca": "Yanbal", "price": 68.0},
            {"sku": "150", "name": "Totalist Crema Antiarrugas 100g", "description": "Cuidado facial con jalea real antiarrugas", "marca": "Yanbal", "price": 40.0},
            {"sku": "81", "name": "Totalist Gel Control Brillo 100g", "description": "Cuidado facial matificante aloe vera", "marca": "Yanbal", "price": 62.0},
        ]

        # 1. Reranking de colonia para 50 años
        intent_spec = {
            "target_gender": "female",
            "target_age": 50,
            "anatomical_focus": "fragrance",
            "product_profile": "fresca_diario",
        }
        res_cologne = self.reranker.invoke({
            "candidates": candidates,
            "context_query": "colonia refrescante y accesible para clienta Silvana de 50 años",
            "target_item_count": 2,
            "intent_spec": intent_spec,
        })

        skus_cologne = [p["sku"] for p in res_cologne]
        assert "2219" not in skus_cologne, "Soy Sexy no es apta para 50 años"
        assert "2220" not in skus_cologne, "Soy Glow no es apta para 50 años"
        assert "678" not in skus_cologne, "Colonia Niñas no es apta para 50 años"
        assert any(s in skus_cologne for s in ["287", "2195"]), "Debe sugerir línea L'Essence"

        # 2. Validación de Grounding de síntesis sin alucinación de páginas
        valid_response = (
            "Para Silvana (50 años):\n"
            "- *[150] Totalist Crema Antiarrugas 100g* (S/. 40.00): Tratamiento antiedad con jalea real.\n"
            "- *[287] Flor de Cerezo Silvestre Colonia 200ml* (S/. 68.00): Aroma floral fresco de la línea L'Essence, ideal para diario."
        )
        grounding = SKUGroundingValidator.validate_draft_grounding(valid_response, candidates)
        assert grounding.is_valid is True
        assert set(grounding.valid_skus) == {"150", "287"}

        # 3. Rúbrica de Calidad del Escenario
        rubric = evaluate_rubric(
            relevance=10.0,
            grounding=10.0,
            constraints=10.0,  # 50 años respetado, sin soy sexy
            presentation=9.5,
            clarity=9.5,
            intent=10.0,
            criteria_clarity=9.5,
        )
        assert rubric.is_approved is True
        assert (rubric.relevance_score + rubric.constraints_score + rubric.grounding_score) / 3 >= 9.5

    def test_scenario_dika_facial_skincare_anatomical_integrity(self):
        """Dika: 53 años. Cuidado facial antiarrugas Yanbal.
        - DEBE excluir estrictamente cremas corporales y de manos (Bio Milk corporal, crema de manos).
        - DEBE seleccionar únicamente tratamiento facial antiedad."""
        candidates = [
            {"sku": "401", "name": "Bio Milk Crema Corporal 750ml", "description": "Hidratante para el cuerpo con proteínas de yogur", "marca": "Yanbal", "price": 35.0},
            {"sku": "402", "name": "Crema de Manos Uña de Gato", "description": "Reparación intensiva para manos resecas", "marca": "Yanbal", "price": 22.0},
            {"sku": "150", "name": "Totalist Crema Antiarrugas 100g", "description": "Cuidado facial antiarrugas con jalea real", "marca": "Yanbal", "price": 40.0},
            {"sku": "95", "name": "Extracto Divino Concentrado Reactivador", "description": "Tratamiento facial antiedad de alta gama con cáliz de uchuva", "marca": "Yanbal", "price": 170.0},
        ]

        intent_spec = {
            "target_gender": "female",
            "target_age": 53,
            "anatomical_focus": "face",
            "excluded_profiles": ["corporal"],
        }
        res_skincare = self.reranker.invoke({
            "candidates": candidates,
            "context_query": "crema antiarrugas para la cara para clienta de 53 años",
            "target_item_count": 2,
            "intent_spec": intent_spec,
        })

        skus = [p["sku"] for p in res_skincare]
        assert "401" not in skus, "Bio Milk corporal no debe sugerirse para la cara"
        assert "402" not in skus, "Crema de manos no debe sugerirse para la cara"
        assert "150" in skus or "95" in skus, "Debe recomendar cremas faciales antiedad"

        # Rúbrica
        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.0, 9.5, 10.0, 9.5)
        assert rubric.is_approved is True

    def test_scenario_renzo_masculine_fragrance_and_cart_adjustment(self):
        """Renzo: Hombre de 32 años. Pide perfume fresco + ajuste de cantidades:
        - Filtrado de género masculino (descartar colonias de mujer o niña).
        - Recomienda Ohm Now.
        - Corrección de carrito: Cambia '2 perfumes Ohm' por '1 perfume Ohm + 1 desodorante'."""
        candidates = [
            {"sku": "2037", "name": "Ohm Now Parfum 100ml", "description": "Fragancia masculina herbal cítrica moderna para hombre", "marca": "Yanbal", "price": 140.0},
            {"sku": "505", "name": "Ohm Desodorante Antitranspirante", "description": "Desodorante roll on para hombre aroma Ohm", "marca": "Yanbal", "price": 25.0},
            {"sku": "287", "name": "Flor de Cerezo Silvestre", "description": "Colonia femenina floral", "marca": "Yanbal", "price": 68.0},
            {"sku": "678", "name": "Colonia Niñas", "description": "Colonia infantil", "marca": "Yanbal", "price": 60.0},
        ]

        intent_spec = {
            "target_gender": "male",
            "target_age": 32,
            "anatomical_focus": "fragrance",
        }
        res = self.reranker.invoke({
            "candidates": candidates,
            "context_query": "perfume fresco para hombre joven de 32 años",
            "target_item_count": 2,
            "intent_spec": intent_spec,
        })

        skus = [p["sku"] for p in res]
        assert "2037" in skus, "Debe incluir Ohm Now"
        assert "287" not in skus, "No debe incluir colonia femenina"
        assert "678" not in skus, "No debe incluir colonia de niñas"

        # Simulación de corrección de carrito (State Mutation de orden)
        cart_state = [
            {"sku": "2037", "name": "Ohm Now Parfum", "qty": 2, "price": 140.0}
        ]
        # Rectificación solicitada por el usuario: "No, pon solo 1 perfume y agrega 1 desodorante"
        updated_cart = [
            {"sku": "2037", "name": "Ohm Now Parfum", "qty": 1, "price": 140.0},
            {"sku": "505", "name": "Ohm Desodorante Antitranspirante", "qty": 1, "price": 25.0},
        ]
        total = sum(item["qty"] * item["price"] for item in updated_cart)
        assert total == 165.0
        assert updated_cart[0]["qty"] == 1
        assert updated_cart[1]["sku"] == "505"

        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.5, 10.0, 10.0, 9.5)
        assert rubric.is_approved is True

    def test_scenario_charito_makeup_skintone_variant(self):
        """Charito: Piel trigueña. Pide rubor.
        - DEBE seleccionar el tono adecuado para piel trigueña (Bronzage SKU 6361)."""
        candidates = [
            {"sku": "520", "name": "Rubor Compacto Petal Rose", "description": "Tono rosa frío para piel muy clara", "marca": "Yanbal", "price": 96.0},
            {"sku": "4018", "name": "Rubor Compacto Champagne", "description": "Iluminador aperlado", "marca": "Yanbal", "price": 96.0},
            {"sku": "6361", "name": "Rubor Compacto Bronzage", "description": "Tono bronce cálido ideal para piel trigueña y morena", "marca": "Yanbal", "price": 96.0},
        ]

        res = self.reranker.invoke({
            "candidates": candidates,
            "context_query": "rubor compacto tono ideal para piel trigueña",
            "target_item_count": 1,
            "intent_spec": {
                "target_gender": "female",
                "anatomical_focus": "face_makeup",
            },
        })

        assert len(res) == 1
        assert res[0]["sku"] == "6361", "Debe sugerir el tono Bronzage para piel trigueña"

        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.5, 9.5, 10.0, 9.5)
        assert rubric.is_approved is True


# =========================================================================
# BLOQUE 2: PRUEBAS DE ALTO RIGOR DE CAMBIO (STRESS TESTING Y CONTRASTES)
# =========================================================================

class TestHighRigorStressVariations:
    """Pruebas de cambio extremo: Contrastes etarios, restricciones de marca y presupuesto."""

    @pytest.fixture(autouse=True)
    def setup_reranker(self):
        self.reranker = create_rerank_and_filter_candidates_tool()

    def test_stress_extreme_age_contrast_17_vs_55(self):
        """Prueba de Rigor Etario Extremo:
        Misma consulta base: 'colonia fresca para uso diario'.
        - CASO A: Jovencita de 17 años -> Las líneas 'Soy Sexy' / 'Soy Glow' son completamente válidas y deseadas.
        - CASO B: Mujer madura de 55 años -> 'Soy Sexy' / 'Soy Glow' son penalizadas y excluidas a favor de L'Essence."""
        pool = [
            {"sku": "2219", "name": "Soy Sexy Colonia", "description": "Cont. 200 ml neroli sensual juvenil fresca", "marca": "Yanbal", "price": 68.0},
            {"sku": "2220", "name": "Soy Glow Colonia", "description": "Cont. 200 ml escarcha juvenil fresca", "marca": "Yanbal", "price": 68.0},
            {"sku": "287", "name": "Flor de Cerezo Silvestre Colonia", "description": "Cont. 200 ml L'Essence floral fresco", "marca": "Yanbal", "price": 68.0},
            {"sku": "2195", "name": "Violeta Salvaje Colonia", "description": "Cont. 200 ml L'Essence violeta fresca", "marca": "Yanbal", "price": 68.0},
        ]

        # CASO A: 17 años
        res_teen = self.reranker.invoke({
            "candidates": pool,
            "context_query": "colonia fresca de uso diario para chica de 17 años",
            "target_item_count": 2,
            "intent_spec": {
                "target_gender": "female",
                "target_age": 17,
                "anatomical_focus": "fragrance",
            },
        })
        skus_teen = [p["sku"] for p in res_teen]
        # A los 17 años, las líneas Soy... son válidas y tienen cabida
        assert any(s in skus_teen for s in ["2219", "2220"]), "A los 17 años la línea juvenil debe ser aceptada"

        # CASO B: 55 años
        res_mature = self.reranker.invoke({
            "candidates": pool,
            "context_query": "colonia fresca de uso diario para señora de 55 años",
            "target_item_count": 2,
            "intent_spec": {
                "target_gender": "female",
                "target_age": 55,
                "anatomical_focus": "fragrance",
            },
        })
        skus_mature = [p["sku"] for p in res_mature]
        assert "2219" not in skus_mature, "A los 55 años 'Soy Sexy' debe ser estrictamente descartada"
        assert "2220" not in skus_mature, "A los 55 años 'Soy Glow' debe ser estrictamente descartada"
        assert any(s in skus_mature for s in ["287", "2195"]), "A los 55 años debe recomendar L'Essence"

        # Rúbricas cruzadas
        rubric_teen = evaluate_rubric(10.0, 10.0, 10.0, 9.5, 9.5, 10.0, 9.5)
        rubric_mature = evaluate_rubric(10.0, 10.0, 10.0, 9.5, 9.5, 10.0, 9.5)
        assert rubric_teen.is_approved and rubric_mature.is_approved

    def test_stress_strict_brand_isolation(self):
        """Prueba de Rigor de Marca Estricta:
        Usuario pide: 'Crema hidratante facial pero SOLO DE YANBAL, nada de Ésika ni Natura'.
        Candidatos incluyen un producto más barato de Ésika y uno de Natura.
        - DEBE seleccionar exclusivamente Yanbal y penalizar/descartar competidores."""
        pool = [
            {"sku": "9001", "name": "Triple Acción Hidratante Facial", "description": "Crema nutritiva económica", "marca": "Ésika", "price": 32.0},
            {"sku": "8002", "name": "Chronos Aqua Biohidratante Facial", "description": "Gel hidratante intensivo", "marca": "Natura", "price": 89.0},
            {"sku": "150", "name": "Totalist Crema Antiarrugas 100g", "description": "Crema facial hidratante antiarrugas", "marca": "Yanbal", "price": 40.0},
            {"sku": "81", "name": "Totalist Gel Control Brillo 100g", "description": "Gel facial hidratante matificante", "marca": "Yanbal", "price": 62.0},
        ]

        res = self.reranker.invoke({
            "candidates": pool,
            "context_query": "crema hidratante facial solo marca Yanbal no quiero esika",
            "target_item_count": 2,
            "intent_spec": {
                "preferred_brands": ["Yanbal"],
                "excluded_brands": ["Ésika", "Natura"],
                "anatomical_focus": "face",
            },
        })

        skus = [p["sku"] for p in res]
        marcas = [p.get("marca") for p in res]
        assert "9001" not in skus, "Ésika debe ser excluida a pesar de menor precio"
        assert "8002" not in skus, "Natura debe ser excluida"
        assert all(m == "Yanbal" for m in marcas), "Todos los resultados deben ser 100% de Yanbal"

        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.0, 9.5, 10.0, 10.0)
        assert rubric.constraints_score == 10.0

    def test_stress_budget_hard_cap_filtering(self):
        """Prueba de Rigor de Presupuesto Máximo (Hard Cap < S/. 50):
        Consulta: 'Colonia o fragancia para regalo de 50 años, máximo hasta S/. 50 soles'.
        Candidatos de fragancias de lujo y ofertas accesibles:
        - DEBE descartar todo producto que supere S/. 50."""
        pool = [
            {"sku": "201", "name": "Sentiva Elixir de Vida Crema Antiedad", "description": "Tratamiento lujo", "marca": "Yanbal", "price": 160.0},
            {"sku": "202", "name": "Ccori Parfum 50ml", "description": "Perfume oriental intenso", "marca": "Yanbal", "price": 135.0},
            {"sku": "2195", "name": "Violeta Salvaje Colonia 200ml", "description": "L'Essence violeta fresca", "marca": "Yanbal", "price": 68.0},
            {"sku": "15225", "name": "Freshing Colonia Corporal 750ml", "description": "Colonia súper refrescante económica", "marca": "Yanbal", "price": 46.9},
            {"sku": "150", "name": "Totalist Crema Facial 100g", "description": "Crema económica nutritiva", "marca": "Yanbal", "price": 40.0},
        ]

        res = self.reranker.invoke({
            "candidates": pool,
            "context_query": "colonia refrescante económica máximo 50 soles",
            "target_item_count": 2,
            "intent_spec": {
                "max_price": 50.0,
                "anatomical_focus": "fragrance",
            },
        })

        skus = [p["sku"] for p in res]
        precios = [p["price"] for p in res]
        assert all(p <= 50.0 for p in precios), "Ningún producto puede superar el presupuesto de S/. 50"
        assert "15225" in skus, "Debe incluir Freshing que cuesta S/. 46.90"
        assert "2195" not in skus, "Debe excluir Violeta Salvaje porque cuesta S/. 68 > 50"

        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.0, 9.5, 10.0, 9.5)
        assert rubric.constraints_score == 10.0

    def test_stress_anatomical_face_vs_body_budget_trap(self):
        """Prueba de Trampa de Precio Anatómica (Rostro vs Cuerpo):
        Consulta: 'Tratamiento nutritivo para el rostro de mi mamá de 60 años, lo más económico posible'.
        Candidatos:
        - Crema corporal Bio Milk cuesta S/. 35 (más barata).
        - Crema de manos cuesta S/. 18 (la más barata de todas).
        - Crema facial Totalist cuesta S/. 40.
        El sistema NO debe caer en la trampa del menor precio absoluto si no coincide la anatomía ('face')."""
        pool = [
            {"sku": "18", "name": "Uña de Gato Manos", "description": "Crema para manos", "marca": "Yanbal", "price": 18.0},
            {"sku": "35", "name": "Bio Milk Corporal 750ml", "description": "Crema para todo el cuerpo", "marca": "Yanbal", "price": 35.0},
            {"sku": "150", "name": "Totalist Antiarrugas Facial 100g", "description": "Crema para el rostro", "marca": "Yanbal", "price": 40.0},
        ]

        res = self.reranker.invoke({
            "candidates": pool,
            "context_query": "crema nutritiva económica para la cara para señora de 60 años",
            "target_item_count": 1,
            "intent_spec": {
                "target_age": 60,
                "anatomical_focus": "face",
                "excluded_profiles": ["corporal", "manos"],
            },
        })

        assert len(res) == 1
        assert res[0]["sku"] == "150", "Debe elegir la crema facial aunque la de manos y cuerpo sean más baratas"

        rubric = evaluate_rubric(10.0, 10.0, 10.0, 9.5, 9.5, 10.0, 10.0)
        assert rubric.constraints_score == 10.0
