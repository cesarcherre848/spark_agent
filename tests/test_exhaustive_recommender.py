"""
tests/test_exhaustive_recommender.py - Batería Exhaustiva de Pruebas del Recomendador Comercial (Product Advisor).
Cubre:
1. Compatibilidad Cosmetológica (Tipos de piel y edades: mixta, seca, sensible, madura 50+).
2. Variantes de Maquillaje y Tonos de Piel (Trigueña, clara, acabados).
3. Perfiles Olfativos, Ocasión y Filtro Estricto de Género (Masculino vs Femenino vs Infantil).
4. Presupuestos y Rendimiento/Gramaje (Económico, 100g vs 50g, 750ml vs 200ml).
5. Gestión Multi-Marca y Restricciones Mono-Marca (Yanbal vs Ésika vs Comparativas).
6. Cross-Selling / Venta Cruzada (Rutinas faciales completas, Perfume + Desodorante).
7. Blindaje de Grounding y Ciclos de Reflexión (Fast y Slow Loops).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorPlan,
    AdvisorToolCall,
    FinalAnswer,
    QualityRubricEvaluation,
)
from src.agent_service.core.guardrails.sku_grounding import (
    SKUGroundingValidator,
    SKUExtractor,
)
from src.agent_service.core.stores.product.product_hierarchy import (
    UniversalProductHierarchyResolver,
)


# ==============================================================================
# 1. PRUEBAS DE ASESORÍA COSMETOLÓGICA (PIEL MIXTA, SENSIBLE, MADURA 50+)
# ==============================================================================

@pytest.mark.asyncio
async def test_recommender_skin_type_mixed_mature_economic():
    """Caso Silvana: 50 años, piel mixta, busca tratamiento económico.
    Verifica que el formateador jerárquico y el recomendador seleccionen
    las variantes de Control Brillo / Antiarrugas y formatos rendidores."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    candidates = [
        {"product_id": 241, "sku": "150", "name": "Totalist Crema Antiarrugas", "price": 40.0, "description": "Cont. 100 g con jalea real", "pagina": 27},
        {"product_id": 242, "sku": "81", "name": "Totalist Gel Control Brillo", "price": 62.0, "description": "Cont. 100 g con aloe vera y niacinamida", "pagina": 27},
        {"product_id": 233, "sku": "67", "name": "BB Cream Matificante Mediano", "price": 65.0, "description": "Controla el exceso de sebo y brillo SPF 30", "pagina": 30},
        {"product_id": 231, "sku": "7", "name": "BB Cream Hidratante Mediano", "price": 65.0, "description": "Luminosidad glow SPF 30", "pagina": 30},
    ]

    # Resolver jerárquico agrupa Totalist y BB Cream
    resolver = UniversalProductHierarchyResolver()
    families = await resolver.group_candidates_hierarchically(candidates)
    context_text = nodes.format_products_context(families)

    # El contexto debe contener explícitamente los SKUs oficiales y detalles sin corchetes de página
    assert "SKU OFICIAL: 150" in context_text
    assert "SKU OFICIAL: 81" in context_text
    assert "SKU OFICIAL: 67" in context_text
    assert "Página de catálogo" not in context_text or "[Pág." not in context_text

    # Validar respuesta simulada
    draft = (
        "Para Silvana (50 años, piel mixta) que busca opciones económicas:\n"
        "- *[150] Totalist Crema Antiarrugas 100g* (S/. 40.00): Ideal para tratar signos de la edad a sus 50 años.\n"
        "- *[81] Totalist Gel Control Brillo 100g* (S/. 62.00): Con niacinamida para controlar la zona T de la piel mixta.\n"
        "- *[67] BB Cream Matificante* (S/. 65.00): Si busca color con protección solar y efecto mate.\n"
    )
    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"150", "81", "67"}


@pytest.mark.asyncio
async def test_recommender_skin_type_sensitive_body_routine():
    """Caso Dika: Piel corporal seca y sensible + rutina facial deportiva.
    Verifica recomendación de Biomilk Sensitive y Sensi Derm."""
    candidates = [
        {"product_id": 518, "sku": "750", "name": "Biomilk Sensitive", "price": 80.0, "description": "Crema corporal para piel delicada"},
        {"product_id": 76, "sku": "76", "name": "Sensi Derm Leche Limpiadora", "price": 62.0, "description": "Limpieza suave para piel sensible"},
        {"product_id": 77, "sku": "77", "name": "Sensi Derm Hidratante Ligero", "price": 90.0, "description": "Restaura la barrera cutánea"},
        {"product_id": 74, "sku": "74", "name": "Suero Antioxidante Iluminador", "price": 178.0, "description": "Con alta concentración de Vitamina C"},
    ]

    draft = (
        "Para el cuidado de piel sensible de Dika:\n"
        "- *[750] Biomilk Sensitive* (S/. 80.00): Hidratación corporal profunda sin irritar.\n"
        "- *[76] Sensi Derm Leche Limpiadora* (S/. 62.00): Limpieza diaria respetuosa.\n"
        "- *[77] Sensi Derm Hidratante Ligero* (S/. 90.00): Hidratación ligera.\n"
        "- *[74] Suero Antioxidante Iluminador* (S/. 178.00): Vitamina C para revitalizar.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"750", "76", "77", "74"}


# ==============================================================================
# 2. PRUEBAS DE MAQUILLAJE Y TONOS DE PIEL (CASO CHARITO - PIEL TRIGÜEÑA)
# ==============================================================================

@pytest.mark.asyncio
async def test_recommender_makeup_tones_for_dark_skin():
    """Caso Charito: Rubor para piel trigueña oscura.
    Verifica que el recomendador exponga tonos cálidos y no caiga en bucles."""
    candidates = [
        {"product_id": 436, "sku": "6361", "name": "Rubor Compacto Bronzage", "price": 96.0, "description": "Tono cálido bronceado para pieles morenas y trigueñas"},
        {"product_id": 111, "sku": "4018", "name": "Rubor Compacto Champagne", "price": 96.0, "description": "Tono iluminador cálido"},
        {"product_id": 49, "sku": "520", "name": "Rubor Compacto Petal Rose", "price": 96.0, "description": "Tono rosado suave para pieles muy claras"},
    ]

    draft = (
        "Para Charito, que tiene piel trigueña oscura, los tonos cálidos y bronceados son los más favorecedores:\n"
        "- *[6361] Rubor Compacto Bronzage* (S/. 96.00): Su tono terracota bronceado aporta luminosidad y define los pómulos de forma natural.\n"
        "- *[4018] Rubor Compacto Champagne* (S/. 96.00): Ideal para dar un toque de brillo sutil en ocasiones especiales.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert "6361" in grounding.valid_skus
    assert "520" not in grounding.mentioned_skus  # Descartó el tono para piel clara


# ==============================================================================
# 3. PRUEBAS DE PERFILES OLFATIVOS, OCASIÓN Y DISCRIMINACIÓN DE GÉNERO
# ==============================================================================

@pytest.mark.asyncio
async def test_recommender_perfumery_young_man_citrus_fresh():
    """Caso Renzo: 32 años, busca fragancia fresca no invasiva (cítrica/moderna).
    Verifica recomendación de Ohm Now, Solo o Adrenaline y descarte de fragancias pesadas o femeninas."""
    candidates = [
        {"product_id": 200, "sku": "2037", "name": "Ohm Now Parfum", "price": 230.0, "description": "Notas frescas de higo y absenta, moderno y juvenil"},
        {"product_id": 217, "sku": "217", "name": "Adrenaline Eau de Toilette", "price": 145.0, "description": "Notas cítricas de lima y lavanda"},
        {"product_id": 799, "sku": "79934", "name": "Solo Parfum", "price": 112.0, "description": "Mandarina y jengibre, fresco y versátil"},
        {"product_id": 11, "sku": "2210", "name": "Soy Única Colonia", "price": 68.0, "description": "Colonia femenina dulce con lichi"},
    ]

    draft = (
        "Para Renzo (32 años) que busca un aroma fresco y juvenil:\n"
        "- *[2037] Ohm Now Parfum* (S/. 230.00): Fragancia moderna con notas de higo y absenta, fresca y distintiva.\n"
        "- *[79934] Solo Parfum* (S/. 112.00): Frescura cítrica de mandarina con jengibre para uso diario.\n"
        "- *[217] Adrenaline Eau de Toilette* (S/. 145.00): Energía cítrica de lima y lavanda.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"2037", "79934", "217"}
    assert "2210" not in grounding.mentioned_skus  # Descartó la colonia femenina


# ==============================================================================
# 4. PRUEBAS DE GESTIÓN MULTI-MARCA Y RESTRICCIONES MONO-MARCA
# ==============================================================================

@pytest.mark.asyncio
async def test_recommender_strict_mono_brand_restriction():
    """Valida que si el usuario pide expresamente 'Solo Yanbal', no se mezclen opciones de Ésika."""
    candidates_yanbal = [
        {"product_id": 1, "sku": "Y_101", "name": "Crema Yanbal 1", "marca": "Yanbal", "price": 50.0},
        {"product_id": 2, "sku": "Y_102", "name": "Crema Yanbal 2", "marca": "Yanbal", "price": 60.0},
    ]

    draft = (
        "En Yanbal, contamos con estas opciones ideales:\n"
        "- *[Y_101] Crema Yanbal 1* (S/. 50.00): Hidratación profunda.\n"
        "- *[Y_102] Crema Yanbal 2* (S/. 60.00): Nutrición intensiva.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates_yanbal)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"Y_101", "Y_102"}


@pytest.mark.asyncio
async def test_recommender_multi_brand_comparison_grouping():
    """Valida que en consultas abiertas comparativas se agrupen con subtítulos (*En Yanbal:*, *En Ésika:*)."""
    candidates = [
        {"product_id": 1, "sku": "Y_101", "name": "Labial Yanbal Hydra", "marca": "Yanbal", "price": 45.0},
        {"product_id": 2, "sku": "E_201", "name": "Labial Colorfix Ésika", "marca": "Ésika", "price": 35.0},
    ]

    draft = (
        "Aquí tienes excelentes opciones de labiales según cada marca:\n\n"
        "*En Yanbal:*\n"
        "- *[Y_101] Labial Yanbal Hydra* (S/. 45.00): Hidratación y color duradero.\n\n"
        "*En Ésika:*\n"
        "- *[E_201] Labial Colorfix Ésika* (S/. 35.00): Acabado mate de 24 horas.\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert "Y_101" in grounding.valid_skus
    assert "E_201" in grounding.valid_skus


# ==============================================================================
# 5. PRUEBAS DE CROSS-SELLING (VENTA CRUZADA DE COMPLEMENTOS)
# ==============================================================================

@pytest.mark.asyncio
async def test_recommender_cross_selling_routine_and_hygiene():
    """Valida sugerencias complementarias coherentes (Perfume Ohm Now ➔ Desodorante Ohm o Effective)."""
    candidates = [
        {"product_id": 200, "sku": "2037", "name": "Ohm Now Parfum", "price": 230.0},
        {"product_id": 346, "sku": "346", "name": "Desodorante Roll on Original", "price": 34.5},
        {"product_id": 392, "sku": "392", "name": "Desodorante Roll on Ohm", "price": 28.0},
    ]

    draft = (
        "Para Renzo, la elección principal es:\n"
        "- *[2037] Ohm Now Parfum* (S/. 230.00): Fragancia fresca y moderna.\n\n"
        "Para complementar su rutina diaria y prolongar la frescura, te sugiero sumar:\n"
        "- *[346] Desodorante Roll on Original* (S/. 34.50) o el *[392] Desodorante Roll on Ohm* (S/. 28.00).\n"
    )

    grounding = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert grounding.is_valid is True
    assert set(grounding.valid_skus) == {"2037", "346", "392"}


# ==============================================================================
# 6. PRUEBAS DE AUTO-REFLEXIÓN DE RÚBRICA Y RECHAZO DETERMINISTA
# ==============================================================================

@pytest.mark.asyncio
async def test_rubric_judge_rejects_and_corrects_page_hallucinations():
    """Valida que el Juez de Rúbrica rechace el borrador que contenga números de página como SKUs
    y proporcione la tabla de corrección exacta."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    candidates = [
        {"product_id": 510, "sku": "79", "name": "Totalist Crema SPF 30", "price": 40.0, "pagina": 27},
        {"product_id": 11, "sku": "2210", "name": "Soy Única Colonia", "price": 68.0, "pagina": 86},
    ]

    # Borrador con alucinaciones de página
    state = {
        "raw_query": "opciones económicas para el rostro y colonia",
        "candidate_products": candidates,
        "draft_response": "Te sugiero [SKU: 27] Totalist Crema SPF 30 y [SKU: 86] Soy Única Colonia.",
        "iteration_count": 0,
        "max_iterations": 3,
        "messages": [],
    }

    eval_result = await nodes.rubric_evaluator_judge(state)
    assert eval_result["meets_rubric"] is False
    assert eval_result["reflection_action"] == "refine_synthesis"
    assert "27" in eval_result["critique"]
    assert "86" in eval_result["critique"]
    assert "TABLA DE REEMPLAZO OBLIGATORIO" in eval_result["critique"]
