import pytest
from src.agent_service.tools.product_tools import create_rerank_and_filter_candidates_tool


def test_silvana_mature_cologne_reranker():
    """Valida que para una clienta de 50 años que busca colonia fresca, el reranker
    descarte líneas de concepto juvenil ('Soy Sexy') e infantil ('Colonia Niñas')
    y priorice líneas florales/frescas clásicas como L'Essence."""
    reranker = create_rerank_and_filter_candidates_tool()

    candidates = [
        {
            "product_id": 540,
            "sku": "678",
            "name": "Colonia Niñas",
            "description": "Cont. 100 ml con frutos rojos",
            "marca": "Yanbal",
            "price": 60.0,
        },
        {
            "product_id": 460,
            "sku": "2219",
            "name": "Soy Sexy Colonia",
            "description": "Cont. 200 ml con neroli y sándalo",
            "marca": "Yanbal",
            "price": 68.0,
        },
        {
            "product_id": 459,
            "sku": "2220",
            "name": "Soy Glow Colonia",
            "description": "Cont. 200 ml con escarcha, manzana y guayaba",
            "marca": "Yanbal",
            "price": 68.0,
        },
        {
            "product_id": 40,
            "sku": "287",
            "name": "Flor de Cerezo Silvestre Colonia",
            "description": "Cont. 200 ml L'Essence flor de cerezo fresca",
            "marca": "Yanbal",
            "price": 68.0,
        },
        {
            "product_id": 41,
            "sku": "2195",
            "name": "Violeta Salvaje Colonia",
            "description": "Cont. 200 ml L'Essence violeta fresca",
            "marca": "Yanbal",
            "price": 68.0,
        },
    ]

    intent_spec = {
        "target_gender": "female",
        "target_age": 50,
        "anatomical_focus": "fragrance",
        "product_profile": "fresca_diario",
    }

    result = reranker.invoke({
        "candidates": candidates,
        "context_query": "colonia refrescante y económica para clienta de 50 años",
        "target_item_count": 2,
        "intent_spec": intent_spec,
    })

    assert len(result) > 0
    skus = [p["sku"] for p in result]
    # No debe contener ni Soy Sexy ni Soy Glow ni Colonia Niñas
    assert "2219" not in skus, "Soy Sexy no debe ser recomendada a clienta de 50 años"
    assert "2220" not in skus, "Soy Glow no debe ser recomendada a clienta de 50 años"
    assert "678" not in skus, "Colonia Niñas no debe ser recomendada a clienta de 50 años"
    # Debe contener Flor de Cerezo Silvestre o Violeta Salvaje (L'Essence)
    assert any(sku in skus for sku in ["287", "2195"]), "Debe priorizar la línea L'Essence"


def test_age_inferred_directly_from_query_text():
    """Valida que si intent_spec no provee la edad, el reranker extraiga '52 años'
    directamente del texto de la consulta e infiera el perfil maduro."""
    reranker = create_rerank_and_filter_candidates_tool()

    candidates = [
        {"sku": "678", "name": "Colonia Niñas", "description": "Cont. 100 ml aroma infantil frutal", "price": 60.0},
        {"sku": "2219", "name": "Soy Sexy Colonia", "description": "Cont. 200 ml aroma juvenil sensual", "price": 68.0},
        {"sku": "287", "name": "Flor de Cerezo Silvestre Colonia", "description": "Cont. 200 ml aroma floral fresco y elegante", "price": 68.0},
    ]

    result = reranker.invoke({
        "candidates": candidates,
        "context_query": "Recomiéndame una colonia fresca para una mujer de 52 años",
        "target_item_count": 1,
    })

    assert len(result) == 1
    assert result[0]["sku"] == "287"


def test_renzo_masculine_fragrance_reranker():
    """Valida que para un cliente hombre se filtren fragancias femeninas o infantiles
    sin usar nombres de personas hardcodeados."""
    reranker = create_rerank_and_filter_candidates_tool()

    candidates = [
        {
            "sku": "2037",
            "name": "Ohm Now Parfum",
            "description": "Para hombre aroma fresco y herbal",
            "marca": "Yanbal",
            "price": 230.0,
        },
        {
            "sku": "287",
            "name": "Flor de Cerezo Silvestre Colonia",
            "description": "Para mujer L'Essence",
            "marca": "Yanbal",
            "price": 68.0,
        },
        {
            "sku": "678",
            "name": "Colonia Niñas",
            "description": "Cont. 100 ml",
            "marca": "Yanbal",
            "price": 60.0,
        },
    ]

    intent_spec = {
        "target_gender": "male",
        "target_age_group": "adult",
        "anatomical_focus": "fragrance",
    }

    result = reranker.invoke({
        "candidates": candidates,
        "context_query": "colonia fresca para chico joven",
        "target_item_count": 2,
        "intent_spec": intent_spec,
    })

    skus = [p["sku"] for p in result]
    assert "2037" in skus
    assert "287" not in skus
    assert "678" not in skus
