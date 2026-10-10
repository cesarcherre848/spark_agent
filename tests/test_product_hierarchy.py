"""
tests/test_product_hierarchy.py - Pruebas unitarias para la Búsqueda Jerárquica y Resolución Universal de Familias y Variantes.
"""

import pytest
from src.agent_service.core.stores.product.product_hierarchy import (
    UniversalProductHierarchyResolver,
    extract_base_concept_name,
    extract_attribute_details,
)


def test_extract_base_concept_name():
    """Valida la extracción agnóstica de nombres base de conceptos."""
    assert extract_base_concept_name("Totalist Crema Antiarrugas Cont. 100 g") == "Totalist Crema Antiarrugas"
    assert extract_base_concept_name("Rubor Compacto Tono Bronzage") == "Rubor Compacto"
    assert extract_base_concept_name("Delineador Retráctil para Labios - Nude") == "Delineador Retráctil para Labios"
    assert extract_base_concept_name("Aretes Amour Doré") == "Aretes Amour Doré"


def test_extract_attribute_details():
    """Valida la extracción de detalles de atributos (gramajes, SPF, activos)."""
    details = extract_attribute_details("Totalist Crema SPF 30 Antiarrugas", "Cont. 50 g con jalea real")
    assert "50 g" in details
    assert "SPF 30" in details


@pytest.mark.asyncio
async def test_universal_product_hierarchy_grouping():
    """Valida la agrupación jerárquica de candidatos planos en conceptos canónicos y variantes."""
    resolver = UniversalProductHierarchyResolver()
    candidates = [
        {
            "product_id": 241,
            "sku": "150",
            "name": "Totalist Crema Antiarrugas",
            "marca": "Yanbal",
            "price": 40.0,
            "description": "Cont. 100 g con jalea real y colágeno vegano",
        },
        {
            "product_id": 242,
            "sku": "81",
            "name": "Totalist Gel Control Brillo",
            "marca": "Yanbal",
            "price": 62.0,
            "description": "Cont. 100 g con aloe vera y niacinamida",
        },
        {
            "product_id": 510,
            "sku": "79",
            "name": "Totalist Crema SPF 30 Antiarrugas",
            "marca": "Yanbal",
            "price": 40.0,
            "description": "Cont. 50 g",
        },
        {
            "product_id": 11,
            "sku": "2210",
            "name": "Soy Única Colonia",
            "marca": "Yanbal",
            "price": 68.0,
            "description": "Cont. 200 ml con lichi y piña colada",
        },
    ]

    families = await resolver.group_candidates_hierarchically(candidates)
    assert len(families) >= 2  # Totalist y Soy Única

    # Validar que las variantes de Totalist se agrupen bajo sus conceptos
    totalist_fams = [f for f in families if "totalist" in f.family_name.lower()]
    assert len(totalist_fams) > 0
    all_totalist_skus = {v.sku for f in totalist_fams for v in f.variants}
    assert "150" in all_totalist_skus
    assert "81" in all_totalist_skus
    assert "79" in all_totalist_skus
