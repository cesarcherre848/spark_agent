"""
tests/test_sku_grounding.py - Pruebas unitarias para el Marco de Blindaje y Validación Anti-Alucinación de SKUs.
"""

import pytest
from src.agent_service.core.guardrails.sku_grounding import (
    SKUExtractor,
    SKUGroundingValidator,
    SKUFailSafeSanitizer,
)


def test_sku_extractor_various_formats():
    """Valida la extracción precisa de SKUs en diferentes formatos."""
    text = (
        "Te recomiendo las siguientes opciones: "
        "- *[SKU: 150] Totalist Crema Antiarrugas* (S/. 40.00)\n"
        "- *[81] Totalist Gel Control Brillo* (S/. 62.00)\n"
        "- *Código: 2210 Soy Única Colonia* (S/. 68.00)\n"
        "- *Ref 2195 Violeta Salvaje*\n"
    )
    skus = SKUExtractor.extract_skus(text)
    assert "150" in skus
    assert "81" in skus
    assert "2210" in skus
    assert "2195" in skus


def test_sku_extractor_avoids_false_positives():
    """Valida que el extractor ignore montos en soles, unidades, teléfonos, edades y órdenes."""
    text = (
        "Para Dika de 53 años, la orden S00015 tiene 1 unidad por S/. 471.50 con 100g de crema "
        "y SPF 30. Llama al 975674799 en Campaña C10."
    )
    skus = SKUExtractor.extract_skus(text)
    # Ninguno de los números de arriba debe ser considerado SKU
    assert "53" not in skus
    assert "471" not in skus
    assert "100" not in skus
    assert "30" not in skus
    assert "975674799" not in skus


def test_sku_grounding_validator_approved_when_all_valid():
    """Valida que un borrador con SKUs autorizados sea aprobado."""
    candidates = [
        {"sku": "150", "name": "Totalist Crema Antiarrugas", "price": 40.0},
        {"sku": "81", "name": "Totalist Gel Control Brillo", "price": 62.0},
    ]
    draft = "Te recomiendo el [150] Totalist Crema Antiarrugas y el [81] Totalist Gel."
    result = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    assert result.is_valid is True
    assert result.invalid_skus == []
    assert set(result.valid_skus) == {"150", "81"}


def test_sku_grounding_validator_rejects_hallucinated_page_numbers():
    """Valida que números de página alucinados como SKUs sean rechazados con tabla de reemplazo."""
    candidates = [
        {"sku": "79", "name": "Totalist SPF 30 Antiarrugas", "pagina": 27, "price": 40.0},
        {"sku": "2210", "name": "Soy Única Colonia", "pagina": 86, "price": 68.0},
    ]
    # El LLM alucinó los números de página 27 y 86 como SKUs
    draft = "Opciones: [SKU: 27] Totalist SPF 30 y [SKU: 86] Soy Única Colonia."
    result = SKUGroundingValidator.validate_draft_grounding(draft, candidates)
    
    assert result.is_valid is False
    assert "27" in result.invalid_skus
    assert "86" in result.invalid_skus
    assert result.replacement_map.get("27") == "79"
    assert result.replacement_map.get("86") == "2210"
    assert "TABLA DE REEMPLAZO OBLIGATORIO" in result.critique_message


def test_sku_failsafe_sanitizer():
    """Valida que el sanitizador reemplace SKUs según la tabla de reemplazos y elimine corchetes engañosos."""
    text = "Opciones: [SKU: 27] Totalist y [99999] Producto Desconocido."
    authorized_skus = {"79", "150"}
    replacement_map = {"27": "79"}
    
    sanitized = SKUFailSafeSanitizer.sanitize_text(text, authorized_skus, replacement_map)
    assert "[79]" in sanitized
    assert "[27]" not in sanitized
    assert "[99999]" not in sanitized
