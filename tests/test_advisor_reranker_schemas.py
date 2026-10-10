"""
tests/test_advisor_reranker_schemas.py - Pruebas unitarias para esquemas de Reranker y Rúbrica con Feedback Loops.
"""

import pytest
from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorToolCall,
    QualityRubricEvaluation,
)
from src.agent_service.core.stores.product.schemas import (
    RerankAndFilterInput,
)


def test_advisor_tool_call_supports_reranker():
    """Valida que AdvisorToolCall acepte 'rerank_and_filter_candidates'."""
    call = AdvisorToolCall(
        tool_name="rerank_and_filter_candidates",
        arguments={"context_query": "crema económica", "target_item_count": 2},
        purpose="Reordenar candidatos y aplicar corte Top-2",
    )
    assert call.tool_name == "rerank_and_filter_candidates"
    assert call.arguments["target_item_count"] == 2


def test_quality_rubric_evaluation_new_fields_and_actions():
    """Valida los nuevos puntajes y acciones de reflexión de QualityRubricEvaluation."""
    eval_pass = QualityRubricEvaluation(
        relevance_score=9.0,
        grounding_score=9.5,
        constraints_score=9.0,
        presentation_score=8.5,
        clarity_and_unambiguity_score=9.0,
        intent_fulfillment_score=9.0,
        criteria_clarity_score=10.0,
        is_approved=True,
        reflection_action="approve",
        critique=None,
    )
    assert eval_pass.is_approved is True
    assert eval_pass.reflection_action == "approve"

    # Acción de Medium Loop: re_rerank_with_feedback
    eval_medium = QualityRubricEvaluation(
        relevance_score=7.0,
        grounding_score=8.0,
        constraints_score=5.0,
        presentation_score=7.0,
        clarity_and_unambiguity_score=8.0,
        intent_fulfillment_score=4.0,
        criteria_clarity_score=8.0,
        is_approved=False,
        reflection_action="re_rerank_with_feedback",
        critique="El precio supera el umbral anterior; excluir productos >= S/. 65.",
        remedy_suggestions=["Forzar opciones con precio < S/. 50.0"],
    )
    assert eval_medium.is_approved is False
    assert eval_medium.reflection_action == "re_rerank_with_feedback"
    assert "excluir productos" in eval_medium.critique

    # Acción de Clarificación Enfocada: ask_clarification
    eval_ask = QualityRubricEvaluation(
        relevance_score=7.0,
        grounding_score=8.0,
        constraints_score=7.0,
        presentation_score=7.0,
        clarity_and_unambiguity_score=6.0,
        intent_fulfillment_score=7.0,
        criteria_clarity_score=5.0,
        is_approved=False,
        reflection_action="ask_clarification",
        critique="Criterio ambiguo: existen dos opciones con disyuntiva marcada de acabado y precio.",
    )
    assert eval_ask.reflection_action == "ask_clarification"


def test_rerank_and_filter_input_schema():
    """Valida la estructura de RerankAndFilterInput para candidatos y feedback."""
    data = {
        "candidates": [
            {"sku": "79", "name": "Totalist Antiarrugas", "price": 40.0},
            {"sku": "67", "name": "BB Cream Matificante", "price": 65.0},
        ],
        "context_query": "Silvana, 50 años, cutis mixto, buscando cremas más económicas",
        "target_item_count": 1,
        "feedback": "Excluir BB Cream por no reducir el precio",
        "sort_by": "price_asc",
    }
    obj = RerankAndFilterInput(**data)
    assert len(obj.candidates) == 2
    assert obj.target_item_count == 1
    assert obj.feedback == "Excluir BB Cream por no reducir el precio"
    assert obj.sort_by == "price_asc"
