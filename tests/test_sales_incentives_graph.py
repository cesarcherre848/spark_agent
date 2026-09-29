"""
tests/test_sales_incentives_graph.py - Pruebas unitarias para el subgrafo sales_incentives (Actor-Critic, ε-Greedy y BaseSynthesizerNode).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent_service.graph.sub_graphs.sales_incentives.state import SalesIncentivesState
from src.agent_service.graph.sub_graphs.sales_incentives.nodes import SalesIncentivesNodes
from src.agent_service.graph.sub_graphs.sales_incentives.schemas import (
    IncentiveActionPlan,
    IncentiveMilestone,
    SuggestedProductCombo,
    CriticPlanEvaluation,
)
from src.agent_service.graph.sub_graphs.sales_incentives.graph import build_sales_incentives_graph


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    return llm


@pytest.mark.asyncio
async def test_gather_context_and_trends_node(mock_llm):
    """Valida que gather_context_and_trends_node obtenga órdenes, incentivos y calcule métricas."""
    mock_list_orders = AsyncMock(return_value={
        "sale": [{"id": 1, "amount_total": 450.0, "lines": [
            {"product_id": [11, "[2210] Soy Única Colonia"], "product_uom_qty": 2.0, "price_subtotal": 68.0}
        ]}],
        "draft": [{"id": 2, "amount_total": 120.0, "lines": []}],
        "cancel": []
    })
    mock_get_incentives = AsyncMock(return_value=[
        {
            "id": 42,
            "title": "Colonia Soy Sexy Gratis",
            "observations": "Por 3 unidades de colonias Soy (Cód. 2210)",
            "description": "Colonia gratis",
            "supplier_name": "Unique S.A.",
        }
    ])

    nodes = SalesIncentivesNodes(
        llm=mock_llm,
        list_orders_tool=mock_list_orders,
        get_incentives_tool=mock_get_incentives,
    )

    state = {
        "user_id": 5,
        "supplier_id": 9,
    }

    res = await nodes.gather_context_and_trends_node(state)

    assert "orders_data" in res
    assert "sales_trends" in res
    assert "gap_info" in res
    assert res["gap_info"]["confirmed_total_pen"] == 450.0
    assert res["gap_info"]["draft_total_pen"] == 120.0
    assert res["gap_info"]["current_discount_tier"] == "25% de Descuento"
    assert res["gap_info"]["next_discount_tier"] == "30% de Descuento"
    assert res["gap_info"]["gap_to_next_tier_pen"] == 270.0  # 720 - 450
    assert len(res["sales_trends"]) == 1
    assert res["sales_trends"][0]["sku"] == "2210"


@pytest.mark.asyncio
async def test_actor_planner_node(mock_llm):
    """Valida que el Actor genere un plan de incentivos estructurado."""
    mock_plan = IncentiveActionPlan(
        summary="Plan para alcanzar la escala del 30% en Unique.",
        current_status_overview="Avance de S/. 450.00 confirmado.",
        milestones=[
            IncentiveMilestone(
                milestone_name="Alcanzar 30% de Descuento",
                target_type="discount_tier",
                gap_pen=270.0,
                reward_description="30% de margen comercial en toda la orden",
                action_steps=["Cerrar cotización pendiente de S/. 120", "Vender 2 colonias Soy"],
            )
        ],
        suggested_combos=[
            SuggestedProductCombo(
                combo_name="Dúo Frescura Juvenil",
                skus=["2210", "2218"],
                target_customer="Carmen Saavedra",
                estimated_amount_pen=68.0,
                strategy_type="hybrid",
                rationale="Producto tendencia que suma para el regalo de Vende y Gana",
            )
        ],
        projected_additional_earnings_pen=150.0,
        epsilon_balance_rationale="75% foco en colonias Soy para ganar el premio y 25% tendencia.",
    )

    with patch("src.agent_service.graph.sub_graphs.sales_incentives.nodes.bind_structured_output") as mock_bind:
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = mock_plan
        mock_bind.return_value = mock_runnable

        nodes = SalesIncentivesNodes(llm=mock_llm)
        state = {
            "user_id": 5,
            "raw_query": "¿Qué me falta para ganar más?",
            "gap_info": {"confirmed_total_pen": 450.0, "gap_to_next_tier_pen": 270.0},
            "active_incentives": [],
            "sales_trends": [],
            "iteration_count": 0,
        }

        res = await nodes.actor_planner_node(state)

        assert "candidate_plan" in res
        assert res["iteration_count"] == 1
        assert res["candidate_plan"]["summary"] == "Plan para alcanzar la escala del 30% en Unique."
        assert len(res["candidate_plan"]["milestones"]) == 1


@pytest.mark.asyncio
async def test_critic_evaluator_approval(mock_llm):
    """Valida que el Critic apruebe un plan viable con score >= 8.0."""
    mock_eval = CriticPlanEvaluation(
        is_approved=True,
        score=9.0,
        temporal_feasibility_score=9.5,
        constraint_adherence_score=9.0,
        roi_score=8.5,
        critiques=[],
        suggested_adjustments=[],
    )

    with patch("src.agent_service.graph.sub_graphs.sales_incentives.nodes.bind_structured_output") as mock_bind:
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = mock_eval
        mock_bind.return_value = mock_runnable

        nodes = SalesIncentivesNodes(llm=mock_llm)
        state = {
            "candidate_plan": {"summary": "Buen plan"},
            "gap_info": {"gap_to_next_tier_pen": 200.0},
            "iteration_count": 1,
            "max_iterations": 3,
        }

        res = await nodes.critic_evaluator_node(state)

        assert res["is_approved"] is True
        assert res["critic_score"] == 9.0
        assert res["critique"] is None


@pytest.mark.asyncio
async def test_critic_evaluator_rejection_and_feedback(mock_llm):
    """Valida que el Critic rechace un plan deficiente y devuelva críticas y ajustes para el Actor."""
    mock_eval = CriticPlanEvaluation(
        is_approved=False,
        score=6.5,
        temporal_feasibility_score=5.0,
        constraint_adherence_score=7.0,
        roi_score=7.5,
        critiques=["El monto sugerido en combos (S/. 50) es insuficiente para cubrir la brecha de S/. 270."],
        suggested_adjustments=["Incrementar la cantidad de combos o sugerir productos premium."],
    )

    with patch("src.agent_service.graph.sub_graphs.sales_incentives.nodes.bind_structured_output") as mock_bind:
        mock_runnable = AsyncMock()
        mock_runnable.ainvoke.return_value = mock_eval
        mock_bind.return_value = mock_runnable

        nodes = SalesIncentivesNodes(llm=mock_llm)
        state = {
            "candidate_plan": {"summary": "Plan flojo"},
            "gap_info": {"gap_to_next_tier_pen": 270.0},
            "iteration_count": 1,
            "max_iterations": 3,
        }

        res = await nodes.critic_evaluator_node(state)

        assert res["is_approved"] is False
        assert res["critic_score"] == 6.5
        assert "insuficiente" in res["critique"]
        assert len(res["suggested_improvements"]) == 1


@pytest.mark.asyncio
async def test_critic_evaluator_forces_approval_on_max_iterations(mock_llm):
    """Valida que si se alcanza max_iterations (default 3), el Critic apruebe forzosamente para no bloquear."""
    nodes = SalesIncentivesNodes(llm=mock_llm)
    state = {
        "candidate_plan": {"summary": "Plan al límite"},
        "iteration_count": 3,
        "max_iterations": 3,
    }

    res = await nodes.critic_evaluator_node(state)

    assert res["is_approved"] is True
    assert res["critic_score"] == 8.0


@pytest.mark.asyncio
async def test_synthesize_coaching_node_uses_base_synthesizer(mock_llm):
    """Valida que synthesize_coaching_node aplique BaseSynthesizerNode con divisas S/. y WhatsApp."""
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="¡Excelente progreso! Te faltan S/. 180.00 para la meta de 30%."))

    nodes = SalesIncentivesNodes(llm=mock_llm)
    state = {
        "candidate_plan": {"summary": "Plan de éxito"},
        "gap_info": {"confirmed_total_pen": 540.0, "gap_to_next_tier_pen": 180.0, "current_discount_tier": "25%"},
        "channel": "whatsapp",
    }

    res = await nodes.synthesize_coaching_node(state)

    assert "final_response" in res
    assert "messages" in res
    assert "S/." in res["final_response"]
    assert res["channel"] == "whatsapp"


@pytest.mark.asyncio
async def test_sales_incentives_graph_e2e_approval(mock_llm):
    """Valida el flujo integral compilado de sales_incentives aprobando en la primera iteración."""
    mock_list_orders = AsyncMock(return_value={
        "sale": [{"id": 1, "amount_total": 400.0, "lines": []}],
        "draft": [],
    })
    mock_get_incentives = AsyncMock(return_value=[
        {"id": 44, "title": "Escala 30%", "observations": "Desde S/ 720"}
    ])

    mock_plan = IncentiveActionPlan(
        summary="Plan para alcanzar S/ 720.",
        current_status_overview="S/. 400.00 confirmado.",
        milestones=[],
        suggested_combos=[],
        projected_additional_earnings_pen=100.0,
        epsilon_balance_rationale="Balance óptimo.",
    )
    mock_eval = CriticPlanEvaluation(
        is_approved=True,
        score=9.0,
        temporal_feasibility_score=9.0,
        constraint_adherence_score=9.0,
        roi_score=9.0,
        critiques=[],
        suggested_adjustments=[],
    )

    with patch("src.agent_service.graph.sub_graphs.sales_incentives.nodes.bind_structured_output") as mock_bind:
        mock_actor_runnable = AsyncMock()
        mock_actor_runnable.ainvoke.return_value = mock_plan
        mock_critic_runnable = AsyncMock()
        mock_critic_runnable.ainvoke.return_value = mock_eval

        # El bind devuelve primero al actor y luego al critic
        mock_bind.side_effect = [mock_actor_runnable, mock_critic_runnable]

        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="¡Hola! Te faltan S/. 320.00 para la escala de 30%."))

        graph = build_sales_incentives_graph(
            llm=mock_llm,
            list_orders_tool=mock_list_orders,
            get_incentives_tool=mock_get_incentives,
        )

        initial_state = {
            "user_id": 5,
            "raw_query": "¿Cómo voy con mis metas?",
            "messages": [HumanMessage(content="¿Cómo voy con mis metas?")],
        }

        final_state = await graph.ainvoke(initial_state)

        assert final_state["is_approved"] is True
        assert final_state["final_response"] is not None
        assert "S/." in final_state["final_response"]
