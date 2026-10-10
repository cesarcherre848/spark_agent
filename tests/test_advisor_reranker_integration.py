"""
tests/test_advisor_reranker_integration.py - Pruebas de integración del Asesor con Reranker y Medium Loop.

Escenario Principal:
Replicación del caso de Silvana / user_id = 8 (wa_51975674799):
1. Limpieza de candidatos obsoletos entre turnos para evitar acumulación cruzada.
2. Invocación de rerank_and_filter_candidates con descarte de cremas corporales (Biomilk) ante consulta de rostro.
3. Desencadenamiento del Medium Reflection Loop (re_rerank_with_feedback) cuando la selección de productos
   requiere ajuste de precio relativo ("algo aún más económico") y priorización de Totalist (S/. 40.00).
4. Verificación de acción ask_clarification ante criterios de usuario ambiguos.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorPlan,
    AdvisorToolCall,
    QualityRubricEvaluation,
    FinalAnswer,
)
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.graph import (
    build_product_advisor_graph,
    _route_after_rubric,
)
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState
from src.agent_service.tools.product_tools import (
    create_rerank_and_filter_candidates_tool,
)


@pytest.mark.asyncio
async def test_new_turn_cleans_candidate_products_pool():
    """Valida la Tarea 3.2: Que en un nuevo turno sin crítica previa, candidate_products
    se limpie para evitar la acumulación histórica de 99 candidatos como ocurrió en user_id = 8."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    # Simular candidatos acumulados de un turno previo
    stale_candidates = [
        {"sku": "750", "name": "Biomilk Leche Vegetal Crema Corporal", "price": 80.0},
        {"sku": "47000", "name": "Crema BB Cream Rostro", "price": 65.0},
        {"sku": "48000", "name": "Crema Supreme 8 en 1", "price": 70.0},
    ]

    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(
        reasoning="Nuevo turno: buscar cremas aún más económicas para rostro",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "crema facial economica", "limit": 15},
                purpose="Búsqueda de alternativas económicas",
            ),
            AdvisorToolCall(
                tool_name="rerank_and_filter_candidates",
                arguments={"context_query": "crema rostro economica", "sort_by": "price_asc"},
                purpose="Reranking por precio",
            ),
        ],
    )

    state: ProductAdvisorState = {
        "raw_query": "algo aun mas economico",
        "candidate_products": stale_candidates,
        "critique": None,  # Nuevo turno
        "iteration_count": 0,
        "messages": [
            HumanMessage(content="busco cremas para rostro"),
            AIMessage(content="Te recomiendo BB Cream [47000] a S/. 65 y Supreme [48000] a S/. 70"),
            HumanMessage(content="algo aun mas economico"),
        ],
    }

    result = await nodes.plan_and_select_tools(state)

    # 1. candidate_products debe estar reseteado a lista vacía para la nueva búsqueda
    assert result["candidate_products"] == []
    # 2. Debe haber guardado los SKUs y precios de referencia del turno previo
    assert "47000" in result["previous_turn_skus"]
    assert result["previous_turn_prices"].get("47000") == 65.0
    assert result["previous_turn_prices"].get("48000") == 70.0


@pytest.mark.asyncio
async def test_medium_reflection_loop_re_rerank_with_feedback():
    """Valida la Tarea 4.1, 4.2 y 4.3: Cuando la selección de productos viola la intención
    (ej: omitió el producto más económico de S/. 40.00), el juez emite re_rerank_with_feedback,
    el grafo enruta a execute_tools y el reranker re-ejecuta en memoria con la crítica semántica."""
    
    # 1. Probar enrutamiento del grafo
    state_medium_loop: ProductAdvisorState = {
        "meets_rubric": False,
        "iteration_count": 1,
        "max_iterations": 3,
        "reflection_action": "re_rerank_with_feedback",
        "critique": "El usuario pidió algo aún más económico y se ignoró Totalist de S/. 40.00",
    }
    next_node = _route_after_rubric(state_medium_loop)
    assert next_node == "execute_tools"

    # 2. Probar ejecución en memoria en execute_tools
    mock_llm = MagicMock(spec=BaseChatModel)
    reranker_tool = create_rerank_and_filter_candidates_tool()
    
    nodes = ProductAdvisorNodes(
        llm=mock_llm,
        tools_registry={"rerank_and_filter_candidates": reranker_tool},
    )

    candidate_pool = [
        {"sku": "48000", "name": "Crema Supreme 8 en 1", "price": 70.0, "category": "facial"},
        {"sku": "47000", "name": "Crema BB Cream Rostro", "price": 65.0, "category": "facial"},
        {"sku": "49000", "name": "Totalist Antiarrugas Rostro", "price": 40.0, "category": "facial"},
        {"sku": "750", "name": "Biomilk Leche Vegetal Crema Corporal", "price": 80.0, "category": "corporal"},
    ]

    state_for_execution: ProductAdvisorState = {
        "raw_query": "algo aun mas economico",
        "candidate_products": candidate_pool,
        "reflection_action": "re_rerank_with_feedback",
        "critique": "El usuario pidió algo aún más económico que S/. 65.00 para rostro. Priorizar opciones < S/. 50 y descartar corporales.",
        "iteration_count": 1,
        "planned_tools": [],
    }

    exec_res = await nodes.execute_tools(state_for_execution)
    filtered_products = exec_res["candidate_products"]

    # Debe haber descartado Biomilk (corporal)
    skus = [p["sku"] for p in filtered_products]
    assert "750" not in skus
    # Debe haber posicionado Totalist (S/. 40.00) en primer lugar
    assert filtered_products[0]["sku"] == "49000"
    assert filtered_products[0]["price"] == 40.0


@pytest.mark.asyncio
async def test_ask_clarification_routing_and_synthesis():
    """Valida la acción ask_clarification cuando el criterio es ambiguo."""
    state_clarification: ProductAdvisorState = {
        "meets_rubric": False,
        "iteration_count": 1,
        "max_iterations": 3,
        "reflection_action": "ask_clarification",
        "critique": "El cliente no especificó si busca crema hidratante de día o tratamiento antiedad de noche.",
    }
    # Debe enrutar a synthesize_draft para formular una pregunta calibradora
    next_node = _route_after_rubric(state_clarification)
    assert next_node == "synthesize_draft"

    # Verificar que synthesize_draft instruya la calibración
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="¿Prefieres una crema ligera hidratante para el día o un tratamiento nutritivo antiedad de noche para tu piel?"
    )

    state_synth: ProductAdvisorState = {
        "raw_query": "recomiendame una crema",
        "candidate_products": [{"sku": "49000", "name": "Totalist", "price": 40.0}],
        "reflection_action": "ask_clarification",
        "critique": "Criterio ambiguo en tipo de crema",
        "suggested_improvements": ["Preguntar al cliente por tipo de beneficio"],
    }

    synth_res = await nodes.synthesize_draft(state_synth)
    assert "Prefieres" in synth_res["draft_response"] or "crema" in synth_res["draft_response"]
