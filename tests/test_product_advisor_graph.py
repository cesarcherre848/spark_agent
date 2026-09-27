"""
tests/test_product_advisor_graph.py - Pruebas unitarias para el subgrafo unificado product_advisor.

Cubre:
1. Validación de esquemas Pydantic del plan y la rúbrica.
2. Comprobación de herramientas deterministas (filtro y ordenamiento de precios/marcas/páginas).
3. Comprobación de la herramienta de historial de compras del cliente (Odoo).
4. Ejecución del grafo en búsqueda directa (Aprobada en iteración 1).
5. Ejecución del grafo con recomendación personalizada usando historial de compras.
6. Ciclo de auto-reflexión con rúbrica (Rechazo en iteración 1 y Aprobación en iteración 2 con crítica).
7. Límite estricto de 3 iteraciones (Corte forzoso para evitar bucle infinito).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from src.agent_service.graph.sub_graphs.product_advisor.schemas import (
    AdvisorPlan,
    AdvisorToolCall,
    QualityRubricEvaluation,
    format_products_for_advisor_prompt,
)
from src.agent_service.graph.sub_graphs.product_advisor.tools import (
    build_advisor_tools_registry,
    invoke_advisor_tool,
)
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes
from src.agent_service.graph.sub_graphs.product_advisor.graph import (
    build_product_advisor_graph,
    _route_after_rubric,
)
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState
from src.agent_service.graph.sub_graphs.product_advisor.schemas import FinalAnswer


def test_advisor_plan_and_rubric_schemas():
    """Verifica la instanciación correcta de esquemas Pydantic del Asesor."""
    plan = AdvisorPlan(
        reasoning="Buscar labiales de Yanbal en catálogo",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "labial", "marca": "Yanbal", "pagina": 12},
                purpose="Recuperar candidatos de la página solicitada",
            )
        ],
        extracted_metadata={"marca": "Yanbal", "pagina": 12},
    )
    assert plan.strategy == "direct_search"
    assert len(plan.tool_calls) == 1
    assert plan.tool_calls[0].tool_name == "search_product_catalog"
    assert plan.extracted_metadata["pagina"] == 12

    rubric = QualityRubricEvaluation(
        relevance_score=9.0,
        grounding_score=9.5,
        constraints_score=8.5,
        presentation_score=9.0,
        is_approved=True,
        critique=None,
        remedy_suggestions=[],
    )
    assert rubric.is_approved is True
    assert rubric.relevance_score == 9.0


def test_filter_and_sort_products_tool():
    """Verifica que filter_and_sort_products filtre por precio, marca y ordene correctamente."""
    from src.agent_service.tools.product_tools import create_filter_and_sort_products_tool

    tool = create_filter_and_sort_products_tool()
    sample_products = [
        {"sku": "SKU-A", "name": "Labial Rojo Yanbal", "price": 45.0, "marca": "Yanbal", "pagina": 10},
        {"sku": "SKU-B", "name": "Labial Nude Yanbal", "price": 85.0, "marca": "Yanbal", "pagina": 10},
        {"sku": "SKU-C", "name": "Perfume Ésika", "price": 70.0, "marca": "Ésika", "pagina": 15},
    ]

    # Filtrar por presupuesto máximo S/ 50 y marca Yanbal
    res = tool.invoke({
        "products": sample_products,
        "max_price": 50.0,
        "marca": "Yanbal",
    })
    assert len(res) == 1
    assert res[0]["sku"] == "SKU-A"

    # Ordenar por precio descendente
    res_sorted = tool.invoke({
        "products": sample_products,
        "sort_by": "price_desc",
    })
    assert len(res_sorted) == 3
    assert res_sorted[0]["sku"] == "SKU-B"
    assert res_sorted[-1]["sku"] == "SKU-A"


@pytest.mark.asyncio
async def test_get_customer_purchase_history_tool():
    """Verifica que get_customer_purchase_history procese correctamente los pedidos de Odoo."""
    from src.agent_service.tools.product_tools import create_get_customer_purchase_history_tool

    mock_orders = {
        "sale": [
            {
                "id": 101,
                "amount_total": 120.0,
                "lines": [
                    {"name": "[SKU-1] Labial Mate", "product_uom_qty": 2.0, "price_unit": 35.0},
                    {"name": "[SKU-2] Crema Hidratante", "product_uom_qty": 1.0, "price_unit": 50.0},
                ],
            }
        ]
    }

    with patch("src.agent_service.tools.sales_tools.odoo_list_sales_orders", AsyncMock(return_value=mock_orders)):
        tool = create_get_customer_purchase_history_tool()
        result = await tool.ainvoke({"partner_id": 42, "user_id": 5})

        assert result["orders_count"] == 1
        assert result["total_spent"] == 120.0
        assert len(result["top_purchased_products"]) == 2
        top_prod = result["top_purchased_products"][0]
        assert top_prod["name"] == "Labial Mate"
        assert top_prod["total_qty"] == 2


def test_route_after_rubric_logic():
    """Valida la bifurcación condicional según el resultado de la rúbrica y el límite de iteraciones."""
    # Caso 1: Aprobado -> finalize_response
    state_approved: ProductAdvisorState = {"meets_rubric": True, "iteration_count": 1, "max_iterations": 3}
    assert _route_after_rubric(state_approved) == "finalize_response"

    # Caso 2: Rechazado pero en iteración 1 -> plan_and_select_tools (reintento)
    state_retry: ProductAdvisorState = {"meets_rubric": False, "iteration_count": 1, "max_iterations": 3}
    assert _route_after_rubric(state_retry) == "plan_and_select_tools"

    # Caso 3: Rechazado en iteración 2 -> plan_and_select_tools (reintento)
    state_retry_2: ProductAdvisorState = {"meets_rubric": False, "iteration_count": 2, "max_iterations": 3}
    assert _route_after_rubric(state_retry_2) == "plan_and_select_tools"

    # Caso 4: Rechazado y alcanzó límite 3 -> finalize_response (corte forzoso)
    state_max_reached: ProductAdvisorState = {"meets_rubric": False, "iteration_count": 3, "max_iterations": 3}
    assert _route_after_rubric(state_max_reached) == "finalize_response"


@pytest.mark.asyncio
async def test_advisor_graph_direct_search_flow():
    """Verifica el flujo completo de búsqueda directa aprobado en la primera iteración."""
    mock_llm = MagicMock(spec=BaseChatModel)

    # 1. Mock de Plan
    mock_plan = AdvisorPlan(
        reasoning="Búsqueda de perfumes Ésika",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "perfumes florales", "marca": "Ésika"},
                purpose="Buscar en catálogo",
            )
        ],
        extracted_metadata={"marca": "Ésika"},
    )

    # 2. Mock de Rúbrica (Aprobada en iteración 1)
    mock_rubric = QualityRubricEvaluation(
        relevance_score=9.0,
        grounding_score=9.0,
        constraints_score=9.0,
        presentation_score=9.0,
        is_approved=True,
    )

    # 3. Mock de Síntesis
    mock_synthesis = FinalAnswer(
        response_text="Te recomiendo el Perfume Red Rose de Ésika [SKU: PERF-01] a S/ 65.00."
    )

    # Mock de herramientas
    mock_tools_registry = {
        "search_product_catalog": AsyncMock(return_value=[
            {
                "sku": "PERF-01",
                "name": "Red Rose Eau de Parfum",
                "price": 65.0,
                "marca": "Ésika",
                "description": "Perfume floral intenso con notas de rosas.",
            }
        ]),
        "get_product_odoo_details": AsyncMock(return_value={"products": []}),
    }

    nodes = ProductAdvisorNodes(
        llm=mock_llm,
        tools_registry=mock_tools_registry,
        default_max_iterations=3,
    )
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = mock_plan
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = mock_rubric
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = mock_synthesis

    # Construir grafo inyectando los nodos
    workflow = build_product_advisor_graph(
        llm=mock_llm,
        tools_registry=mock_tools_registry,
        default_max_iterations=3,
    )

    # Inyectar los mocks internos en el grafo compilado usando el wrapper de nodos
    state = {
        "raw_query": "Busco perfumes Ésika",
        "user_id": 5,
        "channel": "whatsapp",
    }

    # Ejecutar secuencialmente los nodos para verificar el flujo paso a paso
    plan_res = await nodes.plan_and_select_tools(state)
    state.update(plan_res)
    assert len(state["planned_tools"]) == 1

    exec_res = await nodes.execute_tools(state)
    state.update(exec_res)
    assert len(state["candidate_products"]) == 1
    assert state["candidate_products"][0]["sku"] == "PERF-01"

    synth_res = await nodes.synthesize_draft(state)
    state.update(synth_res)
    assert "PERF-01" in state["draft_response"]

    judge_res = await nodes.rubric_evaluator_judge(state)
    state.update(judge_res)
    assert state["meets_rubric"] is True
    assert state["iteration_count"] == 1

    final_res = await nodes.finalize_response(state)
    assert "PERF-01" in final_res["final_response"]
    assert final_res["is_sufficient"] is True


@pytest.mark.asyncio
async def test_advisor_graph_personalized_recommendation_with_purchase_history():
    """Verifica que el planificador invoque el historial de compras y personalice la recomendación."""
    mock_llm = MagicMock(spec=BaseChatModel)

    mock_plan = AdvisorPlan(
        reasoning="Cliente habitual solicita recomendaciones personalizadas",
        strategy="personalized_recommendation",
        tool_calls=[
            AdvisorToolCall(
                tool_name="get_customer_purchase_history",
                arguments={"partner_id": 99},
                purpose="Obtener historial previo de compras",
            ),
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "labiales hidratantes", "marca": "Yanbal"},
                purpose="Buscar productos alineados al gusto del cliente",
            ),
        ],
    )

    mock_tools_registry = {
        "get_customer_purchase_history": AsyncMock(return_value={
            "partner_id": 99,
            "orders_count": 3,
            "total_spent": 240.0,
            "top_purchased_products": [{"name": "Labial Hydra", "total_qty": 4}],
            "preference_summary": "Cliente habitual de labiales Yanbal.",
        }),
        "search_product_catalog": AsyncMock(return_value=[
            {
                "sku": "LAB-HYDRA-02",
                "name": "Labial Hydra Velvet Yanbal",
                "price": 38.0,
                "marca": "Yanbal",
                "description": "Nueva versión con ácido hialurónico.",
            }
        ]),
        "get_product_odoo_details": AsyncMock(return_value={"products": []}),
    }

    nodes = ProductAdvisorNodes(
        llm=mock_llm,
        tools_registry=mock_tools_registry,
        default_max_iterations=3,
    )
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = mock_plan
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="Hola! Como sueles comprar labiales Yanbal, te recomiendo el nuevo Labial Hydra Velvet [SKU: LAB-HYDRA-02] a S/ 38.00."
    )
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=9.5,
        grounding_score=9.5,
        constraints_score=9.0,
        presentation_score=9.5,
        is_approved=True,
    )

    state: ProductAdvisorState = {
        "raw_query": "¿Qué me recomiendas para hoy?",
        "partner_id": 99,
        "customer_name": "María Pérez",
        "user_id": 5,
    }

    state.update(await nodes.plan_and_select_tools(state))
    assert len(state["planned_tools"]) == 2

    state.update(await nodes.execute_tools(state))
    assert state["customer_purchase_history"] is not None
    assert state["customer_purchase_history"]["orders_count"] == 3
    assert len(state["candidate_products"]) == 1
    assert state["candidate_products"][0]["sku"] == "LAB-HYDRA-02"

    state.update(await nodes.synthesize_draft(state))
    assert "LAB-HYDRA-02" in state["draft_response"]

    state.update(await nodes.rubric_evaluator_judge(state))
    assert state["meets_rubric"] is True

    final_res = await nodes.finalize_response(state)
    assert "LAB-HYDRA-02" in final_res["final_response"]


@pytest.mark.asyncio
async def test_advisor_graph_reflection_loop_rejection_then_approval():
    """Verifica que el bucle de reflexión repita y logre aprobar tras un rechazo inicial."""
    mock_llm = MagicMock(spec=BaseChatModel)

    nodes = ProductAdvisorNodes(
        llm=mock_llm,
        tools_registry={
            "search_product_catalog": AsyncMock(return_value=[
                {"sku": "SKU-99", "name": "Labial", "price": 40.0, "marca": "Yanbal"}
            ]),
        },
        default_max_iterations=3,
    )

    # Mock planner
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(
        reasoning="Búsqueda de labial",
        strategy="direct_search",
        tool_calls=[],
    )

    # Mock judge: Falla en intento 1, Aprueba en intento 2
    failed_rubric = QualityRubricEvaluation(
        relevance_score=6.0,
        grounding_score=7.0,
        constraints_score=6.0,
        presentation_score=7.0,
        is_approved=False,
        critique="La respuesta no especificó el tono ni la campaña.",
        remedy_suggestions=["Incluir detalles de edición"],
    )
    approved_rubric = QualityRubricEvaluation(
        relevance_score=9.0,
        grounding_score=9.0,
        constraints_score=9.0,
        presentation_score=9.0,
        is_approved=True,
    )

    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.side_effect = [failed_rubric, approved_rubric]

    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(response_text="Labial Yanbal tono frambuesa C10")

    state: ProductAdvisorState = {
        "raw_query": "Quiero un labial frambuesa",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Iteración 1
    state.update(await nodes.plan_and_select_tools(state))
    state.update(await nodes.execute_tools(state))
    state.update(await nodes.synthesize_draft(state))
    state.update(await nodes.rubric_evaluator_judge(state))

    assert state["meets_rubric"] is False
    assert state["iteration_count"] == 1
    assert state["critique"] == "La respuesta no especificó el tono ni la campaña."
    assert _route_after_rubric(state) == "plan_and_select_tools"

    # Iteración 2 (Reintento con feedback)
    state.update(await nodes.plan_and_select_tools(state))
    state.update(await nodes.execute_tools(state))
    state.update(await nodes.synthesize_draft(state))
    state.update(await nodes.rubric_evaluator_judge(state))

    assert state["meets_rubric"] is True
    assert state["iteration_count"] == 2
    assert _route_after_rubric(state) == "finalize_response"


@pytest.mark.asyncio
async def test_advisor_graph_max_iterations_cutoff():
    """Verifica que el subgrafo se detenga exactamente en la tercera iteración sin bucle infinito."""
    mock_llm = MagicMock(spec=BaseChatModel)

    nodes = ProductAdvisorNodes(
        llm=mock_llm,
        tools_registry={"search_product_catalog": AsyncMock(return_value=[])},
        default_max_iterations=3,
    )

    always_fail_rubric = QualityRubricEvaluation(
        relevance_score=5.0,
        grounding_score=5.0,
        constraints_score=5.0,
        presentation_score=5.0,
        is_approved=False,
        critique="No se hallaron productos con esas especificaciones.",
    )

    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(reasoning="Reintento", strategy="direct_search", tool_calls=[])
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = always_fail_rubric
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(response_text="Disculpa, no encontré productos exactos.")

    state: ProductAdvisorState = {
        "raw_query": "producto imposible xyz",
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # Bucle simulando las 3 iteraciones del grafo
    for i in range(1, 4):
        state.update(await nodes.plan_and_select_tools(state))
        state.update(await nodes.execute_tools(state))
        state.update(await nodes.synthesize_draft(state))
        state.update(await nodes.rubric_evaluator_judge(state))

        if i < 3:
            assert _route_after_rubric(state) == "plan_and_select_tools"
        else:
            # En la iteración 3 debe cortar forzosamente
            assert _route_after_rubric(state) == "finalize_response"

    assert state["iteration_count"] == 3
    assert state["meets_rubric"] is False
    final_res = await nodes.finalize_response(state)
    assert final_res["final_response"] is not None


@pytest.mark.asyncio
async def test_advisor_multi_vendor_handling():
    """Valida el manejo comercial de múltiples proveedores y marcas:
    1. Filtrado determinista por vendor_name en filter_and_sort_products.
    2. Inyección de etiquetas de proveedor y marca en format_products_for_advisor_prompt.
    3. Conformidad whitelabel (sin filtración de vendor_id ni tablas de backend).
    """
    from src.agent_service.tools.product_tools import create_filter_and_sort_products_tool

    # 1. Test filtrado determinista por vendor_name
    filter_tool = create_filter_and_sort_products_tool()
    sample_products = [
        {"sku": "Y-101", "name": "Perfume Yanbal", "marca": "Yanbal", "vendor_name": "Unique S.A.", "price": 120.0},
        {"sku": "E-202", "name": "Perfume Ésika", "marca": "Ésika", "vendor_name": "CETCO S.A.", "price": 95.0},
        {"sku": "C-303", "name": "Labial Cyzone", "marca": "Cyzone", "vendor_name": "CETCO S.A.", "price": 25.0},
    ]

    filtered_unique = filter_tool.invoke({
        "products": sample_products,
        "vendor_name": "Unique S.A.",
    })
    assert len(filtered_unique) == 1
    assert filtered_unique[0]["sku"] == "Y-101"

    filtered_cetco = filter_tool.invoke({
        "products": sample_products,
        "vendor_name": "CETCO S.A.",
    })
    assert len(filtered_cetco) == 2
    assert {p["sku"] for p in filtered_cetco} == {"E-202", "C-303"}

    # 2. Test formateo para prompt con marcas y proveedores diferenciados
    formatted_prompt = format_products_for_advisor_prompt(sample_products)
    assert "Marca: Yanbal" in formatted_prompt
    assert "Proveedor: Unique S.A." in formatted_prompt
    assert "Marca: Ésika" in formatted_prompt
    assert "Proveedor: CETCO S.A." in formatted_prompt

    # 3. Test de síntesis de respuesta comercial multi-proveedor con Whitelabel
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm, default_max_iterations=3)
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text=(
            "¡Claro que sí! Para perfumes disponemos de opciones destacadas de diferentes casas:\n"
            "- [Yanbal] Perfume Yanbal a S/ 120.00.\n"
            "- [Ésika] Perfume Ésika a S/ 95.00.\n"
            "¿Cuál de las dos marcas prefieres?"
        )
    )

    state: ProductAdvisorState = {
        "raw_query": "¿Qué perfumes tienes disponibles en catálogo?",
        "candidate_products": sample_products,
        "iteration_count": 0,
        "max_iterations": 3,
    }

    draft_result = await nodes.synthesize_draft(state)
    draft_text = draft_result["draft_response"]

    # Validar que especifica las marcas comerciales [Yanbal] y [Ésika]
    assert "[Yanbal]" in draft_text
    assert "[Ésika]" in draft_text

    # Validar 100% Whitelabel: sin filtración de backend
    assert "vendor_id" not in draft_text.lower()
    assert "res_partner" not in draft_text.lower()
    assert "partner_id" not in draft_text.lower()
    assert "odoo" not in draft_text.lower()


@pytest.mark.asyncio
async def test_advisor_multi_turn_context_retention():
    """Valida la retención de contexto en conversaciones multi-turno:
    1. plan_and_select_tools inyecta los mensajes previos de la conversación al planificador.
    2. synthesize_draft inyecta los mensajes previos al sintetizador.
    3. rubric_evaluator_judge inyecta los mensajes previos al evaluador de calidad.
    """
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm, default_max_iterations=3)

    turn1_user = HumanMessage(content="Hola, busco perfumes florales para mujer")
    turn1_ai = AIMessage(content="Tenemos opciones florales como Vibranza de Ésika y Ccori de Yanbal. ¿Cuál marca prefieres?")
    turn2_user = HumanMessage(content="de essika o de yambal ?")

    state: ProductAdvisorState = {
        "raw_query": "de essika o de yambal ?",
        "messages": [turn1_user, turn1_ai, turn2_user],
        "iteration_count": 0,
        "max_iterations": 3,
        "candidate_products": [
            {"sku": "E-01", "name": "Vibranza Ésika", "marca": "Ésika", "price": 150.0},
            {"sku": "Y-02", "name": "Ccori Yanbal", "marca": "Yanbal", "price": 210.0},
        ],
    }

    # 1. Verificar plan_and_select_tools
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(
        reasoning="El usuario responde a la pregunta de marca para la búsqueda previa de perfumes florales",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "perfumes florales mujer", "marca": "Ésika"},
                purpose="Buscar perfumes florales Ésika",
            ),
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "perfumes florales mujer", "marca": "Yanbal"},
                purpose="Buscar perfumes florales Yanbal",
            ),
        ],
    )

    plan_res = await nodes.plan_and_select_tools(state)
    assert len(plan_res["planned_tools"]) == 2

    # Verificar que _planner.ainvoke recibió turn1_user y turn1_ai en su lista de mensajes
    call_args = nodes._planner.ainvoke.call_args[0][0]
    message_contents = [getattr(m, "content", "") for m in call_args]
    assert any("perfumes florales para mujer" in c for c in message_contents)
    assert any("Vibranza de Ésika y Ccori de Yanbal" in c for c in message_contents)

    # 2. Verificar synthesize_draft
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="¡Excelente! Te muestro ambas opciones de perfumes florales: Vibranza [Ésika] a S/ 150 y Ccori [Yanbal] a S/ 210."
    )

    draft_res = await nodes.synthesize_draft(state)
    assert "Vibranza" in draft_res["draft_response"]

    synth_call_args = nodes._synthesizer.ainvoke.call_args[0][0]
    synth_contents = [getattr(m, "content", "") for m in synth_call_args]
    assert any("perfumes florales para mujer" in c for c in synth_contents)

    # 3. Verificar rubric_evaluator_judge
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=10.0,
        grounding_score=10.0,
        constraints_score=10.0,
        presentation_score=10.0,
        is_approved=True,
    )

    rubric_res = await nodes.rubric_evaluator_judge(state)
    assert rubric_res["meets_rubric"] is True

    judge_call_args = nodes._rubric_judge.ainvoke.call_args[0][0]
    judge_contents = [getattr(m, "content", "") for m in judge_call_args]
    assert any("perfumes florales para mujer" in c for c in judge_contents)


@pytest.mark.asyncio
async def test_multi_turn_gender_context_propagation():
    """Verifica que el diálogo multi-turno preserve y propague el género/público objetivo (masculino/hombre)."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    turn1_user = HumanMessage(content="perfumes masculinos con precio menor de 50 soles")
    turn1_ai = AIMessage(content="Actualmente, no contamos con perfumes masculinos por debajo de los S/. 50.00.")
    turn2_user = HumanMessage(content="y colonias ?")

    state: ProductAdvisorState = {
        "raw_query": "y colonias ?",
        "messages": [turn1_user, turn1_ai, turn2_user],
        "iteration_count": 0,
        "max_iterations": 3,
        "candidate_products": [
            {"sku": "208", "name": "Temptation Hombre Eau de Parfum", "marca": "Yanbal", "price": 192.0},
            {"sku": "15225", "name": "Freshing Colonia corporal", "marca": "Ésika", "price": 46.90},
        ],
    }

    # 1. Verificar plan_and_select_tools recibe historial y prompt de género
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(
        reasoning="El usuario pregunta por colonias en seguimiento a su búsqueda de fragancias masculinas",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "colonia hombre masculino eau de toilette fragancia fresca"},
                purpose="Buscar colonias y fragancias frescas para hombre",
            )
        ],
    )

    plan_res = await nodes.plan_and_select_tools(state)
    assert len(plan_res["planned_tools"]) == 1
    assert "hombre" in plan_res["planned_tools"][0]["arguments"]["query"]

    # Verificar que el prompt del sistema enviado al planner contiene la directriz de género
    planner_calls = nodes._planner.ainvoke.call_args[0][0]
    system_prompt_content = planner_calls[0].content
    assert "PROPAGACIÓN OBLIGATORIA DE PÚBLICO OBJETIVO Y GÉNERO" in system_prompt_content

    # 2. Verificar synthesize_draft contiene la salvaguarda de género
    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="En colonias masculinas y unisex contamos con: *En Ésika:* [15225] Freshing Colonia corporal (S/. 46.90)."
    )

    draft_res = await nodes.synthesize_draft(state)
    synth_calls = nodes._synthesizer.ainvoke.call_args[0][0]
    synth_sys_prompt = synth_calls[0].content
    assert "CONSISTENCIA DE PÚBLICO OBJETIVO Y GÉNERO" in synth_sys_prompt
    assert "15225" in draft_res["draft_response"]

    # 3. Verificar rubric_evaluator_judge contiene la auditoría de género
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=9.5,
        grounding_score=9.5,
        constraints_score=9.0,
        presentation_score=9.5,
        is_approved=True,
    )

    rubric_res = await nodes.rubric_evaluator_judge(state)
    judge_calls = nodes._rubric_judge.ainvoke.call_args[0][0]
    judge_sys_prompt = judge_calls[0].content
    assert "público objetivo / género hombre/mujer/niños" in judge_sys_prompt
    assert rubric_res["meets_rubric"] is True


@pytest.mark.asyncio
async def test_rubric_rejects_gender_mismatch():
    """Verifica que el Juez de Rúbrica rechace una respuesta cuando se recomiendan artículos femeninos a una consulta masculina."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    turn1_user = HumanMessage(content="perfumes masculinos")
    turn1_ai = AIMessage(content="Tenemos opciones como Ohm y Solo.")
    turn2_user = HumanMessage(content="y colonias ?")

    state: ProductAdvisorState = {
        "raw_query": "y colonias ?",
        "draft_response": "*En Yanbal:* [2210] Soy Única Colonia (S/. 68.00): Para resaltar tu feminidad.",
        "candidate_products": [{"sku": "2210", "name": "Soy Única Colonia", "marca": "Yanbal", "price": 68.0}],
        "messages": [turn1_user, turn1_ai, turn2_user],
        "iteration_count": 0,
        "max_iterations": 3,
    }

    # El LLM de evaluación detecta que Soy Única es femenina mientras se solicitó masculino
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=5.0,
        grounding_score=9.0,
        constraints_score=4.0,
        presentation_score=8.0,
        is_approved=False,
        critique="La respuesta recomienda una colonia femenina ('Soy Única') cuando el usuario busca opciones masculinas.",
        remedy_suggestions=["Buscar fragancias masculinas ligeras como Selecto Eau de Toilette o colonias unisex."],
    )

    rubric_res = await nodes.rubric_evaluator_judge(state)
    assert rubric_res["meets_rubric"] is False
    assert rubric_res["rubric_scores"]["constraints"] == 4.0
    assert "femenina" in rubric_res["critique"]
    assert _route_after_rubric(rubric_res) == "plan_and_select_tools"


@pytest.mark.asyncio
async def test_proactive_top_k_for_exhaustive_and_budget_queries():
    """Verifica que para consultas exhaustivas o con límites de presupuesto, el planificador incorpore
    la directriz de aumentar el top-k/limit a 15-20 y el fallback asigne limit=15."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    # Caso 1: Verificar contenido del System Prompt para Top-K proactivo
    nodes._planner = AsyncMock()
    nodes._planner.ainvoke.return_value = AdvisorPlan(
        reasoning="Consulta exhaustiva: fijar limit=15 para perfumes masculinos Yanbal",
        strategy="direct_search",
        tool_calls=[
            AdvisorToolCall(
                tool_name="search_product_catalog",
                arguments={"query": "perfume hombre", "marca": "Yanbal", "limit": 15},
                purpose="Búsqueda exhaustiva",
            )
        ],
        extracted_metadata={"marca": "Yanbal", "limit": 15},
    )

    state: ProductAdvisorState = {
        "raw_query": "dame todos los perfumes para hombre s de yambal",
        "messages": [],
    }

    plan_res = await nodes.plan_and_select_tools(state)
    assert plan_res["planned_tools"][0]["arguments"]["limit"] == 15

    # Verificar que el system prompt contenga la directriz de top-k exhaustivo
    planner_calls = nodes._planner.ainvoke.call_args[0][0]
    planner_sys_prompt = planner_calls[0].content
    assert "AJUSTE PROACTIVO DE TOP-K / LIMIT (BÚSQUEDA EXHAUSTIVA)" in planner_sys_prompt
    assert "limit=15" in planner_sys_prompt or "limit=20" in planner_sys_prompt

    # Caso 2: Verificar que el fallback automático asigne limit=15 cuando hay palabras de exhaustividad o presupuesto
    nodes._planner.ainvoke.side_effect = Exception("Planner failure simulation")
    fallback_res = await nodes.plan_and_select_tools(state)
    assert len(fallback_res["planned_tools"]) == 1
    assert fallback_res["planned_tools"][0]["arguments"]["limit"] == 15


@pytest.mark.asyncio
async def test_execute_tools_budget_exceeded_fallback():
    """Verifica que si filter_and_sort_products descarta todos los productos por precio,
    execute_tools preserve los 3 candidatos más económicos marcados con budget_exceeded=True."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    # Registro con mock tools
    mock_filter_tool = MagicMock(return_value=[])  # Simula que ninguno estuvo bajo el presupuesto
    nodes._tools_registry = {
        "filter_and_sort_products": mock_filter_tool,
        "get_product_odoo_details": AsyncMock(return_value={"products": []}),
    }

    initial_candidates = [
        {"sku": "SKU-EXPENSIVE", "name": "Perfume Lujo", "price": 180.0},
        {"sku": "SKU-CHEAP1", "name": "Colonia Básica 1", "price": 65.0},
        {"sku": "SKU-CHEAP2", "name": "Colonia Básica 2", "price": 70.0},
        {"sku": "SKU-MID", "name": "Colonia Media", "price": 95.0},
    ]

    state: ProductAdvisorState = {
        "raw_query": "perfumes de hombre menos de 50 soles",
        "planned_tools": [
            {
                "tool_name": "filter_and_sort_products",
                "arguments": {"max_price": 50.0},
            }
        ],
        "candidate_products": initial_candidates,
    }

    result = await nodes.execute_tools(state)
    retained_candidates = result["candidate_products"]

    # Debe haber retenido 3 productos más económicos en lugar de dejar la lista vacía
    assert len(retained_candidates) == 3
    assert retained_candidates[0]["sku"] == "SKU-CHEAP1"
    assert retained_candidates[0].get("budget_exceeded") is True
    assert retained_candidates[1]["sku"] == "SKU-CHEAP2"
    assert retained_candidates[1].get("budget_exceeded") is True


@pytest.mark.asyncio
async def test_synthesize_draft_and_rubric_budget_guideline():
    """Verifica que el prompt del sintetizador contenga la regla de manejo de presupuesto
    y la rúbrica evalúe positivamente la orientación consultiva de rangos."""
    mock_llm = MagicMock(spec=BaseChatModel)
    nodes = ProductAdvisorNodes(llm=mock_llm)

    nodes._synthesizer = AsyncMock()
    nodes._synthesizer.ainvoke.return_value = FinalAnswer(
        response_text="Actualmente las fragancias para hombre inician desde S/. 65.00. Te presento: [SKU-CHEAP1] Colonia Básica (S/. 65.00)."
    )

    state: ProductAdvisorState = {
        "raw_query": "perfumes de hombre de menos de 40 soles",
        "candidate_products": [{"sku": "SKU-CHEAP1", "name": "Colonia Básica", "price": 65.0, "budget_exceeded": True}],
        "draft_response": "Actualmente las fragancias para hombre inician desde S/. 65.00. Te presento: [SKU-CHEAP1] Colonia Básica (S/. 65.00).",
        "messages": [],
    }

    # 1. Verificar synthesize_draft
    draft_res = await nodes.synthesize_draft(state)
    synth_calls = nodes._synthesizer.ainvoke.call_args[0][0]
    synth_sys_prompt = synth_calls[0].content
    assert "MANEJO DE PRESUPUESTO Y RANGOS DE PRECIO" in synth_sys_prompt
    assert "budget_exceeded" in synth_sys_prompt

    # 2. Verificar rubric_evaluator_judge
    nodes._rubric_judge = AsyncMock()
    nodes._rubric_judge.ainvoke.return_value = QualityRubricEvaluation(
        relevance_score=9.0,
        grounding_score=9.5,
        constraints_score=9.0,
        presentation_score=9.0,
        is_approved=True,
    )

    rubric_res = await nodes.rubric_evaluator_judge(state)
    judge_calls = nodes._rubric_judge.ainvoke.call_args[0][0]
    judge_sys_prompt = judge_calls[0].content
    assert "ligeramente mayor" in judge_sys_prompt or "límites de presupuesto" in judge_sys_prompt
    assert rubric_res["meets_rubric"] is True




