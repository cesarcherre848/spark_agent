import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.agent_service.core.hitl.schemas import (
    HITLBinaryDecision,
    HITLEntitySelection,
    HITLOrderChoice,
)
from src.agent_service.graph.sub_graphs.sales_manage.graph import build_sales_manage_graph
from src.agent_service.graph.sub_graphs.sales_manage.schemas import (
    SalesExtractionResult,
    SalesDuplicateCheckResult,
    SalesSynthesizeResponse,
    SalesSKUItem,
)


@pytest.fixture
def mock_list_orders_tool():
    tool = AsyncMock()
    tool.return_value = {
        "draft": [{"id": 101, "name": "SO001", "customer_name": "Empresa Alfa", "amount_total": 450.0}],
        "sale": [{"id": 102, "name": "SO002", "customer_name": "Empresa Alfa", "amount_total": 1200.0}],
        "cancel": [],
    }
    return tool


@pytest.fixture
def mock_list_current_orders_tool():
    tool = AsyncMock()
    tool.return_value = []
    return tool


@pytest.fixture
def mock_create_quotation_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "created",
        "order_id": 105,
        "name": "SO005",
        "amount_total": 250.0,
    }
    return tool


@pytest.fixture
def mock_update_quotation_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "updated",
        "order_id": 101,
        "name": "SO001",
        "amount_total": 600.0,
    }
    return tool


@pytest.fixture
def mock_view_quotation_tool():
    tool = AsyncMock()
    tool.return_value = {
        "id": 101,
        "name": "SO001",
        "customer_name": "Empresa Alfa",
        "state": "draft",
        "amount_total": 600.0,
        "amount_untaxed": 508.47,
        "amount_tax": 91.53,
        "lines": [{"product_name": "SKU 1", "quantity": 2.0, "price_unit": 300.0, "price_subtotal": 600.0}],
    }
    return tool


@pytest.fixture
def mock_confirm_order_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "confirmed",
        "order_id": 101,
        "name": "SO001",
        "state": "sale",
        "amount_total": 600.0,
    }
    return tool


@pytest.fixture
def mock_unlock_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "unlocked", "order_id": 102}
    return tool


@pytest.fixture
def mock_update_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "updated", "order_id": 102, "name": "SO002"}
    return tool


@pytest.fixture
def mock_lock_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "locked", "order_id": 102}
    return tool


@pytest.fixture
def mock_remove_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "cancelled", "order_id": 101}
    return tool


@pytest.fixture
def mock_list_customers_tool():
    tool = AsyncMock()
    tool.return_value = [{"id": 14, "name": "Empresa Alfa", "phone": "987654321"}]
    return tool


@pytest.fixture
def mock_upsert_customer_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "id": 99, "name": "Nuevo Cliente"}
    return tool


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_sales_manage_list_branch(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la rama 'list' para consultar pedidos del vendedor."""
    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(action="list", customer=None, items=[]),
        # 2. Synthesizer
        SalesSynthesizeResponse(response_text="Tienes 1 cotización y 1 pedido confirmado."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    result = await graph.ainvoke(
        {"raw_query": "muéstrame mis pedidos", "user_id": 5},
        config={"configurable": {"thread_id": "t-list"}},
    )

    mock_list_orders_tool.assert_awaited_once()
    assert result.get("sales_action") == "list"
    assert "draft" in result.get("orders_list", {})
    assert "Tienes 1 cotización" in result.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_remove_branch_accept(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la rama 'remove' con interrupción HITL aprobada en lenguaje natural ('dale, cancélala')."""
    # 1. Extractor
    # Interrupción HITL
    # 2. Binary parser para confirmar cancelación
    # 3. Synthesizer
    mock_llm.ainvoke.side_effect = [
        SalesExtractionResult(action="remove", order_name="SO001"),
        HITLBinaryDecision(decision="accept", reasoning="El usuario confirma cancelar."),
        SalesSynthesizeResponse(response_text="La orden SO001 ha sido cancelada correctamente."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    config = {"configurable": {"thread_id": "t-remove-accept"}}

    # Turno 1: se detiene en interrupt
    state_1 = await graph.ainvoke({"raw_query": "cancela la orden SO001", "user_id": 5}, config=config)
    assert "__interrupt__" in state_1

    # Turno 2: reanuda con lenguaje natural
    state_2 = await graph.ainvoke(Command(resume="dale porfa, cancélala"), config=config)

    mock_remove_order_tool.assert_awaited_once()
    assert state_2.get("remove_confirmed") is True
    assert "cancelada" in state_2.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_upsert_branch_new(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la creación de una cotización nueva cuando no hay duplicados."""
    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(action="upsert", customer="Empresa Alfa", items=[SalesSKUItem(sku="SKU1", qty=5.0)]),
        # 2. Synthesizer
        SalesSynthesizeResponse(response_text="Cotización SO005 creada con éxito para Empresa Alfa."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    result = await graph.ainvoke(
        {"raw_query": "cotízame 5 del SKU1 para Empresa Alfa", "user_id": 5},
        config={"configurable": {"thread_id": "t-upsert-new"}},
    )

    mock_create_quotation_tool.assert_awaited_once()
    assert result.get("sales_action") == "upsert"
    assert "SO005" in result.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_upsert_branch_duplicate_selection(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la rama upsert cuando hay duplicados y el usuario selecciona actualizar la orden existente."""
    mock_list_current_orders_tool.return_value = [
        {"id": 101, "name": "SO001", "amount_total": 450.0, "date_order": "2026-09-18"}
    ]

    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(action="upsert", customer="Empresa Alfa", items=[SalesSKUItem(sku="SKU1", qty=2.0)]),
        # 2. Judge (detecta duplicados)
        SalesDuplicateCheckResult(has_duplicates=True, matched_order_name="SO001", reasoning="Cliente tiene SO001 abierta."),
        # Interrupción HITL
        # 3. Order choice parser en lenguaje natural
        HITLOrderChoice(choice="selected", selected_order_name="SO001", reasoning="El usuario pide actualizar la existente."),
        # 4. Synthesizer
        SalesSynthesizeResponse(response_text="La cotización SO001 ha sido actualizada con éxito."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    config = {"configurable": {"thread_id": "t-upsert-dup"}}

    # Turno 1: pausa en duplicados
    state_1 = await graph.ainvoke({"raw_query": "agrega 2 del SKU1 a Empresa Alfa", "user_id": 5}, config=config)
    assert "__interrupt__" in state_1

    # Turno 2: respuesta en lenguaje natural
    state_2 = await graph.ainvoke(Command(resume="actualiza la orden que teníamos abierta SO001"), config=config)

    mock_update_quotation_tool.assert_awaited_once()
    assert state_2.get("duplicate_choice") == "selected"
    assert "SO001" in state_2.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_edit_order_guardrail(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la rama 'edit_order': pasa por guardrail HITL ➔ desbloqueo ➔ actualización ➔ re-bloqueo automático."""
    mock_view_quotation_tool.return_value = {
        "id": 102,
        "name": "SO002",
        "partner_id": 14,
        "customer_name": "Empresa Alfa",
        "state": "sale",
        "amount_total": 1200.0,
        "lines": [],
    }

    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(action="edit_order", order_name="SO002", items=[SalesSKUItem(sku="SKU1", qty=10.0)]),
        # Interrupción HITL
        # 2. Binary parser confirmando desbloqueo
        HITLBinaryDecision(decision="accept", reasoning="El usuario acepta desbloquear."),
        # 3. Synthesizer
        SalesSynthesizeResponse(response_text="El pedido SO002 fue actualizado y re-bloqueado con éxito."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    config = {"configurable": {"thread_id": "t-edit-order"}}

    # Turno 1: se detiene en guardrail unlock
    state_1 = await graph.ainvoke({"raw_query": "cambia a 10 unidades en el pedido confirmado SO002", "user_id": 5}, config=config)
    assert "__interrupt__" in state_1

    # Turno 2: reanudación en lenguaje natural
    state_2 = await graph.ainvoke(Command(resume="sí, no hay problema, desbloquéalo"), config=config)

    mock_unlock_order_tool.assert_awaited_once()
    mock_update_order_tool.assert_awaited_once()
    mock_lock_order_tool.assert_awaited_once()
    assert state_2.get("unlock_confirmed") is True
    assert "re-bloqueado" in state_2.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_ambiguous_customer_disambiguation(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba la desambiguación de cliente mediante HITL en lenguaje natural sin exponer IDs numéricos."""
    mock_list_customers_tool.return_value = [
        {"id": 14, "name": "Carlos Pérez", "phone": "987654321"},
        {"id": 28, "name": "Carlos Perú SAC", "phone": "912345678"},
    ]

    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(action="upsert", customer="Carlos", items=[SalesSKUItem(sku="SKU1", qty=2.0)]),
        # Interrupción HITL
        # 2. Entity selection parser
        HITLEntitySelection(action="select", selected_index=0, selected_entity_name="Carlos Pérez"),
        # 3. Judge
        SalesDuplicateCheckResult(has_duplicates=False),
        # 4. Synthesizer
        SalesSynthesizeResponse(response_text="Cotización creada para Carlos Pérez."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    config = {"configurable": {"thread_id": "t-ambiguous-cust"}}

    # Turno 1: pausa en desambiguación
    state_1 = await graph.ainvoke({"raw_query": "cotízale 2 del SKU1 a Carlos", "user_id": 5}, config=config)
    assert "__interrupt__" in state_1

    # Turno 2: respuesta en lenguaje natural
    state_2 = await graph.ainvoke(Command(resume="al primero, a Carlos Pérez"), config=config)

    assert state_2.get("partner_id") == 14
    assert state_2.get("customer_name") == "Carlos Pérez"
    mock_create_quotation_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_sales_manage_upsert_subtract_lines(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Prueba que el subgrafo ejecute la reducción/sustracción de líneas sobre una cotización existente."""
    mock_view_quotation_tool.return_value = {
        "id": 101,
        "name": "S00003",
        "customer_name": "Adhara Banda",
        "partner_id": 12,
        "state": "draft",
        "amount_total": 692.30,
        "amount_untaxed": 602.0,
        "amount_tax": 90.30,
        "lines": [
            {"product_name": "[6189] Sexy Glam", "quantity": 2.0, "price_unit": 179.0, "price_subtotal": 358.0},
            {"product_name": "[5104] Delineador Plumón Tattoo", "quantity": 4.0, "price_unit": 61.0, "price_subtotal": 244.0},
        ],
    }

    mock_llm.ainvoke.side_effect = [
        # 1. Extractor con items con action="subtract"
        SalesExtractionResult(
            action="upsert",
            customer="Adhara Banda",
            order_name="S00003",
            items=[
                SalesSKUItem(sku="6189", qty=2.0, action="subtract"),
                SalesSKUItem(sku="5104", qty=2.0, action="subtract"),
            ],
        ),
        # 2. Synthesizer
        SalesSynthesizeResponse(
            response_text="He actualizado la cotización S00003 restando 2 unidades de cada producto.",
        ),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    result = await graph.ainvoke(
        {"raw_query": "elimina el 2 items de 6189 y 2 items de 5104 de S00003", "user_id": 5},
        config={"configurable": {"thread_id": "t-upsert-subtract"}},
    )

    mock_update_quotation_tool.assert_awaited_once()
    called_items = mock_update_quotation_tool.call_args.kwargs["items"]
    assert len(called_items) == 2
    assert called_items[0]["action"] == "subtract"
    assert called_items[1]["action"] == "subtract"
    assert "S00003" in result.get("final_response", "")


@pytest.mark.asyncio
async def test_sales_manage_hybrid_multi_turn_view_and_remove(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Valida el patrón híbrido en dos turnos conversacionales:
    Turno 1: 'muestrame la cotizacion 03 de Adhara' -> Fast-Path a view_quotation_node sin auditoría de duplicados ni interrupciones.
    Turno 2: 'quita 2 unidades del Delineador' -> Fast-Path a update_quotation_node heredando S00003 sin falsos HITL de duplicados.
    """
    mock_view_quotation_tool.return_value = {
        "id": 3,
        "name": "S00003",
        "customer_name": "Adhara Banda",
        "state": "draft",
        "amount_total": 692.30,
        "lines": [
            {"product_name": "Sexy Glam", "quantity": 2.0, "price_unit": 179.0},
            {"product_name": "Delineador Plumón Tattoo", "quantity": 4.0, "price_unit": 61.0},
        ],
    }

    mock_llm.ainvoke.side_effect = [
        # Turno 1: Extractor (view)
        SalesExtractionResult(action="view", customer="Adhara", order_name="03"),
        # Turno 1: Synthesizer
        SalesSynthesizeResponse(response_text="Detalle de cotización S00003 para Adhara Banda..."),
        # Turno 2: Extractor (remove_items con nombre comercial 'Delineador')
        SalesExtractionResult(
            action="remove_items",
            customer=None,
            order_name=None,
            items=[SalesSKUItem(sku="Delineador", qty=2.0, action="subtract")],
        ),
        # Turno 2: Synthesizer
        SalesSynthesizeResponse(response_text="He quitado 2 unidades del Delineador de la cotización S00003."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    thread_cfg = {"configurable": {"thread_id": "t-hybrid-multi-turn"}}

    # Turno 1: Ver cotización 03
    t1_res = await graph.ainvoke(
        {"raw_query": "muestrame la cotizacion 03 de Adhara", "user_id": 5},
        config=thread_cfg,
    )
    assert "__interrupt__" not in t1_res
    assert t1_res.get("target_order_name") == "S00003"
    mock_view_quotation_tool.assert_awaited()
    # No debió llamar a list_current_orders ni al duplicate judge
    mock_list_current_orders_tool.assert_not_awaited()

    # Turno 2: Quitar 2 unidades del delineador
    t2_res = await graph.ainvoke(
        {"raw_query": "quita 2 unidades del Delineador"},
        config=thread_cfg,
    )
    assert "__interrupt__" not in t2_res
    assert t2_res.get("target_order_name") == "S00003"
    mock_update_quotation_tool.assert_awaited_once()
    called_items = mock_update_quotation_tool.call_args.kwargs["items"]
    assert len(called_items) == 1
    assert called_items[0]["sku"] == "Delineador"
    assert called_items[0]["qty"] == 2.0
    assert called_items[0]["action"] == "subtract"


@pytest.mark.asyncio
async def test_sales_manage_hybrid_add_items_fast_path(
    mock_llm,
    mock_list_orders_tool,
    mock_list_current_orders_tool,
    mock_create_quotation_tool,
    mock_update_quotation_tool,
    mock_view_quotation_tool,
    mock_confirm_order_tool,
    mock_unlock_order_tool,
    mock_update_order_tool,
    mock_lock_order_tool,
    mock_remove_order_tool,
    mock_list_customers_tool,
    mock_upsert_customer_tool,
):
    """Valida que agregar productos a una orden existente use la ruta Fast-Path sin chequear duplicados."""
    mock_llm.ainvoke.side_effect = [
        SalesExtractionResult(
            action="add_items",
            order_name="S00003",
            items=[SalesSKUItem(sku="6189", qty=3.0, action="add")],
        ),
        SalesSynthesizeResponse(response_text="He agregado 3 unidades de Sexy Glam a S00003."),
    ]

    graph = build_sales_manage_graph(
        llm=mock_llm,
        list_orders_tool=mock_list_orders_tool,
        list_current_orders_tool=mock_list_current_orders_tool,
        create_quotation_tool=mock_create_quotation_tool,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
        confirm_order_tool=mock_confirm_order_tool,
        unlock_order_tool=mock_unlock_order_tool,
        update_order_tool=mock_update_order_tool,
        lock_order_tool=mock_lock_order_tool,
        remove_order_tool=mock_remove_order_tool,
        list_customers_tool=mock_list_customers_tool,
        upsert_customer_tool=mock_upsert_customer_tool,
    )

    res = await graph.ainvoke(
        {"raw_query": "agrega 3 unidades de 6189 a S00003", "user_id": 5},
        config={"configurable": {"thread_id": "t-hybrid-add"}},
    )
    assert "__interrupt__" not in res
    mock_update_quotation_tool.assert_awaited_once()
    mock_list_current_orders_tool.assert_not_awaited()
    assert "S00003" in res.get("final_response", "")

