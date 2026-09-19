import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command

from src.agent_service.core.hitl.schemas import (
    HITLBinaryDecision,
    HITLEntitySelection,
    HITLOrderChoice,
)
from src.agent_service.graph.sub_graphs.sales_manage.graph import build_sales_manage_graph
from src.agent_service.graph.sub_graphs.sales_manage.nodes import SalesManageNodes
from src.agent_service.graph.sub_graphs.sales_manage.schemas import (
    SalesExtractionResult,
    SalesDuplicateCheckResult,
    SalesSynthesizeResponse,
    SalesSKUItem,
)


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=BaseChatModel)
    llm.bind = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_list_orders_tool():
    tool = AsyncMock()
    tool.return_value = {
        "draft": [{"id": 3, "name": "S00003", "customer_name": "Adhara Banda", "amount_total": 652.63}],
        "sale": [],
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
        "order_id": 108,
        "name": "S00004",
        "amount_total": 192.50,
    }
    return tool


@pytest.fixture
def mock_update_quotation_tool():
    tool = AsyncMock()
    tool.return_value = {
        "success": True,
        "action": "updated",
        "order_id": 3,
        "name": "S00003",
        "amount_total": 652.63,
    }
    return tool


@pytest.fixture
def mock_view_quotation_tool():
    tool = AsyncMock()

    async def _view_quotation(order_id=None, order_name=None):
        if (order_name and "04" in str(order_name)) or order_id == 108:
            return {
                "id": 108,
                "name": "S00004",
                "customer_name": "Janet Pupuche",
                "state": "draft",
                "amount_total": 192.50,
                "lines": [],
            }
        return {
            "id": 3,
            "name": "S00003",
            "customer_name": "Adhara Banda",
            "state": "draft",
            "amount_total": 652.63,
            "lines": [
                {"product_name": "Producto Anterior", "quantity": 1.0, "price_unit": 100.0},
            ],
        }

    tool.side_effect = _view_quotation
    return tool


@pytest.fixture
def mock_confirm_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "confirmed", "order_id": 3, "name": "S00003", "state": "sale"}
    return tool


@pytest.fixture
def mock_unlock_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "unlocked", "order_id": 3}
    return tool


@pytest.fixture
def mock_update_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "updated", "order_id": 3, "name": "S00003"}
    return tool


@pytest.fixture
def mock_lock_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "locked", "order_id": 3}
    return tool


@pytest.fixture
def mock_remove_order_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "action": "cancelled", "order_id": 3}
    return tool


@pytest.fixture
def mock_list_customers_tool():
    tool = AsyncMock()

    async def _list_customers(user_id=None, name=None, limit=10):
        if name and "janet" in name.lower():
            return [{"id": 13, "name": "Janet Pupuche", "phone": "999888777"}]
        if name and "adhara" in name.lower():
            return [{"id": 3, "name": "Adhara Banda", "phone": "987654321"}]
        return [{"id": 13, "name": "Janet Pupuche", "phone": "999888777"}]

    tool.side_effect = _list_customers
    return tool


@pytest.fixture
def mock_upsert_customer_tool():
    tool = AsyncMock()
    tool.return_value = {"success": True, "id": 99, "name": "Nuevo Cliente"}
    return tool


@pytest.mark.asyncio
async def test_user_scenario_switch_customer_creates_new_quotation_and_protects_existing(
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
    """Prueba el escenario exacto del usuario:
    Turno 1: Se consulta o visualiza la cotización S00003 de Adhara Banda.
    Turno 2: El usuario dice 'agrega estos productos a la cotizacion de Janet Pupuche'.
    Resultado esperado:
    - La reflexión y el enrutador purgan S00003 y no contaminan la orden de Adhara.
    - Se resuelve Janet Pupuche (partner_id: 13).
    - Al no tener cotizaciones abiertas, se crea una nueva cotización (S00004).
    - mock_update_quotation_tool NUNCA se llama para S00003.
    - mock_create_quotation_tool se llama con partner_id=13.
    """
    mock_llm.ainvoke.side_effect = [
        # Turno 1: Extractor (view S00003)
        SalesExtractionResult(action="view", customer="Adhara", order_name="03"),
        # Turno 1: Synthesizer
        SalesSynthesizeResponse(response_text="Detalle de cotización S00003 de Adhara Banda."),
        # Turno 2: Extractor (agrega productos a Janet Pupuche sin código de orden)
        SalesExtractionResult(
            action="upsert",
            customer="Janet Pupuche",
            order_name=None,
            items=[
                SalesSKUItem(sku="775", qty=1.0, action="add"),
                SalesSKUItem(sku="14", qty=1.0, action="add"),
                SalesSKUItem(sku="6400", qty=1.0, action="add"),
            ],
        ),
        # Turno 2: Synthesizer
        SalesSynthesizeResponse(
            response_text="He creado exitosamente una nueva cotización para Janet Pupuche con los 3 productos."
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

    thread_cfg = {"configurable": {"thread_id": "thread-user-bug-reproduction"}}

    # Turno 1: Ver cotización 03 de Adhara
    t1_res = await graph.ainvoke(
        {"raw_query": "muestrame la cotizacion 03 de Adhara", "user_id": 5},
        config=thread_cfg,
    )
    assert "__interrupt__" not in t1_res
    assert t1_res.get("target_order_name") == "S00003"
    assert t1_res.get("customer_name") == "Adhara"

    # Turno 2: "agrega estos productos a la cotizacion de Janet Pupuche"
    t2_res = await graph.ainvoke(
        {"raw_query": "agrega estos productos a la cotizacion de Janet Pupuche"},
        config=thread_cfg,
    )

    assert "__interrupt__" not in t2_res
    # Verificamos que NO se haya llamado a update_quotation_tool sobre S00003
    mock_update_quotation_tool.assert_not_awaited()
    # Verificamos que SÍ se haya llamado a create_quotation_tool con partner_id=13
    mock_create_quotation_tool.assert_awaited_once()
    create_call_args = mock_create_quotation_tool.await_args
    assert create_call_args.kwargs.get("partner_id") == 13
    assert len(create_call_args.kwargs.get("items", [])) == 3

    assert t2_res.get("partner_id") == 13
    assert t2_res.get("customer_name") == "Janet Pupuche"
    assert t2_res.get("target_order_name") == "S00004"


@pytest.mark.asyncio
async def test_reflection_conflicting_order_and_customer_escalates_to_hitl(
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
    """Valida el Patrón de Reflexión + HITL:
    El usuario dice: 'agrega 1 del 775 a la orden S00003 de Janet Pupuche'.
    Como S00003 en Odoo pertenece a Adhara Banda, la reflexión detecta el conflicto
    explícito entre cliente solicitado (Janet Pupuche) y titular de la orden (Adhara Banda),
    marcando is_intent_clear=False e interrumpiendo con HITL para aclaración del usuario.
    """
    mock_view_quotation_tool.return_value = {
        "id": 3,
        "name": "S00003",
        "customer_name": "Adhara Banda",
        "state": "draft",
        "amount_total": 652.63,
    }

    mock_llm.ainvoke.side_effect = [
        # 1. Extractor
        SalesExtractionResult(
            action="upsert",
            customer="Janet Pupuche",
            order_name="S00003",
            items=[SalesSKUItem(sku="775", qty=1.0, action="add")],
        ),
        # 2. Synthesizer tras cancelación o resumen
        SalesSynthesizeResponse(
            response_text="Operación cancelada conforme a tus instrucciones."
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

    thread_cfg = {"configurable": {"thread_id": "thread-reflection-hitl"}}

    res = await graph.ainvoke(
        {"raw_query": "agrega 1 del 775 a la orden S00003 de Janet Pupuche", "user_id": 5},
        config=thread_cfg,
    )

    # Debe haber pausado en interrupción HITL
    assert "__interrupt__" in res
    interrupt_info = res["__interrupt__"][0].value
    assert interrupt_info["type"] == "intent_clarification"
    assert "S00003" in interrupt_info["question"]
    assert "Adhara Banda" in interrupt_info["question"]
    assert "Janet Pupuche" in interrupt_info["question"]

    # Ni update ni create deben haberse llamado
    mock_update_quotation_tool.assert_not_awaited()
    mock_create_quotation_tool.assert_not_awaited()

    # Usuario cancela la operación
    res_cancel = await graph.ainvoke(
        Command(resume="cancelar"),
        config=thread_cfg,
    )
    assert "__interrupt__" not in res_cancel
    mock_update_quotation_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_quotation_node_guardrail_blocks_foreign_customer():
    """Valida la defensa en profundidad en update_quotation_node:
    Si por cualquier motivo una orden S00003 (titular: Adhara Banda) llega con customer_name='Janet Pupuche',
    el nodo aborta la operación de forma segura y no invoca la herramienta de Odoo.
    """
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_update_quotation_tool = AsyncMock()
    mock_view_quotation_tool = AsyncMock(
        return_value={
            "id": 3,
            "name": "S00003",
            "customer_name": "Adhara Banda",
            "state": "draft",
        }
    )

    nodes = SalesManageNodes(
        llm=mock_llm,
        update_quotation_tool=mock_update_quotation_tool,
        view_quotation_tool=mock_view_quotation_tool,
    )

    state = {
        "raw_query": "agrega 1 del 775",
        "sales_action": "add_items",
        "target_order_name": "S00003",
        "customer_name": "Janet Pupuche",
        "items": [{"sku": "775", "qty": 1.0}],
    }

    result = await nodes.update_quotation_node(state)

    assert result["operation_result"]["success"] is False
    assert "Adhara Banda" in result["operation_result"]["error"]
    assert "Janet Pupuche" in result["operation_result"]["error"]
    mock_update_quotation_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflection_customer_pipeline_relaxes_status_filter_to_include_draft(
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
    """Caso 1 del usuario:
    Al preguntar 'cómo van los Pedidos de Janet Pupuche', aunque el extractor marque status='sale',
    la reflexión evalúa que el vendedor desea ver el pipeline completo del cliente y rectifica
    status_filter=None, permitiendo que se recupere la cotización S00004 en borrador.
    """
    mock_list_orders_tool.return_value = {
        "draft": [
            {
                "id": 4,
                "name": "S00004",
                "customer_name": "Janet Pupuche",
                "amount_total": 163.30,
                "date_order": "2026-09-19",
                "lines": [{"product_name": "Amore Pink", "quantity": 2.0, "price_unit": 71.0}],
            }
        ],
        "sale": [],
        "cancel": [],
    }

    mock_llm.ainvoke.side_effect = [
        # Extractor (marca sale por la palabra 'Pedidos')
        SalesExtractionResult(action="list", customer="Janet Pupuche", status="sale"),
        # Synthesizer
        SalesSynthesizeResponse(
            response_text="Janet Pupuche tiene 1 cotización en borrador: S00004 por un total de S/ 163.30 con 2 unidades de Amore Pink."
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

    thread_cfg = {"configurable": {"thread_id": "thread-pipeline-relax"}}

    res = await graph.ainvoke(
        {"raw_query": "cómo van los Pedidos de Janet Pupuche", "user_id": 5},
        config=thread_cfg,
    )

    assert "__interrupt__" not in res
    # La herramienta debe haberse llamado con status=None gracias a la reflexión
    call_kwargs = mock_list_orders_tool.await_args.kwargs
    assert call_kwargs.get("status") is None
    assert call_kwargs.get("customer_name") == "Janet Pupuche"
    assert "S00004" in res.get("final_response", "")
    assert "Operación comercial procesada" not in res.get("final_response", "")


@pytest.mark.asyncio
async def test_reflection_global_query_purges_previous_customer_and_order(
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
    """Caso 2 del usuario:
    Turno 1: Se confirma o visualiza S00004 de Janet Pupuche.
    Turno 2: El usuario dice 'dame todos mis pedidos'.
    La reflexión detecta que es una consulta global del vendedor ('mis pedidos') y purga
    customer_name y target_order_name para consultar todas las órdenes en Odoo.
    """
    mock_list_orders_tool.return_value = {
        "draft": [
            {"id": 4, "name": "S00004", "customer_name": "Janet Pupuche", "amount_total": 163.30},
            {"id": 3, "name": "S00003", "customer_name": "Adhara Banda", "amount_total": 431.25},
        ],
        "sale": [],
        "cancel": [],
    }

    mock_llm.ainvoke.side_effect = [
        # Turno 1: Extractor (confirm S00004)
        SalesExtractionResult(action="confirm", customer="Janet Pupuche", order_name="S00004"),
        # Turno 1: Synthesizer
        SalesSynthesizeResponse(response_text="¡Excelente noticia! El pedido S00004 de Janet Pupuche ha sido confirmado."),
        # Turno 2: Extractor (list 'dame todos mis pedidos' con customer=None)
        SalesExtractionResult(action="list", customer=None),
        # Turno 2: Synthesizer
        SalesSynthesizeResponse(
            response_text="Aquí tienes tus pedidos: S00004 de Janet Pupuche y S00003 de Adhara Banda."
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

    thread_cfg = {"configurable": {"thread_id": "thread-global-list-purge"}}

    # Turno 1: Confirmar S00004 de Janet Pupuche
    await graph.ainvoke(
        {"raw_query": "confirma la cotización S00004 de Janet Pupuche", "user_id": 5},
        config=thread_cfg,
    )

    # Turno 2: "dame todos mis pedidos"
    t2_res = await graph.ainvoke(
        {"raw_query": "dame todos mis pedidos"},
        config=thread_cfg,
    )

    assert "__interrupt__" not in t2_res
    # Verificamos que list_orders_tool se haya llamado SIN customer_name (búsqueda global de user_id=5)
    call_kwargs = mock_list_orders_tool.await_args.kwargs
    assert call_kwargs.get("customer_name") is None
    assert call_kwargs.get("user_id") == 5
    assert t2_res.get("customer_name") is None
    assert "Operación comercial procesada en Odoo para Janet Pupuche (S00004)" not in t2_res.get("final_response", "")


@pytest.mark.asyncio
async def test_synthesize_sales_response_fallback_formats_list_deterministically():
    """Valida que si _synthesizer.ainvoke falla, el fallback determinista formatee
    correctamente las órdenes y nunca emita 'Operación comercial procesada en Odoo'.
    """
    mock_llm = MagicMock(spec=BaseChatModel)
    # Mock LLM que falla para forzar el fallback determinista
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("Timeout simulado"))
    mock_llm.bind = MagicMock(return_value=mock_llm)
    mock_llm.with_structured_output = MagicMock(return_value=mock_llm)

    nodes = SalesManageNodes(llm=mock_llm)

    state = {
        "raw_query": "dame mis pedidos",
        "sales_action": "list",
        "customer_name": None,
        "target_order_name": None,
        "orders_list": {
            "draft": [
                {
                    "name": "S00004",
                    "customer_name": "Janet Pupuche",
                    "amount_total": 163.30,
                    "lines": [{"product_name": "Amore Pink", "quantity": 2.0}],
                }
            ],
            "sale": [
                {
                    "name": "S00003",
                    "customer_name": "Adhara Banda",
                    "amount_total": 431.25,
                }
            ],
            "cancel": [],
        },
    }

    res = await nodes.synthesize_sales_response(state)
    final_text = res.get("final_response", "")

    assert "Operación comercial procesada" not in final_text
    assert "Cotizaciones en Borrador" in final_text
    assert "S00004" in final_text
    assert "Janet Pupuche" in final_text
    assert "163.30" in final_text
    assert "Pedidos Confirmados" in final_text
    assert "S00003" in final_text
    assert "Adhara Banda" in final_text
