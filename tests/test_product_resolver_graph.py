import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.agent_service.graph.sub_graphs.product_resolver.graph import build_product_resolver_graph
from src.agent_service.graph.sub_graphs.product_resolver.schemas import (
    ExtractionResult,
    ExtractedSKUItem,
    SynthesizeResponse,
)


@pytest.fixture
def mock_product_tool():
    tool = MagicMock()
    tool.ainvoke = AsyncMock()
    tool.ainvoke.return_value = {
        "products": [
            {
                "id": 257,
                "sku": "1",
                "name": "Loción de Seda",
                "price": 210.0,
                "currency": "PEN",
                "uom": "Units",
                "sales_description": "Loción hidratante",
            },
            {
                "id": 228,
                "sku": "11",
                "name": "Desmaquillador Doble Fase",
                "price": 39.5,
                "currency": "PEN",
                "uom": "Units",
                "sales_description": "Desmaquillador bifásico",
            },
        ],
        "not_found_skus": [],
    }
    return tool


@pytest.fixture
def mock_supplier_tool():
    tool = MagicMock()
    tool.ainvoke = AsyncMock()
    tool.ainvoke.return_value = {
        "suppliers": [
            {"id": 101, "name": "Siderperu S.A.", "display_name": "Siderperu S.A.", "vat": "20100102413"},
            {"id": 202, "name": "Aceros Arequipa S.A.", "display_name": "Aceros Arequipa S.A.", "vat": "20100102414"},
        ],
        "query": "",
    }
    return tool


@pytest.fixture
def mock_ownership_tool():
    tool = MagicMock()
    tool.ainvoke = AsyncMock()
    tool.ainvoke.return_value = {
        "user_id": 5,
        "results": {
            "1": {
                "sku": "1",
                "product_id": 257,
                "is_owner": True,
                "candidate_partners": [
                    {"partner_id": "101", "vendor_id": 101, "vendor_name": "Siderperu S.A."}
                ],
                "has_conflict": False,
            },
            "11": {
                "sku": "11",
                "product_id": 228,
                "is_owner": True,
                "candidate_partners": [
                    {"partner_id": "202", "vendor_id": 202, "vendor_name": "Aceros Arequipa S.A."}
                ],
                "has_conflict": False,
            },
        },
    }
    return tool


@pytest.mark.asyncio
async def test_product_resolver_happy_path(mock_product_tool, mock_ownership_tool, mock_supplier_tool):
    """Test 1: Flujo feliz directo sin ambigüedades.
    input -> extract -> check_partners (sin conflictos) -> tool -> group -> synthesize -> END
    """
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = ExtractionResult(
        items=[
            ExtractedSKUItem(
                sku="1",
                attributes={"cantidad": 10},
                partner_name="Siderperu",
            )
        ],
        is_complete=True,
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = SynthesizeResponse(
        response_text="Cotización para Siderperu: 10 unidades de Loción de Seda por un total de 2100.00 PEN."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == ExtractionResult:
            return mock_extractor
        elif schema == SynthesizeResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_product_resolver_graph(
        llm=mock_llm,
        product_tool=mock_product_tool,
        ownership_tool=mock_ownership_tool,
        supplier_tool=mock_supplier_tool,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-happy-path"}}
    initial_state = {
        "raw_query": "Quiero 10 unidades del SKU 1 de Siderperu",
        "user_id": "5",
    }

    result = await app.ainvoke(initial_state, config=config)

    # Verificaciones
    assert result["is_extraction_complete"] is True
    assert result["has_partner_conflicts"] is False
    assert "101" in result["grouped_products"]
    
    sider_prods = result["grouped_products"]["101"]
    assert len(sider_prods) == 1
    assert sider_prods[0]["sku"] == "1"
    assert sider_prods[0]["requested_qty"] == 10.0
    assert sider_prods[0]["subtotal"] == 2100.0
    assert "2100.00 PEN" in result["final_response"]

    mock_extractor.ainvoke.assert_awaited_once()
    mock_ownership_tool.ainvoke.assert_awaited_once()
    mock_product_tool.ainvoke.assert_awaited_once()
    mock_synthesizer.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_product_resolver_hitl_missing_info_interrupt_and_resume(mock_product_tool, mock_ownership_tool, mock_supplier_tool):
    """Test 2: Human-in-the-Loop en el Bucle 1 (Falta de datos).
    1. Consulta inicial ambigua -> is_complete=False -> interrupt() en feedback_ask_missing.
    2. Usuario responde con la información -> Command(resume=...) -> avanza y concluye.
    """
    mock_extractor = AsyncMock()
    # Primera vuelta falla (incompleto); segunda vuelta tiene éxito tras la aclaración del usuario
    mock_extractor.ainvoke.side_effect = [
        ExtractionResult(items=[], is_complete=False, missing_reason="No especificó códigos SKU"),
        ExtractionResult(
            items=[ExtractedSKUItem(sku="1", attributes={"cantidad": 2}, partner_name="Siderperu")],
            is_complete=True,
        ),
    ]

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = SynthesizeResponse(
        response_text="Pedido confirmado para 2 unidades de Loción de Seda."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == ExtractionResult:
            return mock_extractor
        elif schema == SynthesizeResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_product_resolver_graph(
        llm=mock_llm,
        product_tool=mock_product_tool,
        ownership_tool=mock_ownership_tool,
        supplier_tool=mock_supplier_tool,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-hitl-missing"}}
    
    # 1. Ejecutar consulta incompleta
    await app.ainvoke({"raw_query": "hola, quiero comprar", "user_id": "5"}, config=config)

    # Debe haberse pausado por HITL (interrupt)
    state = await app.aget_state(config)
    assert len(state.tasks) > 0
    assert state.tasks[0].interrupts
    interrupt_info = state.tasks[0].interrupts[0].value
    assert interrupt_info["type"] == "missing_info"

    # 2. Reanudar enviando la aclaración del usuario
    resumed_run = await app.ainvoke(Command(resume="Quiero 2 unidades del SKU 1"), config=config)

    # Ahora el flujo culmina exitosamente
    assert resumed_run["is_extraction_complete"] is True
    assert "Loción de Seda" in resumed_run["final_response"]
    assert mock_extractor.ainvoke.await_count == 2
    mock_product_tool.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_product_resolver_hitl_unauthorized_sku_interrupt_and_resume(mock_product_tool, mock_supplier_tool):
    """Test 3: Human-in-the-Loop cuando un SKU no está autorizado en view_user_authorized_products."""
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.side_effect = [
        # Vuelta 1: usuario pide SKU 99999 (no autorizado)
        ExtractionResult(
            items=[ExtractedSKUItem(sku="99999", attributes={"cantidad": 1}, partner_name=None)],
            is_complete=True,
        ),
        # Vuelta 2: usuario corrige a SKU 1 (autorizado)
        ExtractionResult(
            items=[ExtractedSKUItem(sku="1", attributes={"cantidad": 1}, partner_name="Siderperu")],
            is_complete=True,
        ),
    ]

    mock_ownership_tool = MagicMock()
    mock_ownership_tool.ainvoke = AsyncMock()
    mock_ownership_tool.ainvoke.side_effect = [
        # Vuelta 1: no autorizado
        {
            "user_id": 5,
            "results": {
                "99999": {"sku": "99999", "is_owner": False, "candidate_partners": []}
            },
        },
        # Vuelta 2: autorizado
        {
            "user_id": 5,
            "results": {
                "1": {
                    "sku": "1",
                    "product_id": 257,
                    "is_owner": True,
                    "candidate_partners": [{"partner_id": "101", "vendor_id": 101, "vendor_name": "Siderperu"}],
                }
            },
        },
    ]

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = SynthesizeResponse(
        response_text="Cotización completada para SKU 1."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == ExtractionResult:
            return mock_extractor
        elif schema == SynthesizeResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_product_resolver_graph(
        llm=mock_llm,
        product_tool=mock_product_tool,
        ownership_tool=mock_ownership_tool,
        supplier_tool=mock_supplier_tool,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-hitl-unauthorized"}}
    await app.ainvoke({"raw_query": "Quiero 1 del SKU 99999", "user_id": "5"}, config=config)

    # Se pausa con mensaje de SKU no autorizado
    state = await app.aget_state(config)
    assert state.tasks[0].interrupts
    interrupt_info = state.tasks[0].interrupts[0].value
    assert interrupt_info["type"] == "unauthorized_or_missing"
    assert "99999" in interrupt_info["unauthorized_skus"]

    # Reanuda con SKU válido
    resumed = await app.ainvoke(Command(resume="Cámbialo por el SKU 1 de Siderperu"), config=config)
    assert resumed["is_extraction_complete"] is True
    assert "101" in resumed["grouped_products"]


@pytest.mark.asyncio
async def test_product_resolver_hitl_partner_conflicts_interrupt_and_resume(mock_product_tool):
    """Test 4: Human-in-the-Loop en el Bucle 2 (Conflicto de Partners).
    1. SKU con 2 proveedores candidatos y sin partner asignado -> has_partner_conflicts=True -> interrupt() en feedback_clarify_partners.
    2. Usuario selecciona proveedor -> Command(resume='p_siderperu') -> asigna partner y completa.
    """
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = ExtractionResult(
        items=[ExtractedSKUItem(sku="1", attributes={"cantidad": 5}, partner_name=None)],
        is_complete=True,
    )

    # Resolver simulado que inyecta conflicto intencional (2 proveedores disponibles)
    async def conflicting_partner_resolver(sku, user_id, mentioned_partner):
        return True, None, ["p_siderperu", "p_aceros"], {"p_siderperu": "Siderperu", "p_aceros": "Aceros Arequipa"}

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = SynthesizeResponse(
        response_text="Cotización procesada con el proveedor elegido Siderperu."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == ExtractionResult:
            return mock_extractor
        elif schema == SynthesizeResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_product_resolver_graph(
        llm=mock_llm,
        product_tool=mock_product_tool,
        partner_resolver=conflicting_partner_resolver,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-hitl-conflict"}}

    # 1. Ejecución inicial -> detecta conflicto y se pausa
    await app.ainvoke({"raw_query": "Quiero 5 unidades del SKU 1"}, config=config)

    state = await app.aget_state(config)
    assert len(state.tasks) > 0
    assert state.tasks[0].interrupts
    interrupt_info = state.tasks[0].interrupts[0].value
    assert interrupt_info["type"] == "partner_conflict"
    assert interrupt_info["sku"] == "1"
    assert "p_siderperu" in interrupt_info["options"]

    # 2. El usuario elige 'p_siderperu'
    resumed_run = await app.ainvoke(Command(resume="p_siderperu"), config=config)

    # Ahora el flujo continúa sin conflictos con el partner asignado
    assert resumed_run["has_partner_conflicts"] is False
    assert "p_siderperu" in resumed_run["grouped_products"]
    assert resumed_run["items"]["1"]["partner_id"] == "p_siderperu"


@pytest.mark.asyncio
async def test_product_resolver_group_by_multiple_partners(mock_product_tool, mock_ownership_tool, mock_supplier_tool):
    """Test 5: Verifica que productos con distintos partners se agrupen por separado."""
    mock_extractor = AsyncMock()
    mock_extractor.ainvoke.return_value = ExtractionResult(
        items=[
            ExtractedSKUItem(sku="1", attributes={"cantidad": 2}, partner_name="Siderperu"),
            ExtractedSKUItem(sku="11", attributes={"cantidad": 3}, partner_name="Aceros Arequipa"),
        ],
        is_complete=True,
    )

    mock_synthesizer = AsyncMock()
    mock_synthesizer.ainvoke.return_value = SynthesizeResponse(
        response_text="Cotización consolidada para Siderperu y Aceros Arequipa."
    )

    mock_llm = MagicMock(spec=BaseChatModel)
    def side_effect(schema):
        if schema == ExtractionResult:
            return mock_extractor
        elif schema == SynthesizeResponse:
            return mock_synthesizer
        return AsyncMock()

    mock_llm.with_structured_output.side_effect = side_effect

    app = build_product_resolver_graph(
        llm=mock_llm,
        product_tool=mock_product_tool,
        ownership_tool=mock_ownership_tool,
        supplier_tool=mock_supplier_tool,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-multi-partner"}}
    result = await app.ainvoke(
        {"raw_query": "2 de SKU 1 con Siderperu y 3 de SKU 11 con Aceros Arequipa", "user_id": "5"},
        config=config,
    )

    grouped = result["grouped_products"]
    assert "101" in grouped
    assert "202" in grouped

    sider_item = grouped["101"][0]
    assert sider_item["sku"] == "1"
    assert sider_item["requested_qty"] == 2.0
    assert sider_item["subtotal"] == 420.0  # 210.0 * 2

    aceros_item = grouped["202"][0]
    assert aceros_item["sku"] == "11"
    assert aceros_item["requested_qty"] == 3.0
    assert aceros_item["subtotal"] == 118.5  # 39.5 * 3
