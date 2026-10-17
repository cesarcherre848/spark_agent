import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agent_service.graph.sub_graphs.contact_manage.schemas import (
    CustomerItem,
    CustomerExtractionResult,
)
from src.agent_service.graph.sub_graphs.contact_manage.nodes import ContactManageNodes
from src.agent_service.graph.sub_graphs.contact_manage.state import ContactManageState


@pytest.mark.asyncio
async def test_batch_customer_schemas_backward_compatibility():
    """Valida que CustomerExtractionResult admita tanto el cliente individual como la lista de clientes."""
    # Caso 1: Individual
    single = CustomerExtractionResult(
        action="upsert",
        name="Valeria Osores",
        phones=["999111222"],
    )
    assert single.action == "upsert"
    assert single.name == "Valeria Osores"
    assert single.phones == ["999111222"]
    assert len(single.customers) == 0

    # Caso 2: Batch
    batch = CustomerExtractionResult(
        action="upsert",
        customers=[
            CustomerItem(name="Valeria Osores", phones=["999111222"]),
            CustomerItem(name="Francia", phones=["988333444"]),
        ],
    )
    assert batch.action == "upsert"
    assert len(batch.customers) == 2
    assert batch.customers[0].name == "Valeria Osores"
    assert batch.customers[1].name == "Francia"


@pytest.mark.asyncio
async def test_extract_customer_info_batch():
    """Valida que extract_customer_info empaquete correctamente la lista de clientes en el estado."""
    mock_llm = MagicMock()
    nodes = ContactManageNodes(llm=mock_llm)

    # Mock extractor output
    mock_extraction = CustomerExtractionResult(
        action="upsert",
        customers=[
            CustomerItem(name="Valeria Osores", phones=["999111222"]),
            CustomerItem(name="Francia", phones=["988333444"]),
        ],
    )
    nodes._extractor.ainvoke = AsyncMock(return_value=mock_extraction)

    state: ContactManageState = {
        "raw_query": "Registra a Valeria Osores al 999111222 y a Francia al 988333444",
        "messages": [HumanMessage(content="Registra a Valeria Osores al 999111222 y a Francia al 988333444")],
    }

    result = await nodes.extract_customer_info(state)

    assert result["contact_action"] == "upsert"
    assert result["extracted_name"] == "Valeria Osores"
    assert len(result["extracted_customers"]) == 2
    assert result["extracted_customers"][0]["name"] == "Valeria Osores"
    assert result["extracted_customers"][1]["name"] == "Francia"


@pytest.mark.asyncio
async def test_upsert_customer_node_batch_execution():
    """Valida que upsert_customer_node ejecute la herramienta de Odoo para todos los clientes de la lista."""
    mock_llm = MagicMock()
    mock_upsert_tool = AsyncMock()
    
    # Respuestas simuladas de Odoo para cada llamada
    mock_upsert_tool.side_effect = [
        {"success": True, "id": 101, "name": "Valeria Osores", "phone": "999111222", "action": "created"},
        {"success": True, "id": 102, "name": "Francia", "phone": "988333444", "action": "created"},
    ]

    nodes = ContactManageNodes(
        llm=mock_llm,
        upsert_customer_tool=mock_upsert_tool,
    )

    state: ContactManageState = {
        "user_id": 7,
        "contact_action": "upsert",
        "extracted_customers": [
            {"name": "Valeria Osores", "phones": ["999111222"], "contact_id": None},
            {"name": "Francia", "phones": ["988333444"], "contact_id": None},
        ],
    }

    res = await nodes.upsert_customer_node(state)

    assert mock_upsert_tool.call_count == 2
    assert len(res["batch_operation_results"]) == 2
    assert res["batch_operation_results"][0]["id"] == 101
    assert res["batch_operation_results"][1]["id"] == 102
    assert res["active_customer"] == "Francia"
    assert res["active_partner_id"] == 102


@pytest.mark.asyncio
async def test_synthesize_response_batch_summary():
    """Valida que synthesize_response reporte ambos clientes cuando batch_operation_results contiene múltiples elementos."""
    mock_llm = MagicMock()
    nodes = ContactManageNodes(llm=mock_llm)

    state: ContactManageState = {
        "contact_action": "upsert",
        "raw_query": "Registra a Valeria y a Francia",
        "batch_operation_results": [
            {"success": True, "id": 101, "name": "Valeria Osores", "phone": "999111222", "action": "created"},
            {"success": True, "id": 102, "name": "Francia", "phone": "988333444", "action": "created"},
        ],
    }

    # Mock synthesize LLM call
    mock_synth_response = MagicMock()
    mock_synth_response.response_text = "He registrado a Valeria Osores y a Francia en tu cartera comercial."
    nodes._synthesizer.ainvoke = AsyncMock(return_value=mock_synth_response)

    res = await nodes.synthesize_response(state)

    assert "final_response" in res
    assert "Valeria Osores y a Francia" in res["final_response"]
