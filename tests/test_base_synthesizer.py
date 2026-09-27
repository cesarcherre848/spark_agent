"""
tests/test_base_synthesizer.py - Pruebas unitarias para BaseSynthesizerNode, CurrencyFormatter y sensibilidad de canal.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.agent_service.core.formatters.currency import (
    CURRENCY_SYMBOLS,
    get_currency_symbol,
    format_currency,
    normalize_currencies_in_text,
)
from src.agent_service.graph.base_synthesizer import BaseSynthesizerNode
from src.agent_service.soul import SoulRole
from src.agent_service.graph.sub_graphs.sales_manage.nodes import SalesManageNodes
from src.agent_service.graph.sub_graphs.product_resolver.nodes import ProductResolverNodes
from src.agent_service.graph.sub_graphs.contact_manage.nodes import ContactManageNodes
from src.agent_service.graph.sub_graphs.product_advisor.nodes import ProductAdvisorNodes


# ==============================================================================
# 1. PRUEBAS DE FORMATEADOR Y NORMALIZADOR DE MONEDAS
# ==============================================================================
def test_currency_symbol_lookup():
    assert get_currency_symbol("PEN") == "S/."
    assert get_currency_symbol("sol") == "S/."
    assert get_currency_symbol("SOLES") == "S/."
    assert get_currency_symbol("USD") == "$"
    assert get_currency_symbol("EUR") == "€"
    assert get_currency_symbol(None) == "S/."
    assert get_currency_symbol("XYZ") == "XYZ"


def test_format_currency_values():
    assert format_currency(35, "PEN") == "S/. 35.00"
    assert format_currency(1250.5, "PEN") == "S/. 1,250.50"
    assert format_currency("45.8", "PEN") == "S/. 45.80"
    assert format_currency(100, "USD") == "$ 100.00"
    assert format_currency(50, "EUR") == "€ 50.00"
    assert format_currency(None, "PEN") == "S/. 0.00"


def test_normalize_currencies_in_text():
    # 1. 35.00 PEN -> S/. 35.00
    t1 = "El precio unitario es 35.00 PEN por unidad."
    assert "S/. 35.00" in normalize_currencies_in_text(t1)
    assert "PEN" not in normalize_currencies_in_text(t1)

    # 2. $35.00 (PEN) -> S/. 35.00
    t2 = "- [6189] Delineador: $35.00 (PEN)"
    assert normalize_currencies_in_text(t2) == "- [6189] Delineador: S/. 35.00"

    # 3. PEN 120 -> S/. 120.00
    t3 = "Total a pagar: PEN 120"
    assert normalize_currencies_in_text(t3) == "Total a pagar: S/. 120.00"

    # 4. S/ 45.00 -> S/. 45.00
    t4 = "El total es S/ 45.00"
    assert normalize_currencies_in_text(t4) == "El total es S/. 45.00"

    # 5. S/. 45.00 (ya correcto) no debe mutar a S/.. 45.00
    t5 = "El total ya formateado es S/. 45.00 y debe mantenerse intacto."
    assert normalize_currencies_in_text(t5) == t5


# ==============================================================================
# 2. PRUEBAS DE BaseSynthesizerNode: CANAL Y POST-PROCESAMIENTO
# ==============================================================================
def test_base_synthesizer_channel_detection():
    node = BaseSynthesizerNode()

    assert node.get_channel({"channel": "whatsapp"}) == "whatsapp"
    assert node.get_channel({"channel": "wa"}) == "whatsapp"
    assert node.get_channel({"channel": "web"}) == "web"
    assert node.get_channel({"channel": "webchat"}) == "web"
    assert node.get_channel({"channel": "other"}) == "other"
    assert node.get_channel({}) == "whatsapp"  # default seguro
    assert node.get_channel(None) == "whatsapp"


def test_base_synthesizer_prompt_instructions():
    wa_prompt = BaseSynthesizerNode.get_channel_prompt_instructions("whatsapp")
    assert "WHATSAPP" in wa_prompt
    assert "tablas Markdown" in wa_prompt
    assert "S/." in wa_prompt

    web_prompt = BaseSynthesizerNode.get_channel_prompt_instructions("web")
    assert "WEB" in web_prompt
    assert "tablas Markdown" in web_prompt


def test_base_synthesizer_post_processing_whitelabel_and_currency():
    node = BaseSynthesizerNode()
    raw = "En Odoo ERP el precio es 50.00 PEN."
    processed = node.post_process_response(raw, channel="whatsapp")

    assert "Odoo" not in processed
    assert "ERP" not in processed
    assert "S/. 50.00" in processed


def test_base_synthesizer_table_adaptation_whatsapp():
    node = BaseSynthesizerNode()
    table_text = (
        "| Producto | Cantidad | Precio |\n"
        "| --- | --- | --- |\n"
        "| Crema Bio Milk | 2 | 35.00 PEN |\n"
        "| Delineador Tattoo | 1 | 25.00 PEN |"
    )
    adapted = node.post_process_response(table_text, channel="whatsapp")
    # No deben quedar delimitadores de tabla en formato crudo
    assert "| --- |" not in adapted
    assert "*Producto*: Crema Bio Milk" in adapted
    assert "S/. 35.00" in adapted


def test_base_synthesizer_whatsapp_formatting_and_inline_bullet_separation():
    node = BaseSynthesizerNode()
    raw = (
        "¡Excelente elección! Aquí tienes las opciones: *En Ésika:* "
        "- **[01426] You Live** (S/. 107.00) - Pág. 60 "
        "- **[08218] You Good Vibes** (S/. 107.00) - Pág. 60. "
        "Para complementar tu rutina, también contamos con el **[392] Desodorante** (S/. 28.00). "
        "¿Te gustaría que preparemos la cotización formal de alguno de estos productos? 💼"
    )
    processed = node.post_process_response(raw, channel="whatsapp")

    # 1. Separación de subtítulo de marca
    assert "*En Ésika:*" in processed
    # 2. Separación de viñetas en líneas independientes
    assert "\n- *[01426] You Live*" in processed
    assert "\n- *[08218] You Good Vibes*" in processed
    # 3. Conversión de doble asterisco a simple (WhatsApp bold)
    assert "**" not in processed
    assert "*[01426] You Live*" in processed
    # 4. Separación de nota de transición y pregunta final
    assert "\n\nPara complementar tu rutina" in processed
    assert "\n\n¿Te gustaría que preparemos" in processed


def test_base_synthesizer_web_preserves_markdown_bold():
    node = BaseSynthesizerNode()
    raw = "- **[01426] You Live** (35.00 PEN)"
    processed = node.post_process_response(raw, channel="web")
    # En canal web se debe preservar el doble asterisco para negrita Markdown
    assert "**[01426] You Live**" in processed
    assert "S/. 35.00" in processed


def test_format_final_response():
    node = BaseSynthesizerNode()
    result = node.format_final_response(
        raw_text="Total: 100.00 PEN",
        state={"channel": "whatsapp"},
        extra={"custom_flag": True},
    )

    assert result["final_response"] == "Total: S/. 100.00"
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Total: S/. 100.00"
    assert result["custom_flag"] is True


# ==============================================================================
# 3. PRUEBAS DE HERENCIA EN LOS SUBGRAFOS
# ==============================================================================
def test_all_subgraph_nodes_inherit_from_base_synthesizer():
    assert issubclass(SalesManageNodes, BaseSynthesizerNode)
    assert issubclass(ProductResolverNodes, BaseSynthesizerNode)
    assert issubclass(ContactManageNodes, BaseSynthesizerNode)
    assert issubclass(ProductAdvisorNodes, BaseSynthesizerNode)


def test_base_synthesizer_build_system_prompt_contract():
    prompt = BaseSynthesizerNode.build_synthesizer_system_prompt(
        task_specific_rules="Presenta las mejores opciones de labiales.",
        role=SoulRole.RECOMMENDER,
        state={"channel": "whatsapp"},
        include_multi_vendor=True,
    )

    # 1. Identidad de MIA y SOUL
    assert "MIA" in prompt
    assert "ESTRUCTURA Y AGRUPACIÓN DE PRODUCTOS" in prompt
    # 2. Formato estándar de viñetas con [SKU] y S/.
    assert "*[SKU] Nombre Comercial*" in prompt
    assert "S/." in prompt
    # 3. Directrices de canal para WhatsApp
    assert "DIRECTRICES OBLIGATORIAS DE FORMATO PARA WHATSAPP" in prompt
    # 4. Agrupación por marca cuando hay 2 o más marcas
    assert "AGRÚPALAS de forma limpia y ordenada por cada marca comercial" in prompt
    # 5. Continuidad multi-turno
    assert "NO repitas saludos de bienvenida" in prompt
    # 6. Whitelabel y neutralidad comercial sin sesgo de marcas hardcodeadas
    assert "100% Whitelabel" in prompt
    assert "NEUTRALIDAD COMERCIAL" in prompt
    assert "Ésika" not in prompt
    assert "Yanbal" not in prompt


def test_base_synthesizer_format_products_context():
    prods = [
        Document(
            page_content="Labial de larga duración acabado mate.",
            metadata={
                "name": "Hydra-Lip Líquido",
                "sku": "761",
                "marca": "Yanbal",
                "edicion": "C10",
                "pagina": 15,
                "price": 45.0,
                "currency": "PEN",
            },
        ),
        {
            "name": "Colorfix 24H",
            "sku": "9812",
            "marca": "Ésika",
            "edicion": "C15",
            "price": 32.5,
            "description": "Labial indeleble a prueba de agua.",
        },
    ]

    formatted = BaseSynthesizerNode.format_products_context(prods)
    assert "[1] [SKU: 761] Hydra-Lip Líquido" in formatted
    assert "Marca: Yanbal" in formatted
    assert "Campaña C10" in formatted
    assert "Pág. 15" in formatted
    assert "Precio: S/. 45.00" in formatted

    assert "[2] [SKU: 9812] Colorfix 24H" in formatted
    assert "Marca: Ésika" in formatted
    assert "Precio: S/. 32.50" in formatted



@pytest.mark.asyncio
async def test_sales_manage_synthesize_currency_standardization():
    mock_llm = MagicMock()
    # Mock synthesizer to trigger fallback logic
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM failure for testing deterministic fallback"))

    nodes = SalesManageNodes(llm=mock_llm)
    # Simulate a list response with draft orders
    state = {
        "raw_query": "mis pedidos",
        "sales_action": "list",
        "channel": "whatsapp",
        "customer_name": "Carlos Gomez",
        "orders_list": {
            "draft": [
                {"name": "SO001", "customer_name": "Carlos Gomez", "amount_total": 150.0, "lines": []}
            ],
            "sale": [],
            "cancel": [],
        },
    }

    res = await nodes.synthesize_sales_response(state)
    assert "S/. 150.00" in res["final_response"]
    assert "PEN" not in res["final_response"]
    assert "Odoo" not in res["final_response"]


@pytest.mark.asyncio
async def test_product_resolver_synthesize_currency_formatting():
    mock_llm = MagicMock()
    nodes = ProductResolverNodes(llm=mock_llm)
    # Test format_currency helper available on self
    assert nodes.format_currency(35.5, "PEN") == "S/. 35.50"
    assert nodes.get_channel({"channel": "whatsapp"}) == "whatsapp"


@pytest.mark.asyncio
async def test_product_advisor_synthesize_currency_formatting():
    mock_llm = MagicMock()
    nodes = ProductAdvisorNodes(llm=mock_llm)
    # Test currency formatting helper
    assert nodes.format_currency(29.9, "PEN") == "S/. 29.90"
    assert nodes.get_channel({"channel": "web"}) == "web"
