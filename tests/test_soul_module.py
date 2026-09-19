"""
tests/test_soul_module.py - Pruebas Unitarias para el Módulo Global SOUL (MIA)
"""

import pytest
from src.agent_service.soul import (
    SoulPersona,
    SoulToneAndStyle,
    SoulSubtlety,
    SoulRole,
    get_soul_prompt,
    inject_soul,
)


def test_soul_persona_attributes():
    """Verifica que la persona del agente contenga su identidad como MIA y principios."""
    summary = SoulPersona.get_summary()
    assert summary["name"] == "MIA"
    assert summary["role"] == "Mi Asistente IA para gestión de pedidos multimarca"
    assert "gestión de pedidos multimarca" in summary["mission"]

    principles = SoulPersona.INTERACTION_PRINCIPLES
    assert "MIA" in principles
    assert "Mi Asistente IA para gestión de pedidos multimarca" in principles
    assert "Escucha Activa" in principles
    assert "Continuidad Conversacional" in principles
    assert "saludos de bienvenida" in principles
    assert "Manejo Elegante de Ambigüedades" in principles


def test_soul_forbids_erp_and_odoo():
    """Verifica la regla estricta de cero menciones a ERP, Odoo o backend."""
    principles = SoulPersona.INTERACTION_PRINCIPLES
    assert "ESTRICTAMENTE PROHIBIDO mencionar términos técnicos o de backend como 'ERP', 'Odoo'" in principles

    writing = SoulToneAndStyle.WRITING_STYLE_GUIDELINES
    assert "NUNCA menciones nombres de tecnologías backend como 'ERP', 'Odoo'" in writing


def test_soul_tone_and_style():
    """Verifica directrices de tono y estilo de redacción."""
    tone = SoulToneAndStyle.TONE_GUIDELINES
    assert "Profesional, Cercano y Colaborativo" in tone
    assert "Asertivo y con Autoridad Comercial" in tone
    assert "Celebración Ejecutiva" in tone

    writing = SoulToneAndStyle.WRITING_STYLE_GUIDELINES
    assert "Brevity with Substance" in writing
    assert "Markdown" in writing
    assert "user_id" in writing  # Mencionado en la regla de prohibición
    assert "partner_id" in writing
    assert "Privacidad" in writing


def test_soul_subtlety():
    """Verifica directrices de persuasión consultiva y sutileza."""
    subtlety = SoulSubtlety.SUBTLETY_GUIDELINES
    assert "Venta Consultiva vs. Venta Agresiva" in subtlety
    assert "Invitaciones de Valor" in subtlety
    assert "Cross-Selling y Up-Selling Orgánico" in subtlety
    assert "Sensibilidad de Presupuesto" in subtlety
    assert "Diplomacia en Alertas de Riesgo" in subtlety


def test_get_soul_prompt_composition():
    """Verifica que get_soul_prompt ensamble los 4 pilares fundamentales y la identidad de MIA."""
    prompt = get_soul_prompt()
    assert "MIA" in prompt
    assert "Mi Asistente IA para gestión de pedidos multimarca" in prompt
    assert "IDENTIDAD E INTERACCIÓN" in prompt
    assert "TONO Y ESTILO" in prompt
    assert "FORMATO Y PRIVACIDAD" in prompt
    assert "SUTILEZA Y VENTA CONSULTIVA" in prompt


@pytest.mark.parametrize(
    "role, expected_keyword",
    [
        (SoulRole.GENERAL, "Orientadora y Asistente Comercial General"),
        (SoulRole.CATALOG_RAG, "Consultora Experta en Catálogo"),
        (SoulRole.RECOMMENDER, "Estratega Consultiva de Recomendación"),
        (SoulRole.QUOTATION, "Asesora de Consolidación de Cotizaciones"),
        (SoulRole.SALES_ORDERS, "Copiloto Ejecutiva de Gestión de Ventas y Pedidos Multimarca"),
        (SoulRole.CONTACTS, "Gestora Diplomática de Cartera Comercial"),
        ("general", "Orientadora y Asistente Comercial General"),
    ],
)
def test_get_soul_prompt_roles(role, expected_keyword):
    """Verifica que cada rol especializado incorpore su contexto particular con identidad femenina MIA."""
    prompt = get_soul_prompt(role=role)
    assert expected_keyword in prompt
    assert "MIA" in prompt


def test_inject_soul_basic():
    """Verifica la inyección limpia de SOUL en directrices de tarea."""
    task_instructions = """
    Tareas específicas:
    - Listar los productos recuperados con SKU.
    - Explicar precios en moneda oficial.
    """
    injected = inject_soul(task_instructions, role=SoulRole.CATALOG_RAG)

    # Debe contener el núcleo de SOUL
    assert "MIA" in injected
    assert "Consultora Experta en Catálogo" in injected
    assert "SUTILEZA Y VENTA CONSULTIVA" in injected

    # Debe contener las directrices de la tarea
    assert "TAREA ESPECÍFICA:" in injected
    assert "Listar los productos recuperados con SKU" in injected


def test_inject_soul_with_extra_context():
    """Verifica la inyección con bloque de contexto complementario."""
    task_instructions = "Confirmar la orden de venta."
    extra_context = "Orden objetivo: SO001 para cliente Acme Corp."

    injected = inject_soul(
        task_instructions,
        role=SoulRole.SALES_ORDERS,
        extra_context=extra_context,
    )

    assert "Copiloto Ejecutiva de Gestión de Ventas y Pedidos Multimarca" in injected
    assert "Confirmar la orden de venta." in injected
    assert "CONTEXTO COMPLEMENTARIO:" in injected
    assert "Orden objetivo: SO001 para cliente Acme Corp." in injected
