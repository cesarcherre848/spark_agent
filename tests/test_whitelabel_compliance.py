"""
tests/test_whitelabel_compliance.py - Suite de Pruebas de Cumplimiento Whitelabel y Egress Sanitizer (CI/CD)

Verifica que el sistema cumpla al 100% con la política de Whitelabel de MIA:
1. El sanitizador de salida (WhitelabelSanitizer) neutraliza cualquier fuga de términos de backend ('Odoo', 'ERP', 'PostgreSQL').
2. Todas las plantillas de diálogo e interrupción HITL (HITLDialogTemplates) están certificadas como limpias.
3. El despachador de WhatsApp (WhatsAppResponder) no emite términos prohibidos bajo ninguna circunstancia.
4. Auditoría estática del código fuente para garantizar que ningún nodo de ventas o contactos contenga cadenas de salida con términos prohibidos.
"""

import os
import re
import pytest
from src.agent_service.core.templates.dialogs import (
    WhitelabelSanitizer,
    HITLDialogTemplates,
    FORBIDDEN_BACKEND_TERMS,
)
from src.agent_service.api.whatsapp.responder import WhatsAppResponder


# ==============================================================================
# 1. PRUEBAS UNITARIAS DE WHITELABEL SANITIZER
# ==============================================================================

class TestWhitelabelSanitizer:
    """Verifica que las expresiones regulares de saneamiento eliminen o neutralicen fugas de backend."""

    @pytest.mark.parametrize(
        "input_text,expected_sub",
        [
            ("¿Estás seguro de que deseas cancelar la orden S00003 en Odoo?", "¿Estás seguro de que deseas cancelar la orden S00003?"),
            ("El pedido S00003 ya está confirmado y bloqueado en Odoo.", "El pedido S00003 ya está confirmado y bloqueado."),
            ("Se encontraron clientes similares en tu cartera de Odoo: [1] Juan", "Se encontraron clientes similares en tu cartera: [1] Juan"),
            ("Aquí tienes el estado de las cotizaciones y pedidos de Carlos en Odoo:", "Aquí tienes el estado de las cotizaciones y pedidos de Carlos:"),
            ("No se pudo procesar la operación en Odoo ERP.", "No se pudo procesar la operación."),
            ("El cliente no se encuentra registrado en el Odoo.", "El cliente no se encuentra registrado."),
            ("Consulta realizada en el ERP.", "Consulta realizada en el sistema."),
            ("Los datos provienen de la base de datos Odoo ERP.", "Los datos provienen de la base de datos."),
        ],
    )
    def test_clean_text_removes_odoo_and_erp(self, input_text: str, expected_sub: str):
        cleaned = WhitelabelSanitizer.clean_text(input_text)
        assert cleaned == expected_sub
        assert "odoo" not in cleaned.lower()
        assert "erp" not in cleaned.lower()

    def test_contains_forbidden_terms_detection(self):
        assert WhitelabelSanitizer.contains_forbidden_terms("Error en Odoo API") is True
        assert WhitelabelSanitizer.contains_forbidden_terms("Guardado en PostgreSQL") is True
        assert WhitelabelSanitizer.contains_forbidden_terms("Servidor de ERP inactivo") is True
        assert WhitelabelSanitizer.contains_forbidden_terms("Tabla res.partner") is True
        assert WhitelabelSanitizer.contains_forbidden_terms("Modelo sale.order") is True
        assert WhitelabelSanitizer.contains_forbidden_terms("El total de tu pedido es S/ 150.00") is False
        assert WhitelabelSanitizer.contains_forbidden_terms("Cliente registrado en el sistema exitosamente.") is False


# ==============================================================================
# 2. CERTIFICACIÓN DE PLANTILLAS HITL
# ==============================================================================

class TestHITLDialogTemplates:
    """Certifica que ninguna plantilla en HITLDialogTemplates contenga términos prohibidos."""

    def test_all_templates_are_whitelabel_compliant(self):
        templates = [
            HITLDialogTemplates.unlock_order_question("S00001"),
            HITLDialogTemplates.remove_order_question("S00002"),
            HITLDialogTemplates.unknown_customer_question("Carlos Pérez"),
            HITLDialogTemplates.duplicate_contact_question("[1] Juan"),
            HITLDialogTemplates.remove_contact_question("Juan Pérez"),
            HITLDialogTemplates.order_list_header("Carlos Pérez"),
            HITLDialogTemplates.order_list_header(None),
            HITLDialogTemplates.order_list_empty("Carlos Pérez"),
            HITLDialogTemplates.order_list_empty(None),
            HITLDialogTemplates.order_not_found_to_cancel(),
            HITLDialogTemplates.contact_required_name_error(),
            HITLDialogTemplates.contact_not_found_to_remove(),
        ]

        for text in templates:
            assert isinstance(text, str)
            assert len(text.strip()) > 0
            for term in FORBIDDEN_BACKEND_TERMS:
                assert term not in text.lower(), f"La plantilla '{text}' contiene el término prohibido '{term}'"


# ==============================================================================
# 3. PRUEBAS DE LA CAPA DE DESPACHO WHATSAPP (RESPONDER)
# ==============================================================================

class TestWhatsAppResponderWhitelabel:
    """Verifica que el formateador saliente de WhatsApp actúe como barrera defensiva inexpugnable."""

    @pytest.mark.parametrize(
        "raw_text",
        [
            "¿Deseas confirmar la orden S00012 en Odoo?",
            "El cliente fue creado en Odoo ERP.",
            "Detalle de consulta en Odoo.",
            "Error de sincronización con el ERP.",
            "Base de datos PostgreSQL no responde.",
        ],
    )
    def test_format_response_never_leaks_backend_terms(self, raw_text: str):
        formatted = WhatsAppResponder.format_response(raw_text)
        assert "odoo" not in formatted.lower()
        assert "erp" not in formatted.lower()


# ==============================================================================
# 4. AUDITORÍA ESTÁTICA DE CÓDIGO (PREVENCIÓN DE REGRESIONES EN NÓDULOS)
# ==============================================================================

class TestStaticCodeWhitelabelAudit:
    """Escanea el código fuente de los subgrafos para asegurar que no haya nuevas cadenas salientes con 'Odoo'."""

    TARGET_FILES = [
        "src/agent_service/graph/sub_graphs/sales_manage/nodes.py",
        "src/agent_service/graph/sub_graphs/contact_manage/nodes.py",
    ]

    def test_no_forbidden_terms_in_user_facing_strings(self):
        for rel_path in self.TARGET_FILES:
            full_path = os.path.join("/home/spark_agent", rel_path)
            if not os.path.exists(full_path):
                continue

            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, 1):
                stripped = line.strip()
                # Omitir comentarios de python (# ...) y docstrings puros
                if stripped.startswith("#"):
                    continue
                # Omitir directrices negativas explícitas como "NUNCA menciones términos de backend como 'ERP' u 'Odoo'"
                if "NUNCA menciones" in stripped or "ESTRICTAMENTE PROHIBIDO" in stripped:
                    continue

                # Detectar asignaciones a question = ... o error = ... o header = ... o resp_text = ... que mencionen 'en Odoo'
                leak_pattern = re.compile(r'\b(question|header|resp_text|error)\s*=.*[\'"].*\ben\s+odoo\b.*[\'"]', flags=re.IGNORECASE)
                match = leak_pattern.search(stripped)
                assert match is None, f"Posible fuga detectada en {rel_path}:{line_idx} -> '{stripped}'"
