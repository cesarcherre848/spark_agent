"""
src/agent_service/core/templates/dialogs.py - Catálogo Centralizado de Diálogos, Interrupciones HITL y Sanitizador Whitelabel

Centraliza todas las preguntas de confirmación (Human-in-the-Loop), textos deterministas y encabezados
de respuesta comercial. Garantiza el 100% de cumplimiento Whitelabel para MIA, evitando que desarrolladores
hardcodeen términos de backend ('Odoo', 'ERP', 'PostgreSQL') en nodos del grafo o controladores de API.
"""

import re
from typing import Optional, List, Tuple


# Lista canónica de términos de infraestructura y backend estrictamente prohibidos en mensajes al usuario
FORBIDDEN_BACKEND_TERMS: Tuple[str, ...] = (
    "odoo",
    "erp",
    "postgresql",
    "postgres",
    "database",
    "res.partner",
    "sale.order",
)


class WhitelabelSanitizer:
    """Utilidad de sanitización y auditoría de textos salientes para garantizar identidad de MIA."""

    _PATTERN_ODOO = re.compile(r"\b(?:en\s+el\s+|en\s+|de\s+)?Odoo(?:\s+ERP)?\b", flags=re.IGNORECASE)
    _PATTERN_ERP = re.compile(r"\bERP\b", flags=re.IGNORECASE)
    _PATTERN_SPACES = re.compile(r"[ \t]+")
    _PATTERN_ORPHAN_PUNCT = re.compile(r" +([?.!,;:])")

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Sanitiza cualquier texto saliente neutralizando fugas accidentales de backend."""
        if not text:
            return ""

        clean = str(text)
        clean = cls._PATTERN_ODOO.sub("", clean)
        clean = cls._PATTERN_ERP.sub("sistema", clean)
        clean = cls._PATTERN_SPACES.sub(" ", clean)
        clean = cls._PATTERN_ORPHAN_PUNCT.sub(r"\1", clean)
        return clean.strip()

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Alias de clean_text para consistencia semántica."""
        return cls.clean_text(text)

    @classmethod
    def contains_forbidden_terms(cls, text: str) -> bool:
        """Verifica si un texto contiene algún término prohibido de backend."""
        if not text:
            return False

        lower_text = text.lower()
        for term in FORBIDDEN_BACKEND_TERMS:
            # Búsqueda por límite de palabra para evitar falsos positivos
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, lower_text):
                return True
        return False


class HITLDialogTemplates:
    """Plantillas estándar y validadas para confirmaciones interactivas (HITL) y respuestas de sistema."""

    @staticmethod
    def unlock_order_question(order_name: str) -> str:
        """Pregunta antes de desbloquear un pedido confirmado para edición."""
        name = order_name or "la orden"
        return (
            f"El pedido {name} ya está confirmado y bloqueado en el sistema. "
            f"Para modificar sus productos es necesario desbloquearlo primero. "
            f"¿Deseas desbloquearlo y aplicar las modificaciones?"
        )

    @staticmethod
    def remove_order_question(order_name: str) -> str:
        """Pregunta antes de anular/cancelar una orden de venta."""
        name = order_name or "la orden seleccionada"
        return f"¿Estás seguro de que deseas cancelar la orden {name}?"

    @staticmethod
    def unknown_customer_question(customer_name: str) -> str:
        """Pregunta cuando un cliente no existe en la cartera comercial del vendedor."""
        name = customer_name or "el cliente"
        return (
            f"El cliente '{name}' no se encuentra en tu cartera comercial. "
            f"¿Deseas darlo de alta en este momento para continuar con la venta?"
        )

    @staticmethod
    def duplicate_contact_question(candidates_desc: str) -> str:
        """Pregunta cuando se detecta un posible cliente duplicado al registrar."""
        desc = candidates_desc or "existente"
        return (
            f"Se encontraron clientes similares en tu cartera: {desc}. "
            f"¿Deseas actualizar el registro existente ('accept') o cancelar ('reject')?"
        )

    @staticmethod
    def remove_contact_question(target_name: str) -> str:
        """Pregunta antes de archivar un cliente de la cartera."""
        name = target_name or "seleccionado"
        return f"¿Estás seguro de que deseas archivar al cliente '{name}' de tu cartera? ('yes' / 'no')"

    @staticmethod
    def order_list_header(customer_name: Optional[str] = None, status_filter: Optional[str] = None) -> str:
        """Encabezado determinista para listados de órdenes según el filtro de estado."""
        st = (status_filter or "").lower().strip()
        if st in ("draft", "cotizacion", "cotizaciones"):
            if customer_name:
                return f"Aquí tienes las cotizaciones activas de **{customer_name}**:"
            return "Aquí tienes el listado de cotizaciones activas:"
        elif st in ("sale", "confirmado", "confirmados", "aprobado", "aprobados"):
            if customer_name:
                return f"Aquí tienes los pedidos confirmados de **{customer_name}**:"
            return "Aquí tienes el listado de pedidos confirmados:"
        elif st in ("cancel", "cancelado", "cancelados", "anulado", "anulados"):
            if customer_name:
                return f"Aquí tienes las órdenes canceladas de **{customer_name}**:"
            return "Aquí tienes el listado de órdenes canceladas:"
        else:
            if customer_name:
                return f"Aquí tienes el estado de las cotizaciones y pedidos de **{customer_name}**:"
            return "Aquí tienes el listado general de tus cotizaciones y pedidos:"

    @staticmethod
    def order_list_empty(
        customer_name: Optional[str] = None,
        status_filter: Optional[str] = None,
        has_cancelled: bool = False,
        cancel_count: int = 0,
    ) -> str:
        """Mensaje cuando no se encuentran órdenes activas o según el filtro solicitado."""
        st = (status_filter or "").lower().strip()
        target_str = f" para **{customer_name}**" if customer_name else ""

        if st in ("cancel", "cancelado", "cancelados", "anulado", "anulados"):
            return f"No se encontraron órdenes canceladas registradas{target_str}."
        elif st in ("draft", "cotizacion", "cotizaciones"):
            return f"Actualmente no se encontraron cotizaciones activas registradas{target_str}."
        elif st in ("sale", "confirmado", "confirmados", "aprobado", "aprobados"):
            return f"Actualmente no se encontraron pedidos confirmados registrados{target_str}."
        else:
            msg = f"Actualmente no se encontraron cotizaciones ni pedidos activos registrados{target_str}."
            if has_cancelled and cancel_count > 0:
                msg += f" (Existen {cancel_count} orden(es) cancelada(s); si deseas revisarlas, indícamelo explícitamente)."
            return msg

    @staticmethod
    def order_not_found_to_cancel() -> str:
        """Error cuando no se identificó la orden a cancelar."""
        return "No se identificó la orden a cancelar."

    @staticmethod
    def contact_required_name_error() -> str:
        """Error cuando no se envió nombre de cliente para alta."""
        return "El nombre del cliente es obligatorio para registrarlo en el sistema."

    @staticmethod
    def contact_not_found_to_remove() -> str:
        """Error cuando no se identificó el cliente a archivar."""
        return "No se pudo identificar el cliente a archivar en el sistema."
