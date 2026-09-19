"""
src/agent_service/graph/sub_graphs/contact_manage/nodes.py - Nodos ejecutores del subgrafo contact_manage

Implementa el flujo 1:1 de flow_diagrams/contact_manage.drawio con herramientas Odoo desacopladas.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.types import interrupt

from src.agent_service.core.llms import bind_temperature, bind_structured_output
from src.agent_service.tools.contact_tools import (
    odoo_get_customers,
    odoo_list_current_customers,
    odoo_upsert_customer,
    odoo_remove_customer,
)
from src.agent_service.graph.sub_graphs.contact_manage.state import ContactManageState
from src.agent_service.graph.sub_graphs.contact_manage.schemas import (
    CustomerExtractionResult,
    DuplicateCheckResult,
    CustomerSynthesizeResponse,
)
from src.agent_service.core.hitl import parse_hitl_binary_decision
from src.agent_service.soul import inject_soul, SoulRole

logger = logging.getLogger(__name__)


class ContactManageNodes:
    """Nodos del subgrafo de gestión de clientes/contactos en Odoo ERP."""

    def __init__(
        self,
        llm: BaseChatModel,
        get_customers_tool: Callable[..., Awaitable[List[Dict[str, Any]]]] = odoo_get_customers,
        list_current_customers_tool: Callable[..., Awaitable[List[Dict[str, Any]]]] = odoo_list_current_customers,
        upsert_customer_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_upsert_customer,
        remove_customer_tool: Callable[..., Awaitable[Dict[str, Any]]] = odoo_remove_customer,
    ):
        self._llm = llm
        self._get_customers_tool = get_customers_tool
        self._list_current_customers_tool = list_current_customers_tool
        self._upsert_customer_tool = upsert_customer_tool
        self._remove_customer_tool = remove_customer_tool

        # Modelos estructurados
        self._extractor = bind_structured_output(bind_temperature(llm, 0.0), CustomerExtractionResult)
        self._judge = bind_structured_output(bind_temperature(llm, 0.1), DuplicateCheckResult)
        self._synthesizer = bind_structured_output(bind_temperature(llm, 0.35), CustomerSynthesizeResponse)

    async def extract_customer_info(self, state: ContactManageState) -> dict:
        """Nodo 1 (extraction): Extrae atributos (nombre obligatorio, phones múltiple) y la acción."""
        raw_query = state.get("raw_query")
        if not raw_query and state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, HumanMessage):
                    raw_query = str(m.content)
                    break
        raw_query = (raw_query or "").strip()

        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=6,
            strategy="last",
            token_counter=len,
        )

        system_prompt = """
            Eres un asistente especializado en CRM y gestión de cartera de clientes (Customers) para Odoo ERP.
            Tu labor es clasificar la acción solicitada por el vendedor y extraer los datos del cliente:

            1. 'action':
                - 'list': si el vendedor pide ver, listar o consultar clientes de su cartera (ej. 'dame mis clientes', 'ver cartera', 'muéstrame a Juan').
                - 'upsert': si el vendedor desea agregar, registrar, crear o modificar los datos de un cliente (ej. 'agrega el cliente Pedro', 'actualiza el teléfono de Carlos').
                - 'remove': si el vendedor pide eliminar, archivar o borrar un cliente de su cartera.
            2. 'name': Nombre de la persona o razón social de la empresa cliente. Es OBLIGATORIO para upsert y remove (salvo que proporcione ID).
            3. 'phones': Lista con los números telefónicos o celulares detectados (ej. ['987654321', '912345678']).
            4. 'contact_id': ID numérico de Odoo si el usuario hace referencia a un ID específico (ej. 'cliente ID 6').
        """

        user_prompt = f"Consulta o instrucción del vendedor:\n{raw_query}"

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_prompt),
        ]

        extraction: Optional[CustomerExtractionResult] = await self._extractor.ainvoke(messages)
        if extraction is None:
            extraction = CustomerExtractionResult(action="list")
        elif isinstance(extraction, dict):
            extraction = CustomerExtractionResult(**extraction)

        return {
            "contact_action": getattr(extraction, "action", "list"),
            "extracted_name": getattr(extraction, "name", None),
            "extracted_phones": getattr(extraction, "phones", []) or [],
            "target_contact_id": getattr(extraction, "contact_id", None),
            "has_possible_duplicates": False,
            "duplicate_rationale": None,
            "upsert_confirmed": True,
            "remove_confirmed": True,
            "operation_result": None,
            "customers_list": [],
            "current_candidates": [],
        }

    async def get_contacts_node(self, state: ContactManageState) -> dict:
        """Rama List: Invoca la herramienta get_contacts conectada a Odoo res.partner."""
        user_id = int(state.get("user_id") or 5)
        # Si extrajo un nombre para filtrar en la lista, lo usamos como query; si no, listamos cartera
        query = state.get("extracted_name")
        contacts = await self._get_customers_tool(user_id=user_id, query=query, limit=15)
        return {
            "customers_list": contacts,
        }

    async def list_current_contacts_node(self, state: ContactManageState) -> dict:
        """Rama Upsert: Invoca tool (list_current_contacts) para buscar candidatos existentes."""
        user_id = int(state.get("user_id") or 5)
        name = state.get("extracted_name")
        phones = state.get("extracted_phones", [])

        candidates = await self._list_current_customers_tool(
            user_id=user_id,
            name=name,
            phones=phones,
            limit=5,
        )
        return {
            "current_candidates": candidates,
        }

    async def duplicate_candidates_judge_node(self, state: ContactManageState) -> dict:
        """Rama Upsert: duplicate candidates (llm as judge)."""
        candidates = state.get("current_candidates", [])
        if not candidates:
            return {
                "has_possible_duplicates": False,
                "duplicate_rationale": "No se encontraron clientes existentes con datos coincidentes.",
                "upsert_confirmed": True,
            }

        name = state.get("extracted_name", "")
        phones = state.get("extracted_phones", [])
        cand_text = "\n".join(
            f"- [ID: {c['id']}] {c['name']} (Teléfono: {c.get('phone') or 'N/A'})"
            for c in candidates
        )

        system_prompt = """
            Eres un auditor de bases de datos de clientes CRM.
            Evalúa si el cliente nuevo que se desea registrar coincide o es un duplicado de los clientes existentes en la cartera.
            Determina:
            - 'has_duplicates': True si el nombre o teléfono apuntan verosímilmente a la misma persona/empresa; False si son claramente distintos.
            - 'matched_customer_id': ID de Odoo del cliente coincidente, o None.
            - 'reasoning': Breve explicación de la coincidencia.
        """

        user_prompt = f"""
            Nuevo cliente solicitado:
            Nombre: {name}
            Teléfonos: {', '.join(phones) if phones else 'Ninguno'}

            Candidatos existentes en Odoo:
            {cand_text}
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        judge: Optional[DuplicateCheckResult] = await self._judge.ainvoke(messages)
        if judge is None:
            # Fallback seguro: si hay candidatos, asumir posible duplicado para confirmación
            judge = DuplicateCheckResult(
                has_duplicates=len(candidates) > 0,
                matched_customer_id=candidates[0]["id"] if candidates else None,
                reasoning="Coincidencia detectada por datos similares.",
            )
        elif isinstance(judge, dict):
            judge = DuplicateCheckResult(**judge)

        target_id = state.get("target_contact_id") or getattr(judge, "matched_customer_id", None)
        return {
            "has_possible_duplicates": getattr(judge, "has_duplicates", False),
            "duplicate_rationale": getattr(judge, "reasoning", ""),
            "target_contact_id": target_id,
        }

    async def feedback_user_duplicate_node(
        self,
        state: ContactManageState,
        config: RunnableConfig,
    ) -> dict:
        """Rama Upsert: feedback usuario (accept/reject) mediante Human-in-the-Loop interrupt."""
        var_child_runnable_config.set(config)
        candidates = state.get("current_candidates", [])
        cand_desc = ", ".join(f"[{c['id']}] {c['name']} (Tel: {c.get('phone', 'N/A')})" for c in candidates)
        question = (
            f"Se encontraron clientes similares en tu cartera de Odoo: {cand_desc}. "
            f"¿Deseas actualizar el registro existente ('accept') o cancelar ('reject')?"
        )

        user_resume = interrupt({
            "type": "duplicate_customer_conflict",
            "question": question,
            "candidates": candidates,
            "suggested_actions": ["accept", "reject"],
        })

        is_accepted = await parse_hitl_binary_decision(
            user_input=str(user_resume),
            context_question=question,
            llm=self._llm,
        )
        return {
            "upsert_confirmed": is_accepted,
        }

    async def feedback_user_remove_node(
        self,
        state: ContactManageState,
        config: RunnableConfig,
    ) -> dict:
        """Rama Remove: feedback usuario (confirmación de eliminación/archivado con HITL)."""
        var_child_runnable_config.set(config)
        target_name = state.get("extracted_name") or f"ID {state.get('target_contact_id')}"
        question = f"¿Estás seguro de que deseas archivar al cliente '{target_name}' de tu cartera de Odoo? ('yes' / 'no')"

        user_resume = interrupt({
            "type": "remove_customer_confirmation",
            "question": question,
            "suggested_actions": ["yes", "no"],
        })

        is_confirmed = await parse_hitl_binary_decision(
            user_input=str(user_resume),
            context_question=question,
            llm=self._llm,
        )
        return {
            "remove_confirmed": is_confirmed,
        }

    async def upsert_customer_node(self, state: ContactManageState) -> dict:
        """Rama Upsert: Invoca tool (upsert_contact) en Odoo."""
        user_id = int(state.get("user_id") or 5)
        name = (state.get("extracted_name") or "").strip()
        phones = state.get("extracted_phones", [])
        contact_id = state.get("target_contact_id")

        if not name and not contact_id:
            return {
                "operation_result": {
                    "success": False,
                    "error": "El nombre del cliente es obligatorio para registrarlo en Odoo.",
                }
            }

        result = await self._upsert_customer_tool(
            user_id=user_id,
            name=name,
            phones=phones,
            contact_id=contact_id,
        )
        return {
            "operation_result": result,
        }

    async def remove_customer_node(self, state: ContactManageState) -> dict:
        """Rama Remove: Invoca tool (remove_contact) en Odoo."""
        user_id = int(state.get("user_id") or 5)
        contact_id = state.get("target_contact_id")

        if not contact_id:
            # Si no tenemos el contact_id pero sí el nombre, buscarlo primero en Odoo
            name = state.get("extracted_name")
            if name:
                candidates = await self._list_current_customers_tool(user_id=user_id, name=name, limit=1)
                if candidates:
                    contact_id = candidates[0]["id"]

        if not contact_id:
            return {
                "operation_result": {
                    "success": False,
                    "error": "No se pudo identificar el ID del cliente a archivar en Odoo.",
                }
            }

        result = await self._remove_customer_tool(
            contact_id=int(contact_id),
            user_id=user_id,
        )
        return {
            "operation_result": result,
        }

    async def synthesize_response(self, state: ContactManageState) -> dict:
        """Synthesize response: Redacta una respuesta amigable, ejecutiva y clara."""
        raw_query = state.get("raw_query", "")
        action = state.get("contact_action", "list")
        customers = state.get("customers_list", [])
        op_result = state.get("operation_result")
        upsert_confirmed = state.get("upsert_confirmed", True)
        remove_confirmed = state.get("remove_confirmed", True)

        context_lines = [f"Acción procesada: {action}"]

        if action == "list":
            if customers:
                context_lines.append(f"Clientes encontrados en cartera ({len(customers)}):")
                for c in customers:
                    context_lines.append(f"- ID {c['id']}: {c['name']} | Teléfono: {c.get('phone') or 'Sin teléfono registrado'}")
            else:
                context_lines.append("No se encontraron clientes asignados en tu cartera comercial.")

        elif action == "upsert":
            if op_result and op_result.get("success"):
                act_str = "actualizado" if op_result.get("action") == "updated" else "creado exitosamente"
                context_lines.append(
                    f"Cliente {act_str} en el sistema: ID {op_result.get('id')} - {op_result.get('name')} "
                    f"(Tel: {op_result.get('phone', 'N/A')}) asignado a tu usuario."
                )
            elif not upsert_confirmed:
                context_lines.append("Operación cancelada por el usuario tras la alerta de duplicados.")
            elif op_result and not op_result.get("success"):
                context_lines.append(f"Error al guardar cliente: {op_result.get('error')}")

        elif action == "remove":
            if op_result and op_result.get("success"):
                context_lines.append(f"El cliente ID {op_result.get('id')} ha sido archivado exitosamente en la cartera.")
            elif not remove_confirmed:
                context_lines.append("Eliminación/archivado cancelado por el usuario.")
            elif op_result and not op_result.get("success"):
                context_lines.append(f"Error al archivar cliente: {op_result.get('error')}")

        trimmed_history = trim_messages(
            state.get("messages", []),
            max_tokens=10,
            strategy="last",
            token_counter=len,
        )

        system_prompt = inject_soul(
            """
            Directrices de gestión de cartera de clientes:
            - Informa con claridad y cordialidad ejecutiva el resultado de la gestión de clientes.
            - Presenta los clientes ordenadamente con viñetas claras si se solicitaron listas, o confirma las altas/modificaciones con sus nombres y teléfonos.
            - NUNCA menciones términos de backend como 'ERP' u 'Odoo'.
            - NUNCA reveles identificadores técnicos internos como 'user_id' o 'partner_id'.
            """,
            role=SoulRole.CONTACTS,
        )

        user_prompt = f"""
            Instrucción del vendedor: {raw_query}
            Datos del resultado:
            {chr(10).join(context_lines)}
        """

        messages = [
            SystemMessage(content=system_prompt),
            *trimmed_history,
            HumanMessage(content=user_prompt),
        ]

        answer: Optional[CustomerSynthesizeResponse] = await self._synthesizer.ainvoke(messages)
        final_text = (
            answer.response_text
            if answer and hasattr(answer, "response_text") and answer.response_text
            else "\n".join(context_lines)
        )

        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }
