"""
tests/test_sales_list_filtering.py - Pruebas Unitarias de Filtrado de Órdenes y Exclusión de Canceladas por Defecto

Verifica que:
1. En consultas generales (status_filter=None), las órdenes canceladas NUNCA se muestren por defecto.
2. Si el usuario solicita explícitamente canceladas (status_filter='cancel'), únicamente se listen las canceladas.
3. Si el usuario solicita cotizaciones (status_filter='draft'), únicamente se listen cotizaciones activas.
4. Si un cliente no tiene órdenes activas pero sí canceladas, se informe sutilmente sin listar las canceladas.
5. El 100% de las respuestas generadas cumpla con la directriz de Whitelabel.
"""

import pytest
from unittest.mock import MagicMock
from src.agent_service.graph.sub_graphs.sales_manage.nodes import SalesManageNodes
from src.agent_service.core.templates.dialogs import WhitelabelSanitizer


@pytest.fixture
def sample_orders_list():
    return {
        "draft": [
            {
                "id": 101,
                "name": "S00001",
                "customer_name": "Adhara Banda",
                "amount_total": 150.00,
                "lines": [{"product_name": "Delineador Plumón", "quantity": 2.0}],
            }
        ],
        "sale": [
            {
                "id": 102,
                "name": "S00002",
                "customer_name": "Adhara Banda",
                "amount_total": 450.00,
            }
        ],
        "cancel": [
            {
                "id": 103,
                "name": "S00003",
                "customer_name": "Adhara Banda",
                "amount_total": 300.00,
            },
            {
                "id": 104,
                "name": "S00004",
                "customer_name": "Adhara Banda",
                "amount_total": 120.00,
            }
        ],
    }


@pytest.fixture
def sales_nodes():
    return SalesManageNodes(llm=MagicMock())


class TestSalesListFiltering:

    @pytest.mark.asyncio
    async def test_default_query_excludes_cancelled_orders(self, sales_nodes, sample_orders_list):
        """Por defecto (status_filter=None), NO se deben mostrar las órdenes canceladas."""
        state = {
            "sales_action": "list",
            "orders_list": sample_orders_list,
            "customer_name": "Adhara Banda",
            "status_filter": None,
            "operation_result": None,
            "messages": [],
        }

        result = await sales_nodes.synthesize_sales_response(state)
        resp = result["final_response"]

        assert "Cotizaciones en Borrador (Activas)" in resp
        assert "S00001" in resp
        assert "Pedidos Confirmados" in resp
        assert "S00002" in resp

        # Regla de oro: NO deben figurar las órdenes canceladas
        assert "Órdenes Canceladas" not in resp
        assert "S00003" not in resp
        assert "S00004" not in resp

        # Cumplimiento Whitelabel
        assert not WhitelabelSanitizer.contains_forbidden_terms(resp)

    @pytest.mark.asyncio
    async def test_explicit_cancel_query_shows_only_cancelled(self, sales_nodes, sample_orders_list):
        """Si el usuario solicita 'cancel', se deben mostrar ÚNICAMENTE las canceladas."""
        state = {
            "sales_action": "list",
            "orders_list": sample_orders_list,
            "customer_name": "Adhara Banda",
            "status_filter": "cancel",
            "operation_result": None,
            "messages": [],
        }

        result = await sales_nodes.synthesize_sales_response(state)
        resp = result["final_response"]

        assert "Órdenes Canceladas" in resp
        assert "S00003" in resp
        assert "S00004" in resp

        # No deben figurar las activas
        assert "Cotizaciones en Borrador" not in resp
        assert "Pedidos Confirmados" not in resp

        # Cumplimiento Whitelabel
        assert not WhitelabelSanitizer.contains_forbidden_terms(resp)

    @pytest.mark.asyncio
    async def test_explicit_draft_query_shows_only_drafts(self, sales_nodes, sample_orders_list):
        """Si el usuario solicita 'draft' (cotizaciones activas), no se muestran confirmados ni cancelados."""
        state = {
            "sales_action": "list",
            "orders_list": sample_orders_list,
            "customer_name": "Adhara Banda",
            "status_filter": "draft",
            "operation_result": None,
            "messages": [],
        }

        result = await sales_nodes.synthesize_sales_response(state)
        resp = result["final_response"]

        assert "Cotizaciones en Borrador (Activas)" in resp
        assert "S00001" in resp

        # No deben figurar ni pedidos confirmados ni cancelados
        assert "Pedidos Confirmados" not in resp
        assert "S00002" not in resp
        assert "Órdenes Canceladas" not in resp
        assert "S00003" not in resp

        # Cumplimiento Whitelabel
        assert not WhitelabelSanitizer.contains_forbidden_terms(resp)

    @pytest.mark.asyncio
    async def test_no_active_orders_but_has_cancelled(self, sales_nodes):
        """Si no hay pedidos activos pero sí cancelados, informar sutilmente sin listar las canceladas."""
        orders_with_only_cancel = {
            "draft": [],
            "sale": [],
            "cancel": [
                {"id": 99, "name": "S00099", "customer_name": "Carlos Pérez", "amount_total": 50.0}
            ],
        }

        state = {
            "sales_action": "list",
            "orders_list": orders_with_only_cancel,
            "customer_name": "Carlos Pérez",
            "status_filter": None,
            "operation_result": None,
            "messages": [],
        }

        result = await sales_nodes.synthesize_sales_response(state)
        resp = result["final_response"]

        # Debe avisar que no se encontraron activos
        assert "no se encontraron cotizaciones ni pedidos activos" in resp.lower()
        # Debe mencionar sutilmente la existencia de canceladas
        assert "cancelada" in resp.lower()
        # Pero NO debe desplegar la lista de órdenes canceladas
        assert "Órdenes Canceladas" not in resp
        assert "S00099" not in resp

        # Cumplimiento Whitelabel
        assert not WhitelabelSanitizer.contains_forbidden_terms(resp)
