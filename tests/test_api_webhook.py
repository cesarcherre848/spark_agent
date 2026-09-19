"""
tests/test_api_webhook.py - Pruebas unitarias para Webhook y PhoneUserResolver con caché
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agent_service.api.schemas import WebhookRequest, WebhookResponse
from src.agent_service.api.service import (
    PhoneUserResolver,
    normalize_phone,
)
from src.agent_service.core.stores.product.odoo_client import OdooClient


# ==============================================================================
# 1. PRUEBAS DE NORMALIZACIÓN Y ESQUEMAS PYDANTIC
# ==============================================================================

def test_normalize_phone():
    assert normalize_phone("+51 987 654 321") == "51987654321"
    assert normalize_phone("+51-987-654-321") == "51987654321"
    assert normalize_phone("(01) 456-7890") == "014567890"
    assert normalize_phone("987654321") == "987654321"
    assert normalize_phone("") == ""
    assert normalize_phone(None) == ""


def test_webhook_request_schema_valid():
    req = WebhookRequest(
        raw_query="¿Tienes bloqueador solar?",
        phone_number="+51 987 654 321",
    )
    assert req.raw_query == "¿Tienes bloqueador solar?"
    assert req.phone_number == "+51 987 654 321"
    assert req.session_id is None


def test_webhook_request_schema_aliases():
    # Valida alias como phone y thread_id
    req = WebhookRequest.model_validate({
        "raw_query": "Hola catálogo",
        "phone": "999888777",
        "thread_id": "openclaw-session-123",
    })
    assert req.phone_number == "999888777"
    assert req.session_id == "openclaw-session-123"


def test_webhook_request_validation_errors():
    # raw_query vacío debe fallar
    with pytest.raises(ValueError):
        WebhookRequest(raw_query="   ", phone_number="987654321")

    # phone_number vacío debe fallar
    with pytest.raises(ValueError):
        WebhookRequest(raw_query="Hola", phone_number="")


# ==============================================================================
# 2. PRUEBAS DEL RESOLVEDOR PHONEUSERRESOLVER Y CACHETOOLS
# ==============================================================================

@pytest.mark.asyncio
async def test_resolver_cache_miss_queries_odoo_and_stores_in_cache():
    mock_http = AsyncMock()
    mock_auth_resp = MagicMock()
    mock_auth_resp.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 1}}
    mock_auth_resp.raise_for_status = MagicMock()

    # Respuesta de search_read en res.users con id=5
    mock_search_resp = MagicMock()
    mock_search_resp.json.return_value = {
        "jsonrpc": "2.0",
        "result": [
            {
                "id": 5,
                "name": "Vendedor Principal",
                "login": "vendedor@test.com",
            }
        ],
    }
    mock_search_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_auth_resp, mock_search_resp]

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=99)

    assert not resolver.is_cached("987654321")

    # 1. Primera llamada: Cache Miss -> consulta Odoo res.users
    user_id = await resolver.resolve_user_id("+51 987-654-321", client=odoo_client)

    assert user_id == 5
    assert resolver.is_cached("51987654321")
    assert resolver.get_cached("51987654321") == 5

    # 2. Segunda llamada: Cache Hit -> NO vuelve a consultar HTTP
    mock_http.post.reset_mock()
    user_id_cached = await resolver.resolve_user_id("51987654321", client=odoo_client)

    assert user_id_cached == 5
    mock_http.post.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_unregistered_phone_returns_none_and_not_cached():
    mock_http = AsyncMock()
    mock_auth_resp = MagicMock()
    mock_auth_resp.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 1}}
    mock_auth_resp.raise_for_status = MagicMock()

    # Odoo no encuentra ningún partner
    mock_search_resp = MagicMock()
    mock_search_resp.json.return_value = {"jsonrpc": "2.0", "result": []}
    mock_search_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_auth_resp, mock_search_resp]

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    # Por defecto sin default_user_id
    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=None)

    user_id = await resolver.resolve_user_id("999111222", client=odoo_client)

    # Debe retornar None y no debe enmascararse en caché
    assert user_id is None
    assert not resolver.is_cached("999111222")


@pytest.mark.asyncio
async def test_resolver_unregistered_phone_with_explicit_default_returns_default():
    mock_http = AsyncMock()
    mock_auth_resp = MagicMock()
    mock_auth_resp.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 1}}
    mock_auth_resp.raise_for_status = MagicMock()

    mock_search_resp = MagicMock()
    mock_search_resp.json.return_value = {"jsonrpc": "2.0", "result": []}
    mock_search_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_auth_resp, mock_search_resp]

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    # Si explícitamente se configuró un default_user_id
    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=12)

    user_id = await resolver.resolve_user_id("999111222", client=odoo_client)

    assert user_id == 12
    assert resolver.is_cached("999111222")


@pytest.mark.asyncio
async def test_resolver_network_error_returns_none_when_no_default():
    mock_http = AsyncMock()
    mock_http.post.side_effect = ConnectionError("Fallo de red simulado")

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=None)

    user_id = await resolver.resolve_user_id("987000111", client=odoo_client)
    assert user_id is None
    assert not resolver.is_cached("987000111")


@pytest.mark.asyncio
async def test_resolver_internal_salesperson_partner_resolves_user_id():
    """Valida que si el teléfono pertenece a un vendedor interno en res.users,
    el resolvedor obtenga su user_id directamente."""
    mock_http = AsyncMock()
    mock_auth_resp = MagicMock()
    mock_auth_resp.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 1}}
    mock_auth_resp.raise_for_status = MagicMock()

    # res.users retorna el usuario vendedor interno
    mock_user_resp = MagicMock()
    mock_user_resp.json.return_value = {
        "jsonrpc": "2.0",
        "result": [
            {
                "id": 5,
                "name": "Cesar Cherre Vendedor",
                "login": "cesarcherre@gmail.com",
            }
        ],
    }
    mock_user_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_auth_resp, mock_user_resp]

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=None)

    user_id = await resolver.resolve_user_id("+51983689215", client=odoo_client)

    assert user_id == 5
    assert resolver.is_cached("51983689215")
    assert resolver.get_cached("51983689215") == 5


@pytest.mark.asyncio
async def test_resolver_customer_phone_is_rejected_as_none():
    """Valida que un cliente final (no registrado como vendedor interno en res.users)
    sea rechazado retornando None, asegurando exclusividad para vendedores."""
    mock_http = AsyncMock()
    mock_auth_resp = MagicMock()
    mock_auth_resp.json.return_value = {"jsonrpc": "2.0", "result": {"uid": 1}}
    mock_auth_resp.raise_for_status = MagicMock()

    # res.users no encuentra a ningún vendedor con ese teléfono
    mock_user_resp = MagicMock()
    mock_user_resp.json.return_value = {"jsonrpc": "2.0", "result": []}
    mock_user_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_auth_resp, mock_user_resp]

    odoo_client = OdooClient(
        base_url="http://localhost:8069",
        db="test_db",
        username="user",
        password="pwd",
        client=mock_http,
    )

    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=None)

    # Teléfono de un cliente como Adhara Banda
    user_id = await resolver.resolve_user_id("987654321", client=odoo_client)

    assert user_id is None
    assert not resolver.is_cached("987654321")


def test_resolver_clear_and_set_cache():
    resolver = PhoneUserResolver(ttl_seconds=3600, default_user_id=5)
    resolver.set_cache("+51 900-100-200", 25)

    assert resolver.is_cached("51900100200")
    assert resolver.get_cached("51900100200") == 25

    resolver.clear_cache()
    assert not resolver.is_cached("51900100200")


# ==============================================================================
# 3. PRUEBAS DE INTEGRACIÓN DE ENDPOINTS HTTP FASTAPI CON ASGIRANSPORT
# ==============================================================================

from httpx import AsyncClient, ASGITransport
from src.agent_service.api.webhook import app, get_agent_graph
from src.agent_service.api.service import get_phone_user_resolver


@pytest.mark.asyncio
async def test_api_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"
        assert data.get("service") == "spark_agent_webhook"


@pytest.mark.asyncio
async def test_api_webhook_post_success_and_thread_id():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "final_response": "¡Hola! Con gusto te muestro los bloqueadores disponibles.",
        "intent": "rag",
        "is_topic_finished": False,
    }

    mock_resolver = AsyncMock()
    mock_resolver.resolve_user_id.return_value = 5

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    app.dependency_overrides[get_phone_user_resolver] = lambda: mock_resolver

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "phone_number": "+51 987 654 321",
                "raw_query": "¿Tienes bloqueador solar?",
            }
            resp = await client.post("/api/v1/webhook", json=payload)
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "success"
            assert data["user_id"] == 5
            assert data["phone_number"] == "51987654321"
            assert data["session_id"] == "wa_51987654321"
            assert data["intent"] == "rag"
            assert "bloqueadores disponibles" in data["response"]
            assert data["is_topic_finished"] is False

            # Verificar que ainvoke fue llamado con thread_id derivado y session_id en estado
            mock_graph.ainvoke.assert_awaited_once_with(
                {"raw_query": "¿Tienes bloqueador solar?", "user_id": 5, "session_id": "wa_51987654321"},
                config={"configurable": {"thread_id": "wa_51987654321"}},
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_webhook_post_custom_session_id_from_openclaw():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "final_response": "Cotización generada exitosamente.",
        "intent": "resolver",
        "is_topic_finished": True,
    }

    mock_resolver = AsyncMock()
    mock_resolver.resolve_user_id.return_value = 10

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    app.dependency_overrides[get_phone_user_resolver] = lambda: mock_resolver

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "phone": "+51912345678",
                "raw_query": "Confirmo la orden",
                "session_id": "openclaw-ticket-888",
            }
            resp = await client.post("/webhook", json=payload)
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "success"
            assert data["user_id"] == 10
            assert data["session_id"] == "openclaw-ticket-888"
            assert data["is_topic_finished"] is True

            mock_graph.ainvoke.assert_awaited_once_with(
                {"raw_query": "Confirmo la orden", "user_id": 10, "session_id": "openclaw-ticket-888"},
                config={"configurable": {"thread_id": "openclaw-ticket-888"}},
            )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_webhook_validation_errors():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Falta raw_query
        resp1 = await client.post("/api/v1/webhook", json={"phone": "987654321"})
        assert resp1.status_code == 422

        # 2. Falta phone
        resp2 = await client.post("/api/v1/webhook", json={"raw_query": "Hola"})
        assert resp2.status_code == 422

        # 3. Teléfono sin dígitos
        resp3 = await client.post("/api/v1/webhook", json={"phone": "+--", "raw_query": "Hola"})
        assert resp3.status_code == 422


@pytest.mark.asyncio
async def test_api_webhook_internal_error_handling():
    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = RuntimeError("Fallo simulado del grafo")

    mock_resolver = AsyncMock()
    mock_resolver.resolve_user_id.return_value = 5

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    app.dependency_overrides[get_phone_user_resolver] = lambda: mock_resolver

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/webhook",
                json={"phone": "987654321", "raw_query": "Hola fallo"},
            )
            assert resp.status_code == 500
            assert "Fallo simulado del grafo" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sanitized_tool_call_artifacts_in_webhook():
    """Valida que respuestas contaminadas con artefactos internos de tool calls ('call:default_api:...')
    sean sanitizadas antes de enviarse al cliente WhatsApp/OpenClaw."""
    contaminated_response = (
        "['call:default_api:RecommendationSynthesisResponse{response_text:', "
        "'He seleccionado para ti las opciones más versátiles y rentables de nuestro catálogo actual. 📦\\n\\n"
        "* **[5441] Rosa** - **$29.00 PEN**\\n"
        "* **[6262] Toasted Orange** - **$29.00 PEN**', '}', '']"
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "final_response": contaminated_response,
        "intent": "recommender",
        "is_topic_finished": False,
    }

    mock_resolver = AsyncMock()
    mock_resolver.resolve_user_id.return_value = 5

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    app.dependency_overrides[get_phone_user_resolver] = lambda: mock_resolver

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/webhook",
                json={
                    "phone": "51983689215",
                    "raw_query": "Recomiéndame labiales versátiles",
                    "session_id": "openclaw_ticket_uuid_456",
                },
            )
            assert resp.status_code == 200
            data = resp.json()

            assert data["status"] == "success"
            assert data["session_id"] == "openclaw_ticket_uuid_456"
            assert data["intent"] == "recommender"

            # Verificación de sanitización estricta:
            # 1. No debe contener fragmentos técnicos
            assert "call:default_api:" not in data["response"]
            assert "RecommendationSynthesisResponse" not in data["response"]
            assert "['" not in data["response"]
            assert "']" not in data["response"]
            assert "}" not in data["response"]

            # 2. Debe contener el contenido limpio y formateado
            assert "He seleccionado para ti las opciones más versátiles y rentables" in data["response"]
            assert "**[5441] Rosa** - **$29.00 PEN**" in data["response"]
            assert "**[6262] Toasted Orange** - **$29.00 PEN**" in data["response"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_unauthorized_phone_returns_401_without_invoking_graph():
    """Verifica que un número no registrado o sin comercial asignado (user_id=None)
    sea rechazado inmediatamente con HTTP 401 Unauthorized sin ejecutar el grafo principal."""
    mock_graph = AsyncMock()
    mock_resolver = AsyncMock()
    mock_resolver.resolve_user_id.return_value = None

    app.dependency_overrides[get_agent_graph] = lambda: mock_graph
    app.dependency_overrides[get_phone_user_resolver] = lambda: mock_resolver

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/webhook",
                json={
                    "phone": "51999999999",
                    "raw_query": "¿Puedo hacer una consulta?",
                },
            )
            assert resp.status_code == 401
            data = resp.json()
            assert "no está registrado o no cuenta con autorización" in data["detail"]

            # Garantizar que el grafo de agentes NUNCA fue invocado
            mock_graph.ainvoke.assert_not_called()
    finally:
        app.dependency_overrides.clear()

