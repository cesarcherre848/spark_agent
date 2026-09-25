"""
tests/test_whatsapp_api.py - Pruebas Unitarias para el Módulo Exclusivo de WhatsApp Cloud API (Meta)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from src.agent_service.api.webhook import app
from src.agent_service.api.whatsapp.config import WhatsAppSettings, get_whatsapp_settings
from src.agent_service.api.whatsapp.client import WhatsAppClient, get_whatsapp_client
from src.agent_service.api.whatsapp.service import WhatsAppService, get_whatsapp_service
from src.agent_service.api.service import PhoneUserResolver


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def mock_whatsapp_settings():
    return WhatsAppSettings(
        access_token="test_meta_token_12345",
        phone_number_id="109823485721903",
        business_account_id="WABA_99999",
        verify_token="test_secret_verify_token",
        api_version="v21.0",
        api_base_url="https://graph.facebook.com",
        auto_reply=True,
    )


@pytest.fixture
def mock_whatsapp_client():
    client = MagicMock(spec=WhatsAppClient)
    client.send_text_message = AsyncMock(return_value={
        "messaging_product": "whatsapp",
        "contacts": [{"input": "51999999999", "wa_id": "51999999999"}],
        "messages": [{"id": "wamid.TEST_MESSAGE_ID_001"}],
    })
    client.mark_message_as_read = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_resolver():
    resolver = MagicMock(spec=PhoneUserResolver)
    resolver.resolve_user_id = AsyncMock(return_value=5)
    return resolver


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "final_response": "Hola, el precio del SKU 1 es S/ 35.00 con stock disponible.",
        "intent": "resolver",
    })
    return graph


@pytest.fixture
def test_client(mock_whatsapp_settings, mock_whatsapp_client, mock_resolver, mock_graph):
    """Cliente de pruebas FastAPI con dependencias inyectadas."""
    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=mock_whatsapp_settings,
    )

    app.dependency_overrides[get_whatsapp_settings] = lambda: mock_whatsapp_settings
    app.dependency_overrides[get_whatsapp_client] = lambda: mock_whatsapp_client
    app.dependency_overrides[get_whatsapp_service] = lambda: service

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. PRUEBAS DE VERIFICACIÓN DE WEBHOOK (HANDSHAKE GET)
# ==============================================================================

def test_verify_webhook_success(test_client):
    """Valida que Meta reciba hub.challenge en texto plano cuando el token es correcto."""
    response = test_client.get(
        "/api/v1/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "test_secret_verify_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 200
    assert response.text == "1158201444"
    assert "text/plain" in response.headers["content-type"]


def test_verify_webhook_forbidden_on_invalid_token(test_client):
    """Valida que peticiones con token inválido sean rechazadas con HTTP 403."""
    response = test_client.get(
        "/api/v1/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token_attack",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403
    assert "Verificación de token fallida" in response.json()["detail"]


def test_verify_webhook_forbidden_on_invalid_mode(test_client):
    """Valida que peticiones con hub.mode distinto a 'subscribe' sean rechazadas."""
    response = test_client.get(
        "/api/v1/whatsapp/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "test_secret_verify_token",
            "hub.challenge": "1158201444",
        },
    )
    assert response.status_code == 403


# ==============================================================================
# 2. PRUEBAS DE RECEPCIÓN DE WEBHOOK (POST)
# ==============================================================================

def test_receive_webhook_ignores_status_notifications(test_client):
    """Valida que notificaciones de estado (delivered, read) devuelvan 200 OK inmediatamente."""
    status_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_99999",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "109823485721903"},
                            "statuses": [
                                {
                                    "id": "wamid.HBgLM...",
                                    "status": "delivered",
                                    "timestamp": "1727280001",
                                    "recipient_id": "51999999999",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    response = test_client.post("/api/v1/whatsapp/webhook", json=status_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_receive_webhook_acknowledges_incoming_message(test_client):
    """Valida que un mensaje entrante devuelva 200 OK de inmediato y despache en background."""
    message_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_99999",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "109823485721903"},
                            "contacts": [{"profile": {"name": "Cesar"}, "wa_id": "51999999999"}],
                            "messages": [
                                {
                                    "from": "51999999999",
                                    "id": "wamid.TEST_MSG_INBOUND_001",
                                    "timestamp": "1727280000",
                                    "type": "text",
                                    "text": {"body": "Precio del SKU 1"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    response = test_client.post("/api/v1/whatsapp/webhook", json=message_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ==============================================================================
# 3. PRUEBAS DEL SERVICIO ORQUESTADOR WHATSAPP
# ==============================================================================

@pytest.mark.asyncio
async def test_whatsapp_service_handles_message_flow(mock_whatsapp_settings, mock_whatsapp_client, mock_resolver, mock_graph):
    """Valida el ciclo completo de procesamiento: Odoo -> Grafo -> WhatsAppClient."""
    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=mock_whatsapp_settings,
    )

    response = await service.handle_incoming_message(
        sender_phone="+51 999 999 999",
        text="¿Tienen protector solar?",
        message_id="wamid.MSG_123",
    )

    # 1. Verificó marcado de lectura
    mock_whatsapp_client.mark_message_as_read.assert_awaited_once_with("wamid.MSG_123")

    # 2. Verificó resolución de vendedor
    mock_resolver.resolve_user_id.assert_awaited_once_with("51999999999")

    # 3. Verificó llamada al grafo con thread_id derivado
    mock_graph.ainvoke.assert_awaited_once()
    call_args, call_kwargs = mock_graph.ainvoke.call_args
    assert call_args[0]["raw_query"] == "¿Tienen protector solar?"
    assert call_args[0]["user_id"] == 5
    assert call_args[0]["session_id"] == "wa_51999999999"

    # 4. Verificó despacho de respuesta a Meta Cloud API
    mock_whatsapp_client.send_text_message.assert_awaited_once_with(
        to="51999999999",
        text="Hola, el precio del SKU 1 es S/ 35.00 con stock disponible.",
    )
    assert response == "Hola, el precio del SKU 1 es S/ 35.00 con stock disponible."


# ==============================================================================
# 4. PRUEBAS DEL CLIENTE HTTP DE WHATSAPP
# ==============================================================================

@pytest.mark.asyncio
async def test_whatsapp_client_send_text_message_payload():
    """Valida que WhatsAppClient formule correctamente la petición HTTP hacia Meta."""
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messaging_product": "whatsapp",
        "contacts": [{"input": "51999999999", "wa_id": "51999999999"}],
        "messages": [{"id": "wamid.TEST_ID_OK"}],
    }
    mock_http_client.post.return_value = mock_response

    settings = WhatsAppSettings(
        access_token="EAAX_TOKEN",
        phone_number_id="109823485721903",
        api_version="v21.0",
        api_base_url="https://graph.facebook.com",
    )

    client = WhatsAppClient(settings=settings, client=mock_http_client)
    res = await client.send_text_message(to="+51-999-999-999", text="Mensaje de prueba")

    assert res["messages"][0]["id"] == "wamid.TEST_ID_OK"
    mock_http_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_http_client.post.call_args
    assert call_args[0] == "https://graph.facebook.com/v21.0/109823485721903/messages"
    assert call_kwargs["headers"]["Authorization"] == "Bearer EAAX_TOKEN"
    assert call_kwargs["json"]["to"] == "51999999999"
    assert call_kwargs["json"]["text"]["body"] == "Mensaje de prueba"


# ==============================================================================
# 5. PRUEBA DE ENDPOINT MANUAL DE ENVÍO (/send)
# ==============================================================================

def test_send_manual_endpoint_success(test_client, mock_whatsapp_client):
    """Valida el endpoint de envío directo /api/v1/whatsapp/send."""
    response = test_client.post(
        "/api/v1/whatsapp/send",
        json={"to": "51999999999", "message": "Aviso de pedido listo"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "sent"
    mock_whatsapp_client.send_text_message.assert_awaited_once_with(
        to="51999999999",
        text="Aviso de pedido listo",
        preview_url=False,
    )
