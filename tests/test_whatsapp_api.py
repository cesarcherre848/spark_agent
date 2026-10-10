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
    graph.aget_state = AsyncMock(return_value=None)
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
    mock_whatsapp_client.send_typing_indicator = AsyncMock(return_value=True)
    mock_whatsapp_settings.delay_warning_seconds = 0  # Desactivar delay para aislar la prueba

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

    # 1. Verificó envío de typing indicator (reemplaza mark_message_as_read cuando typing está habilitado)
    mock_whatsapp_client.send_typing_indicator.assert_awaited()

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
        text="Hola, el precio del SKU 1 es S/. 35.00 con stock disponible.",
    )
    assert response == "Hola, el precio del SKU 1 es S/. 35.00 con stock disponible."


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


# ==============================================================================
# 6. PRUEBAS DEL MÓDULO DEDICADO WHATSAPP_RESPONDER
# ==============================================================================

@pytest.mark.asyncio
async def test_responder_chunk_and_format(mock_whatsapp_client):
    from src.agent_service.api.whatsapp.responder import WhatsAppResponder
    settings = WhatsAppSettings(auto_reply=True)
    responder = WhatsAppResponder(client=mock_whatsapp_client, settings=settings)

    # Texto con artefacto interno de LLM (dict de Gemini) que debe ser limpiado
    raw = "{'type': 'text', 'text': 'Hola, esta es tu cotización formal.'}"
    formatted = responder.format_response(raw)
    assert formatted == "Hola, esta es tu cotización formal."

    # Prueba de chunking de mensaje largo
    long_msg = "A" * 8500
    chunks = responder.chunk_message(long_msg, max_length=4000)
    assert len(chunks) == 3
    assert len(chunks[0]) == 4000
    assert len(chunks[1]) == 4000
    assert len(chunks[2]) == 500


@pytest.mark.asyncio
async def test_responder_sends_with_dynamic_phone_number_id(mock_whatsapp_client):
    from src.agent_service.api.whatsapp.responder import WhatsAppResponder
    settings = WhatsAppSettings(auto_reply=True)
    responder = WhatsAppResponder(client=mock_whatsapp_client, settings=settings)

    await responder.send_response(
        to="51988776655",
        text="Respuesta personalizada",
        phone_number_id="109823485721903",
    )

    mock_whatsapp_client.send_text_message.assert_awaited_once_with(
        to="51988776655",
        text="Respuesta personalizada",
        phone_number_id="109823485721903",
    )


# ==============================================================================
# 7. PRUEBAS DE TYPING INDICATOR
# ==============================================================================

@pytest.mark.asyncio
async def test_whatsapp_client_send_typing_indicator_payload():
    """Valida que WhatsAppClient envíe correctamente el payload de typing indicator a Meta."""
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_http_client.post.return_value = mock_response

    settings = WhatsAppSettings(
        access_token="EAAX_TOKEN",
        phone_number_id="109823485721903",
        api_version="v21.0",
        api_base_url="https://graph.facebook.com",
    )

    client = WhatsAppClient(settings=settings, client=mock_http_client)
    result = await client.send_typing_indicator(message_id="wamid.TYPING_TEST_001")

    assert result is True
    mock_http_client.post.assert_awaited_once()
    call_args, call_kwargs = mock_http_client.post.call_args
    assert call_args[0] == "https://graph.facebook.com/v21.0/109823485721903/messages"
    payload = call_kwargs["json"]
    assert payload["messaging_product"] == "whatsapp"
    assert payload["status"] == "read"
    assert payload["message_id"] == "wamid.TYPING_TEST_001"
    assert payload["typing_indicator"] == {"type": "text"}


@pytest.mark.asyncio
async def test_incoming_message_triggers_typing_indicator(mock_whatsapp_settings, mock_whatsapp_client, mock_resolver, mock_graph):
    """Valida que handle_incoming_message envíe typing indicator al recibir un message_id."""
    mock_whatsapp_client.send_typing_indicator = AsyncMock(return_value=True)
    mock_whatsapp_settings.typing_indicator_enabled = True
    mock_whatsapp_settings.delay_warning_seconds = 0  # Desactivar delay para aislar la prueba

    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=mock_whatsapp_settings,
    )

    await service.handle_incoming_message(
        sender_phone="+51 999 999 999",
        text="¿Cuánto cuesta el protector solar?",
        message_id="wamid.TYPING_TRIGGER_001",
        phone_number_id="109823485721903",
    )

    # Debe haber llamado send_typing_indicator con el message_id del mensaje entrante
    mock_whatsapp_client.send_typing_indicator.assert_awaited()
    first_call = mock_whatsapp_client.send_typing_indicator.call_args_list[0]
    assert first_call[0][0] == "wamid.TYPING_TRIGGER_001"


# ==============================================================================
# 8. PRUEBAS DE TIMER DE DEMORA EXPLICATIVA (>60s)
# ==============================================================================

@pytest.mark.asyncio
async def test_delay_timer_fires_creative_message_when_exceeds_threshold(
    mock_whatsapp_settings, mock_whatsapp_client, mock_resolver
):
    """Valida que si el grafo tarda más que delay_warning_seconds, se envíe un mensaje de demora."""
    import asyncio

    mock_whatsapp_client.send_typing_indicator = AsyncMock(return_value=True)

    # Grafo que demora 0.3s para simular una consulta lenta
    slow_graph = MagicMock()
    slow_graph.aget_state = AsyncMock(return_value=None)
    async def _slow_ainvoke(*args, **kwargs):
        await asyncio.sleep(0.3)
        return {
            "final_response": "El protector solar FPS 50 cuesta S/ 42.00.",
            "intent": "resolver",
        }
    slow_graph.ainvoke = AsyncMock(side_effect=_slow_ainvoke)

    # Configurar umbral muy bajo (0.05s) para que el timer se dispare durante el test
    mock_whatsapp_settings.typing_indicator_enabled = True
    mock_whatsapp_settings.typing_refresh_interval = 60.0  # Evitar refresco para aislar el test
    mock_whatsapp_settings.delay_warning_seconds = 0.05
    mock_whatsapp_settings.delay_warning_temperature = 0.85

    from src.agent_service.api.whatsapp.responder import WhatsAppResponder
    responder = WhatsAppResponder(client=mock_whatsapp_client, settings=mock_whatsapp_settings)

    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=slow_graph,
        settings=mock_whatsapp_settings,
        responder=responder,
    )

    # Mockear _generate_delay_message para evitar llamada real al LLM
    service._generate_delay_message = AsyncMock(
        return_value="⏳ Sigo revisando las opciones de protector solar para ti... ¡Un momento más! ✨"
    )

    response = await service.handle_incoming_message(
        sender_phone="+51 999 999 999",
        text="¿Cuánto cuesta el protector solar?",
        message_id="wamid.DELAY_TEST_001",
        phone_number_id="109823485721903",
    )

    # Debe haber invocado la generación del mensaje de demora
    service._generate_delay_message.assert_awaited_once_with("¿Cuánto cuesta el protector solar?")

    # Debe haber enviado al menos 2 mensajes: el de demora y la respuesta final
    assert mock_whatsapp_client.send_text_message.await_count >= 2

    # La respuesta final debe ser la del grafo
    assert "protector solar" in response.lower() or "42" in response


@pytest.mark.asyncio
async def test_delay_timer_cancelled_on_fast_response(
    mock_whatsapp_settings, mock_whatsapp_client, mock_resolver, mock_graph
):
    """Valida que si el grafo responde rápido, el timer de demora se cancele sin enviar mensaje intermedio."""
    mock_whatsapp_client.send_typing_indicator = AsyncMock(return_value=True)

    # Configurar umbral alto para que NO se dispare
    mock_whatsapp_settings.typing_indicator_enabled = True
    mock_whatsapp_settings.typing_refresh_interval = 60.0
    mock_whatsapp_settings.delay_warning_seconds = 10.0  # 10s — el mock graph responde instant

    from src.agent_service.api.whatsapp.responder import WhatsAppResponder
    responder = WhatsAppResponder(client=mock_whatsapp_client, settings=mock_whatsapp_settings)

    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=mock_whatsapp_settings,
        responder=responder,
    )

    service._generate_delay_message = AsyncMock(return_value="No debería verse este mensaje")

    response = await service.handle_incoming_message(
        sender_phone="+51 999 999 999",
        text="¿Precio del SKU 1?",
        message_id="wamid.FAST_TEST_001",
    )

    # El mensaje de demora NO debe haberse generado
    service._generate_delay_message.assert_not_awaited()

    # Solo debe haber 1 envío de mensaje: la respuesta final
    assert mock_whatsapp_client.send_text_message.await_count == 1
    assert response is not None


# ==============================================================================
# 8. PRUEBAS DE SOPORTE PARA BSUID / USERNAMES DE META
# ==============================================================================

def test_extract_messages_handles_bsuid_and_username_payload():
    """Valida la extracción correcta cuando Meta envía from_user_id/user_id en lugar de from/wa_id."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "3711775495658495",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551563540",
                                "phone_number_id": "1240195469188096",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Giu", "username": "Giuliana.Y"},
                                    "user_id": "PE.2285075095400313",
                                }
                            ],
                            "messages": [
                                {
                                    "from_user_id": "PE.2285075095400313",
                                    "id": "wamid.TEST_BSUID_001",
                                    "timestamp": "1791327482",
                                    "text": {"body": "Hola"},
                                    "type": "text",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    extracted = WhatsAppService.extract_messages(payload)
    assert len(extracted) == 1
    assert extracted[0]["from"] == "PE.2285075095400313"
    assert extracted[0]["body"] == "Hola"
    assert extracted[0]["phone_number_id"] == "1240195469188096"


@pytest.mark.asyncio
async def test_whatsapp_service_handles_bsuid_flow(mock_whatsapp_settings, mock_whatsapp_client, mock_resolver, mock_graph):
    """Valida que un remitente BSUID se procese y despache la respuesta al BSUID intacto."""
    mock_whatsapp_client.send_typing_indicator = AsyncMock(return_value=True)
    mock_whatsapp_settings.delay_warning_seconds = 0

    service = WhatsAppService(
        client=mock_whatsapp_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=mock_whatsapp_settings,
    )

    response = await service.handle_incoming_message(
        sender_phone="PE.2285075095400313",
        text="Hola",
        message_id="wamid.TEST_BSUID_002",
        phone_number_id="1240195469188096",
    )

    # 1. Verificó resolución de vendedor
    mock_resolver.resolve_user_id.assert_awaited_once_with("PE.2285075095400313")

    # 2. Verificó llamada al grafo con thread_id
    mock_graph.ainvoke.assert_awaited_once()
    call_args, _ = mock_graph.ainvoke.call_args
    assert call_args[0]["raw_query"] == "Hola"
    assert call_args[0]["session_id"] == "wa_PE.2285075095400313"

    # 3. Verificó despacho de respuesta a Meta Cloud API con to=BSUID
    mock_whatsapp_client.send_text_message.assert_awaited_once_with(
        to="PE.2285075095400313",
        text="Hola, el precio del SKU 1 es S/. 35.00 con stock disponible.",
        phone_number_id="1240195469188096",
    )
    assert response is not None


@pytest.mark.asyncio
async def test_whatsapp_client_send_text_message_to_bsuid():
    """Valida que WhatsAppClient no trunque el prefijo ni punto de un BSUID."""
    mock_http_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messaging_product": "whatsapp",
        "contacts": [{"input": "PE.2285075095400313", "wa_id": "PE.2285075095400313"}],
        "messages": [{"id": "wamid.TEST_BSUID_OUT"}],
    }
    mock_http_client.post.return_value = mock_response

    settings = WhatsAppSettings(
        access_token="test_token",
        phone_number_id="1240195469188096",
        business_account_id="WABA_1",
        verify_token="verify_1",
    )
    client = WhatsAppClient(settings=settings, client=mock_http_client)

    result = await client.send_text_message(to="PE.2285075095400313", text="Hola respuesta")

    assert result["messages"][0]["id"] == "wamid.TEST_BSUID_OUT"
    mock_http_client.post.assert_awaited_once()
    _, kwargs = mock_http_client.post.call_args
    assert kwargs["json"]["to"] == "PE.2285075095400313"


