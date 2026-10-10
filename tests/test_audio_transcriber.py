"""
tests/test_audio_transcriber.py - Pruebas unitarias para el transcriber y pipeline de audio de WhatsApp
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from src.agent_service.api.whatsapp.audio_transcriber import (
    transcribe_audio,
    ogg_to_wav_bytes,
    get_whisper_server_url,
)
from src.agent_service.api.whatsapp.client import WhatsAppClient
from src.agent_service.api.whatsapp.config import WhatsAppSettings
from src.agent_service.api.whatsapp.service import WhatsAppService


@pytest.mark.asyncio
async def test_transcribe_audio_empty_bytes():
    """Debe lanzar ValueError si el buffer de audio está vacío."""
    with pytest.raises(ValueError, match="audio_bytes no puede estar vacío"):
        await transcribe_audio(b"", mime_type="audio/ogg")


@pytest.mark.asyncio
async def test_transcribe_audio_wav_direct():
    """Valida que un WAV con cabecera RIFF se envíe directamente al endpoint de Whisper."""
    fake_wav_bytes = b"RIFF" + b"\x00" * 40

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "hola necesito cotización de cables"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await transcribe_audio(
            fake_wav_bytes,
            mime_type="audio/wav",
            whisper_url="http://whisper_internal:9000",
        )
        assert result == "hola necesito cotización de cables"
        assert mock_post.called
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://whisper_internal:9000/inference"


@pytest.mark.asyncio
async def test_transcribe_audio_with_segments_format():
    """Valida extracción de texto cuando whisper devuelve formato 'transcription' en lugar de 'text'."""
    fake_wav_bytes = b"RIFF" + b"\x00" * 40

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcription": [
            {"text": "buenos días"},
            {"text": "tienen stock de interruptores"},
        ]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await transcribe_audio(fake_wav_bytes, mime_type="audio/wav")
        assert result == "buenos días tienen stock de interruptores"


@pytest.mark.asyncio
async def test_transcribe_audio_ogg_pipeline():
    """Valida la conversión ffmpeg + llamada a whisper para audios OGG."""
    fake_ogg = b"OggS" + b"\x00" * 50
    fake_converted_wav = b"RIFF" + b"\x00" * 40

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "quiero comprar 10 unidades"}

    with patch("src.agent_service.api.whatsapp.audio_transcriber.ogg_to_wav_bytes", new_callable=AsyncMock) as mock_convert:
        mock_convert.return_value = fake_converted_wav
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            result = await transcribe_audio(fake_ogg, mime_type="audio/ogg; codecs=opus")
            assert result == "quiero comprar 10 unidades"
            assert mock_convert.called


@pytest.mark.asyncio
async def test_whatsapp_client_download_audio():
    """Valida que WhatsAppClient.download_audio consulte metadatos y luego descargue los bytes."""
    settings = WhatsAppSettings(
        access_token="test_token",
        api_base_url="https://graph.facebook.com",
        api_version="v21.0",
    )

    mock_meta_resp = MagicMock()
    mock_meta_resp.status_code = 200
    mock_meta_resp.json.return_value = {
        "url": "https://lookaside.fbsbx.com/temp_audio.ogg",
        "mime_type": "audio/ogg; codecs=opus",
    }

    mock_audio_resp = MagicMock()
    mock_audio_resp.status_code = 200
    mock_audio_resp.content = b"OGG_RAW_BYTES"

    async def custom_get(url, *args, **kwargs):
        if "lookaside" in url:
            return mock_audio_resp
        return mock_meta_resp

    client = WhatsAppClient(settings=settings)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=custom_get):
        audio_bytes, mime = await client.download_audio("media_id_999")
        assert audio_bytes == b"OGG_RAW_BYTES"
        assert mime == "audio/ogg; codecs=opus"


@pytest.mark.asyncio
async def test_whatsapp_service_extract_audio_messages():
    """Valida que extract_messages extraiga correctamente eventos de tipo audio y voice."""
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "ENTRY_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "PID_123"},
                            "messages": [
                                {
                                    "from": "51983689215",
                                    "id": "wamid.AUDIO_MSG_1",
                                    "timestamp": "1710000001",
                                    "type": "audio",
                                    "audio": {
                                        "id": "MEDIA_ID_AUDIO",
                                        "mime_type": "audio/ogg; codecs=opus",
                                        "voice": True,
                                    },
                                },
                                {
                                    "from": "51983689215",
                                    "id": "wamid.TEXT_MSG_1",
                                    "timestamp": "1710000002",
                                    "type": "text",
                                    "text": {"body": "hola"},
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }

    extracted = WhatsAppService.extract_messages(payload)
    assert len(extracted) == 2
    assert extracted[0]["type"] == "audio"
    assert extracted[0]["media_id"] == "MEDIA_ID_AUDIO"
    assert extracted[0]["phone_number_id"] == "PID_123"
    assert extracted[1]["type"] == "text"
    assert extracted[1]["body"] == "hola"


@pytest.mark.asyncio
async def test_whatsapp_service_process_audio_webhook():
    """Valida el flujo end-to-end de audio en WhatsAppService: descarga -> transcripción -> grafo -> respuesta."""
    mock_client = MagicMock(spec=WhatsAppClient)
    mock_client.download_audio = AsyncMock(return_value=(b"OGG_BYTES", "audio/ogg"))
    mock_client.send_typing_indicator = AsyncMock(return_value=True)
    mock_client.send_text_message = AsyncMock(return_value={"messages": [{"id": "wamid.OUT"}]})

    mock_resolver = MagicMock()
    mock_resolver.resolve_user_id = AsyncMock(return_value=1)

    mock_graph = MagicMock()
    mock_graph.aget_state = AsyncMock(return_value=None)
    mock_graph.ainvoke = AsyncMock(return_value={"final_response": "Sí tenemos stock del producto."})

    service = WhatsAppService(
        client=mock_client,
        resolver=mock_resolver,
        graph_app=mock_graph,
        settings=WhatsAppSettings(auto_reply=True),
    )

    with patch("src.agent_service.api.whatsapp.service.normalize_phone", return_value="51983689215"):
        with patch("src.agent_service.api.whatsapp.audio_transcriber.transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "¿Tienen stock de producto?"

            payload = {
                "object": "whatsapp_business_account",
                "entry": [
                    {
                        "id": "ENTRY_ID",
                        "changes": [
                            {
                                "field": "messages",
                                "value": {
                                    "messaging_product": "whatsapp",
                                    "metadata": {"phone_number_id": "PID_123"},
                                    "messages": [
                                        {
                                            "from": "51983689215",
                                            "id": "wamid.AUDIO_MSG_1",
                                            "timestamp": "1710000001",
                                            "type": "audio",
                                            "audio": {
                                                "id": "MEDIA_ID_AUDIO",
                                                "mime_type": "audio/ogg",
                                                "voice": True,
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }

            responses = await service.process_webhook_payload(payload)
            assert len(responses) == 1
            assert "Sí tenemos stock" in responses[0]
            mock_client.download_audio.assert_called_once_with("MEDIA_ID_AUDIO")
            mock_transcribe.assert_called_once_with(b"OGG_BYTES", mime_type="audio/ogg")
