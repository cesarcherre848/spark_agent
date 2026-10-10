"""
test_health.py – Smoke test del endpoint /health de whisper.cpp.

Verifica que el servidor esté disponible y responda correctamente
antes de ejecutar cualquier test de transcripción.
"""

import pytest
import httpx
from tests.whisper.conftest import WHISPER_BASE_URL


def test_whisper_server_health():
    """El servidor debe responder HTTP 200 en GET /health."""
    try:
        resp = httpx.get(f"{WHISPER_BASE_URL}/health", timeout=10.0)
    except httpx.ConnectError:
        pytest.fail(
            f"No se puede conectar al servidor whisper en {WHISPER_BASE_URL}. "
            "Asegúrate de que el contenedor esté corriendo:\n"
            "  docker compose -f docker_compose.dev.yml up whisper -d"
        )
    assert resp.status_code == 200, (
        f"Se esperaba HTTP 200, se obtuvo {resp.status_code}. Body: {resp.text}"
    )
    body = resp.json()
    assert "status" in body, f"Respuesta sin campo 'status': {body}"
    assert body["status"] == "ok", f"Status inesperado: {body['status']}"
    print(f"\n✅ Servidor whisper.cpp disponible en {WHISPER_BASE_URL} → {body}")


def test_whisper_inference_endpoint_exists():
    """El endpoint POST /inference debe existir (no 404)."""
    try:
        resp = httpx.post(f"{WHISPER_BASE_URL}/inference", timeout=10.0)
    except httpx.ConnectError:
        pytest.fail("No se puede conectar al servidor whisper.")
    # Sin archivo adjunto debe retornar 400 o 422, nunca 404
    assert resp.status_code != 404, (
        "El endpoint /inference no existe (404). "
        "Verifica que el servidor se inició con el flag correcto."
    )
    print(f"\n✅ Endpoint /inference existe (status: {resp.status_code})")
