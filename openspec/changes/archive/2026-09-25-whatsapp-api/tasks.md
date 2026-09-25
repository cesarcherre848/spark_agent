# Tasks: Implementación del Módulo de WhatsApp Cloud API

## 1. Estructura, Configuración y Esquemas

- [x] 1.1 Crear la estructura de directorios `src/agent_service/api/whatsapp/` e implementar `config.py` con `WhatsAppSettings` basado en `pydantic_settings.BaseSettings`
- [x] 1.2 Implementar en `schemas.py` los modelos Pydantic v2 para el payload del Webhook de Meta (`MetaWebhookPayload`, `MetaMessage`, `MetaValue`, etc.) y esquemas de mensaje saliente

## 2. Cliente HTTP y Servicio Orquestador

- [x] 2.1 Implementar `client.py` con la clase `WhatsAppClient` utilizando `httpx.AsyncClient` para enviar mensajes de texto a `https://graph.facebook.com/{version}/{phone_number_id}/messages` y marcar mensajes como leídos
- [x] 2.2 Implementar `service.py` con `WhatsAppService` para parsear mensajes entrantes, resolver vendedor en Odoo vía `PhoneUserResolver`, invocar asíncronamente `main_graph` y despachar la respuesta al cliente mediante `WhatsAppClient`

## 3. Router FastAPI e Integración

- [x] 3.1 Implementar `router.py` con el endpoint `GET /webhook` para verificación de handshake (`hub.challenge`) y `POST /webhook` para recepción de eventos con despacho en background (`BackgroundTasks`)
- [x] 3.2 Montar el router de WhatsApp en `src/agent_service/api/webhook.py` bajo el prefijo `/api/v1/whatsapp`

## 4. Pruebas Unitarias y Validación

- [x] 4.1 Crear `tests/test_whatsapp_api.py` con pruebas unitarias para verificación de handshake (token válido e inválido), recepción de mensaje de texto, descarte de eventos `statuses` y envío saliente mockeado
- [x] 4.2 Ejecutar la suite de pruebas unitarias `.venv/bin/pytest tests/test_whatsapp_api.py` y `.venv/bin/pytest -m "not real_db"` verificando 100% de éxito
