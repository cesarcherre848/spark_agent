# Proposal: Módulo Dedicado de Integración con WhatsApp Cloud API (Meta)

## Why

Actualmente, Spark Agent cuenta con un endpoint genérico de webhook (`/api/v1/webhook`) que espera un JSON plano (`raw_query`, `phone_number`). Sin embargo, para conectar el agente directamente con la plataforma oficial de WhatsApp Cloud API de Meta (WhatsApp Business Platform), se requiere un módulo especializado capaz de:
1. Gestionar el protocolo de verificación de suscripción de Webhook de Meta (`GET` con `hub.mode`, `hub.verify_token`, `hub.challenge`).
2. Recibir e interpretar la estructura compleja anidada de eventos entrantes de Meta (`entry.changes.value.messages`).
3. Responder inmediatamente con `HTTP 200 OK` para cumplir la SLA de 3 segundos de Meta y evitar reenvíos duplicados.
4. Despachar las respuestas del agente de forma asíncrona mediante llamadas HTTP salientes a la Graph API de Meta (`POST /v21.0/{phone_number_id}/messages`).

La creación de un módulo exclusivo y desacoplado dentro de `src/agent_service/api/whatsapp/` permite integrar WhatsApp de forma nativa sin alterar la API genérica existente y manteniendo una arquitectura limpia y modular.

## What Changes

- **Nuevo módulo `src/agent_service/api/whatsapp/`**: Módulo dedicado para la integración con Meta.
- **Configuración (`config.py`)**: Parámetros de entorno tipados para WhatsApp (`WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_API_VERSION`, etc.).
- **Esquemas Pydantic (`schemas.py`)**: Modelos para payloads entrantes de Meta (mensajes de texto, estados, contactos) y respuestas salientes.
- **Cliente HTTP Asíncrono (`client.py`)**: Cliente `httpx` para envío de mensajes y marcado de lectura en WhatsApp Cloud API.
- **Servicio Orquestador (`service.py`)**: Procesamiento asíncrono en background: extracción de mensajes de texto, resolución de vendedor en Odoo vía `PhoneUserResolver`, invocación del grafo `main_graph` y envío de la respuesta al cliente.
- **Router FastAPI (`router.py`)**:
  - `GET /api/v1/whatsapp/webhook`: Handshake de verificación de Meta.
  - `POST /api/v1/whatsapp/webhook`: Recepción de eventos/mensajes de Meta con respuesta rápida 200 OK.
- **Montaje en FastAPI**: Integración del router de WhatsApp en `src/agent_service/api/webhook.py`.
- **Suite de Pruebas**: Tests unitarios completos en `tests/test_whatsapp_api.py`.

## Capabilities

### New Capabilities
- `whatsapp-integration`: Módulo exclusivo para conexión bidireccional con WhatsApp Cloud API de Meta, incluyendo handshake de verificación, procesamiento de mensajes entrantes y despacho de respuestas salientes vía Graph API.

### Modified Capabilities
*(Ninguna especificación existente modificada en sus requerimientos contractuales)*

## Impact

- Código afectado:
  - `src/agent_service/api/whatsapp/` (nuevo directorio con configuración, esquemas, cliente, servicio y router)
  - `src/agent_service/api/webhook.py` (inclusión del router de WhatsApp)
  - `tests/test_whatsapp_api.py` (nueva suite de pruebas unitarias)
- Dependencias: Utiliza `httpx` y `fastapi` (ya existentes en `requirements.txt`).
- Compatibilidad: Cero impacto en el endpoint genérico existente `/api/v1/webhook`.
