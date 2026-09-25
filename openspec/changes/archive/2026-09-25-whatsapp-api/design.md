# Design: Módulo de Integración con WhatsApp Cloud API

## Context

El sistema cuenta con un webhook genérico en `src/agent_service/api/webhook.py` y resolución de vendedor vía `src/agent_service/api/service.py` (`PhoneUserResolver`). Para soportar WhatsApp Cloud API de Meta, se debe procesar el handshake GET de verificación (`hub.challenge`), interpretar el payload anidado de Meta en POST y despachar respuestas salientes mediante llamadas HTTP a Graph API.

## Goals / Non-Goals

**Goals:**
- Módulo autocontenido y desacoplado en `src/agent_service/api/whatsapp/`.
- Verificación conforme a especificación de Meta con `hub.challenge` y `hub.verify_token`.
- Despacho no bloqueante con `BackgroundTasks` de FastAPI para asegurar respuesta HTTP 200 a Meta en < 1 segundo.
- Reutilización de `PhoneUserResolver` para vincular automáticamente el teléfono con el vendedor asignado en Odoo ERP.
- Invocación limpia de `main_graph.ainvoke` y envío de respuesta saliente vía cliente HTTP `httpx.AsyncClient`.

**Non-Goals:**
- Envío de mensajes multimedia complejos (audio, video, stickers) en esta primera versión (se prioriza texto plano conversacional).
- Administración de plantillas HSM (Business-Initiated templates) desde el backend (se opera dentro de la ventana de servicio de 24 horas Customer-Initiated).

## Decisions

### 1. Desacoplamiento en Submódulo `api/whatsapp/`
- **Decisión:** Crear `src/agent_service/api/whatsapp/` con `config.py`, `schemas.py`, `client.py`, `service.py` y `router.py`.
- **Razón:** Mantiene el código de Meta aislado de la API genérica y facilita futuras extensiones (ej: Telegram, Slack) sin ensuciar `webhook.py`.
- **Alternativa:** Colocar todo en `webhook.py`. Descartado por saturar el archivo y mezclar protocolos externos.

### 2. Procesamiento Asíncrono en Background para SLA de Meta
- **Decisión:** El endpoint `POST /webhook` valida el payload, retorna de inmediato `{"status": "ok"}` (HTTP 200) y delega la ejecución de `main_graph` y el envío saliente a `BackgroundTasks`.
- **Razón:** Meta impone un timeout estricto de 3-5 segundos. Si el agente tarda 4 segundos en consultar Odoo o la base vectorial, Meta reintentaría el mensaje generando bucles infinitos de respuestas duplicadas.
- **Alternativa:** Esperar la respuesta completa antes de responder a Meta. Descartado por riesgo de reintentos agresivos de Meta.

### 3. Cliente Saliente con `httpx.AsyncClient`
- **Decisión:** Implementar `WhatsAppClient` con `httpx.AsyncClient` asíncrono para enviar `POST https://graph.facebook.com/{version}/{phone_number_id}/messages`.
- **Razón:** No bloquea el event loop de FastAPI y permite control fino de timeouts, reintentos y headers de autenticación Bearer.

### 4. Filtrado de Eventos de Solo Lectura
- **Decisión:** Ignorar de forma segura eventos de `statuses` (mensajes entregados, leídos) respondiendo 200 sin invocar el grafo.
- **Razón:** Meta envía notificaciones de estado por cada mensaje emitido; procesarlos como consultas saturaría innecesariamente el agente.

## Risks / Trade-offs

- **[Riesgo: Token de Meta expirado]** → Mitigación: Uso de System User Token permanente en producción y logging detallado con advertencia cuando la Graph API retorne 401 Unauthorized.
- **[Riesgo: Mensaje duplicado por reintento de Meta]** → Mitigación: Respuesta 200 instantánea mediante `BackgroundTasks` antes de comenzar la inferencia del agente.
- **[Riesgo: Error en red hacia Graph API]** → Mitigación: Captura de excepciones `httpx.HTTPError` en el cliente con log informativo sin derribar la aplicación.

## Migration Plan

1. Definir variables en `.env.dev`:
   ```bash
   WHATSAPP_ACCESS_TOKEN="<token>"
   WHATSAPP_PHONE_NUMBER_ID="<id>"
   WHATSAPP_VERIFY_TOKEN="spark_verify_token_default"
   WHATSAPP_API_VERSION="v21.0"
   ```
2. Montar el router en `src/agent_service/api/webhook.py`.
3. Configurar la URL de Webhook en el panel de Meta for Developers apuntando a `https://<dominio>/api/v1/whatsapp/webhook`.
