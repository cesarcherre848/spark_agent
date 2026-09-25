# Spec Delta: whatsapp-integration

## Purpose

Proporciona integración nativa y bidireccional con WhatsApp Cloud API de Meta, soportando la verificación de webhooks, recepción de mensajes de clientes, invocación asíncrona de Spark Agent y despacho de respuestas salientes mediante la Graph API.

## ADDED Requirements

### Requirement: Webhook Handshake Verification
The system SHALL expose an HTTP GET endpoint at `/api/v1/whatsapp/webhook` that validates Meta webhook subscription requests by checking `hub.mode` equals `"subscribe"` and `hub.verify_token` matches the configured `WHATSAPP_VERIFY_TOKEN`, returning the integer or text value of `hub.challenge` as plain text with status `200 OK`.

#### Scenario: Successful webhook verification
- **WHEN** Meta sends a GET request with `hub.mode="subscribe"`, a valid `hub.verify_token`, and `hub.challenge="1158201444"`
- **THEN** the system returns status 200 with body `"1158201444"` and `Content-Type: text/plain`

#### Scenario: Verification fails on invalid token
- **WHEN** Meta or a third party sends a GET request with an invalid or missing `hub.verify_token`
- **THEN** the system returns status 403 Forbidden and does not echo `hub.challenge`

### Requirement: Inbound Event Processing
The system SHALL expose an HTTP POST endpoint at `/api/v1/whatsapp/webhook` that accepts Meta WhatsApp Cloud API event notifications, immediately returning `{"status": "ok"}` with status `200 OK` within 3 seconds, and processes valid incoming text messages asynchronously.

#### Scenario: Incoming user text message
- **WHEN** Meta sends a POST webhook payload containing an incoming message of type `"text"` from phone `"51999999999"` with body `"Hola, qué precio tiene el SKU 1?"`
- **THEN** the system acknowledges receipt immediately with status 200, extracts the sender phone and query, resolves the salesperson in Odoo, invokes the Spark Agent graph, and dispatches the generated answer back to the sender via WhatsApp Cloud API

#### Scenario: Non-message status event
- **WHEN** Meta sends a POST webhook payload containing only delivery status updates (`statuses` such as `"sent"`, `"delivered"`, or `"read"`)
- **THEN** the system returns status 200 immediately without executing the conversational graph or sending duplicate messages

### Requirement: Outbound Message Dispatch via Meta Graph API
The system SHALL provide an asynchronous client to dispatch text messages to Meta WhatsApp Cloud API endpoint `POST https://graph.facebook.com/{version}/{phone_number_id}/messages` using the configured bearer token.

#### Scenario: Successfully sending a text message
- **WHEN** the agent or an internal service invokes the outbound send method with recipient phone `"51999999999"` and text `"El SKU 1 tiene un costo de S/ 35.00"`
- **THEN** the client sends an authenticated POST request to Meta's messages endpoint with payload `{"messaging_product": "whatsapp", "to": "51999999999", "type": "text", "text": {"body": "..."}}` and returns the message ID confirmation

### Requirement: Response Sanitization for WhatsApp
The system SHALL ensure that all agent responses dispatched to WhatsApp are formatted as clean conversational text, stripping internal system artifacts, tool calls, and structured dicts before dispatch.

#### Scenario: Response containing clean text
- **WHEN** the Spark Agent graph completes processing and emits a response containing text
- **THEN** any internal tool artifacts or markers are stripped so that the customer on WhatsApp receives exclusively polite, formatted, commercial plain text
