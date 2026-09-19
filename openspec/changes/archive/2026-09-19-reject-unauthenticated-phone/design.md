# Design: Rejection of Unauthenticated / Unregistered Phone Numbers

## Context

Actualmente, `PhoneUserResolver` en `src/agent_service/api/service.py` tiene configurado un `DEFAULT_FALLBACK_USER_ID` (por defecto `5`). Cuando un teléfono no tiene registro en `res.partner` de Odoo o carece de comercial asignado, se retorna dicho fallback y se guarda en `TTLCache`.
En `src/agent_service/api/webhook.py`, `process_webhook` invoca al resolvedor y pasa directamente el `user_id` resultante a `get_agent_graph().ainvoke(...)`, permitiendo que remitentes desconocidos ejecuten el agente como si fueran el usuario 5.

## Goals / Non-Goals

**Goals:**
- Modificar `PhoneUserResolver.resolve_user_id(phone)` para que su tipo de retorno sea `Optional[int]`, retornando `None` cuando el teléfono no exista en Odoo o no tenga `user_id` comercial asociado.
- Evitar cachear números inexistentes (`None`) para no bloquear altas inmediatas de clientes en Odoo.
- En `src/agent_service/api/webhook.py`: verificar si `user_id is None` inmediatamente después de la resolución y antes de cualquier llamada al grafo, levantando `HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="El número de teléfono no está registrado o no cuenta con autorización.")`.
- Asegurar cobertura de pruebas unitarias exhaustivas con `pytest` para rechazo HTTP 401 y resolución de usuarios.

**Non-Goals:**
- No alterar la lógica interna del grafo principal ni subgrafos, los cuales seguirán recibiendo un `user_id: int` garantizado y validado.
- No implementar flujos de autorregistro interactivo en este cambio (fuera del alcance del Webhook de Spark Agent).

## Decisions

### Decisión 1: Retornar `Optional[int]` desde `resolve_user_id` en lugar de valor por defecto
- **Alternativa considerada**: Lanzar una excepción de dominio (`UnregisteredPhoneError`) dentro de `resolve_user_id`.
- **Elección**: Retornar `Optional[int]` (`None` ante fallas de resolución). Es idiomático, simplifica la lógica y desacopla el resolvedor de la capa de transporte HTTP.

### Decisión 2: Rechazo con código estándar HTTP 401 Unauthorized en el Webhook
- **Alternativa considerada**: Retornar HTTP 200 con `{ "status": "unauthorized", "response": "No tienes acceso" }`.
- **Elección**: Retornar HTTP 401 (`HTTP_401_UNAUTHORIZED`). Permite que pasarelas como OpenClaw o middleware de mensajería identifiquen de forma estándar que la petición fue rechazada por falta de autenticación y manejen el fallo adecuadamente a nivel de protocolo.

### Decisión 3: No almacenar `None` en la caché TTL
- **Alternativa considerada**: Cachear respuestas negativas (`None`) durante un TTL corto.
- **Elección**: No registrar en `TTLCache` si `user_id` es `None`. Si un asesor registra al cliente en Odoo en ese instante, el siguiente mensaje del cliente podrá autenticarse sin tener que esperar a que expire la caché.

## Risks / Trade-offs

- **[Riesgo] Pruebas anteriores que dependían del fallback a `DEFAULT_USER_ID` fallarán**:
  - *Mitigación*: Actualizar la suite de pruebas unitarias en `tests/test_api_webhook.py` para adaptar los tests existentes a la nueva semántica (verificar que números no registrados retornen HTTP 401 y resolver retorne `None`).
