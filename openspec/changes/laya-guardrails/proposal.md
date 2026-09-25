# Proposal: Laya Guardrails Decision Engine

## Why

Actualmente, el sistema de guardrails de `spark_agent` depende exclusivamente de una Capa 1 basada en expresiones regulares (`re`). Aunque este enfoque es ultrarrápido (< 1ms) para patrones explícitos conocidos (inyecciones SQL destructivas, jailbreaks directos), no detecta ataques semánticos parafraseados ni permite una graduación del riesgo. Además, opera de forma puramente binaria (permitir o bloquear de tajo), impidiendo advertir al usuario (`WARN`) cuando una consulta se desvía levemente del ámbito comercial sin necesidad de interrumpir el flujo natural de la conversación.

Integrar el modelo **Laya** (`convaiinnovations/laya-multilingual`) desde Hugging Face como motor de decisiones no-autorregresivo (Sistema 1) permite evaluar preguntas estructuradas tipo `noul` (probabilidades calibradas booleanas) y `score` (rúbrica ordinal de riesgo). Esto habilita una política de decisión tripartita (`CONTINUE`, `WARN`, `BLOCK`), robusteciendo la seguridad comercial sin degradar la experiencia de usuario.

## What Changes

- **Integración de Dependencia Laya:** Añadir el paquete `laya` a `requirements.txt` y configurar la carga del modelo (`convaiinnovations/laya-multilingual` por defecto, configurable mediante `LAYA_GUARDRAIL_MODEL` y directorio de caché local / Hugging Face).
- **Esquema de Decisión Tripartita:** Extender `GuardrailResult` y definir `GuardrailAction` (`ALLOW`/`CONTINUE`, `WARN`, `BLOCK`), incluyendo mensajes amigables de advertencia comercial (`warning_message`), flags de advertencia y metadata de scores numéricos calculados.
- **Definición de Reglas con Primitivas Laya:** Crear especificaciones de preguntas estructuradas en Laya:
  - `jailbreak` (`noul`): Probabilidad de que el prompt busque ignorar reglas o personificar roles no autorizados.
  - `prompt_injection` (`noul`): Probabilidad de instrucciones dirigidas a vulnerar directivas del agente.
  - `sensitive_data` (`noul`): Probabilidad de extracción de contraseñas, variables de entorno o system prompts.
  - `out_of_scope` (`noul`): Probabilidad de que la consulta pertenezca a un dominio no comercial (código de software, tareas escolares, poemas, etc.).
  - `harm_severity` (`score`): Nivel de daño potencial en una escala ordinal calibrada.
- **Motor de Reglas y Umbrales Calibrados:**
  - Fast-Path Capa 1: Mantiene el filtrado determinista regex para ataques destructivos instantáneos (`BLOCK`).
  - Capa 2 (Laya Decision Engine):
    - **`BLOCK`**: Si `jailbreak` $\ge 0.8$, `sensitive_data` $\ge 0.8$, o `harm_severity` $\ge 2.0$. Corta el flujo y emite rechazo educado.
    - **`WARN`**: Si `out_of_scope` $\ge 0.5$, o $0.4 \le \text{riesgo} < 0.8$, o $1.0 \le \text{harm\_severity} < 2.0$. Permite que el flujo comercial continúe naturalmente, pero adjunta una advertencia/disclaimer comercial en la interacción.
    - **`CONTINUE`**: Si los scores se mantienen en zona segura. Flujo regular sin alteraciones.
- **Actualización del Grafo Principal (`main_graph.py`):** Modificar `input_guardrail_node` y el estado `MainGraphState` para soportar `guardrail_action`, `is_warning` y `guardrail_warning`, garantizando que las advertencias fluyan hacia la respuesta final sin romper la ejecución de los subgrafos comerciales.

## Capabilities

### New Capabilities
- `guardrails-decision-engine`: Motor de evaluación de guardrails de dos niveles (Capa 1 Regex + Capa 2 Laya con primitivas `score` y `noul`) con soporte para decisiones de `CONTINUE`, `WARN` y `BLOCK`.

### Modified Capabilities
<!-- Ninguna capacidad previa en openspec/specs/ se modifica directamente -->

## Impact

- **Código Afectado:**
  - `src/agent_service/core/guardrails/schemas.py`: Nuevos enums y campos en `GuardrailResult`.
  - `src/agent_service/core/guardrails/rules.py`: Preguntas de Laya (`noul`/`score`), mensajes de advertencia y funciones de cálculo de umbrales.
  - `src/agent_service/core/guardrails/evaluator.py`: Lógica de evaluación combinada (Capa 1 Regex + Capa 2 Laya con fallback seguro).
  - `src/agent_service/graph/main_graph.py`: Soporte de `is_warning` y `guardrail_warning` en el nodo de guardrail y la propagación de advertencias al usuario.
- **Dependencias:**
  - `requirements.txt`: Inclusión de `laya`.
- **APIs y Subgrafos:**
  - Totalmente retrocompatible. Los subgrafos de ventas, productos y memoria reciben el flujo normalmente si no está bloqueado.
