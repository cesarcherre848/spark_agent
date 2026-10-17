# Proposal: Reemplazo de Guardrails Laya por Guardrails LLM y Adición de Juez de Contexto Conversacional

## 1. Motivación y Problema Actual
El análisis detallado de las conversaciones reales de consultoras (como Gemma) ha revelado dos problemas críticos que afectan gravemente la experiencia de usuario:

1. **Falsos Positivos Severos en Guardrails (Laya):**
   - El modelo Laya (`convaiinnovations/laya-multilingual`) presenta un sesgo crítico donde solicitudes imperativas cotidianas en español (ej: *"Si por favor! Agregar Cliente a Valeria osores"*, *"Agregar Cliente a Valeria Osores"*, *"Quiero el ganamas nivel 2"*) son clasificadas como intentos de evasión (`jailbreak = 0.9999`).
   - Esto activa un bloqueo defensivo inmediato que responde a las consultoras con mensajes hostiles sobre directivas de seguridad, imposibilitando la operación comercial normal.

2. **Contaminación de Contexto y Respuestas Desfasadas/Solapadas:**
   - En conversaciones de WhatsApp persistentes, los turnos anteriores (incluso ocurridos días atrás, como un diálogo sobre un cliente con acné) quedan en el historial sin expiración ni delimitación.
   - Cuando la consultora inicia un nuevo tema comercial (ej: incentivos de GanaMás para ella misma), el LLM arrastra indebidamente el contexto previo y responde refiriéndose al cliente anterior.
   - Adicionalmente, el router clasifica erróneamente la intención porque mezcla elementos del mensaje anterior con el actual, generando respuestas desfasadas (efecto *off-by-one*).

## 2. Objetivos de la Propuesta
1. **Eliminar el modelo Laya** y sustituirlo por un evaluador de Guardrails basado en **LLM rápido (Gemini 2.5 Flash / Flash-Lite)** estructurado mediante Pydantic, manteniendo la Capa 1 de Regex rápida para ataques deterministas (SQLi destructivo, OS commands, env leak).
2. **Acotar la Memoria de Corto Plazo a un Máximo de 1 Día (Ventana de 24 Horas):**
   - El historial de turnos conversacionales inmediatos (`messages`) se delimita a un máximo de 24 horas desde la última interacción. Mensajes con mayor antigüedad se descartan del historial inmediato, iniciando una nueva sesión conversacional limpia y evitando la contaminación por temas de días pasados.
3. **Potenciar la Memoria Semántica a Largo Plazo (`user_memory`):**
   - La persistencia duradera entre semanas y meses se delega a la memoria semántica (`user_memories` con pgvector). Esta memoria recupera antecedentes de perfil y preferencias relevantes mediante búsqueda semántica, siendo la fuente legítima de contexto histórico sin arrastrar diálogos obsoletos.
4. **Implementar un Juez de Contexto Conversacional (`ContextJudgeNode`):**
   - Para las interacciones dentro de la ventana de 24 horas, un juez LLM evalúa la consulta entrante frente al historial reciente.
   - Determina explícitamente:
     - `is_standalone`: Si la consulta es independiente o depende del hilo previo.
     - `context_usefulness`: Nivel de utilidad (`fully_useful`, `partially_useful`, `irrelevant_or_obsolete`).
     - `cleaned_context_summary`: Resumen exclusivo de lo que realmente aplica (descartando clientes o temas previos si cambió el foco).
5. **Delimitación Textual Estricta para el Router y Sintetizadores:**
   - Indicar explícitamente al LLM qué información es el turno actual (prioridad máxima) y qué es contexto histórico, prohibiendo asumir clientes, pedidos o temas antiguos ante un cambio de intención.
6. **Batería de Pruebas Detallada y Exhaustiva:**
   - Validar tanto consultas comerciales del mundo real (CRM, pedidos, incentivos, imperativos) para garantizar 0% falsos positivos, como intentos reales de jailbreak y exfiltración para asegurar 100% de protección.
