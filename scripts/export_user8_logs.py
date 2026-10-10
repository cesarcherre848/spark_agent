import asyncio
import json
from collections import OrderedDict
from src.agent_service.graph.sub_graphs.user_memory.checkpointer import get_postgres_checkpointer
from src.agent_service.config.database import get_db_pool

async def export_logs():
    pool = get_db_pool()
    await pool.open()
    checkpointer = await get_postgres_checkpointer(pool)

    # El thread_id de WhatsApp de Giuliana Yañez (user_id=8)
    config = {"configurable": {"thread_id": "wa_51975674799"}}
    history = []
    async for state in checkpointer.alist(config):
        history.append(state)

    print(f"Total estados en historial para wa_51975674799: {len(history)}")

    # Extraer turnos únicos cronológicos
    seen_queries = set()
    turns = []
    for s in reversed(history):
        cv = s.checkpoint.get("channel_values", {})
        query = cv.get("raw_query")
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)

        turns.append({
            "ts": s.checkpoint.get("ts"),
            "query": query,
            "intent": cv.get("intent"),
            "meets_rubric": cv.get("meets_rubric"),
            "rubric_scores": cv.get("rubric_scores"),
            "critique": cv.get("critique"),
            "reflection_action": cv.get("reflection_action"),
            "response": cv.get("final_response"),
            "user_id": cv.get("user_id"),
            "session_id": cv.get("session_id"),
        })

    print(f"Total turnos conversacionales únicos: {len(turns)}")

    md_lines = [
        "# Registro Detallado de Conversaciones: Consultora Giuliana Yañez (user_id=8)",
        "",
        "> **Consultora:** Giuliana Yañez  ",
        "> **User ID:** `8`  ",
        "> **Canal:** WhatsApp (`+51975674799` / `wa_51975674799`)  ",
        f"> **Total de Turnos Conversacionales:** {len(turns)}  ",
        "",
        "---",
        ""
    ]

    for i, t in enumerate(turns, 1):
        md_lines.append(f"## Turno #{i}")
        md_lines.append(f"- **Timestamp:** `{t.get('ts') or 'N/A'}`")
        md_lines.append(f"- **Intención:** `{t.get('intent')}`")
        md_lines.append(f"- **Aprobación de Rúbrica:** `{'APROBADO' if t.get('meets_rubric') else 'RECHAZADO'}`")
        scores = t.get("rubric_scores")
        if isinstance(scores, dict):
            scores_formatted = ", ".join([f"{k}: {v}" for k, v in scores.items()])
            md_lines.append(f"- **Puntajes de Rúbrica:** `{scores_formatted}`")
        else:
            md_lines.append(f"- **Puntajes de Rúbrica:** `{scores}`")

        if t.get("critique"):
            md_lines.append(f"- **Observación Crítica (Rubric Judge):** *{t.get('critique')}*")
        if t.get("reflection_action"):
            md_lines.append(f"- **Acción de Reflexión:** `{t.get('reflection_action')}`")

        md_lines.append("")
        md_lines.append("### 👤 Mensaje de Giuliana:")
        md_lines.append(f"> {t.get('query')}")
        md_lines.append("")
        md_lines.append("### 🤖 Respuesta de Spark Agent:")
        resp = t.get("response") or "*(Sin respuesta final registrada)*"
        md_lines.append(resp)
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    content = "\n".join(md_lines)

    # Guardar en local del proyecto
    with open("/home/spark_agent/conversaciones_user_8_log.md", "w", encoding="utf-8") as f:
        f.write(content)

    # Guardar en directorio de artefactos
    with open("/root/.gemini/antigravity-cli/brain/75cbea57-4d2f-415b-8053-ffbc312b1c31/conversaciones_user_8_log.md", "w", encoding="utf-8") as f:
        f.write(content)

    print("Archivo generado exitosamente.")
    await pool.close()

if __name__ == "__main__":
    asyncio.run(export_logs())
