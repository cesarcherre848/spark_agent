"""
scripts/run_stress_scenarios_log.py - Ejecutor y generador de logs detallados
de las pruebas de alto rigor de cambio y escenarios de estrés.
"""

import asyncio
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv(".env.qa")
load_dotenv(".env.dev")

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.tools.sales_tools import get_shared_odoo_client
from src.agent_service.graph.sub_graphs.product_advisor.graph import build_product_advisor_graph
from src.agent_service.graph.sub_graphs.product_advisor.schemas import QualityRubricEvaluation
from src.agent_service.tools.product_tools import create_rerank_and_filter_candidates_tool
from src.agent_service.core.guardrails.sku_grounding import SKUGroundingValidator


async def main():
    print("Iniciando componentes del grafo para pruebas de alto rigor...")
    llm = get_default_llm()
    pool = get_db_pool()
    await pool.open()
    embeddings = get_embedding_service()
    vector_store = ProductVectorStore(pool=pool, embedding_service=embeddings)
    odoo_client = get_shared_odoo_client()

    advisor_graph = build_product_advisor_graph(
        llm=llm,
        vector_store=vector_store,
        odoo_client=odoo_client,
        default_max_iterations=3,
    )

    test_cases = [
        {
            "id": "PRUEBA-1",
            "categoria": "Contraste Etario Extremo (Adolescente)",
            "query": "Recomiéndame una colonia fresca de uso diario para mi sobrina de 17 años que va al colegio",
            "esperado": "Acepta o prioriza líneas juveniles frescas de fantasía (Soy..., Mix & Chic, etc.)",
            "edad_target": 17,
        },
        {
            "id": "PRUEBA-2",
            "categoria": "Contraste Etario Extremo (Mujer Madura)",
            "query": "Recomiéndame una colonia fresca de uso diario para mi mamá de 55 años elegante",
            "esperado": "Descarta líneas juveniles 'Soy Sexy' / 'Soy Glow' y prioriza líneas sofisticadas adultas (L'Essence, Aires)",
            "edad_target": 55,
        },
        {
            "id": "PRUEBA-3",
            "categoria": "Aislamiento Estricto de Marca (Marca única vs Ruido Competitivo)",
            "query": "Quiero una crema hidratante facial económica PERO SOLO DE YANBAL, nada de Ésika ni Natura",
            "esperado": "Descarta Ésika (Triple Acción S/. 32) y Natura, devuelve 100% productos Yanbal (Totalist S/. 40)",
        },
        {
            "id": "PRUEBA-4",
            "categoria": "Tope Rígido de Presupuesto (Hard Cap < S/. 50)",
            "query": "Busco una colonia o fragancia femenina para regalo de 50 años, mi presupuesto máximo es 50 soles",
            "esperado": "Descarta perfumes de S/. 135-160 y colonias de S/. 68; recomienda opciones <= S/. 50",
        },
        {
            "id": "PRUEBA-5",
            "categoria": "Trampa Anatómica de Precio (Rostro vs Cuerpo/Manos)",
            "query": "Tratamiento nutritivo para la piel de la cara de una señora de 60 años, lo más económico posible",
            "esperado": "Ignora cremas corporales y de manos más baratas (S/. 18-35), prioriza crema facial nutritiva",
        },
        {
            "id": "PRUEBA-6",
            "categoria": "Validación Caso Real Giuliana: Silvana (50 años)",
            "query": "Silvana tiene 50 años y me pidió una colonia fresca y económica para ella, ¿qué opciones tienes en Yanbal?",
            "esperado": "Recomienda línea L'Essence (Flor de Cerezo / Violeta Salvaje S/. 68), 0% Soy Sexy",
        },
        {
            "id": "PRUEBA-7",
            "categoria": "Validación Caso Real Giuliana: Renzo (32 años)",
            "query": "Tengo un cliente se llama Renzo tiene 32 años y busca un perfume fresco herbal cítrico para hombre",
            "esperado": "Filtra fragancias de dama/niñas, recomienda Ohm Now Parfum",
        },
        {
            "id": "PRUEBA-8",
            "categoria": "Validación Caso Real Giuliana: Charito (Piel Trigueña)",
            "query": "Tengo a Charito de piel trigueña oscura me pide un rubor compacto que le quede bien, ¿qué tono recomiendas?",
            "esperado": "Identifica tono Bronzage por coincidencia semántica de tono cálido para piel trigueña",
        },
    ]

    output_lines = [
        "# Registro Detallado de Pruebas de Alto Rigor de Cambio y Casos de Estrés",
        "",
        "> **Entorno de Evaluación:** Contenedor QA `spark_agent_qa` / LLM Gemini 3.1 Flash / PGVector Híbrido  ",
        f"> **Total de Casos Evaluados:** {len(test_cases)}  ",
        "> **Objetivo:** Auditar la robustez del Reranker semántico, la inferencia continua de edad, el aislamiento de marca y la rúbrica de calidad sin hardcoding.  ",
        "",
        "---",
        ""
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\nEjecutando [{idx}/{len(test_cases)}] {tc['categoria']}...")
        state = {
            "raw_query": tc["query"],
            "channel": "whatsapp",
            "user_id": 8,
            "iteration_count": 0,
            "max_iterations": 3,
        }

        try:
            res = await advisor_graph.ainvoke(state)
            
            output_lines.append(f"## {tc['id']}: {tc['categoria']}")
            output_lines.append(f"- **Pregunta de Entrada:** *\"{tc['query']}\"*")
            output_lines.append(f"- **Criterio Esperado:** {tc['esperado']}")
            output_lines.append(f"- **Planificación / Rationale:** `{res.get('plan_rationale')}`")
            
            # Candidatos
            cands = res.get("candidate_products") or []
            output_lines.append("")
            output_lines.append("### 📦 Candidatos Evaluados y Filtrados:")
            if cands:
                for c in cands[:4]:
                    p_sku = c.get("sku") or "Sin SKU"
                    p_name = c.get("name") or "Sin nombre"
                    p_marca = c.get("marca") or "Sin marca"
                    p_price = f"S/. {c.get('price')}" if c.get('price') is not None else "Sin precio"
                    output_lines.append(f"- *[{p_sku}] {p_name}* | Marca: {p_marca} | Precio: {p_price}")
            else:
                output_lines.append("- *(Sin candidatos directos)*")

            # Rúbrica
            output_lines.append("")
            output_lines.append("### 📊 Evaluación de la Rúbrica de Calidad:")
            meets = res.get("meets_rubric")
            output_lines.append(f"- **Estado de Aprobación:** `{'APROBADO' if meets else 'RECHAZADO'}`")
            output_lines.append(f"- **Iteraciones Utilizadas:** `{res.get('iteration_count')} / 3`")
            scores = res.get("rubric_scores") or {}
            scores_str = ", ".join([f"**{k}**: {v}" for k, v in scores.items()])
            output_lines.append(f"- **Puntajes por Dimensión:** {scores_str}")
            if res.get("critique"):
                output_lines.append(f"- **Observación Crítica:** *{res.get('critique')}*")

            # Respuesta comercial
            output_lines.append("")
            output_lines.append("### 💬 Respuesta Comercial Generada para WhatsApp:")
            resp_txt = res.get("final_response") or "*(Sin respuesta generada)*"
            output_lines.append(resp_txt)
            output_lines.append("")
            output_lines.append("---")
            output_lines.append("")

        except Exception as e:
            print(f"Error en {tc['id']}: {e}")
            output_lines.append(f"## {tc['id']}: {tc['categoria']}")
            output_lines.append(f"**ERROR EN EJECUCIÓN:** {e}")
            output_lines.append("---")
            output_lines.append("")

    full_md = "\n".join(output_lines)
    
    with open("/home/spark_agent/registro_pruebas_alto_rigor_log.md", "w", encoding="utf-8") as f:
        f.write(full_md)

    with open("/root/.gemini/antigravity-cli/brain/75cbea57-4d2f-415b-8053-ffbc312b1c31/registro_pruebas_alto_rigor_log.md", "w", encoding="utf-8") as f:
        f.write(full_md)

    print("\nRegistro completo guardado con éxito en /home/spark_agent/registro_pruebas_alto_rigor_log.md")
    await pool.close()
    await odoo_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
