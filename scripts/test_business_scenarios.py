"""
scripts/test_business_scenarios.py

Prueba interactiva y de validación de negocio para el Asesor de Productos (ProductAdvisor)
ejecutando casos complejos, rebuscados y de alta exigencia comercial:
1. Restricción compuesta + error ortográfico ('essika') + presupuesto estricto (< S/ 50).
2. Filtro estricto por campaña y página ('Yanbal C10 página 124').
3. Venta cruzada y asesoría consultiva ('Tengo el labial 761, ¿qué complementario recomiendas?').
4. Caso borde: presupuesto imposible ('Perfume Yanbal por 10 soles').
5. Recomendación personalizada con Historial de Compras en Odoo.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv(".env.qa")
load_dotenv(".env.dev")

from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.config.database import get_db_pool
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.core.stores.product.vector_store import ProductVectorStore
from src.agent_service.tools.sales_tools import get_shared_odoo_client
from src.agent_service.graph.sub_graphs.product_advisor.graph import build_product_advisor_graph
from src.agent_service.graph.sub_graphs.product_advisor.state import ProductAdvisorState

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


async def run_scenario(name: str, query: str, state_extras: dict, graph):
    print("\n" + "=" * 80)
    print(f"ESCENARIO: {name}")
    print(f"PREGUNTA DEL CLIENTE: \"{query}\"")
    print("=" * 80)

    state: ProductAdvisorState = {
        "raw_query": query,
        "channel": "whatsapp",
        "user_id": 5,
        "iteration_count": 0,
        "max_iterations": 3,
        **state_extras,
    }

    try:
        final_state = await graph.ainvoke(state)
        
        print("\n--- 1. DIAGNÓSTICO DEL PLANIFICADOR (PLANNER) ---")
        print(f"Estrategia / Rationale: {final_state.get('plan_rationale')}")
        print("Herramientas planificadas:")
        for t in final_state.get("planned_tools") or []:
            print(f"  * {t.get('tool_name')}({t.get('arguments')}) -> Propósito: {t.get('purpose')}")

        print("\n--- 2. CANDIDATOS RECUPERADOS ---")
        candidates = final_state.get("candidate_products") or []
        print(f"Total candidatos: {len(candidates)}")
        for idx, c in enumerate(candidates[:4], start=1):
            price_str = f"S/ {c.get('price')}" if c.get('price') is not None else "Sin precio"
            marca_str = c.get('marca') or 'Sin marca'
            print(f"  [{idx}] [SKU: {c.get('sku')}] {c.get('name')} | {marca_str} | {price_str}")

        print("\n--- 3. EVALUACIÓN DE LA RÚBRICA DE CALIDAD ---")
        print(f"¿Aprobado por Rúbrica?: {final_state.get('meets_rubric')}")
        print(f"Iteraciones consumidas: {final_state.get('iteration_count')} / 3")
        scores = final_state.get("rubric_scores") or {}
        for k, v in scores.items():
            print(f"  * {k}: {v}")
        if final_state.get("critique"):
            print(f"  * Crítica emitida: {final_state.get('critique')}")

        print("\n--- 4. RESPUESTA COMERCIAL FINAL (WHATSAPP EXPERIENCIA) ---")
        print(final_state.get("final_response"))
        print("-" * 80)

    except Exception as e:
        print(f"ERROR EN ESCENARIO: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("Iniciando componentes del Asesor de Productos (ProductAdvisor)...")
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

    scenarios = [
        (
            "1. RESTRICCIÓN COMPUESTA + MARCA CON ERROR ORTOGRÁFICO + PRESUPUESTO ESTRICTO",
            "busco un rimel o mascara de pestañas a prueba de agua de essika que no pase de 50 soles para ir a la playa",
            {},
        ),
        (
            "2. BÚSQUEDA ESPECÍFICA CON METADATOS DE CATÁLOGO (CAMPAÑA Y PÁGINA)",
            "Búscame labiales de Yanbal de la campaña C10 que estén en la página 124",
            {},
        ),
        (
            "3. ASESORÍA DE VENTA CRUZADA (CROSS-SELL COMPLEMENTARIO)",
            "Ya tengo el labial mate rojo de Yanbal SKU 761, ¿qué delineador o producto complementario me recomiendas para que me dure todo el día?",
            {},
        ),
        (
            "4. CASO BORDE: PRESUPUESTO IMPOSIBLE / SIN COINCIDENCIAS BARATAS",
            "Quiero un perfume fino de Yanbal pero mi presupuesto máximo es 10 soles",
            {},
        ),
        (
            "5. RECOMENDACIÓN PERSONALIZADA CON HISTORIAL DE COMPRAS",
            "¿Qué me recomiendas para comprar hoy según lo que suelo pedir habitualmente?",
            {"partner_id": 9, "customer_name": "Consultora Principal"},
        ),
        (
            "6. CONSULTA COMPARATIVA MULTI-PROVEEDOR Y MULTI-MARCA",
            "¿Qué perfumes para mujer tienes disponibles en catálogo para regalar? Me gustaría comparar opciones de Yanbal y de Ésika",
            {},
        ),
    ]

    for name, query, extras in scenarios:
        await run_scenario(name, query, extras, advisor_graph)

    await pool.close()
    await odoo_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
