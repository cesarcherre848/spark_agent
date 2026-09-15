"""
src/agent_service/main.py - Entrypoint principal para Spark Agent
"""

import sys
from pathlib import Path
import asyncio
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cargar variables de entorno del entorno de desarrollo
load_dotenv(PROJECT_ROOT / ".env.dev")

from src.agent_service.graph.main_graph import (
    build_main_graph,
    get_main_graph,
    MainGraphState,
)


from src.agent_service.config.database import close_db_pool


async def async_main():
    try:
        print("🚀 Inicializando Spark Agent...")
        app = get_main_graph()
        test_query = "Hola, ¿qué servicios y catálogo tienes disponibles?"
        print(f"💬 Consulta de prueba: '{test_query}'")

        result = await app.ainvoke(
            {
                "raw_query": test_query,
                "user_id": 5,
            },
            config={"configurable": {"thread_id": "main-entrypoint-test"}},
        )
        print(f"🎯 Intención detectada: {result.get('intent')}")
        print(f"🤖 Respuesta:\n{result.get('final_response')}")
    finally:
        await close_db_pool()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
