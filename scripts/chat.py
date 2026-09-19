#!/usr/bin/env python3
"""
scripts/chat.py - Spark Agent Interactive Terminal Chat CLI

Permite interactuar en tiempo real con el agente, probando consultas en vivo
con Google Gemini, PostgreSQL (view_user_authorized_products), Odoo ERP y
resolución Human-in-the-Loop (HITL) para desambiguación de productos y proveedores.
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.simplefilter("ignore")

import os
os.environ["PYTHONWARNINGS"] = "ignore"
import sys
import time
import uuid
import asyncio
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Cargar variables de entorno antes de importar módulos internos
load_dotenv(PROJECT_ROOT / ".env.dev")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.agent_service.config.llm import get_llm_settings
from src.agent_service.config.database import get_db_pool, close_db_pool
from src.agent_service.config.odoo import get_odoo_settings
from src.agent_service.core.llms.factory import get_default_llm
from src.agent_service.core.embeddings.factory import get_embedding_service
from src.agent_service.graph.sub_graphs.user_memory import UserMemoryStore, get_postgres_checkpointer
from src.agent_service.graph.main_graph import build_main_graph


# --- Colores ANSI para terminal ---
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"


def print_banner(model_name: str, provider: str, user_id: int, thread_id: str, debug: bool):
    """Imprime el banner inicial de la sesión de chat."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 68}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}   🤖 SPARK AGENT - CHAT UNIFICADO (General, RAG & Resolver)    {Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 68}{Colors.RESET}")
    print(f" {Colors.BOLD}Proveedor LLM:{Colors.RESET} {Colors.GREEN}{provider.upper()}{Colors.RESET} | {Colors.BOLD}Modelo:{Colors.RESET} {Colors.GREEN}{model_name}{Colors.RESET}")
    print(f" {Colors.BOLD}Usuario ID:{Colors.RESET}    {Colors.YELLOW}{user_id}{Colors.RESET} (simulado para permisos de catálogo)")
    print(f" {Colors.BOLD}Sesión:{Colors.RESET}        {Colors.DIM}{thread_id}{Colors.RESET}")
    print(f" {Colors.BOLD}Modo Debug:{Colors.RESET}    {Colors.RED if debug else Colors.DIM}{'ACTIVADO' if debug else 'desactivado'}{Colors.RESET}")
    print(f"{Colors.CYAN}{'-' * 68}{Colors.RESET}")
    print(f" {Colors.ITALIC}Comandos disponibles:{Colors.RESET}")
    print(f"   {Colors.BOLD}/help{Colors.RESET}       - Ver ayuda y ejemplos de consultas")
    print(f"   {Colors.BOLD}/new{Colors.RESET}        - Iniciar un nuevo hilo de conversación")
    print(f"   {Colors.BOLD}/user <id>{Colors.RESET}  - Cambiar el ID de usuario activo (ej: /user 5)")
    print(f"   {Colors.BOLD}/memory{Colors.RESET}     - Ver historial de memorias persistidas del usuario")
    print(f"   {Colors.BOLD}/debug{Colors.RESET}      - Alternar modo de depuración detallado")
    print(f"   {Colors.BOLD}/exit{Colors.RESET}       - Salir del chat (o escribe 'salir')")
    print(f"{Colors.CYAN}{'=' * 68}{Colors.RESET}\n")


def print_help():
    """Muestra la guía de uso y ejemplos prácticos."""
    print(f"\n{Colors.YELLOW}{Colors.BOLD}📖 GUÍA DE USO Y EJEMPLOS DE SPARK AGENT:{Colors.RESET}")
    print(f"  El agente integra Router inteligente, Memoria semántica (pgvector), RAG, Cartera de Clientes y Odoo ERP.")
    print(f"\n  {Colors.BOLD}1. Charla general y orientación (General Chat):{Colors.RESET}")
    print(f"     > {Colors.CYAN}¡Hola! ¿En qué puedes ayudarme?{Colors.RESET}")
    print(f"\n  {Colors.BOLD}2. Exploración y recomendación semántica de catálogo (Product RAG):{Colors.RESET}")
    print(f"     > {Colors.CYAN}¿Qué cremas faciales o lociones hidratantes tienen?{Colors.RESET}")
    print(f"     > {Colors.CYAN}Recomiéndame productos desmaquilladores{Colors.RESET}")
    print(f"\n  {Colors.BOLD}3. Consulta de precios y SKUs (Product Resolver):{Colors.RESET}")
    print(f"     > {Colors.CYAN}Precio del SKU 1 y disponibilidad de Unique{Colors.RESET}")
    print(f"\n  {Colors.BOLD}4. Gestión de cartera comercial de clientes (Contact Manage):{Colors.RESET}")
    print(f"     > {Colors.CYAN}Muéstrame mi cartera de clientes{Colors.RESET}")
    print(f"     > {Colors.CYAN}Agrega al cliente Inversiones Alfa con teléfono 987654321{Colors.RESET}")
    print(f"\n  {Colors.BOLD}5. Ciclo de ventas, cotizaciones y pedidos (Sales Manage):{Colors.RESET}")
    print(f"     > {Colors.CYAN}Muéstrame mis pedidos y cotizaciones de este mes{Colors.RESET}")
    print(f"     > {Colors.CYAN}Cotízale 5 unidades del SKU 1 a Carlos{Colors.RESET}")
    print(f"     > {Colors.CYAN}Confirma la cotización SO001{Colors.RESET}")
    print(f"     > {Colors.CYAN}Cambia la cantidad del pedido confirmado SO002 a 10 unidades{Colors.RESET}")
    print(f"     > {Colors.CYAN}Cancela la orden SO003{Colors.RESET}")
    print(f"\n  {Colors.BOLD}6. Memoria semántica a largo plazo:{Colors.RESET}")
    print(f"     > {Colors.CYAN}/memory{Colors.RESET} (consulta las cotizaciones y preferencias registradas)\n")


async def handle_interrupt(
    app: Any,
    config: dict,
    interrupt_payload: dict,
    debug: bool,
) -> Any:
    """Maneja interactivamente una pausa Human-in-the-Loop (HITL)."""
    i_type = interrupt_payload.get("type", "unknown")
    question = interrupt_payload.get("question", "Se requiere confirmación para continuar.")

    print(f"\n{Colors.YELLOW}{'┌' + '─' * 66 + '┐'}{Colors.RESET}")
    print(f"{Colors.YELLOW}│ ❓ {Colors.BOLD}ACLARACIÓN REQUERIDA POR EL AGENTE (HITL){Colors.RESET}{' ' * 23}{Colors.YELLOW}│{Colors.RESET}")
    print(f"{Colors.YELLOW}{'├' + '─' * 66 + '┤'}{Colors.RESET}")

    # Formatear el mensaje en líneas legibles
    words = question.split()
    current_line = " "
    for w in words:
        if len(current_line) + len(w) + 1 > 64:
            print(f"{Colors.YELLOW}│{Colors.RESET}{current_line:<66}{Colors.YELLOW}│{Colors.RESET}")
            current_line = " " + w
        else:
            current_line += (" " if current_line != " " else "") + w
    if current_line.strip():
        print(f"{Colors.YELLOW}│{Colors.RESET}{current_line:<66}{Colors.YELLOW}│{Colors.RESET}")

    # Mostrar opciones o candidatos si existen
    options = interrupt_payload.get("options") or interrupt_payload.get("candidate_orders") or []
    if options:
        print(f"{Colors.YELLOW}│{Colors.RESET}{' ' * 66}{Colors.YELLOW}│{Colors.RESET}")
        print(f"{Colors.YELLOW}│{Colors.RESET}   {Colors.BOLD}Opciones disponibles:{Colors.RESET}{' ' * 42}{Colors.YELLOW}│{Colors.RESET}")
        for opt in options:
            opt_str = f"   • {opt}"
            print(f"{Colors.YELLOW}│{Colors.RESET}{opt_str:<66}{Colors.YELLOW}│{Colors.RESET}")

    print(f"{Colors.YELLOW}{'└' + '─' * 66 + '┘'}{Colors.RESET}")

    while True:
        try:
            prompt_str = f"{Colors.YELLOW}[Tu respuesta / aclaración] > {Colors.RESET}"
            user_reply = input(prompt_str).strip()
        except EOFError:
            return None

        if not user_reply:
            print(f"{Colors.DIM}Por favor ingresa una respuesta o escribe '/cancel' para cancelar.{Colors.RESET}")
            continue

        if user_reply.lower() in ("/cancel", "/salir", "/exit", "salir"):
            print(f"{Colors.RED}Operación cancelada.{Colors.RESET}")
            return None

        # Reanudar la ejecución con Command(resume=...)
        print(f"\n{Colors.DIM}⏳ Reanudando flujo con tu respuesta...{Colors.RESET}")
        t0 = time.perf_counter()
        try:
            resumed_result = await app.ainvoke(Command(resume=user_reply), config=config)
            t1 = time.perf_counter()
            if debug:
                print(f"{Colors.DIM}[Debug] Reanudación completada en {t1 - t0:.2f}s{Colors.RESET}")
            return resumed_result
        except Exception as e:
            print(f"{Colors.RED}❌ Error al reanudar: {e}{Colors.RESET}")
            return None


async def chat_loop(user_id: int = 5, debug: bool = False):
    """Bucle principal asíncrono del chat."""
    settings = get_llm_settings()

    print(f"{Colors.DIM}Iniciando conexiones (PostgreSQL, Odoo, Google Gemini)...{Colors.RESET}")
    pool = get_db_pool()
    if pool.closed:
        await pool.open()
    odoo_settings = get_odoo_settings()
    llm = get_default_llm()

    # Compilar el grafo con AsyncPostgresSaver (o fallback a MemorySaver)
    try:
        checkpointer = await get_postgres_checkpointer(pool=pool)
    except Exception as e:
        checkpointer = MemorySaver()

    # Inicializar almacén de memoria semántica a largo plazo
    embeddings = get_embedding_service()
    memory_store = UserMemoryStore(pool=pool, embedding_service=embeddings)

    # Compilar el grafo principal unificado
    app = build_main_graph(
        llm=llm,
        memory_store=memory_store,
        checkpointer=checkpointer,
    )

    current_user_id = user_id
    current_thread_id = f"cli-{uuid.uuid4().hex[:8]}"

    print_banner(
        model_name=settings.model_name,
        provider=settings.provider,
        user_id=current_user_id,
        thread_id=current_thread_id,
        debug=debug,
    )

    while True:
        try:
            prompt_label = f"{Colors.BOLD}{Colors.BLUE}[Usuario (ID: {current_user_id})] > {Colors.RESET}"
            user_input = input(prompt_label).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.CYAN}👋 Sesión finalizada. ¡Hasta luego!{Colors.RESET}")
            break

        if not user_input:
            continue

        # Comandos especiales
        cmd_lower = user_input.lower()
        if cmd_lower in ("/exit", "/quit", "salir", "exit", "quit"):
            print(f"\n{Colors.CYAN}👋 Cerrando Spark Agent y liberando recursos...{Colors.RESET}")
            break

        if cmd_lower == "/help":
            print_help()
            continue

        if cmd_lower == "/debug":
            debug = not debug
            status_str = f"{Colors.GREEN}ACTIVADO{Colors.RESET}" if debug else f"{Colors.DIM}desactivado{Colors.RESET}"
            print(f"🔧 Modo Debug ahora está {status_str}")
            continue

        if cmd_lower == "/new":
            current_thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            print(f"🔄 {Colors.GREEN}Nuevo hilo de conversación iniciado:{Colors.RESET} {Colors.DIM}{current_thread_id}{Colors.RESET}")
            continue

        if cmd_lower.startswith("/user"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                current_user_id = int(parts[1])
                print(f"👤 {Colors.GREEN}Usuario activo cambiado a ID:{Colors.RESET} {Colors.YELLOW}{current_user_id}{Colors.RESET}")
            else:
                print(f"{Colors.RED}Uso: /user <id_numérico> (ej: /user 5){Colors.RESET}")
            continue

        if cmd_lower == "/memory":
            mems = await memory_store.get_user_memories(current_user_id, limit=5)
            if mems:
                print(f"\n{Colors.BOLD}🧠 MEMORIA SEMÁNTICA (Usuario {current_user_id}):{Colors.RESET}")
                for m in mems:
                    date_str = str(m['created_at'])[:19]
                    print(f"  • {Colors.DIM}[{date_str}]{Colors.RESET} {m['content']}")
            else:
                print(f"\n{Colors.DIM}No hay memorias registradas para el usuario {current_user_id}.{Colors.RESET}")
            print()
            continue

        if cmd_lower in ("/info", "/status"):
            print(f"\n{Colors.BOLD}ℹ️ ESTADO DEL SISTEMA:{Colors.RESET}")
            print(f"  • Modelo LLM: {settings.model_name} ({settings.provider})")
            print(f"  • Usuario ID: {current_user_id}")
            print(f"  • Thread ID:  {current_thread_id}")
            print(f"  • Debug:      {'Sí' if debug else 'No'}")
            print(f"  • Odoo URL:   {odoo_settings.url} (DB: {odoo_settings.db})")
            print()
            continue

        # Ejecución de la consulta contra el grafo unificado
        config = {"configurable": {"thread_id": current_thread_id}}
        print(f"{Colors.DIM}⏳ Procesando con Spark Agent...{Colors.RESET}")
        t0 = time.perf_counter()

        try:
            await app.ainvoke(
                {
                    "raw_query": user_input,
                    "user_id": current_user_id,
                    "session_id": current_thread_id,
                    "messages": [HumanMessage(content=user_input)],
                },
                config=config,
            )
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error durante la ejecución del grafo: {e}{Colors.RESET}")
            continue

        # Verificar si se produjo una pausa por Human-in-the-Loop (interrupt)
        state = await app.aget_state(config)
        while state.tasks and any(t.interrupts for t in state.tasks):
            interrupt_payload = state.tasks[0].interrupts[0].value
            resume_result = await handle_interrupt(app, config, interrupt_payload, debug)
            if resume_result is None:
                # El usuario canceló la aclaración
                break
            state = await app.aget_state(config)

        t1 = time.perf_counter()

        # Extraer variables del estado unificado
        intent = state.values.get("intent")
        intent_reasoning = state.values.get("intent_reasoning")
        user_context = state.values.get("user_context")
        saved_memory_id = state.values.get("saved_memory_id")
        final_response = state.values.get("final_response")
        grouped_products = state.values.get("grouped_products", {})
        items = state.values.get("items", {})
        matched_skus = state.values.get("matched_skus", [])

        # Impresión de depuración si está activo
        if debug:
            print(f"\n{Colors.CYAN}{'-' * 30} DEBUG INFO ({t1 - t0:.2f}s) {'-' * 30}{Colors.RESET}")
            print(f"{Colors.BOLD}Intención detectada:{Colors.RESET} {intent} ({intent_reasoning})")
            if user_context:
                print(f"{Colors.BOLD}Contexto de memoria inyectado:{Colors.RESET}\n{user_context}")
            if saved_memory_id:
                print(f"{Colors.BOLD}Memoria persistida ID:{Colors.RESET} {saved_memory_id}")
            if matched_skus:
                print(f"{Colors.BOLD}SKUs sugeridos (RAG):{Colors.RESET} {matched_skus}")
            if items:
                print(f"{Colors.BOLD}Items cotizados:{Colors.RESET} {items}")
            if grouped_products:
                print(f"{Colors.BOLD}Grupos por partner:{Colors.RESET} {list(grouped_products.keys())}")
            print(f"{Colors.CYAN}{'-' * 72}{Colors.RESET}\n")

        # Imprimir respuesta al usuario
        print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 Spark Agent:{Colors.RESET}")
        if final_response:
            print(f"{final_response.strip()}\n")
        else:
            print(f"{Colors.DIM}(No se generó respuesta textual){Colors.RESET}\n")


async def main_async(user_id: int, debug: bool):
    try:
        await chat_loop(user_id=user_id, debug=debug)
    finally:
        # Garantizar cierre seguro del pool de conexiones PostgreSQL
        await close_db_pool()


def main():
    parser = argparse.ArgumentParser(description="Spark Agent Interactive Terminal Chat CLI")
    parser.add_argument("--user-id", type=int, default=5, help="ID de usuario simulado (default: 5)")
    parser.add_argument("--debug", action="store_true", help="Activar modo depuración")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(user_id=args.user_id, debug=args.debug))
    except KeyboardInterrupt:
        print("\n👋 Programa interrumpido por el usuario.")
        sys.exit(0)


if __name__ == "__main__":
    main()
