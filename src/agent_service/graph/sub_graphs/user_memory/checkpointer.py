import asyncio
import logging
from typing import Optional
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.agent_service.config.database import get_db_pool

logger = logging.getLogger(__name__)

_GLOBAL_CHECKPOINTER: Optional[AsyncPostgresSaver] = None
_SETUP_LOCK = asyncio.Lock()
_IS_SETUP: bool = False


async def get_postgres_checkpointer(
    pool: Optional[AsyncConnectionPool] = None,
) -> AsyncPostgresSaver:
    """Retorna una instancia singleton de AsyncPostgresSaver con tablas inicializadas."""
    global _GLOBAL_CHECKPOINTER, _IS_SETUP

    target_pool = pool if pool is not None else get_db_pool()
    if target_pool.closed:
        await target_pool.open()

    if _GLOBAL_CHECKPOINTER is None:
        _GLOBAL_CHECKPOINTER = AsyncPostgresSaver(target_pool)

    async with _SETUP_LOCK:
        if not _IS_SETUP:
            logger.info("Inicializando tablas de checkpoints de LangGraph en PostgreSQL...")
            await _GLOBAL_CHECKPOINTER.setup()
            _IS_SETUP = True

    return _GLOBAL_CHECKPOINTER
